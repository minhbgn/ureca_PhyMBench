import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import torch.optim as optim

# Image datasets and image manipulation
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
import pickle

from kan import *

class ureca_KAN():
    def __init__(self, data = None, env_name = 'CartPole-v1', hidden_sizes = None,
                 # Check pykan for the meaning of these parameters 
                grid: int = 3,
                k: int = 3,
                mult_arity = 2,
                noise_scale: float = 0.3,
                scale_base_mu: float = 0.0,
                scale_base_sigma=1.0, 
                base_fun='silu', 
                symbolic_enabled=True, 
                affine_trainable=False, 
                grid_eps=0.02, 
                grid_range=[-1, 1], 
                sp_trainable=True, 
                sb_trainable=True, 
                seed=1, 
                save_act=True, 
                sparse_init=False, 
                auto_save=True, 
                first_init=True, 
                ckpt_path='./model', 
                state_id=0, 
                round = 0):


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data
        if self.data is None:
            try:
                with open(f'./dataset/transition_data_{env_name}.pkl','rb') as file:
                    self.data = pickle.load(file)
            except:
                raise FileNotFoundError("Transition data not found")
        self.env_name = env_name
        self.env = gym.make(self.env_name)

        if type(self.data['action'][0]) == int:
            in_size = len(self.data['state'][0]) + 1
        else:
            in_size = len(self.data['state'][0]) + len(self.data['action'][0])
        out_size = len(self.data['next_state'][0])
        
        self.model = KAN(
            width = [in_size, *hidden_sizes, out_size],
            grid = grid,
            k = k,
            mult_arity = mult_arity,
            noise_scale = noise_scale,
            scale_base_mu = scale_base_mu,
            scale_base_sigma = scale_base_sigma,
            base_fun=base_fun,
            symbolic_enabled=symbolic_enabled,
            affine_trainable=affine_trainable,
            grid_eps=grid_eps,
            grid_range=grid_range,
            sp_trainable=sp_trainable,
            sb_trainable=sb_trainable,
            seed=seed,
            save_act=save_act,
            sparse_init=sparse_init,
            auto_save=auto_save,
            first_init=first_init,
            ckpt_path=ckpt_path,
            state_id=state_id,
            round = round,
            device = self.device
        )

        print(self.model)


    def train_transition_model(self, epochs = 50, optimizer = "LBFGS", lamb = 0.002, lamb_entropy = 2, log=1, lamb_l1=1., lamb_coef=0., 
                                lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1.,
                                start_grid_update_step=-1, stop_grid_update_step=50, batch=-1,
                                metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, 
                                img_folder='./video', singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n', display_metrics=None):
        self.optimizer = optimizer
        
        transition_data_pd = pd.DataFrame(self.data)
        def concat_state_action(row):
            return np.append(row["state"], row["action"])

        X = torch.FloatTensor(np.stack(transition_data_pd[['state', 'action']].apply(concat_state_action, axis = 1).values))
        y = torch.FloatTensor(np.stack(transition_data_pd['next_state'].values))

        def getNewData(X = X, y = y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.FloatTensor(y_train)
            y_test = torch.FloatTensor(y_test)
            return X_train, X_test, y_train, y_test
        
        X_train, X_test, y_train, y_test = getNewData(X, y)
        dataset = {}
        dataset['train_input'] = X_train
        dataset['test_input'] = X_test
        dataset['train_label'] = y_train
        dataset['test_label'] = y_test
        self.dataset = dataset

        print("Dataset processed (1/2)")

        self.model.fit(
                dataset, opt=optimizer, steps=epochs, log=log, lamb=lamb, lamb_l1=lamb_l1, lamb_entropy=lamb_entropy, 
                lamb_coef=lamb_coef, lamb_coefdiff=lamb_coefdiff, update_grid=update_grid, grid_update_num=grid_update_num, loss_fn=loss_fn,
                lr=lr, start_grid_update_step=start_grid_update_step, stop_grid_update_step=stop_grid_update_step, batch=batch,
                metrics=metrics, save_fig=save_fig, in_vars=in_vars, out_vars=out_vars, beta=beta, save_fig_freq=save_fig_freq,
                img_folder=img_folder, singularity_avoiding=singularity_avoiding, y_th=y_th, reg_metric=reg_metric,
                display_metrics=display_metrics
            )
        print("Model fitted (2/2)")

    def refine_transition_model(self, epochs = 50, new_grid = 10):
        self.model = self.model.prune()
        print("Pruning model (1/3)")
        self.model.fit(self.dataset, opt="LBFGS", steps=epochs)
        self.model.refine(new_grid)
        print("Refining model (2/3)")
        self.model.fit(self.dataset, opt="LBFGS", steps=epochs)
        lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
        self.model.auto_symbolic(lib=lib)
        print("Symbolic regressing model (3/3)")
        self.model.fit(self.dataset, opt="LBFGS", steps=epochs)

    
    def create_surrogate_env(self):
        return KANMBRL(self.model, self.env, self.env_name)

class KANMBRL(gym.Env):
    def __init__(self, model, original_env, env_name):
        super().__init__()
        self.model = model
        self.original_env = original_env
        self.env_name = env_name
        self.action_space = self.original_env.action_space
        self.observation_space = self.original_env.observation_space

        if len(self.original_env.reset()) == 2:
            self.state = self.original_env.reset()[0]
            print(self.state)
        else:
            self.state = self.original_env.reset()
            print(self.state)
        
        self.prev_shaping = None
        self.terminated = False
        self.steps_beyond_terminated = None
        self.time = 0
        self.time_limit = 500

    def step(self, action):

        if self.terminated:
            print(
                "You are calling 'step()' even though this "
                "environment has already returned terminated = True. You "
                "should always call 'reset()' once you receive 'terminated = "
                "True' -- any further steps are undefined behavior."
            )
            self.steps_beyond_terminated += 1
            reward = 0.0
            return np.array(self.state, dtype=np.float32), 0, 1, {}

        self.terminated = False
        self.time += 1

        state = self.state
        self.set_original_env_state(state)

        if not isinstance(state, np.ndarray):
            state = state.detach().numpy()
        state_action = np.append(state, action)
        state_action_tensor = torch.FloatTensor(state_action).unsqueeze(0)
        
        self.state = self.model(state_action_tensor).detach().numpy()[0]

        _, reward, terminated, truncated = self.original_env.step(action)
        self.terminated = terminated
        
        if self.terminated == True and self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            return np.array(state, dtype=np.float32), 0, terminated, {}

        if self.time > self.time_limit:
            terminated = True
            reward = 1.0

        return np.array(state, dtype=np.float32), reward, terminated, {}

    def reset(self):
        if len(self.original_env.reset()) == 2:
            self.state = self.original_env.reset()[0]
        else:
            self.state = self.original_env.reset()
        self.terminated = False
        self.prev_shaping = None
        self.time = 0
        return self.state

    def render(self, mode='human'):
        return self.original_env.render(mode)

    def set_original_env_state(self, state):
        if self.env_name == 'LunarLander-v3':
            def set_lunarlander_state(env, lander_state):
                """
                Set LunarLander's lander state and reposition legs accordingly.
            
                Args:
                    env: The LunarLander-v2 environment.
                    lander_state: tuple of
                        (x, y, x_dot, y_dot, angle, angle_dot)
                """
                x, y, x_dot, y_dot, angle, angle_dot = lander_state
            
                lander = env.unwrapped.lander
                legs = env.unwrapped.legs
            
                # Determine relative positions of legs at reset
                # First call: cache initial offsets
                if not hasattr(env.unwrapped, "_leg_offsets"):
                    env.unwrapped._leg_offsets = []
                    for leg in legs:
                        # Offset from lander position at reset
                        offset = leg.position - lander.position
                        env.unwrapped._leg_offsets.append(offset)
            
                # Set lander position, velocity, angle
                lander.position = (x, y)
                lander.linearVelocity = (x_dot, y_dot)
                lander.angle = angle
                lander.angularVelocity = angle_dot
            
                # Update legs based on relative offset and lander angle
                for i, leg in enumerate(legs):
                    local_offset = env.unwrapped._leg_offsets[i]
            
                    # Rotate offset by current lander angle
                    c, s = np.cos(angle), np.sin(angle)
                    rotated_offset = (
                        local_offset[0] * c - local_offset[1] * s,
                        local_offset[0] * s + local_offset[1] * c
                    )
            
                    # Update leg position, angle, velocity
                    leg.position = (x + rotated_offset[0], y + rotated_offset[1])
                    leg.angle = angle
                    leg.linearVelocity = (x_dot, y_dot)
                    leg.angularVelocity = angle_dot
            
                    leg.ground_contact = False
            set_lunarlander_state(self.env, state)
        else:
            self.original_env.unwrapped.state = state