import matplotlib.pyplot as plt
plt.subplot()

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

import pickle

class ureca_PINN():
    def __init__(self, data = None, env_name = 'CartPole-v1', hidden_sizes = None, 
                activation = nn.ReLU, output_activation = None):
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
        
        self.layers = []
        prev_size = in_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(activation())
            prev_size = size
        self.layers.append(nn.Linear(prev_size, out_size))
        if output_activation is not None:
            self.layers.append(output_activation())
        self.model = nn.Sequential(*self.layers)

        print(self.model.parameters)

    def total_loss_fn(self, state, action, pred_next_state, next_state, criterion, physics_loss_fn, alpha = 1):
        # Prediction loss (F1)
        f1 = criterion(pred_next_state, next_state)

        # Physics-informed loss (F2)
        f2 = physics_loss_fn(state, action, pred_next_state)

        return f1 + alpha * f2


    def train_transition_model(self, epochs = 1000, optimizer = None, physics_loss_fn = lambda x : 0, criterion = nn.MSELoss(), physics_loss_weight = 1):
        self.optimizer = optimizer
        self.criterion = criterion
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        transition_data_pd = pd.DataFrame(self.data)
        def concat_state_action(row):
            return np.append(row["state"], row["action"])

        X = torch.FloatTensor(np.stack(transition_data_pd[['state', 'action']].apply(concat_state_action, axis = 1).values))
        y = torch.FloatTensor(np.stack(transition_data_pd['next_state'].values))

        losses = []
        for i in range(epochs):
            y_pred = self.model.forward(X)
            loss = self.total_loss_fn(state = X[:,:-1], action = X[:,-1:], pred_next_state = y_pred, next_state = y, 
                                    criterion = self.criterion, physics_loss_fn = physics_loss_fn, alpha = physics_loss_weight)
            losses.append(loss.detach().numpy())

            if i % 10 == 0:
                print(f'\repoch: {i:2}/{epochs}  loss: {losses[i]}',end='',flush=True)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        plt.plot(range(epochs), losses, label = "Train Loss", color = "blue")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
    
    def create_surrogate_env(self):
        return PINNMBRL(self.model, self.env, self.env_name)

class PINNMBRL(gym.Env):
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
        state_action = np.append(state, action)
        state_action_tensor = torch.FloatTensor(state_action)
        self.set_original_env_state(state)
        
        #Update state using PINN
        with torch.no_grad():
            state_tensor = self.model.forward(state_action_tensor)
            state = state_tensor.numpy()
            self.state = state

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
        if self.env_name == 'CartPole-v1':
            self.original_env.unwrapped.state = state
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
            set_lunarlander_state(env, state)