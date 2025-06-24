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

import numpy as np
import pysindy as ps
from pysindy.optimizers.base import BaseOptimizer
from pysindy.feature_library.base import BaseFeatureLibrary
from pysindy.feature_library import PolynomialLibrary, FourierLibrary, GeneralizedLibrary, CustomLibrary
from pysindy.differentiation import BaseDifferentiation

import pickle

class ureca_SINDy():
    def __init__(self, data = None, model = ps.SINDy(), env_name = "CartPole-v1"):
        self.model = model
        self.data = data
        self.env_name = env_name
        self.env = gym.make(self.env_name)

    def train_transition_model(
        self,
        optimizer = ps.STLSQ(threshold = 0.02),
        feature_library = None,
        differentiation_method = None,
        t_default = 1,
        discrete_time = True,
        feature_names = None
    ):
        def split_trajectories(df):
            trajectories = []
            trajectory = []

            for _, row in df.iterrows():
                trajectory.append(row.values)
                if row['done'] == 1:
                    trajectories.append(np.array(trajectory))
                    trajectory = []

            # Optional: if the last trajectory doesn't end with done==1
            if trajectory:
                trajectories.append(np.array(trajectory))

            return trajectories

        if self.data is None:
            with open(f'./dataset/random_trajectories_{self.env_name}.pkl', 'rb') as f:
                trajectories = pickle.load(f)
        else:
            trajectories = split_trajectories(pd.DataFrame(self.data))
            with open(f'./dataset/random_trajectories_{self.env_name}.pkl','wb') as f:
                pickle.dump(trajectories,f)
            
        print("Data loaded successfully (1/4)")

        states = []
        actions = []

        for traj in trajectories:
            # traj is a 2D array: rows are timesteps, columns are [state, action, next_state, done]
            states.append(traj[:, 0])   # column 0 = state
            actions.append(traj[:, 1])  # column 1 = action

        states = [np.stack(state) for state in states]
        actions = [np.stack(action).reshape(-1,1) for action in actions]

        print("Data processed successfully (2/4)")

        if type(feature_library) == list:
            feature_library = CustomLibrary(library_functions=feature_library).fit(states)
        
        self.model = ps.SINDy(
            optimizer=optimizer,
            feature_library=feature_library,
            differentiation_method=differentiation_method,
            feature_names=feature_names,
            t_default=t_default,
            discrete_time=discrete_time)

        print("Model created successfully (3/4)")

        self.model.fit(states, None, None, actions, True)
        self.model.print()

        print("Model fitted successfully (4/4)")

    def create_surrogate_env(self):
        return SindyMBRL(self.model, self.env, self.env_name)

class SindyMBRL(gym.Env):
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
        
        #Update state using SINDy
        state_tensor = torch.Tensor(self.model.predict(torch.Tensor(self.state).unsqueeze(0), torch.tensor(action).unsqueeze(0)))
        state = state_tensor.numpy()[0]
        self.state = state

        _, reward, terminated, truncated = self.original_env.step(action)
        self.terminated = terminated

        if self.terminated == True:
            if self.steps_beyond_terminated is None:
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