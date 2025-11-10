"""

### NOTICE ###
You DO NOT need to upload this file

"""
import gymnasium as gym
import ale_py
import numpy as np
from atari_wrapper import make_wrap_atari

class Environment(object):
    def __init__(self, env_name, args, atari_wrapper=False, test=False, render_mode=None):
        if atari_wrapper:
            clip_rewards = not test
            self.env = make_wrap_atari(env_name, clip_rewards, render_mode=render_mode)
        else:
            self.env = gym.make(env_name, render_mode=render_mode)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def seed(self, seed: int):
        # Seed Python / NumPy / Torch for full reproducibility
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        # New-style (Gym >=0.21 / Gymnasium) — seeding via reset()
        try:
            # ignore returned value which may be obs or (obs, info)
            self.env.reset(seed=seed)
        except TypeError:
            # Older Gym fallback (if present)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        # Also seed spaces when available
        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(seed)
        if hasattr(self.env, "observation_space") and hasattr(self.env.observation_space, "seed"):
            self.env.observation_space.seed(seed)
        '''
        Control the randomness of the environment
        '''

    def reset(self):
        '''
        When running dqn:
            observation: np.array
                stack 4 last frames, shape: (84, 84, 4)
        '''
        observation, _ = self.env.reset()

        return np.array(observation)


    def step(self,action):
        '''
        When running dqn:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            reward: int
                wrapper clips the reward to {-1, 0, 1} by its sign
                we don't clip the reward when testing
            done: bool
                whether reach the end of the episode?
        '''
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        observation, reward, done, truncated, info = self.env.step(action)

        return np.array(observation), reward, done, truncated, info


    def get_action_space(self):
        return self.action_space


    def get_observation_space(self):
        return self.observation_space


    def get_random_action(self):
        return self.action_space.sample()

    def close(self):
        '''
        close
        '''
        self.env.close()
