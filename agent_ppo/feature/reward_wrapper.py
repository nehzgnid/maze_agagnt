from __future__ import annotations

import gym
import numpy as np

from agent_ppo.conf.conf import Config

try:
    import gymnasium
except ImportError:
    gymnasium = gym

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


class AlignedMiniGridRewardWrapper(gym.Wrapper):
    """Explicit baseline reward, numerically aligned with MiniGrid.

    MiniGrid success reward:
        reward = 1 - 0.9 * step_count / max_steps

    Non-success transitions receive 0 from the original MiniGrid task. The
    scale and bias terms are exposed in Config but default to identity values.
    """

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            reward = self.reward(reward, terminated)
            return obs, reward, terminated, truncated, info

        obs, reward, done, info = step_out
        reward = self.reward(reward, done)
        return obs, reward, done, info

    def reward(self, reward, terminated):
        if terminated and reward > 0:
            step_count = getattr(self.unwrapped, "step_count", 0)
            max_steps = max(getattr(self.unwrapped, "max_steps", 1), 1)
            aligned_reward = Config.REWARD_SUCCESS_BASE - Config.REWARD_TIME_DECAY * (step_count / max_steps)
        else:
            aligned_reward = Config.REWARD_FAILURE

        return Config.REWARD_SCALE * aligned_reward + Config.REWARD_BIAS


class AlignedGymnasiumMiniGridRewardWrapper(gymnasium.Wrapper):
    """Gymnasium version of the explicit MiniGrid reward wrapper."""

    def step(self, action):
        step_out = self.env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            reward = self.reward(reward, terminated)
            return obs, reward, terminated, truncated, info

        obs, reward, done, info = step_out
        reward = self.reward(reward, done)
        return obs, reward, done, info

    def reward(self, reward, terminated):
        if terminated and reward > 0:
            step_count = getattr(self.unwrapped, "step_count", 0)
            max_steps = max(getattr(self.unwrapped, "max_steps", 1), 1)
            aligned_reward = Config.REWARD_SUCCESS_BASE - Config.REWARD_TIME_DECAY * (step_count / max_steps)
        else:
            aligned_reward = Config.REWARD_FAILURE

        return Config.REWARD_SCALE * aligned_reward + Config.REWARD_BIAS
