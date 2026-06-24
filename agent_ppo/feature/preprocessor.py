#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Feature preprocessing used by the MiniGrid PPO baseline."""

from functools import reduce
import operator

import gym
import numpy as np

from agent_ppo.conf.conf import Config

try:
    import gymnasium
except ImportError:
    gymnasium = gym


class LegacyFlatObsWrapper(gym.ObservationWrapper):
    """FlatObsWrapper aligned with the pretrained 2739D checkpoint.

    Current `gym_minigrid.wrappers.FlatObsWrapper` uses 28 character codes,
    producing 2835D observations. The pretrained checkpoint expects the older
    27-code mission encoder:

        7 * 7 * 3 + 96 * 27 = 2739
    """

    def __init__(self, env, maxStrLen=Config.MISSION_MAX_LEN):
        super().__init__(env)
        self.maxStrLen = maxStrLen
        self.numCharCodes = Config.MISSION_CHAR_CODES

        img_space = env.observation_space.spaces["image"]
        img_size = reduce(operator.mul, img_space.shape, 1)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(img_size + self.numCharCodes * self.maxStrLen,),
            dtype="uint8",
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"].lower()

        if mission != self.cachedStr:
            if len(mission) > self.maxStrLen:
                raise ValueError(f"mission string too long: {len(mission)}")
            str_array = np.zeros((self.maxStrLen, self.numCharCodes), dtype="float32")
            for idx, ch in enumerate(mission):
                if "a" <= ch <= "z":
                    ch_no = ord(ch) - ord("a")
                elif ch == " ":
                    ch_no = ord("z") - ord("a") + 1
                else:
                    raise ValueError(f"Unsupported mission character: {ch}")
                str_array[idx, ch_no] = 1
            self.cachedStr = mission
            self.cachedArray = str_array

        return np.concatenate((image.flatten(), self.cachedArray.flatten()))


class LegacyGymnasiumFlatObsWrapper(gymnasium.ObservationWrapper):
    """Gymnasium version of the pretrained 2739D FlatObs encoder."""

    def __init__(self, env, maxStrLen=Config.MISSION_MAX_LEN):
        super().__init__(env)
        self.maxStrLen = maxStrLen
        self.numCharCodes = Config.MISSION_CHAR_CODES

        img_space = env.observation_space.spaces["image"]
        img_size = reduce(operator.mul, img_space.shape, 1)
        self.observation_space = gymnasium.spaces.Box(
            low=0,
            high=255,
            shape=(img_size + self.numCharCodes * self.maxStrLen,),
            dtype="uint8",
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs["image"]
        mission = obs["mission"].lower()

        if mission != self.cachedStr:
            if len(mission) > self.maxStrLen:
                raise ValueError(f"mission string too long: {len(mission)}")
            str_array = np.zeros((self.maxStrLen, self.numCharCodes), dtype="float32")
            for idx, ch in enumerate(mission):
                if "a" <= ch <= "z":
                    ch_no = ord(ch) - ord("a")
                elif ch == " ":
                    ch_no = ord("z") - ord("a") + 1
                else:
                    raise ValueError(f"Unsupported mission character: {ch}")
                str_array[idx, ch_no] = 1
            self.cachedStr = mission
            self.cachedArray = str_array

        return np.concatenate((image.flatten(), self.cachedArray.flatten()))


class Preprocessor:
    """Document the exact FlatObs feature layout used by RL Zoo.

    The actual preprocessing is performed by `gym_minigrid.wrappers.FlatObsWrapper`
    during training. This class exists so the baseline exposes the input design
    in the same package location as the reference project.
    """

    local_view_shape = Config.LOCAL_VIEW_SHAPE
    local_view_dim = Config.LOCAL_VIEW_DIM
    mission_encoding_dim = Config.MISSION_MAX_LEN * Config.MISSION_CHAR_CODES
    feature_dim = Config.OBS_DIM

    def describe(self):
        return {
            "local_view_shape": self.local_view_shape,
            "local_view_dim": self.local_view_dim,
            "mission_encoding_dim": self.mission_encoding_dim,
            "feature_dim": self.feature_dim,
        }
