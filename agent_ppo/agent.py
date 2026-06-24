#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""MiniGrid PPO baseline agent facade."""

from agent_ppo.algorithm.algorithm import RLZooPPO
from agent_ppo.conf.conf import Config
from agent_ppo.model.model import MiniGridPPOModelSpec


class Agent:
    """Facade matching the competition package style.

    Training and evaluation are delegated to RL Baselines3 Zoo so the baseline
    remains aligned with the verified pretrained MiniGrid PPO workflow.
    """

    def __init__(self):
        self.config = Config
        self.model_spec = MiniGridPPOModelSpec()
        self.algorithm = RLZooPPO()

    def train(self):
        self.algorithm.run(self.algorithm.train_cmd())

    def watch(self, no_render=False, n_timesteps=None):
        from agent_ppo.workflow.watch_workflow import workflow

        workflow(no_render=no_render, n_timesteps=n_timesteps, exp_id=Config.EVAL_EXP_ID)

    def describe(self):
        return self.model_spec.describe()
