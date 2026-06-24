#!/usr/bin/env python3
from __future__ import annotations

import argparse

from agent_ppo.workflow.train_workflow import workflow


algorithm_name_list = ["ppo"]
algorithm_name = "ppo"


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniGrid PPO baseline.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true", help="Continue training from the downloaded pretrained model.")
    return parser.parse_args()


if __name__ == "__main__":
    if algorithm_name not in algorithm_name_list:
        raise ValueError(f"Unsupported algorithm_name: {algorithm_name}")
    args = parse_args()
    workflow(timesteps=args.timesteps, n_envs=args.n_envs, pretrained=args.pretrained)
