#!/usr/bin/env python3
from __future__ import annotations

import argparse

from agent_ppo.workflow.watch_workflow import workflow


def parse_args():
    parser = argparse.ArgumentParser(description="Watch/evaluate the MiniGrid PPO model.")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--n-timesteps", type=int, default=None)
    parser.add_argument("--exp-id", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    workflow(no_render=args.no_render, n_timesteps=args.n_timesteps, exp_id=args.exp_id)
