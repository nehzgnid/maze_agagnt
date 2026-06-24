#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Evaluate the MiniGrid PPO baseline without rendering."""

import argparse

from agent_ppo.workflow.evaluate_workflow import workflow


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the MiniGrid PPO baseline.")
    parser.add_argument("--exp-id", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--best", action="store_true", help="Evaluate best_model.zip with best_vecnormalize.pkl.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    workflow(exp_id=args.exp_id, episodes=args.episodes, use_best=args.best)
