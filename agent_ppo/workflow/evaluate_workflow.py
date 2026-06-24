from __future__ import annotations

import pickle
from pathlib import Path

import gym
import gym_minigrid  # noqa: F401
import numpy as np
from stable_baselines3 import PPO

from agent_ppo.conf.conf import Config
from agent_ppo.feature.preprocessor import LegacyFlatObsWrapper
from agent_ppo.feature.reward_wrapper import AlignedMiniGridRewardWrapper

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


def _run_dir(exp_id: int) -> Path:
    return Config.LOG_FOLDER / Config.ALGO / f"{Config.ENV_ID}_{exp_id}"


def _model_path(exp_id: int, use_best: bool = False) -> Path:
    run_dir = _run_dir(exp_id)
    if use_best:
        best_model = run_dir / "best_model.zip"
        if best_model.exists():
            return best_model
        raise FileNotFoundError(f"No best_model.zip found in {run_dir}")

    model_path = run_dir / f"{Config.ENV_ID}.zip"
    if model_path.exists():
        return model_path
    best_model = run_dir / "best_model.zip"
    if best_model.exists():
        return best_model
    raise FileNotFoundError(f"No model zip found in {run_dir}")


def _vecnormalize_path(exp_id: int, use_best: bool = False) -> Path:
    stats_dir = _run_dir(exp_id) / Config.ENV_ID
    if use_best:
        best_stats = stats_dir / "best_vecnormalize.pkl"
        if best_stats.exists():
            return best_stats
    return stats_dir / "vecnormalize.pkl"


def _make_env():
    env = gym.make(Config.ENV_ID)
    env = AlignedMiniGridRewardWrapper(env)
    env = LegacyFlatObsWrapper(env)
    return env


class ObsNormalizer:
    def __init__(self, path: Path):
        with path.open("rb") as f:
            vecnormalize = pickle.load(f)
        self.obs_rms = vecnormalize.obs_rms
        self.clip_obs = vecnormalize.clip_obs
        self.epsilon = vecnormalize.epsilon

    def normalize(self, obs):
        obs = obs.astype(np.float32)
        obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(obs, -self.clip_obs, self.clip_obs).astype(np.float32)


def evaluate(exp_id: int = Config.EVAL_EXP_ID, episodes: int = 20, max_steps: int = 1000, use_best: bool = False):
    model_path = _model_path(exp_id, use_best=use_best)
    normalizer = ObsNormalizer(_vecnormalize_path(exp_id, use_best=use_best))
    model = PPO.load(
        str(model_path),
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
        device="cpu",
    )

    rewards = []
    successes = 0
    lengths = []

    for episode in range(episodes):
        env = _make_env()
        try:
            reset_out = env.reset(seed=episode)
        except TypeError:
            env.seed(episode)
            reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            norm_obs = normalizer.normalize(obs)[None, :]
            action, _ = model.predict(norm_obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out
                terminated = done
            total_reward += float(reward)
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        successes += int(total_reward > 0.0)
        env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    success_rate = successes / episodes
    mean_length = float(np.mean(lengths))

    print(f"model: {model_path}")
    print(f"episodes: {episodes}")
    print(f"mean_reward: {mean_reward:.3f} +/- {std_reward:.3f}")
    print(f"success_rate: {success_rate:.3f}")
    print(f"mean_length: {mean_length:.1f}")
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "mean_length": mean_length,
    }


def workflow(exp_id: int = Config.EVAL_EXP_ID, episodes: int = 20, use_best: bool = False):
    return evaluate(exp_id=exp_id, episodes=episodes, use_best=use_best)
