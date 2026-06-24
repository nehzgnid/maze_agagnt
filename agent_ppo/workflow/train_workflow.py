from __future__ import annotations

from pathlib import Path

import gym
import gym_minigrid  # noqa: F401
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.preprocessor import LegacyFlatObsWrapper
from agent_ppo.feature.reward_wrapper import AlignedMiniGridRewardWrapper

try:
    import gymnasium
except ImportError:
    gymnasium = gym


class SaveBestVecNormalizeCallback(BaseCallback):
    """Save normalization stats that match the current best model."""

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(str(self.save_path))
        return True


def _next_run_id() -> int:
    algo_dir = Config.LOG_FOLDER / Config.ALGO
    algo_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{Config.ENV_ID}_"
    run_ids = []
    for item in algo_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            suffix = item.name[len(prefix) :]
            if suffix.isdigit():
                run_ids.append(int(suffix))
    return max(run_ids, default=0) + 1


def make_env():
    env = gym.make(Config.ENV_ID)
    env = AlignedMiniGridRewardWrapper(env)
    env = LegacyFlatObsWrapper(env)
    return env


def _patch_vec_spaces(env):
    env.action_space = gymnasium.spaces.Discrete(Config.ACTION_NUM)
    env.observation_space = gymnasium.spaces.Box(
        low=-env.clip_obs,
        high=env.clip_obs,
        shape=(Config.OBS_DIM,),
        dtype=np.float32,
    )
    return env


def workflow(timesteps: int | None = None, n_envs: int | None = None, pretrained: bool = False) -> None:
    total_timesteps = int(timesteps or Config.N_TIMESTEPS)
    num_envs = int(n_envs or Config.N_ENVS)

    run_id = _next_run_id()
    run_dir = Config.LOG_FOLDER / Config.ALGO / f"{Config.ENV_ID}_{run_id}"
    model_stats_dir = run_dir / Config.ENV_ID
    model_stats_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(make_env, n_envs=num_envs, seed=Config.TRAIN_SEED)
    if pretrained:
        if not Config.PRETRAINED_VECSNORM_PATH.exists():
            raise FileNotFoundError(f"Missing pretrained VecNormalize stats: {Config.PRETRAINED_VECSNORM_PATH}")
        env = VecNormalize.load(str(Config.PRETRAINED_VECSNORM_PATH), env)
        env.training = True
        env.norm_reward = Config.NORM_REWARD
    else:
        env = VecNormalize(env, norm_obs=Config.NORM_OBS, norm_reward=Config.NORM_REWARD)
    env = _patch_vec_spaces(env)

    eval_env = make_vec_env(make_env, n_envs=1, seed=Config.EVAL_SEED)
    eval_env = VecNormalize(eval_env, norm_obs=Config.NORM_OBS, norm_reward=Config.NORM_REWARD, training=False)
    eval_env = _patch_vec_spaces(eval_env)

    pretrained_path = Config.PRETRAINED_MODEL_PATH if pretrained else None
    agent = Algorithm(env, pretrained_model_path=pretrained_path)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(Config.EVAL_FREQ_STEPS // num_envs, 1),
        n_eval_episodes=Config.N_EVAL_EPISODES,
        deterministic=True,
        callback_on_new_best=SaveBestVecNormalizeCallback(model_stats_dir / "best_vecnormalize.pkl"),
    )

    agent.learn(total_timesteps, callback=eval_callback, progress_bar=True)
    model_path = run_dir / f"{Config.ENV_ID}.zip"
    agent.save(model_path)
    env.save(str(model_stats_dir / "vecnormalize.pkl"))

    env.close()
    eval_env.close()
    print(f"saved: {model_path.resolve()}")
    print(f"exp_id: {run_id}")
