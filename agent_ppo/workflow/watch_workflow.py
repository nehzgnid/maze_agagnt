from __future__ import annotations

import time

import numpy as np
from stable_baselines3 import PPO

from agent_ppo.conf.conf import Config
from agent_ppo.workflow.evaluate_workflow import ObsNormalizer, _make_env, _model_path, _vecnormalize_path


def workflow(
    no_render: bool = False,
    n_timesteps: int | None = None,
    exp_id: int | None = Config.EVAL_EXP_ID,
) -> None:
    total_steps = int(n_timesteps or 1000)
    run_id = int(exp_id or Config.EVAL_EXP_ID)
    model_path = _model_path(run_id)
    normalizer = ObsNormalizer(_vecnormalize_path(run_id))
    model = PPO.load(
        str(model_path),
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
        device="cpu",
    )

    env = _make_env()
    try:
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        episode_reward = 0.0
        episode_len = 0
        episode = 1

        print(f"model: {model_path}")
        print(f"timesteps: {total_steps}")

        for _ in range(total_steps):
            if not no_render:
                env.render()
                time.sleep(0.03)

            norm_obs = normalizer.normalize(obs)[None, :]
            action, _ = model.predict(norm_obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, _info = step_out

            episode_reward += float(reward)
            episode_len += 1

            if done:
                print(f"episode {episode}: reward={episode_reward:.3f}, length={episode_len}")
                reset_out = env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                episode_reward = 0.0
                episode_len = 0
                episode += 1
    finally:
        env.close()
