from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO

from agent_ppo.conf.conf import Config


class RLZooPPO:
    """Thin command wrapper around the official RL Baselines3 Zoo workflow."""

    def __init__(
        self,
        algo: str = Config.ALGO,
        env_id: str = Config.ENV_ID,
        log_folder: str | Path = Config.LOG_FOLDER,
        hyperparams_file: str | Path = Config.HYPERPARAMS_FILE,
    ):
        self.algo = algo
        self.env_id = env_id
        self.log_folder = Path(log_folder)
        self.hyperparams_file = Path(hyperparams_file)

    def train_cmd(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "rl_zoo3.train",
            "--algo",
            self.algo,
            "--env",
            self.env_id,
            "-f",
            str(self.log_folder),
            "--conf-file",
            str(self.hyperparams_file),
        ]

    def enjoy_cmd(
        self,
        no_render: bool = False,
        n_timesteps: int | None = None,
        exp_id: int | None = Config.EVAL_EXP_ID,
    ) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "rl_zoo3.enjoy",
            "--algo",
            self.algo,
            "--env",
            self.env_id,
            "-f",
            str(self.log_folder),
        ]
        if exp_id is not None:
            cmd.extend(["--exp-id", str(exp_id)])
        if no_render:
            cmd.append("--no-render")
        if n_timesteps is not None:
            cmd.extend(["--n-timesteps", str(n_timesteps)])
        return cmd

    def run(self, cmd: list[str]) -> None:
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=Config.ROOT_DIR, check=True)


class Algorithm:
    """Local SB3 PPO trainer aligned with the pretrained RL Zoo setup."""

    def __init__(self, env, verbose: int = 1, pretrained_model_path: str | Path | None = None):
        if pretrained_model_path is not None and Path(pretrained_model_path).exists():
            self.model = PPO.load(
                str(pretrained_model_path),
                env=env,
                custom_objects={
                    "learning_rate": Config.LEARNING_RATE,
                    "lr_schedule": lambda _: Config.LEARNING_RATE,
                    "clip_range": lambda _: Config.CLIP_RANGE,
                    "observation_space": env.observation_space,
                    "action_space": env.action_space,
                },
                device="auto",
            )
            self.model.verbose = verbose
        else:
            self.model = self._create_model(env, verbose=verbose)

    def _create_model(self, env, verbose: int = 1):
        return PPO(
            Config.POLICY,
            env,
            verbose=verbose,
            n_steps=Config.N_STEPS,
            batch_size=Config.BATCH_SIZE,
            n_epochs=Config.N_EPOCHS,
            learning_rate=Config.LEARNING_RATE,
            gamma=Config.GAMMA,
            gae_lambda=Config.GAE_LAMBDA,
            clip_range=Config.CLIP_RANGE,
            ent_coef=Config.ENT_COEF,
            vf_coef=Config.VF_COEF,
            max_grad_norm=Config.MAX_GRAD_NORM,
            policy_kwargs={
                "net_arch": {
                    "pi": Config.ACTOR_HIDDEN_LAYERS,
                    "vf": Config.CRITIC_HIDDEN_LAYERS,
                },
                "activation_fn": nn.Tanh,
                "ortho_init": Config.ORTHO_INIT,
            },
            tensorboard_log=str(Config.ROOT_DIR / "runs"),
            device="auto",
        )

    def learn(self, total_timesteps: int, callback=None, progress_bar: bool = True):
        return self.model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=progress_bar)

    def save(self, path: str | Path):
        self.model.save(str(path))
