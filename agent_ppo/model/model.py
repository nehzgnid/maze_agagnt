from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    """Explicit MLP actor-critic structure aligned with SB3 MlpPolicy."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "minigrid_keycorridor_ppo"
        self.device = device

        self.actor_backbone = nn.Sequential(
            _make_fc(Config.OBS_DIM, Config.ACTOR_HIDDEN_LAYERS[0]),
            nn.Tanh(),
            _make_fc(Config.ACTOR_HIDDEN_LAYERS[0], Config.ACTOR_HIDDEN_LAYERS[1]),
            nn.Tanh(),
        )
        self.critic_backbone = nn.Sequential(
            _make_fc(Config.OBS_DIM, Config.CRITIC_HIDDEN_LAYERS[0]),
            nn.Tanh(),
            _make_fc(Config.CRITIC_HIDDEN_LAYERS[0], Config.CRITIC_HIDDEN_LAYERS[1]),
            nn.Tanh(),
        )
        self.actor_head = _make_fc(Config.ACTOR_HIDDEN_LAYERS[-1], Config.ACTION_NUM, gain=0.01)
        self.critic_head = _make_fc(Config.CRITIC_HIDDEN_LAYERS[-1], Config.VALUE_NUM, gain=1.0)

    def forward(self, s, inference=False):
        x = s.to(torch.float32)
        logits = self.actor_head(self.actor_backbone(x))
        value = self.critic_head(self.critic_backbone(x))
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


@dataclass(frozen=True)
class MiniGridPPOModelSpec:
    """Architecture spec for the RL Zoo MiniGrid PPO checkpoint.

    The actual neural network is Stable-Baselines3's ActorCriticPolicy
    selected by `MlpPolicy`. We keep the spec here so all model-level decisions
    live inside `agent_ppo/model`, matching the competition project layout.
    """

    policy: str = Config.POLICY
    observation_dim: int = Config.OBS_DIM
    local_view_dim: int = Config.LOCAL_VIEW_DIM
    mission_encoding_dim: int = Config.MISSION_MAX_LEN * Config.MISSION_CHAR_CODES
    action_num: int = Config.ACTION_NUM
    value_num: int = Config.VALUE_NUM
    actor_hidden_layers: tuple[int, int] = tuple(Config.ACTOR_HIDDEN_LAYERS)
    critic_hidden_layers: tuple[int, int] = tuple(Config.CRITIC_HIDDEN_LAYERS)

    def describe(self) -> str:
        return (
            f"Policy: {self.policy}\n"
            f"Input: {self.observation_dim}D flat observation "
            f"({self.local_view_dim}D local view + {self.mission_encoding_dim}D mission text)\n"
            f"Actor MLP: {self.actor_hidden_layers} -> {self.action_num} action logits\n"
            f"Critic MLP: {self.critic_hidden_layers} -> {self.value_num} state value\n"
        )
