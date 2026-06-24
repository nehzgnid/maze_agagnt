from __future__ import annotations

from agent_ppo.algorithm.algorithm import RLZooPPO
from agent_ppo.conf.conf import Config


def workflow(no_render: bool = False, n_timesteps: int | None = None, exp_id: int | None = Config.EVAL_EXP_ID) -> None:
    agent = RLZooPPO()
    agent.run(agent.enjoy_cmd(no_render=no_render, n_timesteps=n_timesteps, exp_id=exp_id))
