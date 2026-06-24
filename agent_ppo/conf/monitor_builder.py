from __future__ import annotations

from pathlib import Path

from agent_ppo.conf.conf import Config


def ensure_log_folder() -> Path:
    Config.LOG_FOLDER.mkdir(parents=True, exist_ok=True)
    return Config.LOG_FOLDER

