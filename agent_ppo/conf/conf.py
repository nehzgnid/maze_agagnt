from __future__ import annotations

from pathlib import Path


class Config:
    ROOT_DIR = Path(__file__).resolve().parents[2]

    ALGO = "ppo"
    ENV_ID = "MiniGrid-KeyCorridorS3R1-v0"
    ORG = "sb3"
    LOG_FOLDER = ROOT_DIR / "logs"
    EVAL_EXP_ID = 1
    PRETRAINED_EXP_ID = 1
    PRETRAINED_RUN_DIR = LOG_FOLDER / ALGO / f"{ENV_ID}_{PRETRAINED_EXP_ID}"
    PRETRAINED_MODEL_PATH = PRETRAINED_RUN_DIR / f"{ENV_ID}.zip"
    PRETRAINED_VECSNORM_PATH = PRETRAINED_RUN_DIR / ENV_ID / "vecnormalize.pkl"
    PRETRAINED_REFERENCE_SYSTEM = {
        "python": "3.9.12",
        "stable_baselines3": "1.8.0a11",
        "torch": "1.13.1+cu117",
        "numpy": "1.24.1",
        "gym": "0.21.0",
    }

    # Pretrained/RL Zoo training setup.
    POLICY = "MlpPolicy"
    ENV_WRAPPERS = [
        "agent_ppo.feature.reward_wrapper.AlignedMiniGridRewardWrapper",
        "agent_ppo.feature.preprocessor.LegacyFlatObsWrapper",
    ]
    N_TIMESTEPS = 500_000
    N_ENVS = 8
    N_STEPS = 128
    BATCH_SIZE = 64
    N_EPOCHS = 10
    LEARNING_RATE = 0.00025
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.0
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    NORMALIZE = True
    NORM_OBS = True
    NORM_REWARD = False
    TRAIN_SEED = 0
    EVAL_SEED = 10_000
    EVAL_FREQ_STEPS = 25_000
    N_EVAL_EPISODES = 20

    # Reward design, aligned with MiniGridEnv._reward():
    # success_reward = 1 - 0.9 * step_count / max_steps
    REWARD_SUCCESS_BASE = 1.0
    REWARD_TIME_DECAY = 0.9
    REWARD_FAILURE = 0.0
    REWARD_SCALE = 1.0
    REWARD_BIAS = 0.0

    # FlatObsWrapper layout used by the pretrained model.
    LOCAL_VIEW_SHAPE = (7, 7, 3)
    LOCAL_VIEW_DIM = 7 * 7 * 3
    MISSION_MAX_LEN = 96
    MISSION_CHAR_CODES = 27
    OBS_DIM = LOCAL_VIEW_DIM + MISSION_MAX_LEN * MISSION_CHAR_CODES
    ACTION_NUM = 7
    VALUE_NUM = 1

    # Stable-Baselines3 MlpPolicy defaults for PPO.
    ACTOR_HIDDEN_LAYERS = [64, 64]
    CRITIC_HIDDEN_LAYERS = [64, 64]
    ACTIVATION_FN = "nn.Tanh"
    ORTHO_INIT = True

    HYPERPARAMS_FILE = ROOT_DIR / "agent_ppo" / "conf" / "ppo_minigrid.yml"
    TRAIN_CONF_FILE = ROOT_DIR / "agent_ppo" / "conf" / "train_env_conf.toml"
