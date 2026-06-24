# MiniGrid PPO 二次开发测试工程

本项目保留了 PPO 二次开发和基础测试所需的最小代码。当前实现不是手写 PPO 训练框架，而是基于
Stable-Baselines3 的 PPO，并参考 RL Baselines3 Zoo 的训练配置组织代码。

适合的开发方向：

- 调整 PPO 超参数
- 修改奖励函数
- 修改观测特征处理
- 修改 actor-critic 网络结构
- 做快速训练、评估和观看测试
- 在 MiniGrid 环境上做轻量实验

不太适合的开发方向：

- 深入改写 PPO 损失函数内部细节
- 自己维护 rollout buffer、GAE、policy loss、value loss 等底层算法流程
- 直接套回原始 Robot Vacuum 竞赛框架

如果需要改 PPO 算法底层，建议换成手写 PPO 模板；如果主要是调参、改 reward、改特征和跑实验，当前工程更省心。

## 目录结构

```text
agent_ppo/
  agent.py                     简单 Agent 门面，提供训练和观看接口
  algorithm/
    algorithm.py               Stable-Baselines3 PPO 构建与 RL Zoo 命令包装
  conf/
    conf.py                    环境、模型、PPO 训练参数
    ppo_minigrid.yml           RL Zoo 风格 PPO 超参数配置
    train_env_conf.toml        训练环境配置记录
    monitor_builder.py         监控指标构建
  feature/
    preprocessor.py            MiniGrid 观测编码与 legacy flat obs 兼容处理
    reward_wrapper.py          奖励包装器，可在这里改 reward
    definition.py              特征定义
  model/
    model.py                   actor-critic 网络结构参考
  workflow/
    train_workflow.py          训练流程、VecNormalize、EvalCallback、模型保存
    evaluate_workflow.py       无渲染评估流程
    watch_workflow.py          观看或 no-render 运行流程

train_test.py                  训练入口
evaluate_agent.py              评估入口
watch_agent.py                 观看入口
environment.yml                环境依赖参考
requirements.txt               pip 依赖文件
README.md                      项目说明
```

## 环境配置

建议使用 conda 创建独立虚拟环境。项目默认安装 CPU 版 PyTorch，适合没有 NVIDIA GPU 或只是入门学习的同学。

### 方式一：使用 environment.yml

项目提供了 `environment.yml`，可以直接创建 conda 环境：

```powershell
conda env create -f environment.yml
conda activate rl-maze
```

如果创建环境后发现没有安装 torch，可以在 `rl-maze` 环境中继续安装 CPU 版 PyTorch：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 方式二：手动创建 conda 环境

也可以手动创建环境，然后安装依赖：

```powershell
conda create -n rl-maze python=3.11 -y
conda activate rl-maze
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

如果要使用 GPU，请不要使用上面的 CPU 版 torch 命令。请根据自己的 CUDA 版本，到 PyTorch 官网选择对应安装命令。安装 GPU 版 PyTorch 后，再执行：

```powershell
pip install -r requirements.txt
```

### 检查环境

安装完成后，可以运行：

```powershell
python -c "import torch; import stable_baselines3; import gym; import gym_minigrid; print('ok')"
```

如果输出 `ok`，说明基础依赖已经可用。

## 模型介绍

当前 PPO 目标环境是：

```text
MiniGrid-KeyCorridorS3R1-v0
```

观测输入采用 MiniGrid 的扁平化观测形式，整体维度为：

```text
7 x 7 x 3 局部视野 + mission 文本 one-hot 编码 = 2739 维
```

动作空间为离散动作：

```text
ACTION_NUM = 7
```

策略使用 actor-critic 结构：

```text
输入 2739D flat observation
  ├─ actor MLP  -> 7 个动作 logits
  └─ critic MLP -> 1 个状态价值 value
```

当前模型结构参考在：

```text
agent_ppo/model/model.py
```

训练时实际使用的是 Stable-Baselines3 的 `PPO` 和 `MlpPolicy`。网络层数、隐藏层大小、激活函数等参数主要由
`agent_ppo/conf/conf.py` 和 `agent_ppo/algorithm/algorithm.py` 中传入的 `policy_kwargs` 控制。

默认 actor 和 critic 隐藏层为：

```text
actor:  [64, 64]
critic: [64, 64]
activation: Tanh
```

## 快速运行

快速训练测试：

```powershell
python train_test.py --timesteps 1024 --n-envs 2
```

完整训练会使用 `conf.py` 中的默认步数：

```powershell
python train_test.py
```

评估某个训练结果：

```powershell
python evaluate_agent.py --exp-id 1 --episodes 20
```

评估 best model：

```powershell
python evaluate_agent.py --exp-id 1 --episodes 20 --best
```

观看运行效果：

```powershell
python watch_agent.py --exp-id 1
```

无渲染运行：

```powershell
python watch_agent.py --exp-id 1 --no-render --n-timesteps 5000
```

## 二次开发建议

### 1. 改 PPO 超参数

优先查看：

```text
agent_ppo/conf/conf.py
agent_ppo/conf/ppo_minigrid.yml
```

常改参数包括：

```text
N_TIMESTEPS
N_ENVS
N_STEPS
BATCH_SIZE
N_EPOCHS
LEARNING_RATE
GAMMA
GAE_LAMBDA
CLIP_RANGE
ENT_COEF
VF_COEF
MAX_GRAD_NORM
```

### 2. 改奖励函数

奖励逻辑集中在：

```text
agent_ppo/feature/reward_wrapper.py
```

适合在这里尝试：

- 成功奖励缩放
- 失败惩罚
- 步数惩罚
- 探索奖励
- 距离目标的 shaping reward

改奖励后建议先小步数训练，例如：

```powershell
python train_test.py --timesteps 4096 --n-envs 2
```

### 3. 改观测特征

观测处理集中在：

```text
agent_ppo/feature/preprocessor.py
```

这里负责将 MiniGrid 原始观测转换为模型可用的 flat observation。修改这里时要注意：

- 输入维度需要和 `Config.OBS_DIM` 对齐
- 如果改变观测维度，需要同步修改模型输入维度
- 如果使用已有 checkpoint，观测维度不能随意改变

### 4. 改模型结构

模型结构参考在：

```text
agent_ppo/model/model.py
```

SB3 PPO 的实际网络参数在：

```text
agent_ppo/algorithm/algorithm.py
```

重点关注：

```python
policy_kwargs={
    "net_arch": {
        "pi": Config.ACTOR_HIDDEN_LAYERS,
        "vf": Config.CRITIC_HIDDEN_LAYERS,
    },
    "activation_fn": nn.Tanh,
    "ortho_init": Config.ORTHO_INIT,
}
```

如果只是改隐藏层大小，通常改 `Config.ACTOR_HIDDEN_LAYERS` 和 `Config.CRITIC_HIDDEN_LAYERS` 即可。

### 5. 改训练流程

训练流程在：

```text
agent_ppo/workflow/train_workflow.py
```

这里可以改：

- 训练环境数量
- eval 频率
- 保存路径
- 是否使用 VecNormalize
- callback 逻辑
- best model 保存方式

## 推荐开发流程

```text
1. 先确定要改的方向：reward、特征、网络结构或超参数
2. 只改一个点，避免多个变量同时变化
3. 用小步数快速训练测试
4. 用 evaluate_agent.py 做无渲染评估
5. 指标正常后再加大 timesteps
6. 最后用 watch_agent.py 观察实际行为
```

推荐的快速闭环：

```powershell
python train_test.py --timesteps 4096 --n-envs 2
python evaluate_agent.py --exp-id 1 --episodes 10
python watch_agent.py --exp-id 1 --no-render --n-timesteps 1000
```

## 注意事项

- `logs/` 和 `runs/` 是训练产生的运行产物，当前已清理，不属于核心源码。
- 重新训练后会自动生成新的模型、评估文件和 TensorBoard 日志。
- 如果改变观测维度，需要同步检查 `Config.OBS_DIM`、模型输入层和 wrapper 输出。
- 当前项目更偏实验工程，不是完整比赛提交模板。
- 运行失败时优先检查 Python 环境、MiniGrid/Gym/SB3 版本是否匹配。
