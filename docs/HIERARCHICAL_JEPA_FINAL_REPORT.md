# 技术报告｜基于 Hierarchical JEPA 与 Flow Matching 的潜在世界模型及长程自主导航

**English:** Hierarchical JEPA & Flow Matching for Latent World Modeling and Long-Horizon Navigation  
**实验环境：** MuJoCo / Gymnasium Robotics `PointMaze_Medium-v3`；PyTorch；CUDA，RTX 5060 8GB；WSL  
**报告依据：** 当前仓库的训练与评估代码，以及 2026-11-19 的真实执行日志。完整模型权重、轨迹 NPZ 和 MP4 位于本地 `outputs/`，未随 Git 仓库上传。  
**性质：** 开源架构上的分阶段复现、独立权重训练与视觉动力学扩展；不是原创 JEPA 或 Rectified Flow 算法论文。

## 摘要

本项目复现并整合基于 State JEPA 与 Flow Matching 的分层潜在世界模型，实现 PointMaze Medium 长程闭环自主导航。系统由四个核心组件组成：动作条件 State JEPA、30 步宏时间尺度 Hierarchical World Model（HWM）、条件 Macro Flow 宏动作提议分布、目标条件 Low-level Flow 动作控制器。

项目先使用预训练 State JEPA 与 Low-level Flow 作为底座，独立训练 HWM（2,000 optimizer steps）及 Macro Flow（1,000 steps），在两组共 60 个环境 Episode 中达到 55/60（91.7%）。随后通过完整四阶段训练流水线对 **State JEPA、Low-level Flow、HWM、Macro Flow 全部重新训练**，在独立的 50-Episode 测试块中得到 **50/50（100%）**。

此外，项目实现了基于 Meta JEPA-WM 的视觉编码与动作条件潜在预测扩展：冻结 Meta 视觉 Encoder，微调部分 AdaLN Predictor 并引入零初始化残差动力学校准；最佳 held-out 未来视觉 latent MSE 为 0.043234，对比直接复制当前 latent 的 0.044197（相对降低约 2.18%）。该视觉预测接入一个仍使用真实状态的 Mini 分层规划系统后，10 个 Episode 成功 9 个。该结果**不代表视觉优于状态控制，也不代表端到端 RGB-only 导航已实现**。

## 1. 问题定义与边界

环境提供 `observation`、`achieved_goal`、`desired_goal` 和连续二维动作空间。状态版本的智能体使用仿真环境显式提供的状态/目标信息，目标是在有墙壁与绕行约束的 Medium Maze 中到达目标位置。与仅依据目标方向直接行动不同，长程导航需要生成可行中间子目标，并将其转化为实际控制序列。

本项目讨论三种**不可混同**的评估条件：

1. 部分自主训练：官方 State JEPA + Low-level Flow，自己的 HWM + Macro Flow。
2. 四模块独立权重训练：所有四个模块使用我们训练后保存的权重。
3. 视觉动作条件扩展：Meta 的 RGB 特征与 predictor 参与候选动作评分，但高层与低层仍使用 Mini 原始状态信息。

## 2. 架构与信息流

```text
                simulator state (including current goal)
                              |
                              v
                    State JEPA Encoder
                         z_t = E(s_t)
                              |
                              v
            High Encoder ψ: z_t -> z_t^H
                              |                 desired_goal xy
                              +-----------------------+
                                                      |
                                                      v
                    Goal-conditioned Macro Flow
                    candidate macro-actions m_t
                              |
                              v
                    HWM g(z_t^H,m_t)
                    abstract predicted future
                              |
                              v
                     XY Subgoal Decoder
                              |
                              v
                   Low-level Rectified Flow
                   (state latent + state + subgoal)
                              |
                              v
                      2D continuous actions
                              |
                              v
                       PointMaze_Medium-v3
```

### 2.1 State JEPA

State JEPA 将归一化后的状态编码到潜在空间，并用动作条件 Predictor 对未来 latent 做预测。训练使用在线编码器、目标编码器（EMA/stop-gradient），结合表示正则、状态探针等目标。四模块完整训练调用官方入口 `python -m jepa_robotics.train`，从随机权重训练 State JEPA，生成 `point_medium_jepa_model.pt`。

### 2.2 HWM：抽象潜在状态与宏动作

HWM 由四个子网络组成：

- `psi`：冻结 State JEPA 的 latent 投影到高层抽象状态；
- `macro`：GRU 编码长度为 30 的原始动作序列；
- `g`：动作条件高层 latent 预测器；
- `dec`：解码高层状态对应的二维位置子目标。

形式化表示为：

```text
z_t       = E(s_t)
z_t^H     = ψ(z_t)
m_t       = MacroEncoder(a[t:t+30])
z^H_{t+30}= g(z_t^H, m_t)
subgoal   = dec(z^H_{t+30})
```

代码中 HWM 的训练目标包含 latent 预测归一化 MSE、VICReg 风格方差/协方差正则项和位置解码回归。官方训练入口：`external/mini-jepa/scripts/train/train_hjepa_hwm.py`。

### 2.3 Macro Flow：可行的长程动作提议

Macro Flow 在当前 `z_t^H` 与最终目标位置条件下，用 Rectified Flow 学习高层宏动作分布。训练通过从高斯噪声 `x_0` 与真实宏动作 `x_1` 的插值 `x_τ=(1-τ)x_0+τx_1`，最小化预测速度和 `x_1-x_0` 的 MSE。

与任意 Gaussian 宏动作提议相比，经过轨迹数据训练的 Flow 更倾向于生成训练分布内可行的运动提议。闭环时在候选中通过 HWM 预测及目标距离评分选择下一子目标。这是架构设计动机，**本项目尚未提供严格的 Gaussian-vs-Flow 单因素消融结果**。

### 2.4 Low-level Flow：短时间连续控制

低层控制器读取修改为局部子目标的 goal-conditioned State JEPA latent 与归一化状态条件，采用 Flow 网络采样长度为 8 的二维连续动作块，并循环执行和重新规划。独立训练脚本 `scripts/train_mini_medium_low_flow.py` 从相同 Episode 内的未来 achieved-goal 建立 hindsight goal relabel，进行 rectified-flow 速度监督；保存与 Mini `LowLevelFlow` 兼容的 checkpoint。

**实现差异说明：** 该低层训练数据构造是本项目增加的兼容配方，而非声称完全复制开源作者的 directed-walker 数据和超参数。

## 3. 数据与训练配置

### 3.1 原生 Episode 数据

`scripts/collect_mini_medium_episodes.py` 使用 Mini 官方迷宫 scripted controller 加一定比例随机动作采集连续轨迹，保存为 `medium_episodes.npz`。每条 Episode 保留长度 T+1 的状态以及长度 T 的动作，用于时间窗口及未来目标切片。采集目标约 150,000 transitions，具体已保存条目数以本地运行日志/NPZ 为准；报告不虚构精确 Episode 数量。

### 3.2 快速高层复现（含官方低层/State JEPA）

| 实验项 | 配置/结果 |
|---|---|
| 环境 | PointMaze Medium |
| HWM 宏时间跨度 | 30 simulator steps |
| HWM train/held-out macro samples | 5,539 / 625 |
| HWM 优化步骤 | 2,000 |
| HWM 最后 train prediction loss | 0.0040 |
| HWM 最后 held-out prediction loss | 0.0059 |
| Macro Flow training instances | 20,585 |
| Macro Flow 优化步骤 | 1,000 |
| Batch Size | 128 |
| 30000 起始种子块 | 10/10 |
| 50000 起始种子块 | 45/50 |
| 累计 | **55/60 = 91.7%** |

该阶段的 State JEPA 与 Low-level Flow 是开源预训练权重；HWM 与 Macro Flow 是自主训练权重，因此必须称为**部分自主训练复现**。

### 3.3 完整四阶段独立训练

`scripts/run_mini_four_stage.sh` 定义从头训练顺序：

| 阶段 | 训练器 | 默认步骤 |
|---|---|---:|
| 1. State JEPA | `jepa_robotics.train` | 10,000 |
| 2. Low-level Flow | `scripts/train_mini_medium_low_flow.py` | 12,000 |
| 3. HWM | upstream `train_hjepa_hwm.py` | 4,000 |
| 4. Macro Flow | upstream `train_hwm_macro_flow.py` | 5,000 |

默认 Batch Size 128。四份输出文件分别为：

```text
outputs/mini_scratch/full/
  point_medium_jepa_model.pt
  medium_low_flow.pt
  medium_hwm.pt
  medium_macro_flow.pt
  medium_four_stage_nav.mp4
```

基于用户已完成的真实运行日志：

```text
{"event":"macro_flow_train","step":5000,"loss":0.0986}
{"event":"macro_flow_saved","path":"outputs/mini_scratch/full/medium_macro_flow.pt"}
{"task":"point_medium","episodes":50,"success_rate":1.0}
```

**正式测得：50/50，seed 60000 起的 50 个 Episode。** Macro Flow 在第 5,000 步报告的 train loss 为 0.0986。需注意，短规模的 55/60 与完整四阶段的 50/50 属于**不同模型权重与不同测试种子块**，不应直接计算为“同一模型成功率提升 8.3 个百分点”。

## 4. 视觉表示与动作条件预测扩展

视觉分支使用预训练 Meta JEPA-WM 的视觉 Encoder 将 RGB 映射到 16×16×384 空间 token，并结合此前 2 帧上下文、proprio 与连续动作块做未来 latent 预测。

实验依次包含：

1. 视觉到 Mini latent 的 Bridge/特征映射；
2. 状态+视觉 latent 残差融合；
3. Meta Predictor 动作适配、空间位置读出；
4. 微调部分 AdaLN Predictor block 与 residual-calibration head；
5. 利用未来位置读出对低层 Flow 提供的候选动作进行重新排序。

### 4.1 动力学数值结果

| 指标 | 结果 |
|---|---:|
| 未校准 Meta Predictor 的未来 latent MSE | 约 4.306 |
| 直接复制当前 latent（identity）MSE | 0.044197 |
| 残差初始化/部分微调后最佳 held-out future MSE | **0.043234** |
| 相对 identity 改善 | **约 2.18%** |
| 最佳 Epoch | 11 |
| 最佳 Epoch 反向动作预测 MSE | 0.045291 |
| 正确动作减反向动作 MSE | -0.002056 |
| 视觉候选重排闭环控制 | 9/10（seed 30000–30009） |

**重要解释：** 4.306→0.043 的主要改善来自 identity-initialized 的残差校准，而不是说明微调模型的动作预测提升了近 100 倍。相对静态基线的 2.18% 才是更稳妥的动力学提升量度。反向动作消融表明动作不同会影响预测准确度，但尚不能证明对真实控制有稳定增益。

### 4.2 为什么视觉没有提升 Medium Maze 导航？

Medium Maze 当前环境通过 simulator-state 直接提供准确位置、速度、目标；视觉增加了识别误差与模型偏差。视觉分支在当前系统中参与候选动作重排，但没有取代高层 HWM 与低层 Flow 的真值状态输入。当前已观察到：状态控制性能强于视觉辅助版本；测试样本规模与模型配置不同，不构成严格对照因果结论。

**不允许的成果表述：** “纯视觉闭环自主导航已完成”“RGB 替代所有 oracle state”“视觉提高导航成功率”“原创 JEPA/Flow 核心架构”。

## 5. 验证、可复现性与局限

- **测试环境：** 目前使用本地 `PointMaze_Medium-v3` 的随机起终点 Episodic 任务，而不是官方 Minari 固定起终点评测；日志 `minari_eval_dataset=null`。
- **样本规模：** 50/50 是有限 50-Episode 区间的观察结果，不意味着理论上 100% 泛化成功率。未来若研究需要，可以加 100–500 个不同种子块。
- **独立性：** 训练采集使用 1234 起始的种子，而四模块测试为 60000 起始的种子；这提供不同随机起始配置，但不能代替跨地图拓扑或外部分布评测。
- **数据/权重公开性：** 生成的大型 NPZ、PT、MP4 在 `outputs/`，被 `.gitignore` 排除；公开仓库的脚本可供重新运行，但缺少直接下载好的完整权重。
- **上游源码依赖：** `external/mini-jepa/` 目录被 gitignore，克隆仓库后必须另行获取 [Mini-JEPA](https://github.com/WilliamZhang20/Mini-JEPA)。Meta 实验依赖 [JEPA-WM](https://github.com/facebookresearch/jepa-wms) 及相应 checkpoints。
- **训练归属：** 四阶段权重由本项目独立训练；State JEPA、HWM 与 Macro Flow 复用上游架构/训练器，Low-level Flow 使用本项目编写的目标重标记训练脚本。这是算法复现与系统整合，不是自研全新网络设计。

## 6. 主要文件与运行命令

```text
scripts/collect_mini_medium_episodes.py    # state/action Episode 数据
scripts/run_mini_four_stage.sh              # 四阶段训练与 eval
scripts/train_mini_medium_low_flow.py       # 低层 Flow 的本地训练
scripts/medium_action_predictor.py          # 视觉动作候选预测与排序
scripts/finetune_medium_dynamics.py         # AdaLN 部分微调 + residual dynamics
scripts/visual_medium_bridge.py             # 视觉 latent 与 Mini 表征适配
docs/MINI_FOUR_STAGE_REPRODUCTION.md         # 训练记录
docs/VISUAL_HIERARCHICAL_MEDIUM.md          # 视觉扩展记录
```

```bash
cd ~/projects/JEPA-World-Model-Lab
export PYTHONPATH="$PWD/external/mini-jepa:${PYTHONPATH:-}"

# 只评估四个独立训练权重
bash scripts/run_mini_four_stage.sh eval

# 完整重新训练（会覆盖目标 full/ 下的权重，先做好备份）
bash scripts/run_mini_four_stage.sh all
```

## 7. 结论与简历定位

**已完成核心目标：** 在 Mini-JEPA 的开源算法体系上，独立训练四阶段潜在世界模型和目标条件 Flow 控制器，进行分层抽象状态、宏动作提议和闭环控制，在 PointMaze Medium 的 50 个测试 Episode 中实现 **50/50 成功**。视觉扩展提供了可复用的 RGB 表征、动作条件动力学微调和消融实验，但不构成纯视觉导航成功证明。

### 一页中文简历建议（压缩三条）

> **基于 Hierarchical JEPA 与 Flow Matching 的潜在世界模型及长程自主导航**｜PyTorch、JEPA、Rectified Flow、MuJoCo
>
> - 基于开源 Mini-JEPA 体系，独立训练 State JEPA、Hierarchical World Model（30-step 宏状态预测）、目标条件 Macro Flow 和 Low-level Flow 四模块，完成从潜在状态建模、可行宏动作生成到低层闭环执行的端到端系统集成。
> - 采集约 15 万步 PointMaze Medium 状态—动作轨迹，使用动作条件潜在预测、VICReg 表征约束及 Rectified Flow Matching 训练分层模型；在独立的 50 个随机种子 Episode 中实现 **100%（50/50）导航成功率**。
> - 扩展 Meta JEPA-WM 的 RGB 表征与动作条件 Predictor，采用 AdaLN 部分微调和零初始化残差动力学适配，将验证集未来 latent MSE 相比静态基线降低 **2.18%**；完成状态辅助视觉候选动作排序与消融验证。

如果篇幅不足，优先保留前两条，视觉扩展可以作为技术面试展开内容，避免低估完整四阶段复现的价值。

## 参考

- 上游 Mini-JEPA 源码：<https://github.com/WilliamZhang20/Mini-JEPA>
- Meta JEPA-WM 源码：<https://github.com/facebookresearch/jepa-wms>
- 本项目四阶段实现：`scripts/run_mini_four_stage.sh`
- 本项目视觉动力学微调：`scripts/finetune_medium_dynamics.py`
