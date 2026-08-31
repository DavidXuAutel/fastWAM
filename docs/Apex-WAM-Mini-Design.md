# Apex-WAM-Mini：低资源短周期闭环训练方案

> 文档版本：v1.0  
> 更新日期：2026-07-03  
> 目标：在有限算力和较短训练周期内，验证 WAM 的核心收益，并形成可闭环运行的机器人策略。

---

## 1. 目标

`Apex-WAM-Mini` 是 `Apex-WAM` 的可执行裁剪版。它不追求一次性覆盖 14B 大模型、DINO 动力学、ULA 跨本体和完整 Planner，而是保留最关键、最可能产生收益的部分：

- 使用预训练视频模型作为世界表征骨干。
- 保留 `video co-training`，用未来视频预测塑造动作表征。
- 推理时默认不生成未来视频，以保证低延迟闭环。
- 使用单一目标机器人动作空间，先把仿真/真机闭环跑通。
- 通过最小消融验证 WAM 是否优于普通动作扩散或行为克隆策略。

成功标准：

| 指标 | 目标 |
|------|------|
| 训练资源 | 1-8 张 A100/H100，或 1-4 张 RTX 4090 做小版本 |
| 训练时间 | 3-10 天 |
| 推理延迟 | 150-300ms / action chunk |
| 闭环频率 | 3-7Hz |
| 对比基线 | 优于 Diffusion Policy / ACT 在扰动场景下的表现 |
| 关键消融 | `L_action + L_video` 优于 `L_action only` |

---

## 2. 核心裁剪

相对完整 `Apex-WAM`，Mini 版先移除以下模块：

| 模块 | Mini 版处理 | 原因 |
|------|-------------|------|
| 14B Video Backbone | 暂用 Wan2.2-5B 或更小 DiT | 降低训练与推理成本 |
| DINO Dynamics Expert | 暂不做 | 收益需额外验证，工程复杂 |
| ULA 跨本体动作空间 | 暂不做 | 先固定一个目标机器人动作空间 |
| ACVS Planner | 暂不做 | 候选 rollout 和 value 训练成本高 |
| 复杂 Value/Reward | 第二阶段可选 | 初版优先验证动作闭环 |
| 50K 小时异质数据 | 改为 100-1000 小时量级 | 短周期可执行 |

保留模块：

| 模块 | 保留原因 |
|------|----------|
| Wan VAE / T5 | 复用预训练视频模型接口 |
| Video Backbone | 提供世界表征与视频 co-training |
| Action Expert | 生成可执行 action chunk |
| State Encoder | 注入机器人 proprio / gripper 状态 |
| Fast Mode | 支持实时闭环 |

---

## 3. 模型结构

### 3.1 总体结构

```text
输入:
  多视角图像 obs
  语言指令 prompt
  当前机器人状态 state

编码:
  Wan VAE encode 当前帧和未来视频
  T5 encode prompt
  State MLP encode proprio

核心:
  Wan Video Backbone（冻结或 LoRA）
  + Small Action Expert

输出:
  action chunk [T, C]
```

### 3.2 模块配置

| 模块 | 推荐配置 | 是否训练 |
|------|----------|----------|
| VAE | Wan2.2 VAE | 冻结 |
| Text Encoder | Wan/T5 | 冻结 |
| Video Backbone | Wan2.2-5B 或较小 DiT | 冻结前半部分 + LoRA |
| Action Expert | 12-16 层 Transformer，256M-800M | 全训练 |
| State Encoder | 2 层 MLP | 全训练 |
| Value Head | 小 MLP | 第二阶段可选 |

### 3.3 Video Backbone

训练时：

```text
observed frames + noisy future video latent
  → Video Backbone
  → denoise future video latent
  → L_video
```

推理时：

```text
current observation
  → VAE encode
  → Video Backbone 单次前向
  → latent world features
  → Action Expert
```

关键设计：**训练时学未来，推理时不显式生成未来**。这保留 Fast-WAM 的低延迟优势。

### 3.4 Action Expert

Action Expert 是轻量动作去噪网络：

```text
state token + noisy action tokens
  attend to video world features
  attend to text context
  → predict action noise / velocity
```

推荐配置：

```yaml
action_expert:
  hidden_dim: 768        # 显存紧张可降到 512
  layers: 12             # 推荐 12；资源充足可用 16
  heads: 12
  ffn_dim: 3072
  action_horizon: 16     # 初版推荐 16；稳定后升到 33
  diffusion_steps_train: 1000
  diffusion_steps_infer: 5
```

---

## 4. 动作空间

Mini 版只支持一个目标机器人动作空间，不做跨本体混合。

推荐二选一：

| 使用场景 | 动作空间 |
|----------|----------|
| RoboTwin / 关节控制 | `joint absolute` |
| G1 / 双臂末端控制 | `eef6d relative` |

### 4.1 G1 双臂推荐表示

```text
action_dim = 20

[left_xyz(3), left_rot6d(6), left_gripper(1),
 right_xyz(3), right_rot6d(6), right_gripper(1)]
```

### 4.2 时序

```text
actions: Tensor[T, C]
  T = 16 或 33
  C = action_dim

state: Tensor[1, C]
  当前机器人状态
```

训练和推理都采用 action chunk。闭环时只执行 chunk 的前几步，而不是一次执行完整 chunk。

---

## 5. 训练目标

### 5.1 最小可行损失

```python
L_total = λ_action * L_action + λ_video * L_video
```

推荐初始权重：

```yaml
lambda_action: 1.0
lambda_video: 0.3
```

动作损失：

```text
Flow Matching / Diffusion on action chunk
```

视频损失：

```text
Flow Matching on future Wan VAE latent
```

`L_video` 在 Mini 版中不是为了推理时生成视频，而是作为世界表征正则项，让动作分支使用更有物理含义的 latent features。

### 5.2 第二阶段可选 Value

当动作闭环已经可用后，可以加入轻量 value head：

```python
L_total = L_action + 0.3 * L_video + 0.1 * L_value
```

第一版 value 不做复杂 reward decomposition，只做：

```text
success / failure 二分类
或
progress bucket: 0%, 25%, 50%, 75%, 100%
```

---

## 6. 数据方案

### 6.1 推荐规模

Mini 版不追求 50K 小时数据。第一版建议：

| 数据类型 | 推荐规模 |
|----------|----------|
| 目标机器人高质量遥操 | 20-100 小时 |
| 仿真数据 | 50-200 小时 |
| 公开 robot 数据 | 100-500 小时 |
| failure rollout | 5-30 小时 |
| ego/web 视频 | 可选，100-1000 小时 |

如果目标是快速闭环，目标机器人数据优先级最高。

### 6.2 数据格式

每条样本统一为：

```python
{
    "video": Tensor[C, V, T, H, W],
    "state": Tensor[1, action_dim],
    "actions": Tensor[action_horizon, action_dim],
    "caption": str,
    "success": Optional[bool],
}
```

第一版只需要最小 supervision flags：

```python
{
    "has_action": bool,
    "has_video": bool,
    "has_success": bool,
}
```

### 6.3 采样比例

推荐初始采样比例：

```yaml
robot_target: 0.50
simulation: 0.25
public_robot: 0.20
failure: 0.05
```

如果目标机器人数据很少，可以提高 simulation 和 public robot 占比，但最后必须用目标机器人数据微调。

---

## 7. 训练流程

### Stage A：Action Expert 预热

目标：先检查数据、动作归一化、闭环接口，确认 action policy 能学到基本动作分布。

```yaml
train_mode: action_only
video_backbone: frozen
steps: 10000-30000
effective_batch_size: 64-128
learning_rate: 1e-4
action_horizon: 16
```

### Stage B：WAM 联合训练

目标：验证 video co-training 是否提升泛化。

```yaml
train_mode: action + video
video_backbone: LoRA rank 32-64
action_expert: full
steps: 50000-100000
lr_action: 5e-5
lr_lora: 1e-5
lambda_action: 1.0
lambda_video: 0.3
action_horizon: 16
```

这是 Mini 版的核心阶段。

### Stage C：目标任务微调

目标：提升目标机器人或目标任务闭环成功率。

```yaml
train_mode: action
video_backbone: frozen 或 LoRA
action_expert: full
steps: 5000-20000
learning_rate: 1e-5
data: target robot only
```

如果已有 failure rollout，可在此阶段加轻量 value head。

---

## 8. 推理闭环

### 8.1 Fast Mode

默认推理只走 Fast Mode：

```text
1. 读取当前多视角图像 + state
2. VAE encode 当前帧
3. Video Backbone 单次前向，得到 world features
4. Action Expert 5 步去噪
5. 输出 action chunk
6. 执行前 N 步，例如 2-4 步
7. 重新观测，进入下一轮
```

### 8.2 推荐闭环参数

```yaml
action_horizon: 16
execute_steps_per_chunk: 2-4
denoise_steps: 5
target_control_frequency: 3-7Hz
```

不要一次执行完整 action chunk。只执行前 2-4 步，可以明显降低误差累积。

---

## 9. 资源配置

### 9.1 小资源版本

| 项 | 配置 |
|----|------|
| GPU | 1-4× RTX 4090 / A6000 |
| Backbone | 冻结 Wan 或小 DiT |
| Action Expert | 256M |
| 训练时间 | 3-7 天 |
| 适用 | 仿真、单任务、接口验证 |

### 9.2 推荐版本

| 项 | 配置 |
|----|------|
| GPU | 4-8× A100 80G / H100 |
| Backbone | Wan2.2-5B + LoRA |
| Action Expert | 500M-800M |
| 训练时间 | 7-14 天 |
| 适用 | 多任务、目标机器人闭环 |

---

## 10. 必做消融实验

第一版必须做消融，否则无法判断 WAM 是否真的有效：

| 实验 | 目的 |
|------|------|
| `L_action only` | 普通动作扩散基线 |
| `L_action + L_video` | 验证 video co-training |
| frozen backbone vs LoRA | 判断是否需要改视频骨干 |
| 5 denoise steps vs 10 steps | 延迟与性能权衡 |
| execute 2 steps vs 4 steps | 闭环稳定性 |
| with failure data vs without | 判断失败数据收益 |

最关键的是：

```text
L_action + L_video 是否明显优于 L_action only
```

如果没有提升，就不应该继续扩展 DINO、ULA 或 Planner。

---

## 11. 建议实施里程碑

### Week 1：数据与闭环基线

- 整理目标机器人或仿真数据到统一格式。
- 训练 `L_action only` baseline。
- 打通推理 server/client 和闭环执行。
- 记录成功率、完成时间、延迟。

### Week 2：WAM Mini 联合训练

- 加入 `L_video`。
- 训练 `L_action + L_video`。
- 对比 baseline。
- 做 5-step / 10-step 推理延迟与成功率对比。

### Week 3：微调与失败数据

- 收集失败 rollout。
- 目标任务微调。
- 可选加入 success/failure value head。
- 得到第一版可闭环模型。

---

## 12. 结论

`Apex-WAM-Mini` 的核心取舍是：

```text
保留:
  预训练视频骨干
  video co-training
  action chunk diffusion
  Fast Mode 闭环

暂缓:
  DINO Dynamics
  ULA 跨本体
  14B scale
  ACVS Planner
  复杂 value learning
```

这版方案的价值在于快速回答一个最关键的问题：

> 在当前机器人和数据条件下，视频世界模型表征是否能提升动作泛化和闭环成功率？

如果答案是肯定的，再逐步扩展到完整 `Apex-WAM`；如果答案是否定的，应优先检查数据质量、动作空间和闭环接口，而不是继续堆更复杂的模型模块。
