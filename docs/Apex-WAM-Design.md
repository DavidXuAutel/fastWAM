# Apex-WAM：高能力上限世界动作模型设计方案

> 文档版本：v1.0  
> 更新日期：2026-07-03  
> 参考工作：τ₀-WM、DreamZero、Cosmos Policy、LingBot-VA、Fast-WAM、LDA-1B、Being-H0.7

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [设计哲学](#2-设计哲学)
3. [模型结构](#3-模型结构)
4. [DINO 与 Wan 隐空间分工](#4-dino-与-wan-隐空间分工)
5. [数据准备](#5-数据准备)
6. [完整训练方法](#6-完整训练方法)
7. [推理策略](#7-推理策略)
8. [工程与资源估算](#8-工程与资源估算)
9. [与现有方案对比](#9-与现有方案对比)
10. [实施路线图](#10-实施路线图)
11. [风险与缓解](#11-风险与缓解)
12. [附录](#12-附录)

---

## 1. 背景与目标

### 1.1 什么是 WAM

**World Action Model (WAM)** 是一类机器人基础模型：在预训练视频/世界模型骨干上，联合学习 **未来状态预测** 与 **可执行动作生成**。与 VLA（Vision-Language-Action）从 VLM 出发不同，WAM 从视频扩散模型出发，利用其时空先验理解物理交互。

### 1.2 近期代表性工作

| 工作 | 机构 | 核心贡献 |
|------|------|----------|
| **τ₀-WM** | 上海 AI Lab / AGIBOT | 5B 统一视频-动作模型；ACVS 模拟器；test-time 候选修正 |
| **DreamZero** | — | 14B 自回归 chunk-wise WAM；异质非重复数据；7Hz 实时控制 |
| **Cosmos Policy** | NVIDIA | action/value 作为 latent frame；value-based planning |
| **LingBot-VA** | Ant Group | Wan 2.2-5B + MoT；16K 小时跨本体预训练 |
| **Fast-WAM** | — | 训练时 video co-training，推理时跳过视频生成 |
| **LDA-1B** | — | DINO 结构化动力学；30K 小时异质数据统一摄入 |

### 1.3 Apex-WAM 目标

设计一套 **能力上限优先** 的 WAM，在以下维度追求 SOTA 潜力：

- 跨环境、跨任务、跨本体泛化
- 长视野、细粒度、接触丰富操作
- 支持快速部署与高风险场景下的规划推理
- 充分利用异质数据（含无动作标注视频）

---

## 2. 设计哲学

### 2.1 取各家之长

| 来源 | 吸收的设计 |
|------|-----------|
| **DreamZero** | 14B 级视频骨干、chunk-wise 自回归、teacher forcing、闭环 KV cache 防误差累积 |
| **τ₀-WM** | 层间 cross-attn 耦合、ACVS 模拟器、任务进度评分、test-time 候选修正 |
| **Cosmos Policy** | action/value 作为 latent frame、部署后可做 value-based planning |
| **LingBot-VA / Fast-WAM** | MoT 架构、结构化 attention mask、训练/推理解耦 |
| **LDA-1B** | DINO 结构化 latent 动力学、异质数据分角色使用 |
| **Being-H0.7** | prior/posterior latent bridge、大规模 egocentric 数据 |

### 2.2 相对 τ₀-WM 的关键升级

| 维度 | τ₀-WM | Apex-WAM |
|------|-------|----------|
| 骨干规模 | 5B (Wan2.2-TI2V) | **14B** (Wan2.1-I2V 或同级) |
| 架构 | 并联 cross-attn | **MoT + 共享注意力 + 结构化 mask** |
| 时序建模 | 双向 chunk | **chunk-wise 自回归 + teacher forcing** |
| 动力学表征 | 仅 Wan VAE latent | **Wan + DINO 双轨** |
| 价值学习 | Progress 评分 | **Progress + Value Expert + planning** |
| 推理模式 | 单一 + TTC | **Fast / Joint / Planner 三模式** |
| 数据规模 | ~27K 小时 | **~50K 小时金字塔** |

### 2.3 设计原则

1. **视频质量决定动作质量**（DreamZero 实证）→ 用大骨干 + 保持 video co-training
2. **数据多样性 > 同任务重复**（DreamZero）→ 广覆盖场景/任务/环境
3. **训练时想象，推理时可跳过**（Fast-WAM）→ 三模式可切换
4. **外观与动力学解耦**（LDA-1B）→ Wan + DINO 双轨表征
5. **异质数据统一监督**（τ₀-WM）→ modality-specific supervision mask

---

## 3. 模型结构

### 3.1 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Apex-WAM Core                           │
│                                                                 │
│  输入                                                            │
│  ├─ 多视角观测 o₀:ₗ        (C × V × T × H × W)                   │
│  ├─ 语言指令 ℓ                                                      │
│  ├─ 本体感知 qₗ           (proprio + gripper)                    │
│  └─ Embodiment ID         (机器人/人类/无动作)                    │
│                                                                 │
│  编码器（冻结）                                                   │
│  ├─ Wan2.2 / Cosmos VAE   (空间↓16, 时间↓4)                     │
│  ├─ T5-XXL Text Encoder                                         │
│  ├─ DINOv2 Encoder        (结构化动力学表征)                      │
│  └─ State MLP Encoder                                           │
│                                                                 │
│  MoT DiT Core (~18B total)                                      │
│  ├─ Video Expert          (~10B, 30 layers, 继承 Wan2.1-I2V-14B) │
│  ├─ Action Expert         (~2B, 30 layers, 独立权重)             │
│  ├─ Progress/Value Expert (~1B, 12 layers)                      │
│  └─ DINO Dynamics Expert  (~1B, 8 layers, 轻量 rollout)          │
│                                                                 │
│  输出                                                            │
│  ├─ 未来视频 latent ẑₗ:ₗ₊ₕ                                       │
│  ├─ 动作 chunk âₗ:ₗ₊ₕ                                            │
│  ├─ 任务进度 / 成功概率 V̂                                         │
│  └─ DINO 未来特征 ḋₗ:ₗ₊ₕ                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 MoT Block 细节

每层采用 **Modality-Specific Self-Attention + Shared Cross-Modal Attention**（借鉴 Fast-WAM / LingBot-VA）：

**Token 分组：**

| Token 类型 | 含义 | 训练时 | 推理时 |
|-----------|------|--------|--------|
| Clean Frame | 当前观测（锚点） | ✓ | ✓ |
| Future Video | 未来帧 noisy latent | ✓ | ✗（Fast Mode） |
| Action | noisy action chunk | ✓ | ✓ |
| Value | 进度/回报 | ✓ | ✓（Planner Mode） |
| State | proprio 单 token | ✓ | ✓ |

**结构化 Attention Mask（训练时）：**

| Query → Key | Clean Frame | Future Video | Action | Value | Text |
|-------------|:---:|:---:|:---:|:---:|:---:|
| Clean Frame | ✗ | ✗ | ✗ | ✗ | ✓ |
| Future Video | ✓ | ✓ (双向) | ✗ | ✗ | ✓ |
| Action | ✓ | ✗ | ✓ (双向) | ✗ | ✓ |
| Value | ✓ | ✓ (因果) | ✓ | ✓ | ✓ |

要点：

- Action **不能 attend** Future Video（防止信息泄漏，Fast-WAM 验证有效）
- Value **可以 attend** Action + Future Video（用于评估与规划）
- Clean Frame 是共享视觉锚点，不 attend 其他 token

### 3.3 自回归 Chunk 机制

借鉴 DreamZero 的 chunk-wise 自回归 + teacher forcing：

```
时间轴:  |--chunk 0--|--chunk 1--|--chunk 2--|...
         [o₀:ₖ]     [oₖ:₂ₖ]     [o₂ₖ:₃ₖ]
         [a₀:ₖ]     [aₖ:₂ₖ]     [a₂ₖ:₃ₖ]

训练: teacher forcing — 用 GT 历史 chunk 作条件，去噪当前 chunk
推理: 执行 aₖ:₂ₖ 后，用真实观测 oₖ 替换预测帧写入 KV cache
```

**推荐参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `chunk_size K` | 16 latent frames | 约 2s @ 8fps latent |
| `action_horizon H` | 33 steps | 与 τ₀-WM 一致 |
| `history_chunks` | 4 | 约 8s 历史上下文 |

### 3.4 动作表征：分层统一动作空间

为支持跨本体迁移，采用三层动作表征：

```
Layer 1: Unified Latent Action (ULA)
         32-dim, VAE-style 压缩轨迹窗口
         ← 所有数据源可学（含无动作标注视频）

Layer 2: Embodiment-specific Action
         - dual-arm eef6d: 20-dim
         - single-arm joint: 7-dim
         - human hand: 48-dim MANO
         ← 仅有动作标注的数据

Layer 3: Execution Interface
         目标机器人 FK/IK 映射
         ← 部署时解码
```

### 3.5 多模态预测目标

| 模态 | 表征空间 | 专家 | 作用 |
|------|---------|------|------|
| **Video** | Wan VAE latent | Video Expert | 主世界模型，密集物理监督 |
| **DINO Dynamics** | DINO patch tokens | DINO Dynamics Expert | 轻量语义动力学，忽略外观 |
| **Action** | 连续 chunk | Action Expert | 可执行控制 |
| **Value/Progress** | 标量 + 逐帧进度 | Progress Expert | 规划、候选排序 |

**Value 定义：**

```
V(τ) = Σ γ^t · r_t

r_t = w1·Δtask_progress + w2·contact_success - w3·collision - w4·stall
```

可用自动标注（物体位姿变化、夹爪状态）+ 人工成功标签混合。

### 3.6 损失函数（Stage 1 总览）

```python
L_total = (
    λ_vid  * L_video          # Flow Matching on Wan VAE latent
  + λ_act  * L_action         # Flow Matching on action chunk
  + λ_val  * L_value          # MSE on progress/reward
  + λ_dino * L_dino_dyn       # MSE on DINO future features
  + λ_ula  * L_ula            # VAE reconstruction on latent action
) * sample_mask               # per-sample supervision mask

# 初始权重
λ_vid=1.0, λ_act=0.5, λ_val=0.1, λ_dino=0.3, λ_ula=0.2
```

---

## 4. DINO 与 Wan 隐空间分工

### 4.1 为什么需要双轨表征

**不是不用 Wan 隐空间，而是两者分工：**

| 维度 | Wan VAE Latent | DINO Feature |
|------|----------------|--------------|
| **训练目标** | 视频重建/生成 | 视觉语义自监督 |
| **擅长编码** | 纹理、光照、颜色、风格 | 物体结构、空间关系、语义部件 |
| **对控制相关性** | 大量外观细节，与动作弱相关 | 运动、接触、位姿变化更突出 |
| **时序** | 3D VAE，4× 时间压缩 | 逐帧特征，需另建时序模块 |
| **推理成本** | 高（完整 video diffusion） | 低（小网络单次前向） |

### 4.2 Wan 隐空间：主视频路径

Wan VAE latent 已作为 **Video Expert 的主表征**：

```
Video Expert:  观测 → Wan latent → Flow Matching → 未来视频 latent
```

职责：

- 与预训练 14B 视频骨干对齐（DreamZero 关键）
- 视频-动作联合去噪与一致性检查
- 高保真未来想象（ACVS、re-denoising consistency）
- 视频 co-training 塑造动作表征（Fast-WAM 验证不可或缺）

### 4.3 DINO 隐空间：轻量动力学路径

职责：

- **语义动力学预测**：物体是否移动、夹爪是否接触、相对位姿变化
- **廉价 rollout**：毫秒级前向，用于 Planner Mode 快速筛候选
- **跨域泛化**：对人类视频、不同机器人/相机更鲁棒
- **互补监督**：避免与 Video Expert 功能重叠

```
DINO(o_t) + action → Dynamics Expert → DINO(o_{t+1})   # 单次前向，毫秒级
```

### 4.4 双轨协作（Planner Mode）

```
1. 采样 K=8 个 action candidates
2. DINO Dynamics Expert 对每个 candidate 做轻量 rollout → 快速打分
3. 保留 top-4
4. Wan ACVS 对 top-4 做高保真视频 rollout + Progress Expert 精验
5. 选最优执行；低分则触发修正
```

**直觉例子**（「把红色杯子放到蓝色盘子上」）：

| 表征 | 需要学什么 | 无关干扰 |
|------|-----------|----------|
| Wan latent | 杯子红色色调、桌面纹理、光照、阴影… | 多 |
| DINO | 杯子与盘的相对位置、是否接触、是否在空中 | 少 |

换相机或调曝光时，Wan latent 分布大变，DINO 特征相对稳定。

### 4.5 能否只用其一？

| 方案 | 问题 |
|------|------|
| 只用 Wan | 动力学 rollout 贵；外观与物理纠缠；异质数据泛化弱 |
| 只用 DINO | 失去视频 co-training；无法生成未来视频；细粒度接触建模弱 |
| **Wan + DINO 双轨** | 生成式高保真 + 判别式轻量互补，推荐 |

---

## 5. 数据准备

### 5.1 数据金字塔（目标 ~50K 小时）

```
                    ┌─────────────────┐
                    │ Tier 5: 仿真     │  ~2K h
                   ┌┴─────────────────┴┐
                   │ Tier 4: 失败轨迹     │  ~2K h
                  ┌┴─────────────────────┴┐
                  │ Tier 3: UMI / 手持示范  │  ~3K h
                 ┌┴───────────────────────┴┐
                 │ Tier 2: 多机器人遥操      │  ~8K h
                ┌┴─────────────────────────┴┐
                │ Tier 1: 人类 egocentric     │  ~15K h
               ┌┴───────────────────────────┴┐
               │ Tier 0: Web 操作视频          │  ~20K h
               └─────────────────────────────┘
```

| Tier | 来源 | 小时数 | 有动作 | 有进度 | 用途 |
|------|------|--------|:---:|:---:|------|
| 0 | Ego4D, Something-Something, 互联网操作视频 | ~20K | ✗ | ✗ | 视频动力学先验 |
| 1 | Ego4D hands, EPIC-Kitchens, 自建 egocentric | ~15K | ✗ | △ | 手部交互、工具使用 |
| 2 | AgiBot, OXE, DROID, RoboMIND 等 | ~8K | ✓ | ✓ | 核心动作学习 |
| 3 | UMI, ALOHA portable | ~3K | ✓ | ✓ | 细粒度操作 |
| 4 | 失败、碰撞、恢复、重试轨迹 | ~2K | ✓ | ✓ | 鲁棒性、value 学习 |
| 5 | Isaac / MuJoCo / RoboTwin 大规模 | ~2K | ✓ | ✓ | 长尾场景补充 |

**关键原则**（DreamZero）：数据多样性 > 同任务重复示范。优先覆盖更多场景/任务/环境。

### 5.2 统一数据格式

基于 LeRobot v2 + 扩展字段：

```python
@dataclass
class ApexSample:
    # === 必需 ===
    episode_id: str
    embodiment_id: str          # "agibot_g1" | "franka" | "human_ego" | "umi"
    task_text: str
    timestamp: float

    # === 视觉 ===
    images: dict[str, Tensor]   # cam_name -> [T, C, H, W]
    # 标准视角: head, left_wrist, right_wrist

    # === 动作（可选）===
    actions: Tensor | None      # [T, action_dim]
    states: Tensor | None       # [T, state_dim]

    # === 自动标注 ===
    task_progress: Tensor       # [T], 0→1
    contact_labels: Tensor      # [T]
    is_success: bool
    is_failure: bool
    failure_type: str | None    # "slip" | "collision" | "miss" | "stall"

    # === 监督 mask ===
    supervision_mask: dict
```

### 5.3 Modality-Specific Supervision Mask

```python
SUPERVISION_MASKS = {
    "robot_teleop": {
        "video": True, "action": True, "value": True,
        "dino_dynamics": True, "ula": True,
    },
    "umi": {
        "video": True, "action": True, "value": True,
        "dino_dynamics": True, "ula": True,
    },
    "human_ego": {
        "video": True, "action": False, "value": False,
        "dino_dynamics": True, "ula": True,
    },
    "robot_video_only": {
        "video": True, "action": False, "value": False,
        "dino_dynamics": True, "ula": True,
    },
    "web_video": {
        "video": True, "action": False, "value": False,
        "dino_dynamics": True, "ula": False,
    },
    "failure_rollout": {
        "video": True, "action": True, "value": True,
        "dino_dynamics": True, "ula": True,
        # 负样本：value 目标为低回报
    },
}
```

### 5.4 数据预处理流水线

```
原始数据
  │
  ├─ 1. 质量过滤
  │     - 运动幅度阈值（去掉静止段）
  │     - 模糊/曝光检测
  │     - 任务完成度自动估计（VLM 辅助标注）
  │
  ├─ 2. 标准化
  │     - 统一 256×192 或 320×240
  │     - 多视角时间对齐（±33ms）
  │     - 动作空间统一（per-embodiment 归一化）
  │
  ├─ 3. 自动标注
  │     - task_progress: 物体位姿变化 + 夹爪状态启发式
  │     - contact: 力传感器 或 视觉接触检测
  │     - caption 增强: VLM 生成细粒度描述
  │     - failure_type: 规则 + 小模型分类
  │
  ├─ 4. 统计量计算
  │     - per-embodiment action/state mean/std
  │     - 全局 ULA 统计
  │
  └─ 5. LeRobot 格式导出
        - parquet + mp4
        - delta_timestamps 配置
        - embodiment registry
```

### 5.5 数据混合策略

```python
# Stage 1 (异质预训练)
mixture_weights_s1 = {
    "web_video":        0.35,
    "human_ego":        0.25,
    "robot_teleop":     0.20,
    "umi":              0.08,
    "failure_rollout":  0.07,
    "simulation":       0.05,
}

# Stage 2 (机器人强化)
mixture_weights_s2 = {
    "robot_teleop":     0.45,
    "umi":              0.15,
    "failure_rollout":  0.15,
    "human_ego":        0.10,
    "web_video":        0.08,
    "simulation":       0.07,
}
```

**额外技巧：**

- **课程学习**：先短 horizon (chunk=8)，逐步增到 chunk=16
- **Hard negative mining**：过采样失败轨迹
- **Embodiment dropout**：10% 概率去掉 state，逼模型从视觉推断

---

## 6. 完整训练方法

### 6.1 四阶段训练路线图

```
Stage 0: Backbone 对齐      (~1 周)
    ↓
Stage 1: 异质预训练          (~4-6 周)    ← 核心
    ↓
Stage 2: 机器人共训          (~2-3 周)
    ↓
Stage 3: 任务微调            (~3-5 天/任务)
    ↓
Stage 4: 经验学习            (部署后持续)
```

### 6.2 Stage 0：Backbone 对齐（可选）

| 项 | 配置 |
|----|------|
| **目标** | 让 14B 视频骨干适应机器人观测分布 |
| **数据** | Tier 0+1 视频，无动作 |
| **可训练** | Video Expert 最后 4 层 + LoRA |
| **损失** | Flow Matching video only |
| **学习率** | 1e-5 |
| **步数** | 50K |
| **硬件** | 64× H100, DeepSpeed ZeRO-3 |

### 6.3 Stage 1：异质预训练（核心）

**Flow Matching 细节：**

```python
# 前期：视频和动作共享 timestep（DreamZero: 更快收敛）
t_shared = sample_timestep(batch)

# 后期：解耦（τ₀-WM: 更灵活）
t_video  = sample_timestep(batch)  # shift=5.0
t_action = sample_timestep(batch)  # shift=1.0

# Teacher forcing
for chunk_idx in range(num_chunks):
    context = clean_history_chunks[:chunk_idx]  # GT
    target  = noisy_current_chunk[chunk_idx]
    loss   += flow_matching_loss(model(context, target))
```

**可训练参数：**

| 模块 | Stage 1 |
|------|---------|
| Video Expert (14B) | 全参数微调 或 LoRA rank=256 |
| Action Expert (2B) | 全参数 |
| Progress Expert (1B) | 全参数 |
| DINO Dynamics Expert (1B) | 全参数 |
| VAE, T5, DINO encoder | 冻结 |

**训练配置：**

```yaml
model_size: 14B + 2B + 1B + 1B ≈ 18B total
batch_size: 2 per GPU × 64 GPUs × grad_accum 4 = effective 512
precision: bf16
optimizer: AdamW
learning_rate: 5e-5
warmup_steps: 5000
scheduler: cosine decay
train_steps: 500000
chunk_size: 16
action_horizon: 33
history_chunks: 4
gradient_checkpointing: true
deepspeed: ZeRO-3

# 数据增强
color_jitter: true
random_crop: [0.9, 1.0]
caption_dropout: 0.06
embodiment_dropout: 0.10
temporal_reverse: 0.02
```

**课程学习 schedule：**

| 步数区间 | chunk_size | history | 数据侧重 |
|---------|-----------|---------|---------|
| 0-100K | 8 | 2 | web + ego 为主 |
| 100K-300K | 12 | 3 | 加入 robot |
| 300K-500K | 16 | 4 | 全混合 + failure |

### 6.4 Stage 2：机器人共训

| 项 | 配置 |
|----|------|
| **目标** | 强化动作精度，保持 video co-training 防遗忘 |
| **train_mode** | video + action + value |
| **可训练** | Video Expert: LoRA only (rank=128)；Action/Progress: full |
| **学习率** | 2e-5 |
| **步数** | 200K |
| **损失权重** | λ_act=1.0, λ_vid=0.5 |

> **重要**（Fast-WAM 验证）：去掉 video co-training 的性能损失 **远大于** 去掉 test-time imagination。Stage 2 必须保留 `L_video`。

### 6.5 Stage 3：任务微调

```yaml
# 类似 τ₀-WM post-training
data: 50-500 episodes per task
train_mode: action + value          # 可关掉 video
trainable: action_expert + progress_expert
learning_rate: 1e-5
steps: 10000-50000
action_space: task-specific
action_type: absolute or relative
```

### 6.6 Stage 4：经验学习（部署后）

```
1. 部署收集 rollout（成功 + 失败）
2. 继续训练：
   - 成功轨迹 → 标准 BC + 高 value 目标
   - 失败轨迹 → 低 value 目标 + 对比学习
3. 更新 value function → 启用 test-time planning
4. 可选：RL fine-tuning (GRPO / DPO on action candidates)
```

---

## 7. 推理策略

### 7.1 三模式概览

| 模式 | 延迟 | 视频生成 | 适用场景 |
|------|------|---------|----------|
| **Fast** | ~150ms/chunk | ✗ | 常规操作、实时闭环 |
| **Joint** | ~600ms/chunk | ✓ | 调试、验证 video-action 一致性 |
| **Planner** | ~2-5s/chunk | ✓ (top-K) | 高风险、长视野、精细操作 |

### 7.2 Mode A：Fast（默认部署）

借鉴 Fast-WAM：

```
1. 当前帧 → VAE encode → clean frame tokens
2. Video Expert 单次前向 → latent world features
3. Action Expert flow denoising (5 steps) → action chunk
4. 不生成未来视频
```

### 7.3 Mode B：Joint

借鉴 DreamZero / τ₀-WM：

```
1. 联合去噪 video + action
2. 检查 video-action 一致性
3. 输出 action + 预测视频（可视化）
```

### 7.4 Mode C：Planner

借鉴 τ₀-WM test-time computation + Cosmos Policy：

```
1. 采样 K=8 个 action candidates (不同 seed)
2. DINO Dynamics Expert 轻量 rollout → 快速打分 → 保留 top-4
3. Re-denoising consistency 进一步排序
4. ACVS: 对 top-2 做 action-conditioned Wan 视频 rollout
5. Progress/Value Expert 评分 → 选最优
6. 若最高分 < 阈值 → 用最优 rollout 视觉特征作 guidance 重新生成
```

### 7.5 推理优化清单

| 优化 | 预期加速 | 来源 |
|------|---------|------|
| KV cache（自回归历史） | 3-5× | DreamZero |
| Action cross-attn KV cache | 2× | τ₀-WM |
| Video states buffer 复用 | 2× | τ₀-WM |
| 解耦 video/action timestep | 1.5× | DreamZero-Flash |
| torch.compile | 1.3× | τ₀-WM |
| INT8 量化 Action Expert | 1.5× | 通用 |
| **合计** | **~30-40×** | — |

目标：**Fast Mode 7Hz+ 闭环控制**。

---

## 8. 工程与资源估算

### 8.1 算力需求

| 阶段 | GPU | 时间 | 估算成本 |
|------|-----|------|---------|
| Stage 0 | 64× H100 | 1 周 | ~$50K |
| Stage 1 | 128× H100 | 5 周 | ~$650K |
| Stage 2 | 64× H100 | 2 周 | ~$100K |
| Stage 3 | 8× H100 | 3 天/任务 | ~$2K/任务 |
| **总计** | | | **~$800K** |

若用 5B 骨干（τ₀-WM 规模），成本可降至 ~$150K，但能力上限会明显受限。

### 8.2 推荐硬件配置

| 场景 | 最低配置 | 推荐配置 |
|------|---------|---------|
| Stage 1 训练 | 64× A100 80G | 128× H100 80G |
| Stage 3 微调 | 4× A100 | 8× H100 |
| Fast Mode 部署 | 1× RTX 4090 (量化) | 1× A100 40G |
| Planner Mode 部署 | 1× A100 80G | 1× H100 80G |

---

## 9. 与现有方案对比

| 维度 | τ₀-WM | DreamZero | Fast-WAM | **Apex-WAM** |
|------|-------|-----------|----------|--------------|
| 骨干规模 | 5B | 14B | 5B | **14B** |
| 架构 | 并联 cross-attn | Monolithic DiT | MoT | **MoT + 4 Experts** |
| 时序建模 | 双向 chunk | 自回归 chunk | 单帧 | **自回归 chunk** |
| 动力学表征 | Wan only | Wan only | Wan only | **Wan + DINO** |
| 异质数据 | ✓ mask | ✓ 多样性 | ✗ 较小 | **✓ 50Kh 金字塔** |
| Value/Planning | ✓ progress | ✗ | ✗ | **✓ value + planning** |
| 推理模式 | 1 + TTC | 1 | Fast only | **3 模式** |
| 跨本体 | △ | ✓ video-only | ✗ | **✓ ULA bridge** |
| 失败数据 | ✓ | △ | ✗ | **✓ 显式利用** |
| 开源程度 | 部分 | 部分 | 部分 | 设计方案 |

---

## 10. 实施路线图

### Phase 1：MVP（2-3 月）

- [ ] 基于 Wan2.2-5B 实现 MoT 骨架（可复用 τ₀-WM 代码）
- [ ] 结构化 attention mask
- [ ] 10K 小时异质数据 pipeline
- [ ] Stage 1 训练 + Fast Mode 部署验证

### Phase 2：Scale（2-3 月）

- [ ] 升级到 14B 骨干
- [ ] 自回归 chunk + teacher forcing
- [ ] Value Expert + ACVS
- [ ] Planner Mode

### Phase 3：上限（持续）

- [ ] DINO Dynamics Expert
- [ ] ULA 跨本体迁移
- [ ] Stage 4 经验学习
- [ ] 50K 小时全量数据

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 14B 训练成本过高 | 预算 | 先用 5B 验证 pipeline，再 scale |
| video co-training 被遗忘 | 性能下降 | Stage 2 必须保留 L_video |
| 自回归误差累积 | 长视野漂移 | 闭环 KV cache 用 GT 观测替换 |
| 异质数据动作空间不一致 | 训练不稳定 | ULA 中间层 + embodiment decoder |
| 推理延迟过高 | 无法实时 | 默认 Fast Mode；Planner 仅高风险启用 |
| Value 标注噪声 | 规划失效 | 启发式 + 人工校验；失败轨迹作负样本 |
| MoT 实现复杂度高 | 工程延期 | Phase 1 可先用 τ₀-WM 式并联 cross-attn |

---

## 12. 附录

### 12.1 关键论文链接

| 论文 | 链接 |
|------|------|
| τ₀-WM | https://arxiv.org/html/2606.01027 |
| DreamZero | https://arxiv.org/pdf/2602.15922 |
| Cosmos Policy | https://arxiv.org/html/2601.16163 |
| Fast-WAM | https://arxiv.org/html/2603.16666 |
| LDA-1B | https://arxiv.org/html/2602.12215 |
| τ₀-WM 代码 | https://github.com/sii-research/tau-0-wm |
| NVIDIA WAM 综述 | https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/ |

### 12.2 推荐初始超参（5B MVP 版）

若资源有限，可先用 5B 验证：

```yaml
# 基于 τ₀-WM 扩展
backbone: Wan2.2-TI2V-5B
architecture: MoT (video 5B + action 0.5B + value 0.3B + dino 0.3B)
train_steps: 200000
batch_size: 96 effective
chunk_size: 9          # 与 τ₀-WM 一致
action_horizon: 33
data_hours: 10000
estimated_cost: ~$150K
```

### 12.3 消融实验建议

验证核心设计选择：

| 实验 | 对比 | 验证假设 |
|------|------|----------|
| A1 | MoT vs 并联 cross-attn | MoT 更优 |
| A2 | +DINO vs -DINO | DINO 提升跨域泛化 |
| A3 | 自回归 vs 双向 chunk | 自回归长视野更稳 |
| A4 | +failure data vs -failure | 失败数据提升鲁棒性 |
| A5 | Fast vs Joint vs Planner | 三模式性能-延迟权衡 |
| A6 | 14B vs 5B 骨干 | 规模与视频质量关系 |

### 12.4 术语表

| 术语 | 含义 |
|------|------|
| WAM | World Action Model，世界动作模型 |
| VAM | Video Action Model，视频动作模型（WAM 子集） |
| MoT | Mixture-of-Transformers，多专家 Transformer |
| ACVS | Action-Conditioned Video Simulator，动作条件视频模拟器 |
| ULA | Unified Latent Action，统一隐空间动作 |
| Flow Matching | 连续归一化流匹配，扩散模型训练目标 |
| Teacher Forcing | 训练时用 GT 历史作条件 |
| TTC | Test-Time Computation，测试时额外计算 |

---

*本文档为 Apex-WAM 设计方案 v1.0，基于 2025-2026 年公开研究成果整理。具体实现时应根据可用算力、数据和目标机器人平台调整。*
