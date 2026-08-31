# Apex-WAM-Mini v3.2：低资源短周期闭环训练方案（可复现实施版）

> 文档版本：v3.2（含内部修订 r1）  
> 更新日期：2026-07-03  
> 前身：`Apex-WAM-Mini-Design-v3.1.md`  
> 目标：在有限算力和较短训练周期内，验证 WAM 的核心收益，并形成可闭环、可复现、结论可信的机器人策略。
>
> **v3.2 相对 v3.1 的主要改进：**
> 1. **形式化 FULL / A1c forward spec**，消除 `full_like` 实现歧义。
> 2. **A1c 与 B 的 locked-same 配置表**，明确唯一差异为 `L_video`。
> 3. **`data_compatibility.yaml` registry**，替代仅靠 `action_space` 字符串匹配。
> 4. **Stage C 防遗忘**：`λ_video=0.1` 轻量 video co-training。
> 5. **Checkpoint 选择协议**，统一 A0/A1/A1c/B 模型选取标准。
> 6. **τ₀-WM 最小改造路径**（Phase 0/1/2），对齐文档架构与代码实现。
> 7. **精简 Week 2 排期**，主路径优先 A1c + B_perceiver。
> 8. **分层通过线**：macro + per-task + perturbed 分列。
> 9. **推理延迟预算表**含 multi-seed 选项。

---

## 1. 目标与核心问题

`Apex-WAM-Mini` 的核心问题保持不变：

> **在当前机器人和数据条件下，video co-training 是否能提升动作泛化和闭环成功率？**

v3.2 在 v3.1「实验可信」基础上，进一步解决 **实现可复现** 与 **对照公平性**：

- 消融要公平：A1c 与 B 除 `L_video` 外完全一致。
- Forward 要可复现：FULL / A1c / FAST 有形式化 spec，不同实现者产出相同计算图。
- 数据监督要干净：compatibility registry 而非字符串相等。
- 代码路径要清晰：τ₀-WM 改造分 Phase 0/1/2，Week 0 不踩架构坑。
- 评测要有统计意义：macro + per-task + perturbed 分层通过线。

---

## 2. 版本定位

| 版本 | 定位 | 状态 |
|------|------|------|
| v1 | 裁剪版初稿 | 历史草案 |
| v2 | 工程优化版 | 中间版 |
| v3 | 工程可复现版 | 历史主线 |
| v3.1 | 实验可信版 | 历史主线 |
| **v3.2** | **可复现实施版** | **当前实施主文档** |

v3.2 不推翻 v3.1 的主线与判据，只在 forward spec、数据 registry、改造路径、排期与通过线上补强。

---

## 3. 成功标准与统计通过线

### 3.1 延迟分档

| 配置 | 单次 backbone 前向 | action 去噪 | 目标频率 |
|------|-------------------|-------------|----------|
| 小 DiT + 单视角 + 5 步去噪 | 较低 | 5 步 | 5-7Hz |
| Wan2.2-5B + 单视角 + 5 步去噪 | 较高 | 5 步 | 2-4Hz |
| Wan2.2-5B + 多视角 + 5 步去噪 | 更高 | 5 步 | 1.5-3Hz |

### 3.2 推理延迟预算（含 multi-seed）

| 模式 | 相对基线 | 部署建议 |
|------|----------|----------|
| 单 seed Fast Mode | 1.0×（基线） | **默认部署** |
| 2 seed + candidate_filter | ~1.8× | 中等风险任务 |
| 3 seed + candidate_filter | ~2.5–3× | 高风险/接触任务 |

延迟 benchmark 必须分别记录单 seed 与 3 seed 的 p50/p95；通过线以 **单 seed** 为准，multi-seed 仅作可选安全层。

Perceiver 相对 mean_pool 的推理增量需在 Week 0 实测并写入 `benchmark_latency.json`（目标：增量 < 30ms @ 单视角）。

### 3.3 WAM 主假设通过线（分层）

```yaml
# macro 平均（主判据）
macro_avg:
  vs_A1c_clean: +3%
  vs_A1c_perturbed: +5%
  bootstrap_confidence: 95%
  require_ci_lower_bound_above_zero: true

# per-task 防退化
per_task:
  no_regression_vs_A1c: true
  max_allowed_drop: 5%          # 任一任务不得相对 A1c 下降超过 5%

# 相对外部 baseline（辅助）
vs_baseline_clean: +5%
vs_baseline_perturbed: +8%

# 工程指标
inference_latency: 满足延迟分档（单 seed）
ablation_report: 完成 A0/A1c/B 主路径 + 扩展消融报告
```

解释：

- B 相对 A0 提升 → 比冻结 baseline 好。
- B 相对 A1 提升 → 可能来自 `L_video`，仍可能混入计算量差异。
- **B 相对 A1c 提升（macro + perturbed CI）** → v3.2 最可信的 WAM 核心判据。
- **per-task 不得退化** → 防止 macro +3% 掩盖单任务大幅失败。

---

## 4. 代码复用映射与 τ₀-WM 最小改造路径

### 4.1 复用资产

Mini 不是从零实现 WAM，而是在 **τ₀-WM + FastWAM 现有资产** 上做可控裁剪实验。

| 能力 | 复用来源 | 路径 / 说明 |
|------|----------|-------------|
| WanModel / VAE / T5 | 官方 τ₀-WM | `sii-research/tau-0-wm` |
| 推理 Policy + Server | τ₀-WM | `web_infer_utils/TauPolicy.py`, `server.py` |
| 部署预检 | FastWAM | `experiments/tau0_wm_sim/preflight.py` |
| 闭环 smoke loop | FastWAM | `experiments/tau0_wm_sim/vam_only_loop.py` |
| 候选动作安全过滤 | FastWAM | `experiments/tau0_wm_sim/candidate_filter.py` |
| 延迟 benchmark | FastWAM | `experiments/tau0_wm_sim/benchmark_vam.py` |
| Post-train 数据格式 | τ₀-WM | `data/example_dataset.py`, LeRobot |

Mini 需新增：

```text
configs/mini_stage_{a0,a1,a1c,b,c}.yaml
configs/data_compatibility.yaml
train/infer 对齐模式开关
feature_adapter 配置
supervision mask + compatibility registry 数据加载逻辑
bootstrap 评测脚本
checkpoint 选择 hook
reports/ablation_summary.md 模板
```

### 4.2 τ₀-WM 最小改造路径（v3.2 新增）

官方 τ₀-WM 的 Action 分支是 **逐层 cross-attn 到 video block 中间表示**；v3.2 的 scheme B + Perceiver 是 **hook 抽层 → Feature Adapter → Mini Action Expert**。不能假设直接 fork `TauPolicy` 即可跑 B 实验。

分三阶段改造，避免 Week 0 冒烟通过、Week 2 才发现要改核心 forward：

```text
Phase 0 — Baseline 对齐（Week 0）
  目标: 用官方 τ₀-WM TauPolicy 作 A0 外部 baseline
  改动: 无（或仅 config / 数据路径）
  验收: preflight + vam_only_loop + benchmark 通过

Phase 1 — Feature 接口（Week 0–1）
  目标: 在 WanModel forward 加 layer hooks → Feature Adapter → Mini Action Expert
  改动:
    - WanModel: register_forward_hook on layers [8, 16, 24]
    - 新建 feature_adapter.py (mean_pool / perceiver_resampler)
    - 新建 mini_action_expert.py (cross-attn adapter)
    - TauPolicy: FAST mode 走新 action path，保留官方 path 作对照
  验收: FAST forward 输出 action chunk；Action Interface Tests 全过

Phase 2 — 训练模式（Week 1–2）
  目标: FULL / FAST / A1c 模式 + supervision mask + compatibility registry
  改动:
    - runner/posttrain.py: train_forward_modes, loss masking, A1c no_video_loss
    - dataloader: build_supervision_mask + registry lookup
    - checkpoint selector: perturbed_success_rate
  验收: A1c 与 B 的 backbone_forward_count 在 ±2% 内一致
```

**注意**：Phase 1 完成前不要启动 A1c/B 正式训练。

---

## 5. 模型结构

### 5.1 总体结构

```text
输入: 多视角 obs + prompt + state

编码:
  Wan VAE（当前帧 + 训练时未来帧）
  T5
  State MLP

核心:
  Video Backbone（frozen / LoRA）
  Feature Adapter（mean_pool 或 perceiver_resampler）
  Action Expert（cross-attn adapter，方案 B）

输出:
  action chunk [T, C]
```

### 5.2 模块配置

| 模块 | 配置 | 训练 |
|------|------|------|
| VAE / T5 | Wan2.2 | 冻结 |
| Video Backbone | Wan2.2-5B 或小 DiT | A0 冻结；A1/A1c/B LoRA |
| Feature Adapter | `mean_pool` / `perceiver_resampler` | 全训练 |
| Action Expert | 12 层，256M-800M | 全训练 |
| State Encoder | 2 层 MLP | 全训练 |
| Value Head | 小 MLP | 有条件启动 |

---

## 6. Feature Adapter 设计

### 6.1 Baseline：Mean Pool

```yaml
feature_adapter_baseline:
  type: mean_pool
  input_layers: [8, 16, 24]
  max_feature_tokens: 256
  preserve_view_tokens: false
```

适合：Day-1 冒烟、A0、扩展消融 `B_mean_pool`。

### 6.2 Recommended：Perceiver Resampler

```yaml
feature_adapter_recommended:
  type: perceiver_resampler
  input_layers: [8, 16, 24]
  num_latents: 64          # 资源足可升到 128
  latent_dim: 768
  preserve_view_tokens: true
  cross_attn_heads: 8
  depth: 2
```

结构：

```text
backbone tokens from layers [8,16,24]
  → add view / time / layer embeddings
  → Perceiver latent queries (64)
  → compact world tokens
  → Action Expert cross-attn
```

主路径默认：

```yaml
default_feature_adapter: perceiver_resampler
fallback_feature_adapter: mean_pool
```

扩展消融（Week 2 选做）：`B_mean_pool vs B_perceiver_resampler`。

---

## 7. Forward Spec：FULL / FAST / A1c（v3.2 核心）

### 7.1 问题

v3.1 的 `full_like` 未严格定义，不同实现者可能做出不同的 A1c，结论不可复现。v3.2 将三种 forward 形式化为 **可实现的计算图 spec**。

### 7.2 FULL forward spec

```text
FULL forward（训练模式，B 与 A1c 共用）:
  1. VAE encode: current frames + future frames → latent
  2. sample noise ε on future latent region
  3. backbone forward on [current_cond, noisy_future]
     - LoRA 参与 forward
     - layer hooks 抽取 [8, 16, 24] 特征
  4. [B only] compute L_video on future region, backward through video head + LoRA
     [A1c] skip step 4: L_video = 0, no backward through video head
  5. Feature Adapter → world tokens
  6. Action Expert → L_action
     - action feature path 与 FAST 分支一致（当前帧条件）
  7. total loss:
     B:     L_action + λ_video * L_video
     A1c:   L_action only
```

关键点：

- A1c 的 step 1–3、5–6 与 B **字节级相同**；唯一差异是 step 4 不算 `L_video`。
- LoRA 在 A1c FULL 模式下仍接收 action path 梯度；不接收 video head 梯度（与 B 的 FULL 相比，这是预期差异，由 `L_video` 引起）。

### 7.3 FAST forward spec

```text
FAST forward（训练与推理共用）:
  1. VAE encode: current frames only
  2. backbone forward on current_cond only
  3. Feature Adapter → world tokens
  4. Action Expert → L_action（训练）或 action chunk（推理）
  5. L_video = 0
```

### 7.4 混合训练

Stage B 以概率混合 FULL / FAST：

```yaml
train_forward_modes: [full, fast]
train_forward_probs: [0.5, 0.5]
```

```python
if mode == "full":
    loss = L_action * mask["action"] + 0.3 * L_video * mask["video"]  # B
    # A1c: 同上但 L_video 项恒为 0
else:
    loss = L_action * mask["action"]
```

必做消融：`B_align_on vs B_align_off`（`forward_probs: [0.5, 0.5]` vs `[1.0, 0.0]`）。

### 7.5 A1c 与 B 的 locked-same 表（v3.2 新增）

以下配置项在 A1c 与 B 之间 **必须完全相同**；任何差异需写入实验日志并视为无效对照。

| 配置项 | A1c | B | 锁定 |
|--------|-----|---|------|
| `feature_adapter` | perceiver_resampler | perceiver_resampler | ✓ |
| `forward_modes` | [full, fast] | [full, fast] | ✓ |
| `forward_probs` | [0.5, 0.5] | [0.5, 0.5] | ✓ |
| `steps` | N | N | ✓ |
| `lr_action` / `lr_lora` | 5e-5 / 1e-5 | 5e-5 / 1e-5 | ✓ |
| LoRA rank | 32-64 | 32-64 | ✓ |
| `supervision_mask` | true | true | ✓ |
| Stage B 数据混合 | 相同 | 相同 | ✓ |
| `L_video` on FULL | **0** | **0.3 × L_video** | **唯一差异** |

记录项（每次训练写入 `reports/training_meta.json`）：

```yaml
record:
  backbone_forward_count: true
  gpu_hours: true
  full_vs_fast_ratio: true
```

验收：A1c 与 B 的 `backbone_forward_count` 差异 ≤ 2%。

### 7.6 扩展消融：B_video_grad_only（选做）

用于分离「video loss 对 backbone 的梯度 shaping」效应：

```yaml
id: B_video_grad_only
description: FULL forward + L_video 更新 backbone LoRA，但 L_action 仅用 FAST features
purpose: 若 B_video_grad_only ≈ B 且 >> A1c，说明收益主要来自 video 梯度而非 world tokens
```

---

## 8. 动作空间

Mini 版只支持一个目标机器人 action space。

| 场景 | 推荐 |
|------|------|
| RoboTwin / 关节 | `joint absolute` |
| G1 双臂末端 | `eef6d relative` |

G1 双臂模型内部 20 维：

```text
[left_xyz(3), left_rot6d(6), left_gripper(1),
 right_xyz(3), right_rot6d(6), right_gripper(1)]
```

部署输出可仍为 τ₀-WM 的 16 维，由现有 `TauPolicy` 后处理完成。

---

## 9. Action Interface Tests（必做）

在任何正式训练前必须通过动作接口单元测试。

### 9.1 必测项目

```text
1. rot6d ↔ quaternion round-trip
2. relative action → absolute action 转换
3. absolute → relative → absolute 一致性
4. gripper 归一化与反归一化
5. gripper 开/合方向
6. left/right arm 顺序
7. base frame 坐标系
8. candidate_filter 边界条件
```

### 9.2 通过条件

```yaml
rotation_roundtrip_error: < 1e-4
position_roundtrip_error: < 1e-4 m
gripper_range_valid: true
left_right_order_verified: true
candidate_filter_rejects_invalid: true
```

### 9.3 推荐测试文件

```text
tests/test_action_space_interface.py
tests/test_candidate_filter.py
```

---

## 10. 数据方案、Compatibility Registry 与 Supervision Mask

### 10.1 数据规模

| 类型 | 规模 |
|------|------|
| 目标机器人遥操 | 20-100 h |
| 仿真 | 50-200 h |
| 公开 robot | 100-500 h |
| failure rollout | 5-30 h |

### 10.2 样本格式

```python
{
    "video": Tensor[C, V, T, H, W],
    "state": Tensor[1, action_dim],
    "actions": Tensor[action_horizon, action_dim] | None,
    "caption": str,
    "success": Optional[bool],
    "action_space": str,
    "embodiment_id": str,
    "source_id": str,           # v3.2: 用于 registry 查找
}
```

### 10.3 Compatibility Registry（v3.2 新增）

仅靠 `action_space == target_action_space` 字符串匹配过于粗糙（sim 与真机命名一致不代表坐标系/动力学兼容）。v3.2 引入显式 registry：

```yaml
# configs/data_compatibility.yaml
target:
  action_space: eef6d_relative
  embodiment: agibot_g1
  coord_frame: arm_base
  action_dim: 20

compatible_sources:
  - id: target_robot_teleop
    action_space: eef6d_relative
    embodiment: agibot_g1
    coord_frame: arm_base
    action_supervision: true
    verified: true

  - id: robowin_g1_proxy
    action_space: eef6d_relative
    embodiment: agibot_g1
    coord_frame: arm_base
    action_supervision: true
    verified: true
    notes: "需通过 Action Interface Tests 抽检 100 条"

video_only_sources:
  - id: oxe_franka_joint
  - id: human_ego
  - id: incompatible_public_robot

rules:
  action_supervision: lookup(source_id) in compatible_sources AND verified == true
  video_supervision: all sources except explicitly blocked

# verified 的量化门槛（r1 新增，不得仅靠人工声明）
verification_criteria:
  sample_size: 100                        # 每 source 抽检条数
  replay_eef_position_error: < 5 mm       # 动作重放后 EEF 轨迹位置误差
  replay_eef_rotation_error: < 2 deg      # EEF 姿态误差
  coord_frame_consistency: 100%           # 坐标系一致率必须 100%
  gripper_direction_match: 100%           # 开/合方向一致
  pass_rule: 全部满足才可置 verified: true
  artifact: reports/registry_verification/{source_id}.json
```

`verified` 与 §9 Action Interface Tests 采用同一量化风格；未达标的 source 只能进入 `video_only_sources`。

### 10.4 Supervision Mask

```python
def build_supervision_mask(sample, registry):
    compat = registry.lookup(sample["source_id"])
    action_compatible = (
        sample["actions"] is not None
        and compat.action_supervision
        and compat.verified
    )
    return {
        "video": True,
        "action": action_compatible,
        "success": sample.get("success") is not None,
    }
```

规则：

| 数据类型 | `L_video` | `L_action` |
|----------|-----------|------------|
| target robot（registry verified） | ✓ | ✓ |
| compatible sim（registry verified） | ✓ | ✓ |
| incompatible / video_only | ✓ | ✗ |
| failure rollout | ✓ | 仅 registry action compatible 时 ✓ |

### 10.5 Stage 数据混合

Stage B（验证 WAM，求泛化信号）：

```yaml
target_action_data: 0.25
compatible_action_data: 0.30
video_only_or_incompatible: 0.40
failure: 0.05
```

Stage C（部署微调，求闭环成功率）：

```yaml
robot_target: 0.90
failure: 0.10
# 不混入 public/sim，避免 action space 污染
```

---

## 11. 训练目标

### 11.1 主损失

```python
if mode == "full":
    loss = (
        1.0 * L_action * mask["action"]
        + lambda_video * L_video * mask["video"]   # A1c: lambda_video = 0
    )
else:
    loss = 1.0 * L_action * mask["action"]
```

Stage B：`lambda_video = 0.3`（B）/ `0`（A1c）。  
Stage C：`lambda_video = 0.1`（见 §13.5 防遗忘）。

### 11.2 Value Head 启动条件

Value Head 不在 A0/A1/A1c/B 主路径中启用。启动需满足：

```text
1. B 相对 A1c 已通过主假设通过线（§3.3）
2. 每个任务 >= 50 success + 50 failure，或总样本 >= 500 且类别均衡
3. 标签人工抽检一致率 >= 95%
```

第一版 value 仅做 success/failure 二分类。

**样本来源与评测隔离（r1 新增）**：§16 评测集（3–5 任务 × 50 trials）**不得**复用作 value 训练样本，否则造成评测污染。value 所需的 ≥50 success + ≥50 failure/任务须来自**独立的 rollout 采集**（额外遥操/仿真/failure rollout，见 §10.1），与评测 trials 的种子、初始条件分离并记录在案。若无法获得足量独立样本，则不启动 value head。

---

## 12. 显存与吞吐核算

| 档位 | Backbone | GPU | 说明 |
|------|----------|-----|------|
| Small | 小 DiT <1B | 4090 24G 可行 | 冒烟、A0 |
| Base | Wan2.2-5B + LoRA | A100/H100 80G | A1/A1c/B/C |

> **Compute 预算提示（r1 新增）**：由 §7.5 locked-same 约束，A1c 与 B 的 `forward_modes / forward_probs / steps` 完全一致，因此 **A1c 与 B 的 GPU 小时基本相等**。主判据 `A1c vs B` 需要跑**两组等价的 5B LoRA 混合前向训练**，主路径总 compute ≈ `2 × (B 单组)` + A0 + 外部 baseline + Stage C。排期与卡数规划务必按此计。

正式训练前实测并记录：

```text
- 参数量（backbone / feature adapter / action expert）
- FULL vs FAST forward 显存峰值
- mean_pool vs perceiver_resampler 显存峰值
- perceiver 相对 mean_pool 推理 ms 增量
- 单步 forward+backward 时间
- 单卡最大 batch size
- 单视角 vs 多视角、chunk=8 vs 16
```

---

## 13. 训练流程 A0/A1/A1c/B/C

### 13.0 Week 0 冒烟

```text
1. preflight.py 检查 τ₀-WM + Wan 权重
2. Phase 0: 官方 TauPolicy baseline 跑通
3. Phase 1: layer hooks + Feature Adapter + Mini Action Expert
4. Action Interface Tests 全部通过
5. <100 条数据过拟合单任务
6. vam_only_loop.py --mock 或接 server 打通闭环
7. benchmark_vam.py 记录基线延迟（单 seed + 3 seed）
8. data_compatibility.yaml 初版 + 抽检 100 条
```

### 13.1 A0：Action Baseline

```yaml
id: A0
train_mode: action_only
forward_mode: fast_only
video_backbone: frozen
feature_adapter: mean_pool
action_expert: full
steps: 10000-30000
lr: 1e-4
loss: L_action
```

### 13.2 A1：LoRA Control（扩展实验，Week 2 选做）

```yaml
id: A1
train_mode: action_only
forward_mode: fast_only
video_backbone: LoRA rank 32-64
feature_adapter: mean_pool
action_expert: full
steps: 50000-80000
lr_action: 5e-5
lr_lora: 1e-5
loss: L_action
```

### 13.3 A1c：Compute-Matched Control

A1c 匹配 B 的 backbone forward 次数与训练 compute，但不使用 `L_video`。详见 §7.2–§7.5。

```yaml
id: A1c
train_mode: action_only
forward_modes: [full, fast]
forward_probs: [0.5, 0.5]
video_backbone: LoRA rank 32-64
feature_adapter: perceiver_resampler    # 与 B 相同，见 locked-same 表
action_expert: full
steps: same_as_B
lr_action: 5e-5
lr_lora: 1e-5
lambda_video: 0                         # FULL 模式 L_video = 0
supervision_mask: true
record:
  backbone_forward_count: true
  gpu_hours: true
```

**B 必须相对 A1c 通过，才能说明 video co-training 真有效。**

### 13.4 B：WAM Joint

```yaml
id: B
train_mode: action + video
forward_modes: [full, fast]
forward_probs: [0.5, 0.5]
coupling: cross-attn adapter
video_backbone: LoRA rank 32-64
feature_adapter: perceiver_resampler
action_expert: full
steps: 50000-100000
lr_action: 5e-5
lr_lora: 1e-5
lambda_action: 1.0
lambda_video: 0.3
supervision_mask: true
```

### 13.5 C：Target Deploy Fine-tune（含防遗忘）

v3.2 在 Stage C 保留轻量 `L_video`，防止 target-only 微调遗忘 B 学到的 world features。

**关键一致性约束（r1 修订）**：`L_video` 只能在 **FULL 分支** 计算（§7.2），FAST 分支按 §7.3 恒有 `L_video=0`。因此 Stage C 不能是纯 FAST，否则 `λ_video` 无处施加、防遗忘失效。Stage C 采用 **以 FAST 为主的混合前向**：绝大多数步走 FAST（对齐部署），少量步走 FULL 施加 video 正则。

```yaml
id: C
train_mode: action + light_video
forward_modes: [full, fast]             # 以 FAST 为主，少量 FULL 承载 video 正则
forward_probs: [0.2, 0.8]               # 部署对齐优先，仅 20% 步施加防遗忘
data: Stage C 混合（target + failure）
video_backbone: frozen 或 LoRA
feature_adapter: B_perceiver            # 见 §13.6，固定用主路径胜出 adapter
action_expert: full
steps: 5000-20000
lr: 1e-5
lambda_action: 1.0
lambda_video: 0.1                       # 仅 FULL 分支生效，不追求生成质量
supervision_mask: true
```

说明：

- FULL 步（20%）用 §7.2 spec 施加 `0.1 * L_video`，保持 backbone world 表征稳定。
- FAST 步（80%）与部署路径完全一致，`L_video=0`。
- **替代方案（若严格禁止 C 出现 FULL）**：改用 **feature distillation**——冻结 B 作 teacher，Stage C student 的 FAST world tokens 对齐 teacher 的 world tokens（`L_distill` 替代 `L_video`）。此方案纯 FAST，但需额外常驻 teacher 显存。默认采用上面的混合前向方案。

### 13.6 Stage C 的 `feature_adapter` 与 backbone 来源（r1 新增）

Stage C 的初始化必须**唯一确定**，避免「best_from_B」歧义：

```yaml
stage_c_init:
  feature_adapter: B_perceiver           # 固定为主路径 B（§13.4），非扩展消融变体
  backbone_lora: from B_perceiver best checkpoint
  action_expert: from B_perceiver best checkpoint
  forbidden_sources: [B_mean_pool, B_align_off, B_video_grad_only]  # 不得作为 C 的起点
```

原因：Stage C 是主结论的部署落地延续，必须承接主判据胜出的 `B_perceiver`；若从扩展消融变体初始化，会切断「B 通过 → C 部署」的因果链。

---

## 14. Checkpoint 选择协议（v3.2 新增）

所有 A0/A1/A1c/B/C 使用 **同一协议**，避免 cherry-pick：

```yaml
model_selection:
  primary_metric: perturbed_success_rate
  secondary_metric: clean_success_rate
  eval_every: 2000 steps
  eval_tasks: fixed_task_set          # §15.1
  eval_trials_per_task: 10             # 训练中快速 eval；最终报告用 50
  select: max primary_metric over last 3 eval checkpoints
  tie_breaker: lower val_action_loss
  tie_breaker_2: lower inference_latency_p50

artifacts:
  - checkpoints/best_{A0,A1c,B,C}.pt
  - reports/checkpoint_selection.json  # 记录每步 eval 与选中理由
```

最终 bootstrap 报告必须使用 `best_*` checkpoint，不得事后另选 step。

---

## 15. 推理闭环与安全层

### 15.1 Fast Mode

```text
1. 读取 obs + state + prompt
2. VAE encode 当前帧
3. Backbone FAST forward
4. Feature Adapter → compact world tokens
5. Action Expert 5 步去噪
6. candidate_filter 校验/排序
7. 执行前 2-4 步
8. 重新观测，循环
```

### 15.2 candidate_filter

复用 `experiments/tau0_wm_sim/candidate_filter.py`：

```text
- 惩罚过大 EEF 跳变
- 惩罚 quaternion 跳变
- 拒绝 gripper 超出范围
- 可选：2-3 个 seed 采样，RCS-lite 选最稳候选（见 §3.2 延迟预算）
```

### 15.3 闭环参数

```yaml
action_horizon: 16
execute_steps_per_chunk: 2-4
denoise_steps: 5
target_control_frequency: 见 §3.1 延迟分档
default_inference: single_seed
multi_seed: optional_high_risk_only
```

---

## 16. 评测协议

### 16.1 最小任务集

第一版固定 3-5 个任务：

```text
- pick
- place
- push
- open drawer
- stack（可选）
```

```yaml
tasks: 3-5
trials_per_task: 50
random_seeds: 固定并记录
```

### 16.2 扰动

第一版只做：

```text
- 物体初始位置扰动
- 光照变化
```

### 16.3 对比模型

**主路径（Week 2 必做）：**

```text
- Diffusion Policy 或 ACT（外部 baseline）
- A0
- A1c
- B_perceiver_resampler
```

**扩展（Week 2 选做 / Week 3）：**

```text
- A1
- B_mean_pool
- B_align_off
- B_video_grad_only
- C
```

### 16.4 统计显著性

```yaml
bootstrap:
  num_samples: 1000
  confidence: 0.95

pass_rule:
  macro:
    metric: success_rate_delta_vs_A1c
    require_ci_lower_bound_above_zero: true
  perturbed:
    min_delta: +5%
    require_ci_lower_bound_above_zero: true
  per_task:
    max_regression_vs_A1c: 5%
```

报告必须包含：

```text
- macro 平均成功率 + 95% bootstrap CI
- clean / perturbed 分列
- per-task 成功率表 + 相对 A1c delta
- 完成时间
- 推理延迟 p50/p95（单 seed + 3 seed）
- 实际闭环频率
- gpu_hours / backbone_forward_count
- checkpoint_selection.json 摘要
```

### 16.5 最小可检测效应量与试验数（r1 新增）

3–5 任务 × 50 trials = 150–250 次试验时，需先确认门槛「+3% macro 且 CI 下界 >0」在统计上可检测，避免设定注定通不过的目标。

粗略估算（成功率为二项分布，macro 为任务均值）：

| 基线成功率 | macro trials | 检测 +3% 的把握 | 建议 |
|-----------|-------------|----------------|------|
| ~50% | 250 | 偏低（方差最大） | 提高 trials 或减任务数 |
| ~70% | 250 | 中等 | 可行，建议 trials/task ≥ 75 |
| ~80% | 250 | 中高 | 可行 |
| ~90% | 250 | 较高（方差小） | 可行，+3% 更易分辨 |

规则：

```yaml
power_check:
  # Week 1 用 A0/baseline 估计基线成功率后，反推所需 trials
  target_mde_macro: 3%          # 最小可检测效应
  target_power: 0.8
  action_if_underpowered:
    - increase_trials_per_task: [50 → 75 → 100]
    - or reduce_task_count: [5 → 3]  # 集中方差
  per_task_mde: 5%              # per-task 门槛较宽，接受更高方差
```

**注意**：per-task +5% / 50 trials 方差较大，per-task 判据只作**防退化红线**（不得下降 >5%），不作提升的强证据；提升结论以 macro + perturbed 的 CI 为准。

---

## 17. 消融与回退

### 17.1 必做消融

| ID | 对比 | 目的 | 优先级 |
|----|------|------|--------|
| A1c vs B | +L_video | **WAM 核心判据** | P0 |
| B align on/off | Train-test 对齐 | 验证混合 forward | P0 |
| A0 vs A1c | LoRA + compute | 隔离 backbone 微调 | P1 |
| B mean vs perceiver | Feature adapter | 验证空间结构收益 | P2 |
| B_video_grad_only vs B | video 梯度 shaping | 分离梯度效应 | P2 |
| forward_probs [0.3,0.7] vs [0.5,0.5] | FULL/FAST 比例 | 省算力 + 部署对齐权衡 | P2 |
| 5 vs 10 denoise steps | 延迟 | 工程权衡 | P2 |
| execute 2 vs 4 steps | 闭环稳定性 | 控制策略 | P2 |

### 17.2 回退决策树

若 B 相对 A1c 无提升：

```text
1. 评测：CI 是否跨 0？trial 数是否不足？per-task 是否退化？
2. Checkpoint 选择是否一致？是否用了非 best checkpoint？
3. Action Interface Tests 是否全部通过？
4. A1c 与 B 的 locked-same 表是否被违反？forward_count 差异 > 2%？
5. 数据：registry verified？时间对齐 / 坐标系 / 视角 / 归一化
6. Supervision mask 是否错误监督了不兼容动作？
7. Feature adapter：mean_pool → perceiver_resampler
8. Train-test 对齐：调 forward_probs 或加 feature distillation
9. 耦合：方案 B → 方案 A（MoT）
10. 仍无效 → 暂停 WAM 扩展，勿直接上 DINO/14B/Planner
```

若 B 有提升但 deploy 差：

```text
→ Stage C（λ_video=0.1）target 微调
→ candidate_filter / execute_steps 调整
→ 收集 failure rollout
→ 仍差再考虑 value head
```

---

## 18. 资源配置

### 18.1 Small

| 项 | 配置 |
|----|------|
| GPU | 1-4× 4090 |
| Backbone | 小 DiT |
| 用途 | 冒烟、A0、Phase 0/1 |

### 18.2 Base

| 项 | 配置 |
|----|------|
| GPU | 4-8× A100/H100 80G |
| Backbone | Wan2.2-5B + LoRA |
| Action Expert | 500M-800M |
| Feature Adapter | Perceiver 64 latents |
| 用途 | A1c/B/C 主实验 |

---

## 19. 实施里程碑与交付物

### 19.1 时间表（v3.2 精简排期）

| 周 | 必做 | 选做 |
|----|------|------|
| Week 0 | preflight、Phase 0/1、Action Interface Tests、显存/延迟实测、registry 初版、固定任务集 | — |
| Week 1 | A0、外部 DP/ACT baseline、闭环 benchmark、Phase 2 训练模式 | A1 |
| Week 2 | **A1c + B_perceiver + B_align 消融** | B_mean_pool、A1 |
| Week 3 | Stage C（λ_video=0.1）、完整 bootstrap 报告、MoT/Value 决策 | B_video_grad_only |

原则：**Week 2 主判据 `A1c vs B_perceiver` 不得被排期压垮**；扩展消融可延后至 Week 3 或资源允许时。

> **资源不足时的降级顺序（r1 新增）**：A1c 与 B 各需一组等价 5B LoRA 训练（见 §12 Compute 预算提示）。若 Week 2 卡数吃紧，按以下优先级保主判据：
> 1. 必保：`A1c + B_perceiver`（核心判据）。
> 2. 次保：`B_align` 消融 → 可降为 Week 3。
> 3. 最后：`B_mean_pool`、`A1`、`B_video_grad_only` → 资源允许才做。

### 19.2 交付物

```text
reports/
  preflight.json
  action_interface_tests.json
  benchmark_latency.json          # 含 single_seed + multi_seed
  vam_loop.json
  ablation_summary.md
  bootstrap_ci.json
  checkpoint_selection.json
  training_meta.json              # backbone_forward_count, gpu_hours

configs/
  data_compatibility.yaml
  mini_stage_a0.yaml
  mini_stage_a1.yaml              # 选做
  mini_stage_a1c.yaml
  mini_stage_b_perceiver.yaml
  mini_stage_b_mean.yaml          # 选做
  mini_stage_c.yaml

checkpoints/
  best_A0.pt
  best_A1c.pt
  best_B.pt
  best_C.pt

docs/
  Apex-WAM-Mini-Design-v3.2.md
  experiment_one_page_summary.md
```

---

## 20. 结论

v3.2 在 v3.1「实验可信」基础上，补齐「**可复现实施**」的最后缺口：

```text
v3   → 能不能工程实施
v3.1 → 实验结论是否可信
v3.2 → 对照是否公平、实现是否一致、排期是否可完成
```

v3.2 的关键判据不变，但判定更严格：

> **在 locked-same 协议、形式化 forward spec、registry 监督、bootstrap CI（macro + perturbed + per-task）通过的条件下，B 是否相对 A1c 显著提升？**

若通过 → video co-training 在当前数据与机器人上确实带来收益，可升级 MoT、Value 或完整 Apex-WAM。  
若不通过 → 优先修 forward spec 对齐、registry、Action Interface、Feature Adapter 与 train-test 对齐，**不要**直接堆 DINO、14B 或 Planner。

---

## 附录 A：v3.1 → v3.2 变更摘要

| 主题 | v3.1 | v3.2 |
|------|------|------|
| A1c forward | `full_like` 文字描述 | §7.2 形式化 FULL spec |
| A1c vs B 公平性 | `same_as_B` 一句 | §7.5 locked-same 表 + forward_count 验收 |
| 数据兼容 | `action_space` 字符串 | §10.3 `data_compatibility.yaml` |
| Stage C | `action only` | §13.5 `λ_video=0.1` 防遗忘 |
| 模型选择 | 未规定 | §14 checkpoint 协议 |
| 代码路径 | 复用列表 | §4.2 Phase 0/1/2 改造路径 |
| Week 2 | A1+A1c+B_mean+B_perceiver | §19.1 主路径 A1c+B_perceiver |
| 通过线 | macro only | §3.3 macro + per-task + perturbed |
| 延迟 | 不含 multi-seed | §3.2 含 multi-seed 预算表 |

---

## 附录 B：v3.2 内部修订记录（r1）

针对 v3.2 首版发现的内部一致性与严谨性问题的修订。

| # | 优先级 | 问题 | 修订 | 位置 |
|---|--------|------|------|------|
| 1 | P0 | Stage C 写 `forward_modes:[fast]` 又要 `λ_video=0.1`，但 FAST 恒 `L_video=0`，防遗忘失效 | 改为 `[full,fast]=[0.2,0.8]`，`λ_video` 仅 FULL 生效；给出 feature distillation 替代方案 | §13.5 |
| 2 | P1 | `best_from_B` 在多个 B 变体下歧义 | 固定为 `B_perceiver`，禁用扩展消融变体作 C 起点 | §13.6 |
| 3 | P1 | A1c ≈ B 使主路径 compute 近乎翻倍，预算/排期未反映 | 加 Compute 预算提示 + 资源不足降级顺序 | §12, §19.1 |
| 4 | P2 | 通过线缺最小可检测效应量/功效支撑 | 加 MDE 与 trials 估算表 + power_check | §16.5 |
| 5 | P2 | `forward_probs` 为固定拍脑袋值 | 列入 P2 消融 `[0.3,0.7] vs [0.5,0.5]` | §17.1 |
| 6 | P2 | Value 样本来源与评测集冲突未说明 | 强制独立 rollout，禁止复用评测 trials | §11.2 |
| 7 | P2 | registry `verified` 仅人工声明 | 加重放误差/坐标系一致率等量化门槛 | §10.3 |
