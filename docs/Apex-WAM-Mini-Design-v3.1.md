# Apex-WAM-Mini v3.1：低资源短周期闭环训练方案（实验可信版）

> 文档版本：v3.1  
> 更新日期：2026-07-03  
> 前身：`Apex-WAM-Mini-Design-v3.md`  
> 目标：在有限算力和较短训练周期内，验证 WAM 的核心收益，并形成可闭环、可复现、结论可信的机器人策略。
>
> **v3.1 相对 v3 的主要改进：**
> 1. 将 `mean pooling` 升级为可对照的 **Feature Adapter 双档设计**：`mean_pool` baseline 与 `perceiver_resampler` 推荐方案。
> 2. 新增 **A1c compute-matched control**，避免 B 的提升来自更多训练计算量。
> 3. Stage B 引入 **supervision mask**，防止 public/sim 数据动作空间不兼容时污染 action 监督。
> 4. 评测协议加入 **bootstrap 置信区间**，避免把随机波动当作提升。
> 5. 新增 **Action Interface Tests**，把坐标系、rot6d/quat、gripper、左右臂顺序等风险前置。
> 6. 收紧 **Value Head 启动条件**，要求每任务正负样本覆盖或总量/均衡达标。

---

## 1. 目标与核心问题

`Apex-WAM-Mini` 的核心问题保持不变：

> **在当前机器人和数据条件下，video co-training 是否能提升动作泛化和闭环成功率？**

v3.1 的重点不是继续扩大模型，而是让该问题的实验答案更可信：

- 消融要公平：B 不能因为训练计算量更多而“虚假领先”。
- 数据监督要干净：动作空间不兼容的数据不能监督 `L_action`。
- 特征接口要保留空间结构：不能只依赖全局 mean pooling。
- 评测要有统计意义：提升必须通过置信区间检验。
- 动作接口要先测通：避免坐标/归一化错误伪装成模型失败。

---

## 2. 版本定位

| 版本 | 定位 | 状态 |
|------|------|------|
| v1 | 裁剪版初稿 | 历史草案 |
| v2 | 工程优化版 | 中间版 |
| v3 | 工程可复现版 | 可实施主线 |
| **v3.1** | **实验可信版** | **推荐作为当前实施主文档** |

v3.1 不推翻 v3 的主线，只在实验公平性、特征接口、数据监督边界和测试协议上补强。

---

## 3. 成功标准与统计通过线

### 3.1 延迟分档

| 配置 | 单次 backbone 前向 | action 去噪 | 目标频率 |
|------|-------------------|-------------|----------|
| 小 DiT + 单视角 + 5 步去噪 | 较低 | 5 步 | 5-7Hz |
| Wan2.2-5B + 多视角 + 5 步去噪 | 较高 | 5 步 | 2-4Hz |

### 3.2 WAM 主假设通过线

WAM 主假设通过需同时满足：

```yaml
# 相对外部 baseline
vs_baseline_clean: +5% 平均成功率
vs_baseline_perturbed: +8% 平均成功率

# WAM 核心判据
vs_A1c_compute_matched: +3% 平均成功率

# 统计显著性
bootstrap_confidence: 95%
require_ci_lower_bound_above_zero: true

# 工程指标
inference_latency: 满足延迟分档
ablation_report: 完成 A0/A1/A1c/B 对照报告
```

解释：

- B 相对 A0 提升，只能说明“比冻结 baseline 好”。
- B 相对 A1 提升，说明可能来自 `L_video`，但仍可能混入训练计算量差异。
- **B 相对 A1c 提升**，才是 v3.1 中最可信的 WAM 核心判据。

---

## 4. 代码复用映射

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
train/infer 对齐模式开关
feature_adapter 配置
supervision mask 数据加载逻辑
bootstrap 评测脚本
reports/ablation_summary.md 模板
```

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

## 6. Feature Adapter 设计（v3.1 重点）

### 6.1 问题

v3 默认 `feature_pooling: mean`，实现简单，但会抹掉空间结构。机器人操作依赖：

- 夹爪与物体相对位置
- 接触区域
- 小物体位移
- 多视角差异

因此 v3.1 将 feature adapter 分成 baseline 与推荐方案。

### 6.2 Baseline：Mean Pool

```yaml
feature_adapter_baseline:
  type: mean_pool
  input_layers: [8, 16, 24]
  max_feature_tokens: 256
  preserve_view_tokens: false
```

优点：

- 实现简单
- 显存低
- 适合作为 Day-1 / A0 / A1 冒烟版本

缺点：

- 丢失空间信息
- 多视角关系被弱化
- 对小物体和接触动作不友好

### 6.3 Recommended：Perceiver Resampler

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

v3.1 默认建议：

```yaml
default_feature_adapter: perceiver_resampler
fallback_feature_adapter: mean_pool
```

必须消融：

```text
B_mean_pool vs B_perceiver_resampler
```

---

## 7. Train-Test 对齐训练

### 7.1 问题

训练时 backbone 做未来视频 denoise；推理时 Fast Mode 只看当前帧。Action Expert 读取的 world features 可能分布不一致。

### 7.2 混合 Forward

Stage B 以概率混合两种 forward：

```yaml
train_forward_modes: [full, fast]
train_forward_probs: [0.5, 0.5]
```

FULL：

```text
current + noisy future video latent
→ backbone full forward
→ L_action + L_video
```

FAST：

```text
current frame only
→ backbone fast forward
→ L_action
```

训练损失：

```python
if mode == "full":
    loss = L_action + 0.3 * L_video
else:
    loss = L_action
```

必须消融：

```text
B_align_on vs B_align_off
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

部署输出可仍为 τ₀-WM 的 16 维，由现有 `TauPolicy` 后处理完成：

```text
left xyz + left quat xyzw + left gripper +
right xyz + right quat xyzw + right gripper
```

---

## 9. Action Interface Tests（v3.1 新增，必做）

在任何正式训练前必须通过动作接口单元测试。否则训练失败可能只是坐标/归一化错误。

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

## 10. 数据方案与 Supervision Mask

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
}
```

### 10.3 Supervision Mask（v3.1 新增）

Stage B 会混入 public robot / sim / target robot。动作空间不兼容的数据不能监督 `L_action`。

```python
def build_supervision_mask(sample, target_action_space):
    action_compatible = (
        sample[\"actions\"] is not None
        and sample[\"action_space\"] == target_action_space
    )

    return {
        \"video\": True,
        \"action\": action_compatible,
        \"success\": sample.get(\"success\") is not None,
    }
```

规则：

| 数据类型 | `L_video` | `L_action` |
|----------|-----------|------------|
| target robot | ✓ | ✓ |
| compatible sim | ✓ | ✓ |
| incompatible public robot | ✓ | ✗ |
| ego/web video | ✓ | ✗ |
| failure rollout | ✓ | 仅 action compatible 时 ✓ |

### 10.4 Stage 数据混合

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
if mode == \"full\":
    loss = (
        1.0 * L_action * mask[\"action\"]
        + 0.3 * L_video * mask[\"video\"]
    )
else:
    loss = 1.0 * L_action * mask[\"action\"]
```

如果 `mask["action"] == False`，样本只参与 video co-training。

### 11.2 Value Head 启动条件（v3.1 收紧）

Value Head 不在 A0/A1/A1c/B 主路径中启用。启动需满足：

```text
1. B 相对 A1c 已通过主假设通过线
2. 每个任务 >= 50 success + 50 failure
   或总样本 >= 500 且类别均衡
3. 标签人工抽检一致率 >= 95%
```

第一版 value 仅做 success/failure 二分类。

---

## 12. 显存与吞吐核算

| 档位 | Backbone | GPU | 说明 |
|------|----------|-----|------|
| Small | 小 DiT <1B | 4090 24G 可行 | 冒烟、A0 |
| Base | Wan2.2-5B + LoRA | A100/H100 80G | A1/A1c/B/C |

正式训练前实测并记录：

```text
- 参数量（backbone / feature adapter / action expert）
- FULL vs FAST forward 显存峰值
- mean_pool vs perceiver_resampler 显存峰值
- 单步 forward+backward 时间
- 单卡最大 batch size
- 单视角 vs 多视角、chunk=8 vs 16
```

---

## 13. 训练流程 A0/A1/A1c/B/C

### 13.0 Week 0 冒烟

```text
1. preflight.py 检查 τ₀-WM + Wan 权重
2. Action Interface Tests 全部通过
3. <100 条数据过拟合单任务
4. vam_only_loop.py --mock 或接 server 打通闭环
5. benchmark_vam.py 记录基线延迟
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

### 13.2 A1：LoRA Control

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

### 13.3 A1c：Compute-Matched Control（v3.1 新增）

A1c 用来匹配 B 的 backbone forward 次数与训练 compute，但不使用 `L_video`。

```yaml
id: A1c
train_mode: action_only
forward_modes: [full_like, fast]
forward_probs: [0.5, 0.5]
video_backbone: LoRA rank 32-64
feature_adapter: same_as_B
action_expert: full
steps: same_as_B
lr_action: 5e-5
lr_lora: 1e-5
loss: L_action
no_video_loss: true
record:
  backbone_forward_count: true
  gpu_hours: true
```

解释：

- `full_like` 执行与 B 的 FULL 类似的 backbone forward，但不回传 `L_video`。
- A1c 控制「更多计算量」这一混杂变量。
- **B 必须相对 A1c 通过，才能说明 video co-training 真有效。**

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

### 13.5 C：Target Deploy Fine-tune

```yaml
id: C
train_mode: action
forward_mode: fast_only
data: Stage C 混合
video_backbone: frozen 或 LoRA
feature_adapter: best_from_B
action_expert: full
steps: 5000-20000
lr: 1e-5
```

---

## 14. 推理闭环与安全层

### 14.1 Fast Mode

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

### 14.2 candidate_filter

复用 `experiments/tau0_wm_sim/candidate_filter.py`：

```text
- 惩罚过大 EEF 跳变
- 惩罚 quaternion 跳变
- 拒绝 gripper 超出范围
- 可选：2-3 个 seed 采样，RCS-lite 选最稳候选
```

### 14.3 闭环参数

```yaml
action_horizon: 16
execute_steps_per_chunk: 2-4
denoise_steps: 5
target_control_frequency: 见延迟分档
```

---

## 15. 评测协议

### 15.1 最小任务集

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

### 15.2 扰动

第一版只做：

```text
- 物体初始位置扰动
- 光照变化
```

### 15.3 对比模型

```text
- Diffusion Policy 或 ACT
- A0
- A1
- A1c
- B_mean_pool
- B_perceiver_resampler
- C（可选）
```

### 15.4 统计显著性（v3.1 新增）

```yaml
bootstrap:
  num_samples: 1000
  confidence: 0.95

pass_rule:
  metric: success_rate_delta
  require_ci_lower_bound_above_zero: true
```

报告必须包含：

```text
- 平均成功率
- 95% bootstrap CI
- clean / perturbed 分列
- 完成时间
- 推理延迟 p50/p95
- 实际闭环频率
- gpu_hours / backbone_forward_count
```

---

## 16. 消融与回退

### 16.1 必做消融

| ID | 对比 | 目的 |
|----|------|------|
| A0 vs A1 | LoRA 本身 | 隔离 backbone 微调 |
| A1 vs A1c | 计算量本身 | 控制 compute |
| A1c vs B | +L_video | **WAM 核心判据** |
| B align on/off | Train-test 对齐 | 验证混合 forward |
| B mean vs perceiver | Feature adapter | 验证空间结构收益 |
| 5 vs 10 denoise steps | 延迟 | 工程权衡 |
| execute 2 vs 4 steps | 闭环稳定性 | 控制策略 |

### 16.2 回退决策树

若 B 相对 A1c 无提升：

```text
1. 评测：CI 是否跨 0？trial 数是否不足？
2. Action Interface Tests 是否全部通过？
3. 数据：时间对齐 / 坐标系 / 视角 / 归一化
4. Supervision mask 是否错误监督了不兼容动作？
5. Feature adapter：mean_pool → perceiver_resampler
6. Train-test 对齐：调 p_align 或加 feature distillation
7. 耦合：方案 B → 方案 A（MoT）
8. 仍无效 → 暂停 WAM 扩展，勿直接上 DINO/14B/Planner
```

若 B 有提升但 deploy 差：

```text
→ Stage C target-only 微调
→ candidate_filter / execute_steps 调整
→ 收集 failure rollout
→ 仍差再考虑 value head
```

---

## 17. 资源配置

### 17.1 Small

| 项 | 配置 |
|----|------|
| GPU | 1-4× 4090 |
| Backbone | 小 DiT |
| 用途 | 冒烟、A0、接口验证 |

### 17.2 Base

| 项 | 配置 |
|----|------|
| GPU | 4-8× A100/H100 80G |
| Backbone | Wan2.2-5B + LoRA |
| Action Expert | 500M-800M |
| Feature Adapter | Perceiver 64 latents |
| 用途 | A1/A1c/B/C 主实验 |

---

## 18. 实施里程碑与交付物

### 18.1 时间表

| 周 | 内容 |
|----|------|
| Week 0 | preflight、Action Interface Tests、显存实测、固定任务集 |
| Week 1 | A0、Diffusion Policy/ACT baseline、闭环 benchmark |
| Week 2 | A1、A1c、B_mean_pool、B_perceiver |
| Week 3 | C 微调、统计报告、决定是否升级 MoT / Value |

### 18.2 交付物

```text
reports/
  preflight.json
  action_interface_tests.json
  benchmark_latency.json
  vam_loop.json
  ablation_summary.md
  bootstrap_ci.json

configs/
  mini_stage_a0.yaml
  mini_stage_a1.yaml
  mini_stage_a1c.yaml
  mini_stage_b_mean.yaml
  mini_stage_b_perceiver.yaml
  mini_stage_c.yaml

checkpoints/
  best_A1.pt
  best_A1c.pt
  best_B.pt

docs/
  Apex-WAM-Mini-Design-v3.1.md
  experiment_one_page_summary.md
```

---

## 19. 结论

v3.1 的核心目标是让 Mini WAM 的结论更可信：

```text
v3 解决：能不能工程实施
v3.1 解决：实验结论是否可信
```

v3.1 的关键判据是：

> **在固定协议、compute-matched、bootstrap CI 通过的条件下，B 是否相对 A1c 显著提升？**

如果通过，说明 video co-training 在当前数据与机器人上确实带来收益，可以继续升级 MoT、Value 或完整 Apex-WAM。  
如果不通过，应优先修 action 接口、数据监督、feature adapter 与 train-test 对齐，而不是直接堆 DINO、14B 或 Planner。
