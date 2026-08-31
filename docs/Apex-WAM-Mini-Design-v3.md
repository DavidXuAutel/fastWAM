# Apex-WAM-Mini v3：低资源短周期闭环训练方案（工程可复现版）

> 文档版本：v3.0  
> 更新日期：2026-07-03  
> 前身：`Apex-WAM-Mini-Design-v2.md`  
> 目标：在有限算力和较短训练周期内，验证 WAM 的核心收益，并形成可闭环、可复现的机器人策略。
>
> **v3 相对 v2 的主要变化：**
> 1. 增加 **train-test 对齐训练**，缓解 Fast Mode 与 video co-training 的表征分布不一致。
> 2. 训练阶段重构为 **A0 / A1 / B / C**，LoRA 与 video loss 可单变量归因。
> 3. 新增 **与 τ₀-WM / FastWAM 现有代码的复用映射**，避免重复造轮子。
> 4. 固定 **最小任务集 + 量化通过线**。
> 5. **Stage B / Stage C 数据混合策略拆分**。
> 6. **方案 B feature 接口规格化**（层索引、pooling、token 上限）。
> 7. 推理侧纳入 **candidate_filter 安全层**。
> 8. **Value head 启动条件** 与 **Week 3 交付物清单**。

---

## 目录

1. [目标与通过线](#1-目标与通过线)
2. [核心裁剪](#2-核心裁剪)
3. [代码复用映射](#3-代码复用映射)
4. [模型结构](#4-模型结构)
5. [Train-Test 对齐训练](#5-train-test-对齐训练)
6. [动作空间](#6-动作空间)
7. [训练目标与损失](#7-训练目标与损失)
8. [数据方案](#8-数据方案)
9. [显存与吞吐核算](#9-显存与吞吐核算)
10. [训练流程 A0/A1/B/C](#10-训练流程-a0a1bc)
11. [推理闭环与安全层](#11-推理闭环与安全层)
12. [评测协议](#12-评测协议)
13. [消融实验与回退决策树](#13-消融实验与回退决策树)
14. [资源配置](#14-资源配置)
15. [实施里程碑与交付物](#15-实施里程碑与交付物)
16. [结论](#16-结论)

---

## 1. 目标与通过线

`Apex-WAM-Mini` 是 `Apex-WAM` 的可执行裁剪版，核心问题是：

> **在当前机器人和数据条件下，video co-training 是否能提升动作泛化和闭环成功率？**

### 1.1 设计原则

- 使用预训练视频模型作为世界表征骨干。
- 保留 `video co-training`，用未来视频预测塑造动作表征。
- 推理时默认不生成未来视频（Fast Mode），保证低延迟闭环。
- 使用单一目标机器人动作空间，先把仿真/真机闭环跑通。
- **在固定评测协议下，用 A0/A1/B 单变量消融验证 WAM**，而非跨 Stage 混比。

### 1.2 资源与延迟目标

| 指标 | 目标 |
|------|------|
| 训练资源 | 1-8 张 A100/H100，或 1-4 张 RTX 4090（仅 Small 档） |
| 训练时间 | 3-14 天（含调参与重训余量） |
| 推理延迟 | 见 [1.3 延迟分档](#13-延迟分档) |

### 1.3 延迟分档

| 配置 | 单次 backbone 前向 | action 去噪 | 目标频率 |
|------|-------------------|-------------|----------|
| 小 DiT + 单视角 + 5 步去噪 | 较低 | 5 步 | 5-7Hz |
| Wan2.2-5B + 多视角 + 5 步去噪 | 较高 | 5 步 | 2-4Hz |

> 7Hz 是 DreamZero 用 14B + 系统级优化才达到的目标，Mini 版不直接对标。

### 1.4 量化通过线（v3 新增）

WAM 主假设 **通过** 需同时满足：

```yaml
# 相对 Diffusion Policy / ACT（同一评测协议）
vs_baseline_clean: +5% 平均成功率
vs_baseline_perturbed: +8% 平均成功率   # position + lighting 扰动

# WAM 核心判据（同一 backbone 训练策略下）
vs_A1_L_action_only: +3% 平均成功率     # B 相对 A1，隔离 LoRA 变量

# 工程指标
inference_latency: 满足 1.3 对应档位
ablation_report: 完成 A0/A1/B 对照报告
```

若 B 相对 A0 有提升但相对 A1 无提升，应归因于 **LoRA 微调 backbone**，而非 video co-training。

---

## 2. 核心裁剪

相对完整 `Apex-WAM`，Mini 版移除：

| 模块 | 处理 | 原因 |
|------|------|------|
| 14B Video Backbone | Wan2.2-5B 或小 DiT | 降成本 |
| DINO Dynamics Expert | 暂不做 | 收益未验证 |
| ULA 跨本体 | 暂不做 | 先固定单机器人 action space |
| ACVS Planner | 暂不做 | 用 candidate_filter 作 RCS-lite |
| 复杂 Value/Reward | 后置，有条件启动 | 见 §7.2 |
| 50K 小时数据 | 100-1000 小时 | 短周期可执行 |

保留：

| 模块 | 原因 |
|------|------|
| Wan VAE / T5 | 复用 τ₀-WM 接口 |
| Video Backbone + LoRA | video co-training |
| Action Expert（方案 B） | 可执行 action chunk |
| Fast Mode | 低延迟闭环 |
| candidate_filter | 部署安全，几乎零额外延迟 |

---

## 3. 代码复用映射（v3 新增）

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

**Mini 真正新增（需自行实现或扩展）：**

```text
configs/mini_stage_{a0,a1,b,c}.yaml
train/infer 对齐模式开关（§5）
A0/A1/B 训练脚本（可 fork runner/posttrain.py）
backbone feature layer 配置（§4.3）
评测报告模板 reports/ablation_summary.md
```

---

## 4. 模型结构

### 4.1 总体结构

```text
输入: 多视角 obs + prompt + state

编码:
  Wan VAE（当前帧 + 训练时未来帧）
  T5
  State MLP

核心:
  Video Backbone（frozen / LoRA）
  Action Expert（cross-attn adapter，方案 B）

输出:
  action chunk [T, C]
```

### 4.2 模块配置

| 模块 | 配置 | 训练 |
|------|------|------|
| VAE / T5 | Wan2.2 | 冻结 |
| Video Backbone | Wan2.2-5B 或小 DiT | A0 冻结；A1/B LoRA |
| Action Expert | 12 层，256M-800M | 全训练 |
| State Encoder | 2 层 MLP | 全训练 |
| Value Head | 小 MLP | 有条件启动，§7.2 |

### 4.3 方案 B：Feature 接口规格（v3 新增）

Action Expert 通过 cross-attn 读取 backbone 多层特征，第一版固定：

```yaml
backbone_num_layers: 30          # Wan 示例
backbone_feature_layers: [8, 16, 24]
feature_pooling: mean            # 对 spatial-temporal tokens 做 mean pool
max_feature_tokens: 256          # 每层 pool 后截断/投影到此上限
action_expert_injection: per_block  # 每个 action block cross-attn 到最近层 feature

# 第一版 ablation（可选）
ablation_layers:
  shallow_only: [8]
  multi_scale: [8, 16, 24]
```

Action Expert 每个 block：

```text
self-attn(action tokens + state token)
cross-attn → pooled world features（来自最近 layer index）
cross-attn → text context
FFN
```

### 4.4 耦合方式选型

| 方案 | MVP | 升级条件 |
|------|-----|----------|
| B：cross-attn adapter | **默认** | — |
| A：MoT 共享注意力 | 后置 | B 已通过 §1.4，且需要更高上限 |

---

## 5. Train-Test 对齐训练（v3 新增，P0）

### 5.1 问题

- **训练时**：backbone 对未来 noisy video latent 去噪，算 `L_video`。
- **推理时（Fast Mode）**：backbone 仅对当前帧单次前向，不跑 future branch。

Action Expert 在训练时见到的 world features 与推理时可能分布不一致，削弱 video co-training 收益。

### 5.2 推荐方案：混合 forward（Option 1）

每个 training step 以概率 `p` 选择 forward 模式：

```yaml
p_align: 0.5   # 可调 0.3-0.5

# Mode FULL（video co-training）
- 输入: current + noisy future video latent
- 损失: L_video + L_action
- backbone: 完整 video denoise forward

# Mode FAST（推理对齐）
- 输入: 仅 current frame latent
- 损失: 仅 L_action
- backbone: 单次 forward，与 Fast Mode 一致
- action expert: cross-attn 到 current-only features
```

Stage B 训练配置：

```yaml
train_forward_modes: [full, fast]
train_forward_probs: [0.5, 0.5]
```

**判据**：B（含 align）相对 A1 的提升，才视为 video co-training + 对齐训练的有效组合。

### 5.3 备选方案（按需）

| 选项 | 做法 | 何时用 |
|------|------|--------|
| Option 2 | Feature distillation：约束 fast features ≈ full features | align 混合不够时 |
| Option 3 | B1 正常训练 → B2 冻结 video head，仅 fast forward 微调 action | 部署前最后一轮 |

---

## 6. 动作空间

Mini 版只支持 **一个** 目标机器人 action space。

| 场景 | 推荐 |
|------|------|
| RoboTwin / 关节 | `joint absolute` |
| G1 双臂末端 | `eef6d relative`（与 τ₀-WM 预训练一致） |

G1 双臂 20 维（模型内部）：

```text
[left_xyz(3), left_rot6d(6), left_gripper(1),
 right_xyz(3), right_rot6d(6), right_gripper(1)]
```

部署输出可仍为 τ₀-WM 的 16 维（quat + gripper 0-1），由现有 `TauPolicy` 后处理完成。

时序：

```yaml
action_horizon: 16        # MVP；稳定后 33
execute_steps_per_chunk: 2-4
state: Tensor[1, C]
```

---

## 7. 训练目标与损失

### 7.1 主损失

```python
# Mode FULL
L = λ_action * L_action + λ_video * L_video

# Mode FAST
L = λ_action * L_action
```

```yaml
lambda_action: 1.0
lambda_video: 0.3
```

### 7.2 Value Head（有条件启动）

**不在 A0/A1/B 主路径中启用。** 启动需同时满足：

```text
1. B 相对 A1 已通过 §1.4 通过线
2. 已收集 >= 200 条带 success/failure 标签的 rollout
3. 标签经人工抽检一致率 >= 95%
```

第一版 value 仅 **success / failure 二分类**：

```python
L_total = L_action + 0.3 * L_video + 0.1 * L_value   # 仅 Stage C+ 可选
```

Label 来源：仿真环境 success 标志；真机脚本规则 + 人工抽检。

---

## 8. 数据方案

### 8.1 规模建议

| 类型 | 规模 |
|------|------|
| 目标机器人遥操 | 20-100 h |
| 仿真 | 50-200 h |
| 公开 robot | 100-500 h |
| failure rollout | 5-30 h |

### 8.2 样本格式

```python
{
    "video": Tensor[C, V, T, H, W],
    "state": Tensor[1, action_dim],
    "actions": Tensor[action_horizon, action_dim],
    "caption": str,
    "success": Optional[bool],
    "has_action": bool,
    "has_video": bool,
}
```

### 8.3 按 Stage 拆分混合（v3 新增）

**Stage B（验证 WAM，求泛化信号）：**

```yaml
simulation: 0.40
public_robot: 0.30
robot_target: 0.25
failure: 0.05
```

**Stage C（部署微调，求闭环成功率）：**

```yaml
robot_target: 0.90
failure: 0.10
# 不混入 public/sim，避免 action space 污染
```

---

## 9. 显存与吞吐核算

| 档位 | Backbone | GPU | 说明 |
|------|----------|-----|------|
| Small | 小 DiT <1B | 4090 24G 可行 | 接口验证、A0 冒烟 |
| Base | Wan2.2-5B + LoRA | A100/H100 80G | A1/B 主实验 |

正式训练前实测并记录：

```text
- 参数量（backbone / action expert）
- FULL vs FAST forward 显存峰值
- 单步 forward+backward 时间
- 单卡最大 batch size
- 单视角 vs 多视角、chunk=8 vs 16
```

---

## 10. 训练流程 A0/A1/B/C

v3 用 **单变量友好** 的四阶段替代 v2 的 Stage A/B/C。

### 10.0 Day-1 冒烟测试（Week 0，必做）

```text
1. preflight.py 检查 τ₀-WM + Wan 权重
2. <100 条数据过拟合单任务
3. vam_only_loop.py --mock 或接 server 打通闭环
4. 验证 action 空间、归一化、坐标系、gripper 方向
5. benchmark_vam.py 记录基线延迟
```

### 10.1 A0：Action Baseline（frozen backbone）

```yaml
id: A0
train_mode: action_only
forward_mode: fast_only          # 与推理一致
video_backbone: frozen
action_expert: full
steps: 10000-30000
lr: 1e-4
effective_batch_size: 64-128
loss: L_action
```

目的：纯 action diffusion baseline，不含 LoRA、不含 video loss。

### 10.2 A1：LoRA Control（隔离 LoRA 变量）

```yaml
id: A1
train_mode: action_only
forward_mode: fast_only
video_backbone: LoRA rank 32-64
action_expert: full
steps: 50000-80000
lr_action: 5e-5
lr_lora: 1e-5
loss: L_action
```

目的：测量 **仅 LoRA 微调 backbone** 的收益。B 必须相对 A1 比较，而非相对 A0。

### 10.3 B：WAM Joint（核心实验）

```yaml
id: B
train_mode: action + video
forward_modes: [full, fast]
forward_probs: [0.5, 0.5]
coupling: cross-attn adapter
video_backbone: LoRA rank 32-64
action_expert: full
steps: 50000-100000
lr_action: 5e-5
lr_lora: 1e-5
lambda_action: 1.0
lambda_video: 0.3
backbone_feature_layers: [8, 16, 24]
```

**WAM 主假设判据：B 相对 A1 满足 §1.4。**

### 10.4 C：Target Deploy Fine-tune

```yaml
id: C
train_mode: action          # 可保留少量 L_video（λ=0.1）防遗忘
forward_mode: fast_only
data: Stage C 混合（§8.3）
video_backbone: frozen 或 LoRA
action_expert: full
steps: 5000-20000
lr: 1e-5
```

可选：满足 §7.2 后加 value head。

---

## 11. 推理闭环与安全层

### 11.1 Fast Mode 流程

```text
1. 读取 obs + state + prompt
2. VAE encode 当前帧
3. Backbone FAST forward → world features（layers [8,16,24]）
4. Action Expert 5 步去噪 → action chunk
5. candidate_filter 校验/排序（可选多 seed）
6. 执行前 2-4 步
7. 重新观测，循环
```

### 11.2 candidate_filter 安全层（v3 纳入）

复用 `experiments/tau0_wm_sim/candidate_filter.py`：

```text
- 惩罚过大 EEF 跳变
- 惩罚 quaternion 跳变
- 拒绝 gripper 超出 [0, 120]（或归一化范围）
- 可选：2-3 个 seed 采样，RCS-lite 选最稳候选
```

不改变 WAM 架构，可显著提高真机闭环稳定性，额外延迟可忽略。

### 11.3 推荐闭环参数

```yaml
action_horizon: 16
execute_steps_per_chunk: 2-4
denoise_steps: 5
target_control_frequency: 见 §1.3
```

---

## 12. 评测协议

### 12.1 最小任务集（v3 固定）

第一版固定 **3-5 个任务**，例如从下列类型选取：

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

### 12.2 扰动（第一版只做 2 种）

```text
- 物体初始位置扰动
- 光照变化
```

干扰物、新实例泛化留到 Phase 2。

### 12.3 对比模型

必须在同一协议下评测：

```text
- Diffusion Policy 或 ACT（外部 baseline）
- A0（L_action, frozen）
- A1（L_action, LoRA）
- B（L_action + L_video, LoRA + align）
- C（deploy fine-tune，可选）
```

### 12.4 报告指标

```text
- 平均成功率（clean / perturbed 分列）
- 完成时间
- 干预率（真机）
- 推理延迟 p50/p95
- 实际闭环频率
```

通过线见 [§1.4](#14-量化通过线v3-新增)。

---

## 13. 消融实验与回退决策树

### 13.1 必做消融

| ID | 对比 | 目的 |
|----|------|------|
| A0 vs A1 | LoRA 本身 | 隔离 backbone 微调 |
| A1 vs B | +L_video + align | **WAM 核心** |
| B: align on vs off | forward 混合 | 验证 §5 必要性 |
| B: layers [8] vs [8,16,24] | feature 层 | 规格化调参 |
| denoise 5 vs 10 steps | 延迟 | 工程权衡 |
| execute 2 vs 4 steps | 闭环稳定性 | 控制策略 |

### 13.2 回退决策树

若 **B 相对 A1 无提升**：

```text
1. 评测是否可靠？trial 数、扰动是否有效？
2. 数据：归一化 / 坐标系 / 时间对齐 / 视角
3. 动作空间：joint vs eef6d，absolute vs relative
4. Train-test 对齐：提高 p_align 或 Option 2/3
5. 耦合：方案 B → 方案 A（MoT）
6. 仍无效 → 检查是否 worth continuing WAM；勿直接上 DINO/14B/Planner
```

若 **B 相对 A1 有提升但 deploy 差**：

```text
→ 优先 Stage C target-only 微调 + candidate_filter
→ 勿急于加 Value / Planner
```

---

## 14. 资源配置

### 14.1 Small（冒烟 + A0）

| 项 | 配置 |
|----|------|
| GPU | 1-4× 4090 |
| Backbone | 小 DiT |
| 时间 | 3-7 天 |
| 用途 | 全链路、A0 |

### 14.2 Base（A1/B/C 主实验）

| 项 | 配置 |
|----|------|
| GPU | 4-8× A100 80G / H100 |
| Backbone | Wan2.2-5B + LoRA |
| Action Expert | 500M-800M |
| 时间 | 7-14 天 |

---

## 15. 实施里程碑与交付物

### 15.1 时间表

| 周 | 内容 |
|----|------|
| Week 0 | 冒烟、preflight、显存实测、固定任务集与通过线 |
| Week 1 | 训练 A0；Diffusion Policy baseline；闭环 + benchmark |
| Week 2 | 训练 A1、B（含 align）；同协议对比 |
| Week 3 | Stage C 微调；消融报告；决定是否 MoT / Value |

### 15.2 Week 3 交付物清单（v3 新增）

```text
reports/
  preflight.json
  benchmark_latency.json
  vam_loop.json
  ablation_summary.md          # A0/A1/B vs baseline，含 §1.4 判据

configs/
  mini_stage_a0.yaml
  mini_stage_a1.yaml
  mini_stage_b.yaml
  mini_stage_c.yaml

checkpoints/
  best_A1.pt                   # LoRA control
  best_B.pt                    # WAM main

docs/
  本方案 v3 + 实验结论一页摘要
```

---

## 16. 结论

`Apex-WAM-Mini v3` 在 v2 基础上补齐了工程落地最关键的四块：

```text
1. Train-test 对齐（§5）     → Fast Mode 与 co-training 不打架
2. A0/A1/B 单变量消融（§10） → video loss 可归因
3. 代码复用映射（§3）        → 基于 τ₀-WM + tau0_wm_sim 快速开工
4. 量化通过线（§1.4, §12）   → 3 周有明确终点
```

核心取舍不变：

```text
保留: video co-training + action chunk + Fast Mode + candidate_filter
暂缓: DINO / ULA / 14B / ACVS / 复杂 value
升级: B 通过后 → MoT；C 稳定后 → 有条件 value
```

最终要回答的问题仍然只有一个：

> **在固定协议下，B 是否相对 A1 显著提升？**

若是，继续向完整 Apex-WAM 演进；若否，按 §13.2 回退，优先修数据、对齐与评测，而非堆模型复杂度。
