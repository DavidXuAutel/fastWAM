# Apex-WAM-Mini：Franka 机械臂 Profile 补充

> 相对 `Apex-WAM-Mini-Design-v3.2.md` 的 **双 profile 扩展**  
> 默认仍为 **G1**；Franka 用于 **LIBERO 公开数据可监督 L_action** 的验证路径。

---

## 1. 为什么增加 Franka profile

| 问题 | Franka profile 解法 |
|------|---------------------|
| G1 eef6d 20-dim 无开箱公开集 | LIBERO 四套件 **7-dim delta eef**，FastWAM 已验证 |
| Stage B 缺 L_action 数据 | Franka 档 LIBERO → `compatible_action`（verify 后） |
| 仅 video co-training 无法验证 WAM 主假设 | 可在 Franka 上先跑通 **A1c vs B** 再迁移 G1 |

---

## 2. Profile 对照

| 项 | `g1`（默认） | `franka` |
|----|--------------|----------|
| 目标本体 | AgiBot G1 双臂 | Franka Panda 单臂 |
| action | eef6d **relative** 20-dim | eef **delta** 7-dim |
| 部署输出 | 16-dim (τ₀-WM quat) | 7-dim |
| 视角 | head + 双腕 (3) | agent + wrist (2) |
| LIBERO | **video-only** | **L_action + L_video**（verify 后） |
| RoboTwin | video-only | video-only |
| Hydra 配置 | `configs/data/apex_wam_mini_g1.yaml` | `configs/data/apex_wam_mini_franka.yaml` |

Registry：`configs/data_compatibility.yaml` → `profiles.g1` / `profiles.franka`

---

## 3. Franka Stage B 混合

```yaml
franka_stage_b_mix:
  target_action_data: 0.15      # 可选真机 Franka teleop
  compatible_action_data: 0.45  # LIBERO 四套件
  video_only_or_incompatible: 0.35  # RoboTwin
  failure: 0.05
```

---

## 4. 使用方式

```bash
# 扫描 + 双 profile manifest
/opt/conda/bin/python3 experiments/apex_wam_mini/prepare_data.py all

# 仅 Franka
/opt/conda/bin/python3 experiments/apex_wam_mini/prepare_data.py manifest --stage B --profile franka

# verify LIBERO（Franka 档）
/opt/conda/bin/python3 experiments/apex_wam_mini/prepare_data.py verify --profile franka --mark-verified
```

训练时在 `sources.yaml` 设 `active_profile: franka`，或 Hydra 使用 `data=apex_wam_mini_franka`。

---

## 5. 与 G1 主路径关系

- Franka profile **不替代** G1 部署目标；用于 **公开数据上验证 WAM 假设** 与 **Week 0–2 冒烟**。
- G1 真机 `target_robot_teleop` 仍为 Stage C 必需。
- 两 profile **共享** video-only 池（RoboTwin），**不共享** L_action 监督。

---

## 6. 远端数据（当前）

| 数据 | G1 档 | Franka 档 |
|------|-------|-----------|
| LIBERO (~8.8 GB) | L_video | L_action（verify 后） |
| RoboTwin (~100 GB) | 下载中 | L_video |
