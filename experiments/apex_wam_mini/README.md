# Apex-WAM-Mini 数据准备

实现 `docs/Apex-WAM-Mini-Design-v3.2.md` §10 的数据 registry、supervision mask 与 stage manifest 工具链。

## 文件

| 文件 | 作用 |
|------|------|
| `sources.yaml` | 本地 LeRobot 路径与 stage 混合权重 |
| `registry.py` | 加载 `configs/data_compatibility.yaml` |
| `supervision.py` | `build_supervision_mask()` |
| `manifest.py` | 扫描数据集、生成 stage B/C manifest |
| `verify_source.py` | registry 验证与 artifact |
| `prepare_data.py` | CLI 入口 |
| `prepare_data.sh` | 一键 scan + manifest |

## 快速开始

```bash
cd /Users/xudazhong/Projects/FastWAM

# 扫描现有路径并生成 manifest（缺失数据会标记 MISSING）
bash experiments/apex_wam_mini/prepare_data.sh

# 单独命令
python3 experiments/apex_wam_mini/prepare_data.py scan
python3 experiments/apex_wam_mini/prepare_data.py manifest --stage B
python3 experiments/apex_wam_mini/prepare_data.py manifest --stage C
python3 experiments/apex_wam_mini/prepare_data.py verify --mark-verified
```

## 与训练配置的衔接

- Registry：`configs/data_compatibility.yaml`
- Hydra 数据配置（G1 eef6d）：`configs/data/apex_wam_mini_g1.yaml`
- 数据根目录说明：`data/apex_wam_mini/README.md`

## Week 0 检查项（§13.0 #8）

```bash
python3 experiments/apex_wam_mini/prepare_data.py all --verify --mark-verified
ls data/apex_wam_mini/reports/registry_verification/
```

通过后在 registry 中将对应 source 的 `verified` 设为 `true`，dataloader 才会对该 source 启用 `L_action`。
