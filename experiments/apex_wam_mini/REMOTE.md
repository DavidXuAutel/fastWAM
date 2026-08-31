# 远端数据准备（a25689@10.239.121.11:31126）

## 已部署

```text
/home/a25689/FastWAM/
├── configs/data_compatibility.yaml   # profiles: g1 | franka
├── configs/data/apex_wam_mini_{g1,franka}.yaml
├── docs/Apex-WAM-Mini-Franka-Profile.md
└── data/
    ├── libero_mujoco3.3.2/   # ✅ ~8.8GB（4 套件）
    ├── robotwin2.0/          # ⏬ 下载中
    └── apex_wam_mini/
```

## 下载

```bash
cd ~/FastWAM
export PYTHON=/opt/conda/bin/python3

# LIBERO（已完成）
bash experiments/apex_wam_mini/download_video_only_datasets.sh libero

# RoboTwin（后台）
nohup bash experiments/apex_wam_mini/download_video_only_datasets.sh robotwin \
  > data/apex_wam_mini/reports/robotwin_download.log 2>&1 &
tail -f data/apex_wam_mini/reports/robotwin_download.log
```

## 扫描（双 profile）

```bash
/opt/conda/bin/python3 experiments/apex_wam_mini/prepare_data.py all
# 产出 stage_b_g1.json, stage_b_franka.json, ...

# Franka 档 verify LIBERO（可 L_action）
/opt/conda/bin/python3 experiments/apex_wam_mini/prepare_data.py verify --profile franka --mark-verified
```

## Profile 说明

| Profile | LIBERO | RoboTwin | 文档 |
|---------|--------|----------|------|
| **g1** | video-only | video-only | v3.2 主路径 |
| **franka** | **L_action + L_video** | video-only | `docs/Apex-WAM-Mini-Franka-Profile.md` |
