# Isaac Sim + 智元 Genie G1（USD）— 与 MuJoCo 并行

本目录在 **不改动** `experiments/mujoco_g1_infer/` 的前提下，增加 **NVIDIA Isaac Sim** 作为另一条仿真链路，用于加载 AgiBot 在 **GenieSimAssets** 中发布的 **G1 USD**（与 MuJoCo / MJCF 无关）。

## 与 MuJoCo 的分工

| 环境 | 资产格式 | 本仓库入口 |
|------|----------|------------|
| **MuJoCo** | MJCF（厂商或自带 proxy） | `experiments/mujoco_g1_infer/` |
| **Isaac Sim** | USD（官方 GenieSimAssets） | 本目录 + `mujoco_g1_infer/fetch_official_genie_g1_sim_assets.sh` |

两套环境可装在同一台机器；**不要**用系统 `python3` 混装 Isaac 扩展，请始终用 Isaac 自带的 **`python.sh`**（见下）。

**一键启动 Isaac 并加载桌面+布料实验**：`start_isaac_table_cloth_experiment_on_host.sh`（见 **`TUTORIAL_ISAAC_G1_AND_INFERENCE.md` §4.1**）；本机远端触发：`remote_start_isaac_table_cloth_expect.expect`（需 `SSHPASS`）。

## 1. 安装 Isaac Sim

按 NVIDIA 文档安装 **Isaac Sim 5.x**（需满足 GPU / 驱动要求）：

- [Isaac Sim 下载与安装](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html)
- 安装后设置 **`ISAAC_SIM_PATH`** 指向安装根目录（包含 **`python.sh`** 的目录）。

可选：**Isaac Lab**（强化学习 / 场景编排）见 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)，与 Genie Sim 3 文档中 Isaac 5.1 路线一致。

## 2. 拉取智元 G1 官方 USD

与 MuJoCo 目录共用脚本（或复制路径）：

```bash
cd ../mujoco_g1_infer
bash fetch_official_genie_g1_sim_assets.sh
# 默认 ~/genie_g1_official_hf/robot/G1_omnipicker/robot.usd 等
```

## 3. 启动 Isaac 并加载 G1 USD

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim   # 含 python.sh
export GENIE_G1_USD=$HOME/genie_g1_official_hf/robot/G1_omnipicker/robot.usd

./run_with_isaac_python.sh load_genie_g1_usd_standalone.py --usd "$GENIE_G1_USD"
# 无头可加：  --headless
```

脚本仅在 **Isaac 的 `python.sh`** 下导入 `isaacsim` / `omni.*`；用系统 Python 会直接报错并提示使用 `run_with_isaac_python.sh`。

## 4. 可选：远端（Ubuntu）用 Docker 跑 Isaac Sim（无本机安装时）

若已在远端 **本机安装** Isaac Sim（推荐），实验 USD 写在 **`~/isaac_sim_exports/`**，脚本默认 **不走 Docker**。仅在需要容器化时再使用本节。

NVIDIA 文档：[Container Installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)。脚本构建/加载如需 Docker，请 **`export ISAAC_USE_DOCKER=1`**（见 **`TUTORIAL_ISAAC_G1_AND_INFERENCE.md` §4.1**）。

本目录脚本（需 **NVIDIA 驱动 + Docker + NVIDIA Container Toolkit**，磁盘与镜像体积很大）：

```bash
# 在目标机上（可先只做依赖与目录，镜像 pull 放到 tmux）
bash install_isaac_sim_docker_ubuntu.sh deps
bash install_isaac_sim_docker_ubuntu.sh volumes
bash install_isaac_sim_docker_ubuntu.sh pull    # 耗时长
bash install_isaac_sim_docker_ubuntu.sh smoke  # 可选
```

从本机同步到跳板机（**勿**把密码写进仓库；用环境变量）：

```bash
export SSHPASS='…'
chmod +x sync_isaac_sim_to_remote.expect
./sync_isaac_sim_to_remote.expect yao 10.229.66.70 /绝对路径/FastWAM/experiments/isaac_genie_g1_sim
```

同步脚本只自动执行 **`deps` + `volumes`**；**`docker pull`** 请在服务器 **`tmux`** 里执行，避免 SSH 超时。

生成在 **`~/docker/isaac-sim/`** 的辅助脚本：

- **`run_isaac_bash_interactive.sh`** — 进入容器 bash（调试用）
- **`run_isaac_headless_livestream.sh`** — 无头 + livestream（需按文档装 WebRTC 客户端）

镜像与版本：`ISAAC_SIM_DOCKER_IMAGE`（默认 **`nvcr.io/nvidia/isaac-sim:5.1.0`**）。企业/限流环境可能需要 **`docker login nvcr.io`**（NGC API Key）。

### 终止 MuJoCo、启动 Isaac（远端一键）

在仓库 **`isaac_genie_g1_sim/`** 下（需已加入 **`docker`** 组或可用 **`sg docker`**，且已 **`docker pull`** 镜像）：

```bash
export SSHPASS='…'
chmod +x remote_switch_mujoco_to_isaac.expect
./remote_switch_mujoco_to_isaac.expect 10.229.66.70
```

会 **pkill** `~/mujoco_g1_infer` 下的 viewer / 推理客户端等，再 **`nohup`** 启动 **`~/docker/isaac-sim/run_isaac_headless_livestream.sh`**，日志 **`/tmp/isaac_headless.log`**。本机查看远端状态（推荐专用 expect，避免内联 `-c` 丢输出）：

```bash
export SSHPASS='…'
chmod +x remote_run_isaac_status.expect
./remote_run_isaac_status.expect 10.229.66.70
```

## Isaac 显示「未启动」时排查

1. **看日志**（失败时现在也会写入）：`tail -n 80 /tmp/isaac_headless.log`  
2. **`docker info` 是否可用**：若报 `permission denied` / `docker.sock`，执行  
   `sudo usermod -aG docker "$USER"` 后 **完全退出 SSH 再登录**（或试 `newgrp docker`）。  
3. **镜像是否已拉取**：在 `tmux` 里  
   `bash ~/isaac_genie_g1_sim/install_isaac_sim_docker_ubuntu.sh pull`  
   未完成 pull 时 `docker run` 会失败或长时间无输出。  
4. **本机看远端状态**：`./remote_run_isaac_status.expect`；深度检查：`./remote_run_isaac_diagnose.expect`（需 `SSHPASS`）。

## 5. 后续（未实现）

- 与 `inference_service_api.md` 的相机 / proprio 对齐需另写 **Isaac 传感器 + ROS2/桥** 或 **Replicator** 出图。
- 与 FastWAM HTTP 推理闭环可在 Isaac 侧用独立节点拉流后 POST，与 MuJoCo 客户端对称实现。
