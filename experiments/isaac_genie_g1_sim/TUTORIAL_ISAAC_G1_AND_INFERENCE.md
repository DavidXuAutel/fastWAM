# Isaac Sim 部署智元 Genie G1（USD）并连接 FastWAM HTTP 推理 — 教程

本文说明：**在 Isaac Sim 中加载官方 G1 USD**、**准备推理服务**、以及**与 MuJoCo 客户端对称的闭环思路**（Isaac 侧完整 HTTP 循环脚本见仓库 README「后续」；可按本文自行接传感器与 `requests`）。

---

## 0. 你将得到什么

| 阶段 | 结果 |
|------|------|
| A. 资产 | 本机或服务器上有 **GenieSimAssets 拉取的 G1 USD**（如 `robot/G1_omnipicker/robot.usd`）。 |
| B. Isaac | **Isaac Sim 5.x** 能启动，并用 `python.sh` 跑通 **`load_genie_g1_usd_standalone.py`**（场景里出现 G1）。 |
| C. 推理服务 | 另一进程（可与 Isaac 同机或内网）运行 **`scripts/serve.py`**，对外 **`GET /health`**、**`POST /v1/infer_action`**。 |
| D. 闭环（实现项） | 从 Isaac **渲染 RGB（及 14 维 proprio）** → 拼成服务要求的图像布局 → **POST** → 将返回的 **`[T, 14]`** 动作写回 G1 关节/控制器。 |

接口字段以 **`inference_service_api.md`**（或仓库内同名说明）为准；下表为常见 RoboTwin 3cam 检查点约定。

---

## 1. 前置条件

- **GPU**：NVIDIA 驱动可用（`nvidia-smi`）。
- **磁盘**：Isaac Docker 镜像与缓存体积很大，预留 **百 GB 级** 更稳妥。
- **网络**：拉 USD（Hugging Face）与 `docker pull nvcr.io/nvidia/isaac-sim:5.1.0` 需稳定网络；企业环境常需 **`docker login nvcr.io`**（NGC API Key）。

---

## 2. 拉取智元 G1 官方 USD

与 MuJoCo 目录共用脚本（只需执行一次或更新时重跑）：

```bash
cd /path/to/FastWAM/experiments/mujoco_g1_infer
bash fetch_official_genie_g1_sim_assets.sh
```

默认常见路径示例：

```text
$HOME/genie_g1_official_hf/robot/G1_omnipicker/robot.usd
```

也可设置环境变量 **`GENIE_G1_OFFICIAL_DIR`** 指向你存放资产的根目录。

---

## 3. 安装 Isaac Sim（二选一）

### 3.1 工作站安装（本机带显示器 / 本地调试）

1. 按 NVIDIA 文档安装 **Isaac Sim 5.x**：  
   [Install Workstation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html)
2. 设置环境变量（路径以你机器为准）：

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim   # 目录内必须包含 python.sh
```

### 3.2 Ubuntu + Docker（无头 / 远端服务器）

1. 文档：[Container Installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
2. 使用本目录脚本（在**目标机**上执行）：

```bash
cd ~/isaac_genie_g1_sim   # 或你同步后的路径
bash install_isaac_sim_docker_ubuntu.sh deps
bash install_isaac_sim_docker_ubuntu.sh volumes
# 建议在 tmux 中执行，避免 SSH 断开
bash install_isaac_sim_docker_ubuntu.sh pull
bash install_isaac_sim_docker_ubuntu.sh smoke   # 可选
```

3. **Docker 权限**：若 `docker info` 报 `permission denied`，执行：

```bash
sudo usermod -aG docker "$USER"
```

然后 **完全退出 SSH 再登录**（或 `newgrp docker`）。

4. 生成在 **`~/docker/isaac-sim/`** 的脚本：
   - **`run_isaac_bash_interactive.sh`** — 进容器调试
   - **`run_isaac_headless_livestream.sh`** — 无头 + livestream

从开发机**同步**本目录到远端（密码用环境变量，勿写进仓库）：

```bash
export SSHPASS='…'
chmod +x sync_isaac_sim_to_remote.expect
./sync_isaac_sim_to_remote.expect yao YOUR_HOST /绝对路径/FastWAM/experiments/isaac_genie_g1_sim
```

同步脚本默认只跑 **`deps` + `volumes`**；**`pull` 请在远端 tmux 手动执行**。

---

## 4. 在 Isaac 中加载 G1 USD（烟测）

在**已设置 `ISAAC_SIM_PATH` 的机器**上（工作站路径；Docker 内则用容器里的 `python.sh` 等价路径）：

```bash
cd /path/to/FastWAM/experiments/isaac_genie_g1_sim
export GENIE_G1_USD="$HOME/genie_g1_official_hf/robot/G1_omnipicker/robot.usd"

chmod +x run_with_isaac_python.sh
./run_with_isaac_python.sh load_genie_g1_usd_standalone.py --usd "$GENIE_G1_USD"
# 无 GUI：
# ./run_with_isaac_python.sh load_genie_g1_usd_standalone.py --usd "$GENIE_G1_USD" --headless
```

**必须**通过 **`run_with_isaac_python.sh`** 调用 Isaac 自带的 **`python.sh`**；用系统 `python3` 直接运行会因缺少 `isaacsim` / `omni.*` 而失败。

---

## 4.1 程序化生成「地面 + 穹顶光 + 桌子 + 布料」USD（推荐：远端**本机安装** Isaac）

实验 **USD 默认写在远端磁盘普通目录**，**不依赖 Docker 卷**：  
**`~/isaac_sim_exports/table_cloth_env.usd`**（可用环境变量 **`ISAAC_SIM_EXPORT_ROOT`** 改成其它根路径）。

前提：远端已按官方文档安装 **工作站版 Isaac Sim**，且 **`ISAAC_SIM_PATH`**（或常见路径下的 **`python.sh`**）可用。

**① 生成实验 USD**

```bash
export ISAAC_SIM_PATH=/path/to/isaac-sim   # 含 python.sh；也可依赖脚本自动探测

chmod +x ~/isaac_genie_g1_sim/run_build_table_cloth_env_on_host.sh
bash ~/isaac_genie_g1_sim/run_build_table_cloth_env_on_host.sh
# 默认输出：~/isaac_sim_exports/table_cloth_env.usd
```

自定义输出路径：

```bash
bash ~/isaac_genie_g1_sim/run_build_table_cloth_env_on_host.sh "$HOME/my_scenes/table_cloth_env.usd"
```

若粒子布 API 与当前小版本不兼容，只生成几何 + 刚体：

```bash
bash ~/isaac_genie_g1_sim/run_build_table_cloth_env_on_host.sh "$HOME/isaac_sim_exports/table_cloth_env.usd" skip
```

**② 加载该实验**

```bash
chmod +x ~/isaac_genie_g1_sim/run_load_table_cloth_on_host.sh
bash ~/isaac_genie_g1_sim/run_load_table_cloth_on_host.sh --headless --steps 200
# 或 GUI / 指定 USD：
bash ~/isaac_genie_g1_sim/run_load_table_cloth_on_host.sh --usd "$HOME/isaac_sim_exports/table_cloth_env.usd"
```

说明：构建与加载 **默认仅走本机 `python.sh`**。只有当你**显式**设置 **`export ISAAC_USE_DOCKER=1`** 时，才会使用 `install_isaac_sim_docker_ubuntu.sh` 那套 Docker 镜像（USD 需落在 **`$ISAAC_DOCKER_DATA_ROOT/data/`** 挂载树内才能被容器读到）。

### 一键启动 Isaac 并加载实验（推荐）

```bash
bash ~/isaac_genie_g1_sim/start_isaac_table_cloth_experiment_on_host.sh --headless
# 有显示器 / DISPLAY：
bash ~/isaac_genie_g1_sim/start_isaac_table_cloth_experiment_on_host.sh --gui
```

行为：若 **`ISAAC_EXPERIMENT_USD`**（默认 **`~/isaac_sim_exports/table_cloth_env.usd`**）不存在，会先 **`run_build_*`**（本机 Isaac）；再 **`run_load_*`**。

从开发机触发（同步脚本 + SSH 执行；**本机 Isaac 场景不需要** `ISAAC_DEPLOY_SUDO_PASS`，除非你同时开了 **`ISAAC_USE_DOCKER=1`**）：

```bash
cd /path/to/FastWAM/experiments/isaac_genie_g1_sim
export SSHPASS='…'
chmod +x remote_start_isaac_table_cloth_expect.expect
./remote_start_isaac_table_cloth_expect.expect 10.229.66.70 --headless
```

**从开发机仅同步构建脚本**（可选）：

```bash
cd /path/to/FastWAM/experiments/isaac_genie_g1_sim
export SSHPASS='…'
chmod +x remote_run_build_table_cloth_env.expect
./remote_run_build_table_cloth_env.expect 10.229.66.70
```

**在 Isaac GUI 中打开**：`File → Open` → **`~/isaac_sim_exports/table_cloth_env.usd`**（或你自定义的路径）。需要 **GPU** 与本机 Isaac。

---

## 5. 启动 FastWAM 推理服务

在**装有 FastWAM 环境与权重**的机器上（可与 Isaac 同机，也可仅内网可达），参考 `inference_service_api.md` 中的启动示例，核心是 **`scripts/serve.py`** + **`task` + `ckpt`**，例如 RoboTwin 三相机 384 检查点：

```bash
cd /path/to/FastWAM
CUDA_VISIBLE_DEVICES=0 \
DIFFSYNTH_MODEL_BASE_PATH=/path/to/FastWAM/checkpoints \
.venv/bin/python scripts/serve.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  model.redirect_common_files=false \
  service.device=cuda \
  service.host=0.0.0.0 \
  service.port=8000
```

**健康检查**（将 `HOST` 换成服务实际 IP）：

```bash
curl http://HOST:8000/health
```

应返回 `ready`、`image_shape`（如 `[1, 3, 384, 320]`）、`action_dim`、`proprio_dim`（常见为 **14**）。

---

## 6. HTTP 推理接口（与 Isaac 对接的契约）

### 6.1 `POST /v1/infer_action`

| 字段 | 说明 |
|------|------|
| `image_base64` | 单张 **RGB** 的 base64（可带 data URL 前缀）。服务端会 **center crop + resize** 到检查点期望分辨率（如 **384×320**）。 |
| `prompt` | 语言指令。 |
| `proprio` | 可选，**长度 14** 的数组（与当前示例检查点一致）。 |
| `action_horizon` | 可选，默认 32。 |

响应：`action` 为 **`[T, action_dim]`**（如 `[32, 14]`）。

### 6.2 与 MuJoCo 参考客户端对齐的图像布局

仓库 **`experiments/mujoco_g1_infer/loop_mujoco_g1_inference_client.py`** 中函数 **`build_robotwin_style_image`** 将 **头/左/右** 三路相机拼成与 **`deploy_policy.py`** 一致的 **384×320** 布局。Isaac 侧闭环时建议：

1. 用 **Replicator / Camera sensor / ROS bridge** 从 USD 场景取 **三路 RGB**（分辨率可不同，先 resize 再拼）。
2. 调用同一布局逻辑（可复制该函数到 Isaac 脚本或共享小模块）。
3. 将拼好的图 **JPEG → base64** 后 **POST**。

### 6.3 将动作写回 G1（实现要点）

- 返回的 **14 维**需与**当前训练任务**的关节语义一致（RoboTwin 检查点 ≠ 人形 G1 真机关节顺序时，必须做 **映射表** 或换用 **针对 G1 数据微调的 ckpt**）。
- Isaac 中通常通过 **Articulation / DCU / ROS2 JointTrajectory** 下发；**仿真步长**与 **`action_horizon`** 的时间尺度要对齐（例如每步仿真消费 1 行动作，或做插值）。

---

## 7. 端到端闭环（推荐架构）

当前仓库 **已提供**：Isaac 烟测脚本 + Docker 安装 + 远端同步/切换辅助脚本。  
**未内置**「Isaac 专用 HTTP 循环」时，推荐架构如下：

```text
┌─────────────────────┐     HTTP JSON      ┌──────────────────────┐
│  Isaac Sim (G1 USD) │  ───────────────►   │  FastWAM serve.py    │
│  渲染 + proprio      │  POST /v1/...      │  GPU 推理            │
│  关节控制            │  ◄──────────────   │                      │
└─────────────────────┘     action [T,14]  └──────────────────────┘
```

实现顺序建议：

1. 固定 **单相机** 烟测：一张图 + 零 proprio 调通 **200**。
2. 改为 **三相机拼图** + 真实 **proprio**。
3. 加 **闭环频率** 与 **动作执行**（物理子步、限幅、安全停）。

可参考 **`loop_mujoco_g1_inference_client.py`** 的 `requests.post(..., json=payload)` 与 **`jpeg_b64`** 流程，在 **`load_genie_g1_usd_standalone.py`** 的扩展版里挂 **SimulationApp 主循环**（同一 `python.sh` 解释器）。

---

## 8. 远端一键：停 MuJoCo、起 Isaac 无头（可选）

若同机曾跑 MuJoCo 推理/viewer，可用 expect 切换（需 **`SSHPASS`**、且已 **`docker pull`**）：

```bash
cd /path/to/FastWAM/experiments/isaac_genie_g1_sim
export SSHPASS='…'
chmod +x remote_switch_mujoco_to_isaac.expect
./remote_switch_mujoco_to_isaac.expect YOUR_HOST
```

日志：**`/tmp/isaac_headless.log`**。状态：

```bash
./remote_run_isaac_status.expect YOUR_HOST
```

环境总检（Docker / GPU / 镜像 / 日志）：

```bash
chmod +x remote_run_isaac_env_check.expect isaac_env_check_remote_wrap.sh
./remote_run_isaac_env_check.expect YOUR_HOST
```

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| `permission denied` on `docker.sock` | `usermod -aG docker` 后重登 SSH。 |
| Isaac 容器秒退 | `tail -n 80 /tmp/isaac_headless.log`；确认镜像已 pull、GPU 可用。 |
| 推理 400 / shape 错误 | 检查 **`/health`** 的 `image_shape`；拼图画布与 **JPEG base64** 编码。 |
| 动作乱/机器人发散 | 检查 **14 维语义是否与 ckpt 一致**；降低步长、加关节限幅。 |

---

## 10. 参考链接

- NVIDIA Isaac Sim 文档（安装 / 容器）：见上文链接  
- 本目录 **`README.md`**（与 MuJoCo 分工、脚本索引）  
- 推理 API：**`inference_service_api.md`**  
- MuJoCo 参考闭环：**`experiments/mujoco_g1_infer/loop_mujoco_g1_inference_client.py`**
