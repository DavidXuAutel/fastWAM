# MuJoCo + 智元 Genie G01 + FastWAM HTTP 推理

在跳板机（如 **`yao@10.229.66.70`**）上跑 **MuJoCo**，并按 **`inference_service_api.md`** 调用 **`GET /health`**、**`POST /v1/infer_action`**（`image_base64`、`prompt`、`proprio` 长度 **14**、`action_horizon` 等；图拼成 **384×320** 与 `deploy_policy.py` 一致）。

**另增 Isaac Sim 路线（不替代本目录）**：用官方 **G1 USD** 做仿真见 **`../isaac_genie_g1_sim/README.md`**；本目录 MuJoCo 安装与脚本保持不变。

## 资产说明（重要）

- **AgiBot 公开数据集 [GenieSimAssets](https://huggingface.co/datasets/agibot-world/GenieSimAssets)**（国内镜像：[ModelScope GenieSimAssets](https://modelscope.cn/datasets/agibot_world/GenieSimAssets)）里，G1 条目为 **`robot/G1_omnipicker`**、**`robot/G1_120s`** 等 **USD（Isaac / Genie Sim）**，当前公开树 **未包含** 智元 Genie G01 的 **MuJoCo MJCF** 主文件；与 **宇树 G1 / Menagerie** 不是同一款机器人。
- 拉取上述 **官方 USD 资产**（用于 Isaac / Genie Sim 管线，不是 MuJoCo）：

```bash
bash fetch_official_genie_g1_sim_assets.sh
# 或指定目录：GENIE_G1_OFFICIAL_DIR=$HOME/agibot_hf bash fetch_official_genie_g1_sim_assets.sh
```

- **MuJoCo 用的「官方」MJCF**：请向 **GDK / 厂商文档 / 售后** 索取，拷贝到 **`vendor_genie_g1_mjcf/`**（说明见该目录下 `README.txt`），然后：

```bash
export MUJOCO_GENIE_G1_XML=$HOME/mujoco_g1_infer/vendor_genie_g1_mjcf/g1_scene.xml   # 文件名以你拿到的为准
```

- **默认**仍使用自带 **`scenes/genie_g1_arm14_proxy.xml`**：双臂 14 铰链 **HTTP 联调代理**，非整机外观。
- 使用厂商 MJCF 时务必准备匹配的 **`--joint-names`**（14 行，与 HAL **`/hal/arm_joint_state`** 左 7 + 右 7 一致；语义见 `experiments/genie_g1/g1_fastwam_arm_mapping.py`）。

## 安装

```bash
bash install_on_ubuntu.sh
```

无显示器：

```bash
export MUJOCO_GL=egl   # 或 osmesa
```

若脚本提示未装系统库，请在本机用 `sudo apt install …` 安装 `libegl1`、`libgl1-mesa-glx`、`python3-pip` 后重试。

## 启动 MuJoCo Simulate 并加载 Genie G01（MJCF）

交互式 **Simulate** 窗口使用 **GLFW**（与离屏 `Renderer` 的 `MUJOCO_GL=egl` 不同）。在 **有图形环境** 的机器上：

```bash
cd ~/mujoco_g1_infer
./start_genie_g1_viewer.sh
# 等价：MUJOCO_GL=glfw python3 launch_genie_g1_viewer.py
```

指定厂商 MJCF：

```bash
./start_genie_g1_viewer.sh --mjcf "$HOME/genie_g1_mujoco/your_scene.xml"
```

常见方式：

- 机器人/跳板机已登录 **桌面会话**：在终端里直接运行上面命令（必要时 `export DISPLAY=:0` 或 `:1`，与 `echo $DISPLAY` 在桌面终端里一致）。
- 从笔记本 **X11 转发**：`ssh -Y yao@10.229.66.70`，再执行 `./start_genie_g1_viewer.sh`。
- 仅作无头冒烟（可能因驱动失败）：`sudo apt install -y xvfb && xvfb-run -a ./start_genie_g1_viewer.sh`。

## 运行闭环

```bash
export INFER_API_BASE=http://127.0.0.1:8000
export ROBOTWIN_DATASET_STATS=$HOME/robotwin_uncond_3cam_384_dataset_stats.json   # 可选

python3 loop_mujoco_g1_inference_client.py \
  --api-base "$INFER_API_BASE" \
  --joint-names ./joint_names_genie_g1_proxy_14.txt \
  --steps 5 \
  --dry-run
```

- **`--dry-run`**：只请求推理，不写 `qpos`。
- 去掉 **`--dry-run`** 时会对 **14 个标量关节** 做极小步长写回（演示用，非完整 WBC）。

## 在远端后台开启 Simulate（需已有图形会话）

在机器人上 **`who`** 若能看到 **`(tty)` 或 `(:1)`** 等图形登录，可把查看器挂到对应 **`DISPLAY`**（常见为 **`:1`**，`:0` 在纯 SSH 下常失败）：

```bash
export SSHPASS='你的SSH密码'
chmod +x remote_start_viewer.expect
./remote_start_viewer.expect 10.229.66.70 :1
```

日志：`ssh yao@主机 'tail -f /tmp/mujoco_genie_viewer.log'`。查进程：`bash ~/mujoco_g1_infer/remote_viewer_status.sh`。

## 从本机同步到远程（expect）

勿把密码写进仓库；用环境变量：

```bash
export SSHPASS='你的SSH密码'
chmod +x sync_and_install_remote.expect
./sync_and_install_remote.expect yao 10.229.66.70 /绝对路径/mujoco_g1_infer
```

## API 对齐摘要

| 字段 | 行为 |
|------|------|
| `image_base64` | 头 320×256 + 双腕 160×128 → 384×320 JPEG |
| `proprio` | 14 维 float，来自 `qpos`（`--joint-names`） |
| `action` | 若提供 `dataset_stats` 的 z-score `default`，则 **x×std+mean** 反归一化 |

## 参考文档

本机示例：`/Users/xudazhong/Downloads/inference_service_api.md`（可 scp 到服务器）。
