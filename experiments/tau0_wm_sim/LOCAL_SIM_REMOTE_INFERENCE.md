# τ0-WM 本地仿真连接远端模型实时推理方案

本文档说明如何在**本地运行 Isaac Sim / 仿真环境**，连接**远端 τ0-WM VAM 模型服务**，实现实时或准实时闭环推理。

## 1. 总体架构

```text
本地 Isaac Sim / 仿真
  ├─ 采集多视角 RGB + 双臂 EEF state + gripper
  ├─ 打包 τ0-WM payload
  ├─ 通过 SSH tunnel / websocket 发给远端 VAM server
  ├─ 接收 [T,16] action chunk
  └─ 本地执行前 k 步，然后重新观测并推理
```

推荐先走 **VAM-only** 路线，不依赖尚未完整开源的 ACVS/TTC。

## 2. 远端模型服务

远端服务器：

```text
host: 34.173.35.192
user: tau0
repo: /home/tau0/workspace/tau-0-wm
env: /home/tau0/miniconda3/envs/tau0-wm-py312
checkpoint: /home/tau0/models/tau-0-wm
Wan2.2: /home/tau0/models/Wan2.2-TI2V-5B
preflight: ready=true
```

远端启动服务：

```bash
ssh tau0@34.173.35.192
source ~/miniconda3/bin/activate tau0-wm-py312
cd ~/workspace/tau-0-wm

# 建议选择一张空闲 GPU
export CUDA_VISIBLE_DEVICES=0

# 只监听本机，避免公网暴露
bash run_infer_server.sh 127.0.0.1 8001
```

本地建立 SSH tunnel：

```bash
ssh -N -L 8001:127.0.0.1:8001 tau0@34.173.35.192
```

建立后，本地访问 `127.0.0.1:8001` 即等价于访问远端 τ0-WM server。

## 3. τ0-WM VAM 输入输出格式

### 输入 payload

```python
payload = {
    "obs": obs,                         # [V, 3, 192, 256], float32, range [-1, 1]
    "prompt": "task instruction",
    "state": state,                     # [14]
    "gripper_states": gripper_states,   # [2], range 0-120
    "num_inference_steps": 5,
    "execution_step": 10,
    "sample_solver": "euler",
    "shift": 1.0,
}
```

`state` 顺序：

```text
left_eef_xyz + left_quat_xyzw + right_eef_xyz + right_quat_xyzw
```

`gripper_states`：

```text
left_gripper + right_gripper
范围: 0-120
0   = open
120 = close
```

### 输出 actions

模型返回：

```text
actions: [T, 16]
```

顺序：

```text
left xyz + left quat xyzw + left gripper
right xyz + right quat xyzw + right gripper
```

## 4. 本地连接远端 server

官方使用 websocket client：

```python
from web_infer_utils.openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(
    host="127.0.0.1",
    port=8001,
)

result = policy.infer(obs=payload)
actions = result["actions"]
```

如果本地不想完整放置官方仓库，可以复用本仓库已准备的部署工具：

```text
FastWAM/experiments/tau0_wm_sim/
```

核心文件：

```text
adapters.py           # 观测、state、action 适配
vam_only_loop.py      # VAM-only 闭环脚手架
candidate_filter.py   # RCS-lite / 动作候选过滤
benchmark_vam.py      # 延迟测量
preflight.py          # 部署条件检查
```

## 5. 本地仿真侧 Adapter

本地 Isaac Sim / 仿真侧需要实现三个 adapter。

### 5.1 Observation Adapter

从本地仿真相机读取 RGB 图像，转成 τ0-WM 需要的格式：

```python
obs = rgb.astype(np.float32) / 127.5 - 1.0
obs = obs.transpose(0, 3, 1, 2)  # [V,H,W,3] -> [V,3,H,W]
```

要求：

```text
shape: [V, 3, 192, 256]
dtype: float32
range: [-1, 1]
```

建议先使用 3 路视角：

```text
head / third-person view
left wrist or left-side view
right wrist or right-side view
```

### 5.2 State Adapter

从仿真中读取左右 EEF pose。

要求：

```text
left xyz + left quat xyzw + right xyz + right quat xyzw
```

注意：

- quaternion 顺序必须是 `xyzw`。
- 每个 EEF pose 的坐标原点应尽量对齐到对应 arm base frame。
- 如果仿真只能提供 world frame，需要先转换到 arm base frame。

夹爪映射：

```python
gripper_0_120 = open_fraction * 120.0
```

### 5.3 Action Adapter

把远端返回的 `[T,16]` 转成本地仿真控制命令。

如果 Isaac 支持 EEF pose control：

```text
直接发送 left/right target pose + gripper target
```

如果 Isaac 只支持 joint control：

```text
EEF target pose -> IK / retargeting -> joint target
```

推荐一开始只执行 action chunk 的前 `k` 步，不要一次执行完整 chunk。

## 6. 实时闭环策略

推荐使用 receding horizon：

```text
1. 本地仿真采集 obs/state/gripper
2. 远端 τ0-WM 推理得到 [T,16] action chunk
3. 本地只执行前 k 步，例如 k=5 或 k=10
4. 执行期间异步请求下一次推理
5. 新 action 到达后替换 action buffer
```

推荐初始参数：

```text
num_inference_steps: 5
execution_step: 10 或 30
local_execute_k: 5-10
sample_solver: euler
shift: 1.0
```

如果推理延迟较高：

- 降低 `num_inference_steps`
- 增大本地 action buffer
- 异步请求下一段 action
- 只在 buffer 剩余不足时请求新 chunk
- 固定 prompt，避免重复 text encoding 开销

## 7. 本地 loop 伪代码

```python
action_buffer = []

while sim_running:
    if len(action_buffer) < min_buffer:
        obs = read_sim_cameras()
        state = read_dual_eef_state()
        gripper_states = read_grippers()

        payload = {
            "obs": obs,
            "prompt": prompt,
            "state": state,
            "gripper_states": gripper_states,
            "num_inference_steps": 5,
            "execution_step": 10,
            "sample_solver": "euler",
            "shift": 1.0,
        }

        result = policy.infer(obs=payload)
        action_buffer = postprocess(result["actions"])

    action = action_buffer.pop(0)
    apply_action_to_isaac(action)
    sim.step()
```

## 8. 候选动作过滤

在未接入 ACVS/TTC 前，可以先使用轻量 RCS-lite / 几何过滤：

```text
多次采样候选 action chunks
过滤 EEF 位移跳变
过滤姿态跳变
过滤 gripper 越界
过滤 IK 不可解或碰撞风险
选择最平滑且有效的候选
```

本仓库对应工具：

```text
FastWAM/experiments/tau0_wm_sim/candidate_filter.py
```

## 9. 延迟测量

可以用：

```bash
python3 FastWAM/experiments/tau0_wm_sim/benchmark_vam.py \
  --tau-repo /path/to/tau-0-wm \
  --host 127.0.0.1 \
  --port 8001 \
  --iterations 20 \
  --output reports/runtime.json
```

建议记录：

```text
单次推理 latency
平均 latency
GPU 显存
action chunk 长度
本地执行 k
仿真控制频率
```

## 10. 常见问题

### 10.1 远端端口无法连接

检查 SSH tunnel：

```bash
ssh -N -L 8001:127.0.0.1:8001 tau0@34.173.35.192
```

远端服务建议绑定：

```text
127.0.0.1:8001
```

本地 client 连接：

```text
127.0.0.1:8001
```

### 10.2 动作方向错误

优先检查：

- EEF pose 是否从 world frame 转成 arm base frame
- quaternion 是否为 `xyzw`
- 左右臂顺序是否反了
- gripper open/close 是否反了

### 10.3 推理太慢

优先尝试：

- `num_inference_steps=3` 或 `5`
- `execution_step=10`
- 异步推理
- 使用更空闲的 GPU
- 后续安装 FlashAttention 2/3 优化

### 10.4 模型服务启动失败

远端预检报告：

```text
/home/tau0/reports/tau0_preflight_ready.json
```

当前应为：

```text
ready: true
```

## 11. 当前远端状态

远端 `34.173.35.192` 已准备：

```text
user: tau0
env: /home/tau0/miniconda3/envs/tau0-wm-py312
repo: /home/tau0/workspace/tau-0-wm
checkpoint: /home/tau0/models/tau-0-wm
Wan2.2: /home/tau0/models/Wan2.2-TI2V-5B
preflight report: /home/tau0/reports/tau0_preflight_ready.json
```

已验证：

```text
Python 3.12.13
Torch 2.8.0+cu128
CUDA available: True
BF16 supported: True
TauPolicy import ok
```

注意：

- 当前 VAM-only 环境已就绪。
- 官方 ACVS/TTC 权重和代码仍需等待官方释放，或自行训练/实现替代模块。
- 远端 apt 存在 NVIDIA kernel 包依赖冲突，Docker/Isaac Sim 远端路线需后续单独处理；本方案默认 Isaac Sim 在本地运行。
