# ScoutXWAM DROID-100 ↔ Franka FR3 bridge

Connects H100 ScoutXWAM inference (`:8010`) to the wired FR3 stack on
`yao@10.229.20.125`. Does **not** touch Desk / robot networking or
`10.229.66.70`. Leaves kairos WAM on `:8005` alone.

## Layout

| Piece | Where |
| --- | --- |
| `serve.py` | H100 `/home/a25689/FastWAM/scoutxwam_droid100_inference/` |
| dry-run / open-loop | this directory (sync to 125) |
| tunnel | `tunnel_to_125.sh` (Mac → H100 starts `-R 8010`) |

## 1. Smoke (one-shot CLI)

```bash
ssh -i ~/.ssh/franka_ros2_ed25519 -p 31126 a25689@10.239.121.11
cd /home/a25689/FastWAM/scoutxwam_droid100_inference
export PYTHONPATH="$PWD/src:$PWD/third_party/X-WAM:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=0 /home/a25689/micromamba/envs/mot-wam/bin/python infer.py \
  --input examples/request.npz --output outputs/prediction.npz
```

`infer.py` on the server now also inserts `src/` into `sys.path` automatically.
## 2. HTTP service (H100)

```bash
# from Mac: sync serve.py then start
scp -P 31126 -i ~/.ssh/franka_ros2_ed25519 \
  experiments/scoutxwam_franka_bridge/serve.py \
  a25689@10.239.121.11:/home/a25689/FastWAM/scoutxwam_droid100_inference/serve.py

ssh -i ~/.ssh/franka_ros2_ed25519 -p 31126 a25689@10.239.121.11 \
  'bash /home/a25689/FastWAM/experiments/scoutxwam_franka_bridge/start_serve.sh'
# or after syncing start_serve.sh into the package:
# bash /home/a25689/FastWAM/scoutxwam_droid100_inference/start_serve.sh

curl -sS http://127.0.0.1:8010/health   # on H100
```

## 3. Tunnel to lab PC

```bash
bash experiments/scoutxwam_franka_bridge/tunnel_to_125.sh start
bash experiments/scoutxwam_franka_bridge/tunnel_to_125.sh status
```

## 4. Dry-run on 125 (no motion)

```bash
# sync this folder to 125, then:
export SCOUT_URL=http://127.0.0.1:8010
python3 dryrun_franka.py --prompt "pick up the pen" --num-infer 1
```

## 5. Open-loop (default plan-only)

```bash
python3 open_loop_franka.py --prompt "pick up the pen" --plan-only

# hardware (dual gate):
export SCOUT_ARM_TOKEN='choose-a-secret'
python3 open_loop_franka.py --prompt "pick up the pen" \
  --i-approve-motion --arm-token "$SCOUT_ARM_TOKEN"
```

Requires existing RealSense + Franka ROS stack on 125, and `~/kairos/scripts/phase2`
for IK / GelloTakeover.

## 6. MuJoCo sim only (live sensors → infer → viewer; no real motion)

```bash
export SCOUT_URL=http://127.0.0.1:8010
export DISPLAY=:1
python3 sim_open_loop_mujoco.py --prompt "pick up the pen" --max-steps 32

# headless qpos stepping + full logs (no GUI):
python3 sim_open_loop_mujoco.py --prompt "pick up the pen" --no-viewer
```

Writes `~/scoutxwam_sim_logs/<ts>/` (`frames`, `request.json`, `actions.npy`,
`plan.json`, `playback.jsonl`, `meta.json`). Does **not** publish arm commands.

## API

- `GET /health`
- `POST /v1/infer` with `prompt`, `proprio[8]`, `video_shape[2,H,W,3]`, `video_b64`
- Response: denormalized `actions[32,7]`, `proprios[9,16]`, `infer_s`
