#!/usr/bin/env python3
"""
Genie G01 (智元) oriented MuJoCo loop + FastWAM HTTP inference API.

Default MJCF is a small **arm14 kinematic proxy** (see scenes/genie_g1_arm14_proxy.xml).
Replace with vendor Genie G01 MJCF via --mjcf / MUJOCO_GENIE_G1_XML and matching --joint-names.

API: inference_service_api.md — POST {api_base}/v1/infer_action
Env: MUJOCO_GL=egl|osmesa for headless.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_PROXY_MJCF = _SCRIPT_DIR / "scenes" / "genie_g1_arm14_proxy.xml"

try:
    import mujoco
    from mujoco import Renderer
except ImportError as e:
    raise SystemExit("Install mujoco: pip install 'mujoco>=3.1.0'") from e

try:
    import requests
except ImportError as e:
    raise SystemExit("Install requests") from e

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Install pillow") from e


def _resize_rgb(rgb: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    return np.asarray(im.resize((w, h), resample=Image.BILINEAR), dtype=np.uint8)


def build_robotwin_style_image(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Match experiments/robotwin/fastwam_policy/deploy_policy.py _build_robotwin_image_tensor layout."""
    head_r = _resize_rgb(head, (320, 256))
    left_r = _resize_rgb(left, (160, 128))
    right_r = _resize_rgb(right, (160, 128))
    bottom = np.concatenate([left_r, right_r], axis=1)
    return np.concatenate([head_r, bottom], axis=0)


def jpeg_b64(rgb: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _set_free_cam(cam: Any, *, lookat: Sequence[float], distance: float, azimuth: float, elevation: float) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation


def subtree_com_for_body(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(body_name)
    return np.asarray(data.subtree_com[bid], dtype=np.float64).reshape(3)


def resolve_lookat(
    model: mujoco.MjModel, data: mujoco.MjData, preferred: Optional[str]
) -> np.ndarray:
    names: List[str] = []
    if preferred:
        names.append(preferred)
    names.extend(["torso", "pelvis", "base_link", "root", "world"])
    for n in names:
        try:
            return subtree_com_for_body(model, data, n)
        except KeyError:
            continue
    return np.asarray(data.qpos[:3], dtype=np.float64).reshape(3)


def render_triplet(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: Renderer,
    *,
    lookat_body: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = resolve_lookat(model, data, lookat_body)

    def grab(lookat: np.ndarray, dist: float, azim: float, elev: float, wh: Tuple[int, int]) -> np.ndarray:
        cam = mujoco.MjvCamera()
        _set_free_cam(cam, lookat=lookat, distance=dist, azimuth=azim, elevation=elev)
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        return _resize_rgb(rgb, wh)

    head = grab(mid + np.array([0.0, 0.0, 0.25]), 2.2, 90, -20, (640, 480))
    left = grab(mid + np.array([0.0, 0.15, 0.2]), 1.4, 140, -15, (640, 480))
    right = grab(mid + np.array([0.0, -0.15, 0.2]), 1.4, 40, -15, (640, 480))
    return head, left, right


def load_joint_names(path: Optional[Path]) -> Optional[List[str]]:
    if path is None or not path.is_file():
        return None
    names: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(s)
    return names if len(names) == 14 else None


def _joint_scalar_qposadr(model: mujoco.MjModel, jid: int) -> int:
    t = model.jnt_type[jid]
    adr = int(model.jnt_qposadr[jid])
    if t in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return adr
    raise RuntimeError(f"joint {jid} type {t} not supported (need hinge or slide)")


def resolve_qpos_indices(model: mujoco.MjModel, names: Optional[List[str]]) -> List[int]:
    if names is not None:
        idx: List[int] = []
        for n in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid < 0:
                raise RuntimeError(f"joint not found: {n}")
            idx.append(_joint_scalar_qposadr(model, jid))
        return idx
    out: List[int] = []
    for j in range(model.njnt):
        try:
            out.append(_joint_scalar_qposadr(model, j))
        except RuntimeError:
            continue
        if len(out) >= 14:
            break
    if len(out) < 14:
        raise RuntimeError("Could not find 14 scalar joints; provide --joint-names file")
    return out


def read_proprio(data: mujoco.MjData, qpos_idx: Sequence[int]) -> List[float]:
    return [float(data.qpos[i]) for i in qpos_idx]


def load_action_zscore_stats(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as f:
        stats = json.load(f)
    block = stats.get("action", {}).get("default", {})
    mean = block.get("global_mean")
    std = block.get("global_std")
    if mean is None or std is None:
        raise ValueError("Expected action.default.global_mean / global_std in dataset_stats.json")
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    s = np.asarray(std, dtype=np.float64).reshape(-1)
    if m.size != 14 or s.size != 14:
        raise ValueError(f"mean/std must be length 14, got {m.size}, {s.size}")
    return m, s


def denorm_action_row(norm_row: Sequence[float], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.asarray(norm_row, dtype=np.float64)
    return x * (std + 1e-8) + mean


def post_infer_action(
    api_base: str,
    image_b64: str,
    prompt: str,
    proprio: List[float],
    action_horizon: int,
    num_inference_steps: int,
    timeout: float,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/v1/infer_action"
    payload = {
        "image_base64": image_b64,
        "prompt": prompt,
        "proprio": proprio,
        "action_horizon": int(action_horizon),
        "num_inference_steps": int(num_inference_steps),
        "text_cfg_scale": 1.0,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _resolve_mjcf_path(arg: Optional[str]) -> Path:
    if arg:
        p = Path(arg).expanduser().resolve()
    else:
        env = os.environ.get("MUJOCO_GENIE_G1_XML")
        p = Path(env).expanduser().resolve() if env else _DEFAULT_PROXY_MJCF
    if not p.is_file():
        raise SystemExit(f"MJCF not found: {p}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="Genie G01 MuJoCo + FastWAM /v1/infer_action client")
    ap.add_argument("--api-base", default=os.environ.get("INFER_API_BASE", "http://127.0.0.1:8000"))
    ap.add_argument(
        "--mjcf",
        default=None,
        help="Main MJCF file (default: env MUJOCO_GENIE_G1_XML or bundled arm14 proxy)",
    )
    ap.add_argument(
        "--joint-names",
        type=Path,
        default=None,
        help="14 lines: joint names for proprio / qpos writes (default: proxy list if file next to script)",
    )
    ap.add_argument(
        "--lookat-body",
        default=os.environ.get("MUJOCO_LOOKAT_BODY"),
        help="Body name for camera look-at (default: try torso, pelvis, …)",
    )
    ap.add_argument(
        "--dataset-stats",
        type=Path,
        default=os.environ.get("ROBOTWIN_DATASET_STATS"),
        help="robotwin dataset_stats.json for z-score denorm (or env ROBOTWIN_DATASET_STATS)",
    )
    ap.add_argument("--prompt", default="pick up the object")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--action-horizon", type=int, default=8)
    ap.add_argument("--num-inference-steps", type=int, default=6)
    ap.add_argument("--http-timeout", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="Do not write predicted qpos back to MuJoCo")
    ap.add_argument("--apply-step", type=float, default=0.02, help="Max delta per joint per step when applying")
    args = ap.parse_args()

    xml_path = _resolve_mjcf_path(args.mjcf)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = Renderer(model, height=480, width=640)

    jn_path = args.joint_names
    if jn_path is None:
        cand = _SCRIPT_DIR / "joint_names_genie_g1_proxy_14.txt"
        if xml_path.resolve() == _DEFAULT_PROXY_MJCF.resolve() and cand.is_file():
            jn_path = cand
    names = load_joint_names(Path(jn_path).expanduser() if jn_path else None)
    qidx = resolve_qpos_indices(model, names)

    mean = std = None
    if args.dataset_stats is not None and str(args.dataset_stats).strip():
        mean, std = load_action_zscore_stats(Path(args.dataset_stats).expanduser().resolve())

    health = requests.get(args.api_base.rstrip("/") + "/health", timeout=10)
    health.raise_for_status()
    print("[health]", health.json())
    print("[mjcf]", xml_path)

    look_pref = args.lookat_body if args.lookat_body else None
    for t in range(args.steps):
        head, left, right = render_triplet(model, data, renderer, lookat_body=look_pref)
        comp = build_robotwin_style_image(head, left, right)
        img_b64 = jpeg_b64(comp)
        prop = read_proprio(data, qidx)
        out = post_infer_action(
            args.api_base,
            img_b64,
            args.prompt,
            prop,
            args.action_horizon,
            args.num_inference_steps,
            args.http_timeout,
        )
        act = out.get("action", [])
        print(f"step={t} action_shape={out.get('action_shape')}")
        if not act:
            continue
        row0 = act[0]
        if mean is not None and std is not None:
            row0 = denorm_action_row(row0, mean, std).tolist()
        else:
            print("  (warn) no --dataset-stats: using normalized actions as-is")

        if not args.dry_run:
            for val, qi in zip(row0, qidx):
                cur = float(data.qpos[qi])
                tgt = float(val)
                delta = max(-args.apply_step, min(args.apply_step, tgt - cur))
                data.qpos[qi] = cur + delta
            mujoco.mj_forward(model, data)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
