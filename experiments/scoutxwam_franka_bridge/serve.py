#!/usr/bin/env python3
"""Persistent FastAPI server for ScoutXWAM DROID-100 Franka inference.

Run on the H100 package host:

  cd /home/a25689/FastWAM/scoutxwam_droid100_inference
  CUDA_VISIBLE_DEVICES=0 /home/a25689/micromamba/envs/mot-wam/bin/python serve.py \\
    --host 127.0.0.1 --port 8010
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _package_root(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.resolve()
    here = Path(__file__).resolve().parent
    if (here / "infer.py").is_file() and (here / "model").is_dir():
        return here
    # Running from FastWAM experiments/scoutxwam_franka_bridge/
    sibling = Path("/home/a25689/FastWAM/scoutxwam_droid100_inference")
    if sibling.is_dir():
        return sibling
    raise FileNotFoundError(
        "Cannot locate scoutxwam_droid100_inference; pass --package-root"
    )


class InferRequest(BaseModel):
    prompt: str
    proprio: list[float] = Field(..., min_length=8, max_length=16)
    video_shape: list[int]
    video_b64: str
    seed: int = 42
    action_denoise_steps: int = 10


class ScoutEngine:
    def __init__(self, package: Path) -> None:
        self.package = package
        self.lock = threading.Lock()
        self.model = None
        self.config = None
        self.state_lo = None
        self.state_hi = None
        self.action_lo = None
        self.action_hi = None
        self.has_right = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, *, checkpoint: Path, config_path: Path, wan_dir: Path) -> None:
        # Import infer helpers from package
        sys.path.insert(0, str(self.package))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import infer as infer_mod  # type: ignore

        from omegaconf import OmegaConf

        xwam_root = self.package / "third_party" / "X-WAM"
        src_root = self.package / "src"
        sys.path.insert(0, str(xwam_root))
        sys.path.insert(0, str(src_root))
        from runners.xwam_runner import XWAMRunner

        config = OmegaConf.load(config_path)
        config.wan_checkpoint_dir = str(wan_dir)
        config.sample_steps = 10
        config.action_denoise_steps = 10
        config.use_decoupled_inference = True
        config.action_num = config.dataset.frame_skip // config.dataset.action_skip

        state_lo, state_hi, action_lo, action_hi, has_right = infer_mod.bounds(config)
        model = XWAMRunner(config=config).to(self.device, dtype=torch.bfloat16)
        ckpt = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("module"))
        if state is None:
            raise ValueError(f"Unsupported checkpoint keys: {sorted(ckpt)[:20]}")
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint mismatch: "
                f"missing={incompatible.missing_keys[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
        model.eval()

        self.model = model
        self.config = config
        self.infer_mod = infer_mod
        self.state_lo = state_lo
        self.state_hi = state_hi
        self.action_lo = action_lo
        self.action_hi = action_hi
        self.has_right = has_right

    def infer(
        self,
        *,
        video: np.ndarray,
        proprio_raw: np.ndarray,
        prompt: str,
        seed: int,
        action_denoise_steps: int,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("model not loaded")
        with self.lock:
            t0 = time.time()
            self.config.sample_steps = action_denoise_steps
            self.config.action_denoise_steps = action_denoise_steps

            proprio_raw = np.asarray(proprio_raw, dtype=np.float32).reshape(-1)
            if proprio_raw.size == 8 and not self.has_right:
                proprio_raw = np.concatenate([proprio_raw, np.zeros(8, dtype=np.float32)])
            if proprio_raw.shape != self.state_lo.shape:
                raise ValueError(
                    f"proprio must have {len(self.state_lo)} values (or 8), got {proprio_raw.size}"
                )
            proprio = (
                2
                * (proprio_raw - self.state_lo)
                / np.maximum(self.state_hi - self.state_lo, 1e-8)
                - 1
            )
            if not self.has_right:
                proprio[8:] = 0

            rgb = self.infer_mod.preprocess_rgb(
                video, tuple(self.config.dataset.video_size)
            ).to(self.device, dtype=torch.bfloat16)
            proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(
                self.device, dtype=torch.bfloat16
            )
            with torch.inference_mode():
                _, actions, proprios, _ = self.model.generate(
                    rgb,
                    proprio_t,
                    [prompt],
                    seeds=[seed],
                    early_stop=True,
                    cfg=0.0,
                    run_depth=False,
                )
            actions = actions[0, :, : len(self.action_lo)].float().cpu().numpy()
            actions = (actions + 1) / 2 * (self.action_hi - self.action_lo) + self.action_lo
            predicted = proprios[0].float().cpu().numpy()
            predicted = (predicted + 1) / 2 * (self.state_hi - self.state_lo) + self.state_lo
            return {
                "actions": actions.astype(np.float32).tolist(),
                "proprios": predicted.astype(np.float32).tolist(),
                "actions_shape": list(actions.shape),
                "proprios_shape": list(predicted.shape),
                "infer_s": round(time.time() - t0, 3),
            }


def build_app(engine: ScoutEngine) -> FastAPI:
    app = FastAPI(title="ScoutXWAM DROID-100 Infer", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "model_loaded": engine.model is not None,
            "device": engine.device,
            "package": str(engine.package),
        }

    @app.post("/v1/infer")
    def infer(req: InferRequest) -> dict[str, Any]:
        if engine.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        shape = tuple(int(x) for x in req.video_shape)
        if len(shape) != 4 or shape[0] != 2 or shape[-1] != 3:
            raise HTTPException(
                status_code=400, detail=f"video_shape must be [2,H,W,3], got {shape}"
            )
        try:
            raw = base64.b64decode(req.video_b64, validate=True)
            video = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"bad video payload: {exc}") from exc
        try:
            return engine.infer(
                video=video,
                proprio_raw=np.asarray(req.proprio, dtype=np.float32),
                prompt=req.prompt,
                seed=req.seed,
                action_denoise_steps=req.action_denoise_steps,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model/checkpoints/epoch=0-step=1000.ckpt"),
    )
    parser.add_argument("--config", type=Path, default=Path("model/training/config.yaml"))
    parser.add_argument("--wan-checkpoint-dir", type=Path, default=Path("base/wan22_5b"))
    args = parser.parse_args()

    package = _package_root(args.package_root)
    resolve = lambda p: p if p.is_absolute() else package / p
    engine = ScoutEngine(package)
    print(f"[serve] loading from {package} on {engine.device}", flush=True)
    engine.load(
        checkpoint=resolve(args.checkpoint),
        config_path=resolve(args.config),
        wan_dir=resolve(args.wan_checkpoint_dir),
    )
    print("[serve] model ready", flush=True)
    app = build_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
