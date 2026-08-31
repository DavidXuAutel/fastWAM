#!/usr/bin/env python3
"""Preflight checks for tau0-WM VAM-only deployment."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _check_path(path: Path, *, expect_dir: Optional[bool] = None) -> Dict[str, Any]:
    exists = path.exists()
    ok = exists
    if exists and expect_dir is not None:
        ok = path.is_dir() if expect_dir else path.is_file()
    return {"ok": bool(ok), "path": str(path), "exists": bool(exists)}


def _check_cuda(require_cuda: bool) -> Dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {
            "ok": not require_cuda,
            "required": require_cuda,
            "nvidia_smi": None,
            "message": "nvidia-smi not found",
        }
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - host dependent
        return {"ok": False, "required": require_cuda, "nvidia_smi": nvidia_smi, "message": str(exc)}
    ok = proc.returncode == 0
    return {
        "ok": bool(ok or not require_cuda),
        "required": require_cuda,
        "nvidia_smi": nvidia_smi,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def run_preflight(
    *,
    tau_repo: Path,
    tau_checkpoint: Path,
    wan_root: Path,
    require_cuda: bool = True,
) -> Dict[str, Any]:
    tau_repo = tau_repo.expanduser().resolve()
    tau_checkpoint = tau_checkpoint.expanduser().resolve()
    wan_root = wan_root.expanduser().resolve()

    checks: Dict[str, Dict[str, Any]] = {
        "tau_repo": _check_path(tau_repo, expect_dir=True),
        "deployment_config": _check_path(
            tau_repo / "configs" / "deployment" / "wan_pretrain_rela_eef6d.yaml",
            expect_dir=False,
        ),
        "infer_server_script": _check_path(tau_repo / "run_infer_server.sh", expect_dir=False),
        "simple_client": _check_path(tau_repo / "web_infer_utils" / "simple_client.py", expect_dir=False),
        "tau_checkpoint": _check_path(tau_checkpoint),
        "wan_root": _check_path(wan_root, expect_dir=True),
        "wan_vae": _check_path(wan_root / "Wan2.2_VAE.pth", expect_dir=False),
        "wan_text_encoder": _check_path(
            wan_root / "models_t5_umt5-xxl-enc-bf16.pth",
            expect_dir=False,
        ),
        "wan_tokenizer": _check_path(wan_root / "google" / "umt5-xxl", expect_dir=True),
        "cuda": _check_cuda(require_cuda),
    }
    ready = all(item["ok"] for item in checks.values())
    return {
        "ready": bool(ready),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "checks": checks,
        "next_steps": _next_steps(checks),
    }


def _next_steps(checks: Dict[str, Dict[str, Any]]) -> list[str]:
    steps = []
    if not checks["tau_repo"]["ok"]:
        steps.append("Clone https://github.com/sii-research/tau-0-wm and pass --tau-repo.")
    if not checks["tau_checkpoint"]["ok"]:
        steps.append("Download the tau0-WM VAM checkpoint from Hugging Face and pass --tau-checkpoint.")
    if not checks["wan_root"]["ok"] or not checks["wan_vae"]["ok"]:
        steps.append("Download Wan-AI/Wan2.2-TI2V-5B and pass --wan-root.")
    if not checks["cuda"]["ok"]:
        steps.append("Run on a CUDA host with nvidia-smi available, or use --no-require-cuda for dry-run checks.")
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-repo", type=Path, required=True)
    parser.add_argument("--tau-checkpoint", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, required=True)
    parser.add_argument("--no-require-cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = run_preflight(
        tau_repo=args.tau_repo,
        tau_checkpoint=args.tau_checkpoint,
        wan_root=args.wan_root,
        require_cuda=not args.no_require_cuda,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
