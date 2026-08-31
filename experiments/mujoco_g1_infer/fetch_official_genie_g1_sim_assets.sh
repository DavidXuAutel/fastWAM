#!/usr/bin/env bash
# Download AgiBot-published Genie G1 *simulation assets* (USD) from Hugging Face dataset GenieSimAssets.
# Public tree does NOT ship a Genie G01 MuJoCo MJCF; use vendor MJCF in vendor_genie_g1_mjcf/ for MuJoCo.
set -euo pipefail
DEST="${GENIE_G1_OFFICIAL_DIR:-$HOME/genie_g1_official_hf}"
python3 -m pip install --user -q "huggingface_hub>=0.23"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
python3 << 'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

dest = Path(os.environ.get("GENIE_G1_OFFICIAL_DIR", Path.home() / "genie_g1_official_hf")).expanduser()
dest.mkdir(parents=True, exist_ok=True)
print("Downloading to:", dest.resolve())
snapshot_download(
    repo_id="agibot-world/GenieSimAssets",
    repo_type="dataset",
    local_dir=str(dest),
    allow_patterns=[
        "robot/G1_omnipicker/**",
        "robot/G1_120s/**",
    ],
)
print("Done. Official G1 USD packs under:", dest / "robot")
PY
