#!/usr/bin/env bash
# Ubuntu 22.04+: Python MuJoCo + deps for Genie G01 HTTP inference loop (no Unitree menagerie).
set -euo pipefail

echo "==> APT (GL/EGL for headless Renderer; skipped if no passwordless sudo)"
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq git python3-pip python3-venv libgl1-mesa-glx libegl1 || true
elif [ "$(id -u)" -eq 0 ]; then
  apt-get update -qq
  apt-get install -y -qq git python3-pip python3-venv libgl1-mesa-glx libegl1 || true
else
  echo "WARN: no passwordless sudo — install git python3-pip libgl1-mesa-glx libegl1 yourself, then re-run."
fi

echo "==> pip: mujoco numpy pillow requests"
python3 -m pip install --user -U "mujoco>=3.1.0" "numpy>=1.23" "pillow>=9" "requests>=2.28"

GENIE_DIR="${GENIE_G1_MUJOCO_DIR:-$HOME/genie_g1_mujoco}"
mkdir -p "$GENIE_DIR"
echo "Place vendor Genie G01 MJCF here (optional). Default client uses bundled proxy if unset:"
echo "  export MUJOCO_GENIE_G1_XML=$GENIE_DIR/your_scene.xml"
echo "OK. Typical exports:"
echo "  export MUJOCO_GL=egl"
echo "  export MUJOCO_GENIE_G1_XML=...   # or omit to use proxy MJCF next to client"
