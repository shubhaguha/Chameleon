#!/bin/bash
# Start the local SDXL inpainting backend natively on macOS (needs the Apple GPU, so not in Docker).
# First run: creates the venv and downloads ~7GB of weights into InpaintServer/models/.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi
[ -d models/diffusers ] || ./download_models.sh
# MPS lacks a few ops; fall back to CPU for them instead of failing
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/uvicorn main:app --host 0.0.0.0 --port "${PORT:-8006}"
