#!/bin/bash
# Start the local SDXL inpainting backend natively on macOS (needs the Apple GPU, so not in Docker).
# First run: creates the venv and downloads ~7GB of weights into InpaintServer/models/.
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
# every start, so an existing venv picks up new requirements (a no-op when nothing changed)
uv pip install -q --python .venv/bin/python -r requirements.txt
[ -d models/diffusers ] || ./download_models.sh
# MPS lacks a few ops; fall back to CPU for them instead of failing
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/uvicorn main:app --host 0.0.0.0 --port "${PORT:-8006}"
