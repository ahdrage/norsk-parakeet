#!/usr/bin/env bash
# The full run, as the systemd user service parakeet-train. Waits for data packing, then trains
# one epoch over everything; after a crash systemd starts this again and training resumes from
# runs/full/ckpt. Exits 0 once runs/full/done exists.
set -euo pipefail
cd "$(dirname "$0")/.."
while systemctl --user is-active --quiet parakeet-pack; do sleep 60; done
grep -q '^TOTAL' pack.log || { echo "data packing did not finish; see pack.log" >&2; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Triton builds a small C helper on first use and needs Python headers; this server has no
# python3.12-dev, so use a uv-managed CPython's (`uv python install 3.12`).
uv_include=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*/include/python3.12 2>/dev/null | head -1)
[ -n "$uv_include" ] && export C_INCLUDE_PATH="$uv_include"
exec .venv/bin/python -m norsk_parakeet.train --train-dir data/train --eval-dir data/eval --out runs/full --epochs 1
