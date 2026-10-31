#!/usr/bin/env bash
# Stage 2 again (v3), as the systemd user service parakeet-v3. Restartable stage by stage:
#   1. pack up to MAX_HOURS of clips from timestamped, transcribed recordings
#      (data/bc/episodes_v3.raw, not included in this repository), cut in the middle of pauses,
#      keeping only clips that v1 hears from the first word to the last (CER <= 30%); a few
#      recordings are held out as data/eval/heldout2, and v2's test recordings are left out entirely;
#   2. continue training from v1 (not v2) on those clips mixed with part of the original data;
#   3. count runaway outputs on v2's 3,571 held-out clips, and score the NB eval splits.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv_include=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*/include/python3.12 2>/dev/null | head -1)
[ -n "$uv_include" ] && export C_INCLUDE_PATH="$uv_include"
py=.venv/bin/python
stage() { echo "$(date -Is) == $*"; }

if [ ! -f data/v3/train/PACKED ]; then
  stage pack
  mkdir -p data/v3
  $py -m norsk_parakeet.broadcast pack --episodes data/bc/episodes_v3.raw --out data/v3/train \
    --test-out data/eval/heldout2 --test-episodes 4 \
    --exclude-ids data/bc/test_episodes_v2.txt --filter-model runs/full/final --max-cer 0.3 \
    --max-hours "${MAX_HOURS:?}" --workers 8 | tee -a data/v3/pack.log
  grep -q '^TOTAL' data/v3/pack.log && date -Is > data/v3/train/PACKED
fi

if [ ! -f data/v3/mix/READY ]; then
  stage mix
  rm -rf data/v3/mix && mkdir -p data/v3/mix
  for f in data/v3/train/bc_*.jsonl; do
    ln -s "$PWD/$f" data/v3/mix/; ln -s "$PWD/${f%.jsonl}.i16" data/v3/mix/
  done
  # Part of the original shards, chosen by hash.
  for f in $(for g in data/train/*.jsonl; do echo "$(echo "$g" | md5sum | cut -c1-8) $g"; done | sort | head -310 | cut -d' ' -f2); do
    ln -s "$PWD/$f" data/v3/mix/; ln -s "$PWD/${f%.jsonl}.i16" data/v3/mix/
  done
  date -Is > data/v3/mix/READY
fi

if [ ! -f results/v1_heldout2/summary.json ]; then
  stage baseline on heldout2
  $py -m norsk_parakeet.evaluate --model runs/full/final --eval-dir data/eval --sets heldout2 \
    --limit 1000 --out results/v1_heldout2
fi

stage train
$py -m norsk_parakeet.train --model runs/full/final --train-dir data/v3/mix --eval-dir data/eval \
  --eval-sets fleurs_nb npsc_nb stortinget_val nst_val broadcast_clean heldout2 \
  --out runs/v3 --epochs 1 --lr 3e-5 --warmup 500 --eval-every 5000

if [ ! -f results/v3_runaway.json ]; then
  stage runaway count
  $py -m norsk_parakeet.evaluate --model runs/v3/final --eval-dir data/eval --sets broadcast_test heldout2 \
    --limit 100000 --out results/v3_bt
  $py - <<'PY'
import json
out = {}
for name in ["broadcast_test", "heldout2"]:
    rows = [json.loads(l) for l in open(f"results/v3_bt/{name}.jsonl")]
    n = sum(len(r["hyp"].split()) > 1.6 * len(r["ref"].split()) + 8 for r in rows)
    out[name] = {"clips": len(rows), "runaway": n}
print("RUNAWAY", json.dumps(out), flush=True)
json.dump(out, open("results/v3_runaway.json", "w"))
PY
fi

if [ ! -f results/nbeval_v3/final/summary.json ]; then
  stage nb eval splits
  $py -m norsk_parakeet.nbeval --model runs/v3/final --out results/nbeval_v3 \
    --splits test_nrk_nb test_nrk_nn test_audio_books_nb test_audio_books_nn test_nb_tale_nb test_nb_tale_nn
fi
stage done
date -Is > runs/v3/ALL_DONE
