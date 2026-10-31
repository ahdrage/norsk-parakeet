#!/usr/bin/env bash
# Stage 2 (v2), as the systemd user service parakeet-bc. Every stage is restartable and
# skipped once done, so after a crash systemd simply runs this again:
#   1. pack clips from timestamped, transcribed episodes (data/bc/episodes.raw,
#      not included in this repository), holding out whole episodes as
#      data/eval/broadcast_test;
#   2. drop clips whose audio no longer matches the transcript timestamps (a different version of the audio), and whole
#      misaligned episodes from the test set (broadcast_clean); score the current model on it;
#   3. continue training from runs/full/final on the broadcast clips mixed with part of the
#      original data, one epoch;
#   4. score the result on the NB eval splits (NRK, audiobooks, NB Tale) for comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv_include=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*/include/python3.12 2>/dev/null | head -1)
[ -n "$uv_include" ] && export C_INCLUDE_PATH="$uv_include"
py=.venv/bin/python
stage() { echo "$(date -Is) == $*"; }

if [ ! -f data/bc/train/PACKED ]; then
  stage pack
  $py -m norsk_parakeet.broadcast pack --episodes data/bc/episodes.raw --out data/bc/train \
    --test-out data/eval/broadcast_test --max-hours "${MAX_HOURS:?}" --workers 8 | tee data/bc/pack.log
  grep -q '^TOTAL' data/bc/pack.log && date -Is > data/bc/train/PACKED
fi

if [ ! -f data/bc/filtered/FILTERED ]; then
  stage filter misaligned clips
  $py -m norsk_parakeet.filter --model runs/full/final --src data/bc/train --out data/bc/filtered | tee data/bc/filter.log
  grep -q '^TOTAL' data/bc/filter.log && date -Is > data/bc/filtered/FILTERED
fi

if [ ! -f data/eval/broadcast_clean.jsonl ]; then
  # Some test episodes have the same problem. Whole episodes are dropped when the current
  # model's median CER against the transcript is above 35% (aligned episodes: under 25%; shifted: ~75%),
  # so no single hard clip is removed for being hard.
  stage clean test set
  mkdir -p data/bc/test_src && ln -sf "$PWD/data/eval/broadcast_test.jsonl" "$PWD/data/eval/broadcast_test.i16" data/bc/test_src/
  $py -m norsk_parakeet.filter --model runs/full/final --src data/bc/test_src --out data/bc/test_scored --max-cer 100
  $py - <<'PY'
import json, statistics
from collections import defaultdict
rows = [json.loads(l) for l in open("data/bc/test_scored/broadcast_test.jsonl")]
by_ep = defaultdict(list)
for r in rows:
    by_ep[r["episode"]].append(r["cer"])
keep = {e for e, c in by_ep.items() if statistics.median(c) <= 0.35}
with open("data/eval/broadcast_clean.jsonl.tmp", "w") as f:
    for r in rows:
        if r["episode"] in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"broadcast_clean: {len(keep)} of {len(by_ep)} episodes", flush=True)
PY
  ln -sf broadcast_test.i16 data/eval/broadcast_clean.i16
  mv data/eval/broadcast_clean.jsonl.tmp data/eval/broadcast_clean.jsonl
fi

if [ ! -f data/bc/mix/READY ]; then
  stage mix
  rm -rf data/bc/mix && mkdir -p data/bc/mix
  for f in data/bc/filtered/bc_*.jsonl; do
    ln -s "$PWD/$f" data/bc/mix/; ln -s "$PWD/${f%.jsonl}.i16" data/bc/mix/
  done
  # Part of the original shards, chosen by hash, so the model keeps what it knows.
  for f in $(for g in data/train/*.jsonl; do echo "$(echo "$g" | md5sum | cut -c1-8) $g"; done | sort | head -200 | cut -d' ' -f2); do
    ln -s "$PWD/$f" data/bc/mix/; ln -s "$PWD/${f%.jsonl}.i16" data/bc/mix/
  done
  date -Is > data/bc/mix/READY
fi

if [ ! -f results/bc_base_clean/summary.json ]; then
  stage baseline on broadcast_clean
  $py -m norsk_parakeet.evaluate --model runs/full/final --eval-dir data/eval --sets broadcast_clean \
    --limit 1000 --out results/bc_base_clean
fi

stage train
$py -m norsk_parakeet.train --model runs/full/final --train-dir data/bc/mix --eval-dir data/eval \
  --eval-sets fleurs_nb npsc_nb stortinget_val nst_val broadcast_clean \
  --out runs/bc1 --epochs 1 --lr 3e-5 --warmup 500 --eval-every 2000

if [ ! -f results/nbeval_bc1/final/summary.json ]; then
  stage nb eval splits
  $py -m norsk_parakeet.nbeval --model runs/bc1/final --out results/nbeval_bc1 \
    --splits test_nrk_nb test_nrk_nn test_audio_books_nb test_audio_books_nn test_nb_tale_nb test_nb_tale_nn
fi
stage done
date -Is > runs/bc1/ALL_DONE
