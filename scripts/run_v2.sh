#!/usr/bin/env bash
# Stage 2 (v2), as the systemd user service parakeet-v2. Every stage is restartable and
# skipped once done, so after a crash systemd simply runs this again:
#   1. pack clips from timestamped, transcribed episodes (data/stage2/episodes.raw,
#      not included in this repository), holding out whole episodes as
#      data/eval/recordings_test;
#   2. drop clips whose audio no longer matches the transcript timestamps (a different version of the audio), and whole
#      misaligned episodes from the test set (recordings_clean); score the current model on it;
#   3. continue training from runs/full/final on the new clips mixed with part of the
#      original data, one epoch;
#   4. score the result on the NB eval splits (NRK, audiobooks, NB Tale) for comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv_include=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12*/include/python3.12 2>/dev/null | head -1)
[ -n "$uv_include" ] && export C_INCLUDE_PATH="$uv_include"
py=.venv/bin/python
stage() { echo "$(date -Is) == $*"; }

if [ ! -f data/stage2/train/PACKED ]; then
  stage pack
  $py -m norsk_parakeet.recordings pack --episodes data/stage2/episodes.raw --out data/stage2/train \
    --test-out data/eval/recordings_test --max-hours "${MAX_HOURS:?}" --workers 8 | tee data/stage2/pack.log
  grep -q '^TOTAL' data/stage2/pack.log && date -Is > data/stage2/train/PACKED
fi

if [ ! -f data/stage2/filtered/FILTERED ]; then
  stage filter misaligned clips
  $py -m norsk_parakeet.filter --model runs/full/final --src data/stage2/train --out data/stage2/filtered | tee data/stage2/filter.log
  grep -q '^TOTAL' data/stage2/filter.log && date -Is > data/stage2/filtered/FILTERED
fi

if [ ! -f data/eval/recordings_clean.jsonl ]; then
  # Some test episodes have the same problem. Whole episodes are dropped when the current
  # model's median CER against the transcript is above 35% (aligned episodes: under 25%; shifted: ~75%),
  # so no single hard clip is removed for being hard.
  stage clean test set
  mkdir -p data/stage2/test_src && ln -sf "$PWD/data/eval/recordings_test.jsonl" "$PWD/data/eval/recordings_test.i16" data/stage2/test_src/
  $py -m norsk_parakeet.filter --model runs/full/final --src data/stage2/test_src --out data/stage2/test_scored --max-cer 100
  $py - <<'PY'
import json, statistics
from collections import defaultdict
rows = [json.loads(l) for l in open("data/stage2/test_scored/recordings_test.jsonl")]
by_ep = defaultdict(list)
for r in rows:
    by_ep[r["episode"]].append(r["cer"])
keep = {e for e, c in by_ep.items() if statistics.median(c) <= 0.35}
with open("data/eval/recordings_clean.jsonl.tmp", "w") as f:
    for r in rows:
        if r["episode"] in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"recordings_clean: {len(keep)} of {len(by_ep)} episodes", flush=True)
PY
  ln -sf recordings_test.i16 data/eval/recordings_clean.i16
  mv data/eval/recordings_clean.jsonl.tmp data/eval/recordings_clean.jsonl
fi

if [ ! -f data/stage2/mix/READY ]; then
  stage mix
  rm -rf data/stage2/mix && mkdir -p data/stage2/mix
  for f in data/stage2/filtered/rec_*.jsonl; do
    ln -s "$PWD/$f" data/stage2/mix/; ln -s "$PWD/${f%.jsonl}.i16" data/stage2/mix/
  done
  # Part of the original shards, chosen by hash, so the model keeps what it knows.
  for f in $(for g in data/train/*.jsonl; do echo "$(echo "$g" | md5sum | cut -c1-8) $g"; done | sort | head -200 | cut -d' ' -f2); do
    ln -s "$PWD/$f" data/stage2/mix/; ln -s "$PWD/${f%.jsonl}.i16" data/stage2/mix/
  done
  date -Is > data/stage2/mix/READY
fi

if [ ! -f results/v1_recordings_clean/summary.json ]; then
  stage baseline on recordings_clean
  $py -m norsk_parakeet.evaluate --model runs/full/final --eval-dir data/eval --sets recordings_clean \
    --limit 1000 --out results/v1_recordings_clean
fi

stage train
$py -m norsk_parakeet.train --model runs/full/final --train-dir data/stage2/mix --eval-dir data/eval \
  --eval-sets fleurs_nb npsc_nb stortinget_val nst_val recordings_clean \
  --out runs/v2 --epochs 1 --lr 3e-5 --warmup 500 --eval-every 2000

if [ ! -f results/nbeval_v2/final/summary.json ]; then
  stage nb eval splits
  $py -m norsk_parakeet.nbeval --model runs/v2/final --out results/nbeval_v2 \
    --splits test_nrk_nb test_nrk_nn test_audio_books_nb test_audio_books_nn test_nb_tale_nb test_nb_tale_nn
fi
stage done
date -Is > runs/v2/ALL_DONE
