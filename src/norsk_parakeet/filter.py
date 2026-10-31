"""Drop clips whose transcript does not match the audio.

Audio fetched later can differ from the version the transcript was made from (inserted or
removed segments), which shifts it against the timestamps: from that point on, every clip holds
other words than its transcript.
The current model transcribes every clip; clips with a character error rate above --max-cer
against the transcript are left out. Misaligned clips sit around 75% CER, aligned ones mostly under 20%,
so a loose threshold drops the misaligned ones and keeps clips that are merely hard (dialect,
crosstalk), which are what the model should learn from.

    python -m norsk_parakeet.filter --model runs/full/final --src data/stage2/train --out data/stage2/filtered

Writes <out>/<shard>.jsonl with the kept rows (and their "cer") next to a symlink to the shard's
.i16, so <out> is a PackedSet directory. Restartable per shard.
"""

import argparse
import json
from pathlib import Path

import jiwer
import torch

from norsk_asr.data import PackedSet, bucket_batches
from norsk_asr.text import normalize

from .evaluate import load, transcribe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/full/final")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-cer", type=float, default=0.5)
    p.add_argument("--mem-fraction", type=float, default=0.6)
    a = p.parse_args()
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    processor, model = load(a.model, device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept_h = total_h = 0.0
    for src in sorted(Path(a.src).glob("*.jsonl")):
        dst = out / src.name
        if not dst.exists():
            ds = PackedSet([src])
            hyps = {}
            for idx in bucket_batches(ds, 300, seed=0, shuffle=False):
                for i, h in zip(idx, transcribe(processor, model, [ds.audio(i) for i in idx], device)):
                    hyps[i] = normalize(h)
            tmp = dst.with_suffix(".jsonl.tmp")
            with open(tmp, "w") as f:
                for i, r in enumerate(ds.rows):
                    cer = jiwer.cer(r["text"], hyps[i]) if r["text"] else 1.0
                    if cer <= a.max_cer:
                        row = {k: v for k, v in r.items() if k != "shard"}
                        f.write(json.dumps({**row, "cer": round(cer, 3)}, ensure_ascii=False) + "\n")
            tmp.rename(dst)
            link = out / src.with_suffix(".i16").name
            if not link.exists():
                link.symlink_to(src.with_suffix(".i16").resolve())
        rows = [json.loads(l) for l in open(dst)]
        n_src = sum(json.loads(l)["n"] for l in open(src))
        kept_h += sum(r["n"] for r in rows) / 16000 / 3600
        total_h += n_src / 16000 / 3600
        print(f"{src.stem}: kept {kept_h:.1f} of {total_h:.1f} h", flush=True)
    print(f"TOTAL kept {kept_h:.1f} of {total_h:.1f} h", flush=True)


if __name__ == "__main__":
    main()
