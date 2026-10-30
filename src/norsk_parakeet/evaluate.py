"""Evaluate a Parakeet TDT checkpoint on norsk-asr's eval sets, scored exactly like norsk-asr.

    python -m norsk_parakeet.evaluate --model runs/full/final --eval-dir data/eval --limit 1000 --out results/full
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT

from norsk_asr.data import SR, bucket_batches
from norsk_asr.evaluate import load_eval_sets, score
from norsk_asr.text import normalize

BASE = "nvidia/parakeet-tdt-0.6b-v3"  # parakeet-ultra ships no processor or generation config; they are the base's


def load(model_id: str, device):
    processor = AutoProcessor.from_pretrained(BASE)
    model = ParakeetForTDT.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
    if model.generation_config is None or model.generation_config.decoder_start_token_id is None:
        model.generation_config = GenerationConfig.from_pretrained(BASE)
    return processor, model


@torch.no_grad()
def transcribe(processor, model, wavs: list[np.ndarray], device) -> list[str]:
    inputs = processor([w.astype(np.float32) / 32768.0 for w in wavs], sampling_rate=SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**inputs)
    return processor.batch_decode(out.sequences, skip_special_tokens=True)


def evaluate(processor, model, sets, device, max_batch_seconds=300, out=None, tag=""):
    results = {}
    for name, ds in sets.items():
        hyps = {}
        t0 = time.time()
        for idx in bucket_batches(ds, max_batch_seconds, seed=0, shuffle=False):
            for i, h in zip(idx, transcribe(processor, model, [ds.audio(i) for i in idx], device)):
                hyps[i] = h
        torch.cuda.synchronize()
        wall = time.time() - t0
        order = range(len(ds))
        refs = [ds.rows[i]["text"] for i in order]
        raw = [hyps[i] for i in order]
        res = score(refs, [normalize(h) for h in raw], [ds.rows[i].get("has_digits", False) for i in order])
        res["speed_x"] = ds.hours() * 3600 / wall
        results[name] = res
        print(tag, name, json.dumps(res), flush=True)
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            with open(Path(out) / f"{name}.jsonl", "w") as f:
                for i in order:
                    f.write(json.dumps({"id": ds.rows[i]["id"], "ref": refs[i], "raw": raw[i],
                                        "hyp": normalize(raw[i]), "has_digits": ds.rows[i].get("has_digits", False)},
                                       ensure_ascii=False) + "\n")
    if out:
        (Path(out) / "summary.json").write_text(json.dumps(results, indent=1))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="moondream/parakeet-ultra")
    p.add_argument("--eval-dir", default="eval")
    p.add_argument("--sets", nargs="*")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--out")
    p.add_argument("--mem-fraction", type=float, default=0.6, help="leave room for the embedding worker")
    a = p.parse_args()
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    processor, model = load(a.model, device)
    evaluate(processor, model, load_eval_sets(a.eval_dir, a.sets, a.limit), device, out=a.out, tag=a.model)


if __name__ == "__main__":
    sys.exit(main())
