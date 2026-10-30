"""Score models on splits of NbAiLab/nb-asr-eval-withwav-sorted (e.g. NB Tale), scored like norsk-asr.

    python -m norsk_parakeet.nbeval --model runs/full/final --data data/nbeval --splits test_nb_tale_nb --out results/nbtale
    python -m norsk_parakeet.nbeval --model NbAiLab/nb-whisper-large --data data/nbeval --splits test_nb_tale_nb --out results/nbtale

The dataset's licence file is empty: use it for internal evaluation only and do not redistribute
the audio. Results are reported for all clips, for read and free speech separately (NB Tale ids
contain "free" for spontaneous speech), and without clips whose transcripts cannot fit their audio
(outside 2-30 characters per second, the same rule as the training data), a filter that does not
depend on any model.
"""

import argparse
import json
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from norsk_asr.data import SR
from norsk_asr.evaluate import score
from norsk_asr.text import normalize


def load_split(root: Path, split: str) -> list[dict]:
    rows = [json.loads(l) for l in open(root / "audio" / split / "metadata.jsonl")]
    for r in rows:
        wav, sr = sf.read(root / "audio" / split / r["file_name"], dtype="float32", always_2d=True)
        wav = wav.mean(1)
        if sr != SR:
            g = gcd(sr, SR)
            wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
        r["wav"], r["dur"] = wav, len(wav) / SR
    return rows


def parakeet_transcriber(model_id: str, batch_seconds: float = 300):
    from .evaluate import load

    device = torch.device("cuda")
    processor, model = load(model_id, device)

    @torch.no_grad()
    def run(wavs: list[np.ndarray]) -> list[str]:
        order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
        out, batch, secs = {}, [], 0.0

        def flush():
            inputs = processor([wavs[i] for i in batch], sampling_rate=SR, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                seq = model.generate(**inputs).sequences
            out.update(zip(batch, processor.batch_decode(seq, skip_special_tokens=True)))

        for i in order:
            d = len(wavs[i]) / SR
            if batch and (len(batch) + 1) * max(d, secs) > batch_seconds:
                flush()
                batch, secs = [], 0.0
            batch.append(i)
            secs = max(secs, d)
        if batch:
            flush()
        return [out[i] for i in range(len(wavs))]

    return run


def whisper_transcriber(model_id: str, batch_size: int = 16):
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=model_id, device=0, dtype=torch.float16)
    gen = {"task": "transcribe", "language": "no", "num_beams": 1}  # as norsk_asr.baseline

    def run(wavs: list[np.ndarray]) -> list[str]:
        audio = [{"raw": w, "sampling_rate": SR} for w in wavs]
        return [x["text"] for x in asr(audio, batch_size=batch_size, chunk_length_s=28, generate_kwargs=gen)]

    return run


def report(rows: list[dict], hyps: list[str]) -> dict:
    def sub(keep):
        idx = [i for i, r in enumerate(rows) if keep(r)]
        res = score([normalize(rows[i]["text"]) for i in idx], [hyps[i] for i in idx],
                    [any(c.isdigit() for c in rows[i]["text"]) for i in idx])
        return {k: (round(100 * v, 2) if k != "utts" else v) for k, v in res.items()}

    plausible = lambda r: 2 <= len(r["text"]) / max(r["dur"], 0.1) <= 30
    return {"all": sub(lambda r: True), "plausible": sub(plausible),
            "read": sub(lambda r: plausible(r) and "free" not in r["id"]),
            "free": sub(lambda r: plausible(r) and "free" in r["id"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/nbeval")
    p.add_argument("--splits", nargs="+", default=["test_nb_tale_nb", "test_nb_tale_nn"])
    p.add_argument("--out", required=True)
    p.add_argument("--mem-fraction", type=float, default=0.6)
    a = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    whisper = "whisper" in a.model.lower()
    run = whisper_transcriber(a.model) if whisper else parakeet_transcriber(a.model)
    out = Path(a.out) / Path(a.model).name
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in a.splits:
        rows = load_split(Path(a.data), split)
        t0 = time.time()
        raw = run([r["wav"] for r in rows])
        wall = time.time() - t0
        hyps = [normalize(h) for h in raw]
        summary[split] = {**report(rows, hyps), "speed_x": round(sum(r["dur"] for r in rows) / wall)}
        print(split, json.dumps(summary[split]), flush=True)
        with open(out / f"{split}.jsonl", "w") as f:
            for r, h, x in zip(rows, hyps, raw):
                f.write(json.dumps({"id": r["id"], "ref": r["text"], "raw": x, "hyp": h, "dur": round(r["dur"], 2)},
                                   ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"model": a.model, "results": summary}, indent=1))


if __name__ == "__main__":
    main()
