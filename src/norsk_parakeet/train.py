"""Fine-tune a Parakeet TDT checkpoint on norsk-asr's packed Norwegian shards.

Targets are the original transcripts (cased, punctuated), since that is what Parakeet writes;
scoring still goes through norsk-asr's normalize(), like every other model in its table.

    python -m norsk_parakeet.train --train-dir data/train --eval-dir data/eval --out runs/full --epochs 1

The run is resumable: every --save-every-min minutes it writes <out>/ckpt (weights, optimizer
state and the batch position), and a restart with the same arguments continues from there.
The batch order of an epoch depends only on the epoch number, so a resumed run sees the same
batches as an uninterrupted one.
"""

import argparse
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import ParakeetForTDT

from norsk_asr.data import SR, PackedSet, bucket_batches
from norsk_asr.evaluate import load_eval_sets

from .evaluate import evaluate, load
from .loss import tdt_loss

EVAL_SETS = ["fleurs_nb", "npsc_nb", "stortinget_val", "nst_val"]


def target_text(row: dict) -> str:
    # NPSC's verbatim markers (<inaudible>, <ee>) are not words.
    return " ".join(re.sub(r"<[^>]*>", " ", row["orig"]).split())


def plausible(row: dict) -> bool:
    # Transcripts that cannot fit their audio (NPSC has whole sentences on 0.5 s clips) give
    # loss spikes in the thousands; normal speech is ~15 characters per second.
    return 2 <= len(row["orig"]) / (row["n"] / SR) <= 30


def lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def save_checkpoint(model, opt, out: Path, step: int, extra: dict):
    """Write <out>/ckpt atomically: a crash while saving leaves the previous one intact."""
    tmp, final, old = out / "ckpt.tmp", out / "ckpt", out / "ckpt.old"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps({"step": step, **extra}))
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="moondream/parakeet-ultra")
    p.add_argument("--train-dir", default="data/train")
    p.add_argument("--eval-dir", default="data/eval")
    p.add_argument("--eval-sets", nargs="+", default=EVAL_SETS)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-seconds", type=float, default=64)
    p.add_argument("--max-utt-seconds", type=float, default=30)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-limit", type=int, default=300)
    p.add_argument("--final-eval-limit", type=int, default=1000)
    p.add_argument("--save-every-min", type=float, default=30)
    p.add_argument("--max-token-loss", type=float, default=15.0,
                   help="leave out utterances above this loss per token (mismatched transcripts)")
    p.add_argument("--joint-chunk", type=int, default=24, help="encoder frames per joint chunk in the loss")
    p.add_argument("--mem-fraction", type=float, default=0.6, help="the GPU is shared with other services")
    a = p.parse_args()

    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")

    def emit(rec: dict):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    if (out / "done").exists():
        print("already finished", flush=True)
        return

    ckpt = out / "ckpt"
    resumed = ckpt.exists()
    processor, model = load(a.model, device)
    if resumed:
        generation_config = model.generation_config
        del model
        model = ParakeetForTDT.from_pretrained(ckpt, dtype=torch.float32).to(device)
        model.generation_config = generation_config
    # Small batches: keep BatchNorm's pretrained statistics, and no layer-drop.
    model.encoder.config.layerdrop = 0.0

    def train_mode():
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.98), weight_decay=1e-3, fused=True)
    state = {"step": 0, "audio_s": 0.0, "skipped": 0, "outliers": 0}
    if resumed:
        opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=device))
        state.update(json.loads((ckpt / "state.json").read_text()))

    train = PackedSet.from_dir(Path(a.train_dir), max_seconds=a.max_utt_seconds)
    before = len(train)
    train.rows = [r for r in train.rows if plausible(r)]
    per_epoch = len(bucket_batches(train, a.batch_seconds, seed=0))
    total = int(per_epoch * a.epochs)
    emit({"event": "resume" if resumed else "start", "step": state["step"], "total_steps": total,
          "train_utts": len(train), "train_hours": round(train.hours(), 1), "dropped_implausible": before - len(train),
          "args": vars(a)})
    eval_sets = load_eval_sets(a.eval_dir, a.eval_sets, a.eval_limit)
    train_mode()

    step = start_step = state["step"]
    t0, audio0, last_save = time.time(), state["audio_s"], time.time()
    while step < total:
        epoch, k0 = divmod(step, per_epoch)
        batches = bucket_batches(train, a.batch_seconds, seed=epoch)
        for idx in batches[k0:]:
            if step >= total:
                break
            wavs = [train.audio(i).astype(np.float32) / 32768.0 for i in idx]
            inputs = processor(audio=wavs, text=[target_text(train.rows[i]) for i in idx], sampling_rate=SR,
                               return_tensors="pt", padding=True)
            inputs = {kk: v.to(device, non_blocking=True) for kk, v in inputs.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.lr, a.warmup, total)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, dropped = tdt_loss(model, **inputs, chunk=a.joint_chunk, max_token_loss=a.max_token_loss)
                state["outliers"] += dropped
                if loss is None:
                    step += 1
                    continue
                loss.backward()
            except torch.OutOfMemoryError:
                # The GPU is shared with the embedding worker, whose cache grows; skip the batch.
                del inputs
                loss = None
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                state["skipped"] += 1
                emit({"step": step, "oom_skip": state["skipped"]})
                step += 1
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            state["audio_s"] += sum(len(w) for w in wavs) / SR
            if step % 100 == 0:
                el = time.time() - t0
                emit({"step": step, "of": total, "loss": round(loss.item(), 4), "gnorm": round(gnorm.item(), 3),
                      "lr": round(opt.param_groups[0]["lr"], 9), "audio_h": round(state["audio_s"] / 3600, 2),
                      "audio_h_per_s": round((state["audio_s"] - audio0) / 3600 / el, 4),
                      "eta_h": round((total - step) * el / (step - start_step) / 3600, 2),
                      "outliers": state["outliers"],
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "time": time.strftime("%H:%M")})
            if step % a.eval_every == 0:
                model.eval()
                res = evaluate(processor, model, eval_sets, device, tag=f"step{step}")
                emit({"step": step, "eval": res})
                train_mode()
            if time.time() - last_save > a.save_every_min * 60:
                save_checkpoint(model, opt, out, step, {k: v for k, v in state.items() if k != "step"})
                emit({"step": step, "saved": "ckpt", "time": time.strftime("%H:%M")})
                last_save = time.time()

    save_checkpoint(model, opt, out, step, {k: v for k, v in state.items() if k != "step"})
    model.save_pretrained(out / "final")
    emit({"step": step, "saved": "final"})
    model.eval()
    res = evaluate(processor, model, load_eval_sets(a.eval_dir, a.eval_sets, a.final_eval_limit), device,
                   out=out / "final_eval", tag="final")
    emit({"step": step, "final_eval": res})
    (out / "done").write_text(time.strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()
