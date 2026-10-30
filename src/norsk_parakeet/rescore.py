"""Rescore beam-search n-best lists with a causal language model.

    python -m norsk_parakeet.rescore lm --nbest results/beam8_dev --lm NbAiLab/nb-llama-3.2-1B   # adds "lm" scores
    python -m norsk_parakeet.rescore tune --nbest results/beam8_dev                               # alpha/beta grid
    python -m norsk_parakeet.rescore apply --nbest results/beam8_test --alpha 0.3 --beta 0.5

Each candidate scores am + alpha * lm + beta * words, where am is the transducer's log-prob and lm
the language model's log-prob of the cased, punctuated text. Weights are tuned on dev sets only.
"""

import argparse
import json
from pathlib import Path

import torch

from norsk_asr.evaluate import score
from norsk_asr.text import normalize


def read(nbest_dir: Path) -> dict[str, list[dict]]:
    return {p.stem: [json.loads(l) for l in open(p)] for p in sorted(nbest_dir.glob("*.jsonl"))}


@torch.no_grad()
def cmd_lm(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.lm)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(a.lm, dtype=torch.bfloat16).cuda().eval()
    for name, rows in read(Path(a.nbest)).items():
        texts = sorted({c["text"] for r in rows for c in r["nbest"]}, key=len)
        scores = {}
        for s in range(0, len(texts), a.batch):
            batch = texts[s:s + a.batch]
            enc = tok([tok.bos_token + t for t in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            logits = lm(**enc).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(2, enc.input_ids[:, 1:, None])[..., 0]
            lp = (lp * enc.attention_mask[:, 1:]).sum(1)
            scores.update(zip(batch, lp.tolist()))
        for r in rows:
            for c in r["nbest"]:
                c["lm"] = scores[c["text"]]
        with open(Path(a.nbest) / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(texts), "candidates scored", flush=True)


def pick(rows, alpha, beta):
    return [normalize(max(r["nbest"], key=lambda c: c["am"] + alpha * c["lm"] + beta * len(c["text"].split()))["text"])
            for r in rows]


def wers(sets, alpha, beta):
    out = {}
    for name, rows in sets.items():
        res = score([r["ref"] for r in rows], pick(rows, alpha, beta), [r["has_digits"] for r in rows])
        out[name] = round(100 * res.get("wer_nodigit", res["wer"]), 2)
    return out


def cmd_tune(a):
    sets = read(Path(a.nbest))
    base = wers(sets, 0.0, 0.0)
    print("am only", base)
    best = None
    for alpha in a.alphas:
        for beta in a.betas:
            w = wers(sets, alpha, beta)
            avg = sum(w.values()) / len(w)
            if best is None or avg < best[0]:
                best = (avg, alpha, beta, w)
    print("best alpha=%s beta=%s" % best[1:3], best[3])


def cmd_apply(a):
    sets = read(Path(a.nbest))
    print("am only", wers(sets, 0.0, 0.0))
    print(f"alpha={a.alpha} beta={a.beta}", wers(sets, a.alpha, a.beta))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lm")
    s.add_argument("--nbest", required=True)
    s.add_argument("--lm", default="NbAiLab/nb-llama-3.2-1B")
    s.add_argument("--batch", type=int, default=64)
    s = sub.add_parser("tune")
    s.add_argument("--nbest", required=True)
    s.add_argument("--alphas", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    s.add_argument("--betas", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    s = sub.add_parser("apply")
    s.add_argument("--nbest", required=True)
    s.add_argument("--alpha", type=float, required=True)
    s.add_argument("--beta", type=float, required=True)
    a = p.parse_args()
    {"lm": cmd_lm, "tune": cmd_tune, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
