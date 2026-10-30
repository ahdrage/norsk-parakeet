"""TDT beam search with n-best output, for language-model rescoring.

transformers' ParakeetForTDT decodes greedily only. This keeps `beam` hypotheses per utterance,
each at its own encoder frame: a hypothesis is extended by the top tokens or by blank, each with
every duration, identical (tokens, frame) pairs are merged, and the best `beam` survive. A
hypothesis is finished when its frame reaches the end of the audio.

    python -m norsk_parakeet.beam --model runs/full/final --eval-dir data/dev --beam 8 --out results/beam8_dev
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from norsk_asr.data import SR
from norsk_asr.evaluate import load_eval_sets, score
from norsk_asr.text import normalize

from .evaluate import load


def _words_edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


class Beam:
    def __init__(self, model, beam=8, max_symbols=10):
        cfg = model.config
        self.model, self.beam, self.max_symbols = model, beam, max_symbols
        self.vocab, self.blank = cfg.vocab_size, cfg.blank_token_id
        self.durations = list(cfg.durations)
        dec = model.decoder
        self.emb, self.lstm, self.proj = dec.embedding, dec.lstm, dec.decoder_projector

    def _step(self, tokens, state):
        # One LSTM step for a batch of hypotheses; tokens [N], state (h, c) each [L, N, H].
        out, state = self.lstm(self.emb(tokens)[:, None, :], state)
        return self.proj(out[:, 0]), state

    @torch.no_grad()
    def search(self, enc: torch.Tensor, length: int) -> list[tuple[list[int], float]]:
        """enc [T, H] (projected encoder output) -> n-best [(tokens, log-prob)], best first."""
        dev = enc.device
        start = torch.tensor([self.blank], device=dev)
        dec0, st0 = self._step(start, None)
        # hypothesis: tokens tuple, frame, score, decoder output, (h, c), symbols emitted at this frame
        hyps = [((), 0, 0.0, dec0[0], (st0[0][:, 0], st0[1][:, 0]), 0)]
        d_blank = [i for i, d in enumerate(self.durations) if d > 0]
        while any(h[1] < length for h in hyps):
            live = [h for h in hyps if h[1] < length]
            done = [h for h in hyps if h[1] >= length]
            dec = torch.stack([h[3] for h in live])
            frames = torch.tensor([h[1] for h in live], device=dev)
            logits = self.model.joint(decoder_hidden_states=dec, encoder_hidden_states=enc[frames]).float()
            tok_lp = torch.log_softmax(logits[:, :self.vocab], -1)
            dur_lp = torch.log_softmax(logits[:, self.vocab:], -1)
            # Blank with duration 0 would not advance. Greedy decoding treats it as 1, but adding its
            # mass to duration 1 here favours blank over tokens and drops words; it is left out.
            dur_blank = dur_lp
            top_lp, top_tok = tok_lp[:, :self.blank].topk(self.beam, -1)
            cands = {}

            def add(key, score, parent, token, frame, nsym):
                old = cands.get(key)
                if old is None:
                    cands[key] = [score, parent, token, frame, nsym]
                else:  # same tokens at the same frame: merge paths
                    old[0] = float(np.logaddexp(old[0], score))

            tl, tt, bl, db, dl = (x.tolist() for x in (top_lp, top_tok, tok_lp[:, self.blank], dur_blank, dur_lp))
            for n, h in enumerate(live):
                toks, t, s = h[0], h[1], h[2]
                for i in d_blank:
                    d = self.durations[i]
                    add((toks, min(t + d, length)), s + bl[n] + db[n][i], n, None, min(t + d, length), 0)
                for k in range(self.beam):
                    for i, d in enumerate(self.durations):
                        if d == 0 and h[5] + 1 >= self.max_symbols:
                            continue
                        nt = toks + (tt[n][k],)
                        add((nt, min(t + d, length)), s + tl[n][k] + dl[n][i], n, tt[n][k], min(t + d, length),
                            h[5] + 1 if d == 0 else 0)
            for h in done:
                add((h[0], h[1]), h[2], None, h, h[1], 0)
            best = sorted(cands.items(), key=lambda kv: -kv[1][0])[:self.beam]
            # Run the decoder once for every surviving hypothesis that emitted a token.
            need = [(key, v) for key, v in best if v[1] is not None and v[2] is not None]
            if need:
                idx = [v[1] for _, v in need]
                h0 = torch.stack([live[i][4][0] for i in idx], 1)
                c0 = torch.stack([live[i][4][1] for i in idx], 1)
                out, (hn, cn) = self._step(torch.tensor([v[2] for _, v in need], device=dev), (h0, c0))
                new_state = {key: (out[j], (hn[:, j], cn[:, j])) for j, (key, _) in enumerate(need)}
            hyps = []
            for key, (score, parent, token, frame, nsym) in best:
                if parent is None:  # a finished hypothesis carried over
                    h = token
                    hyps.append((h[0], h[1], score, h[3], h[4], 0))
                elif token is None:  # blank: same decoder state
                    p = live[parent]
                    hyps.append((p[0], frame, score, p[3], p[4], 0))
                else:
                    o, st = new_state[key]
                    hyps.append((key[0], frame, score, o, st, nsym))
        return [(list(h[0]), h[2]) for h in sorted(hyps, key=lambda h: -h[2])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/full/final")
    p.add_argument("--eval-dir", default="data/dev")
    p.add_argument("--sets", nargs="*")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--beam", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--mem-fraction", type=float, default=0.6)
    a = p.parse_args()
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    processor, model = load(a.model, device)
    tok = processor.tokenizer
    search = Beam(model, a.beam)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, ds in load_eval_sets(Path(a.eval_dir), a.sets, a.limit).items():
        t0, rows = time.time(), []
        for i in range(len(ds)):
            wav = ds.audio(i).astype(np.float32) / 32768.0
            x = processor(wav, sampling_rate=SR, return_tensors="pt").to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                enc = model.get_audio_features(input_features=x["input_features"], attention_mask=x["attention_mask"])
                nbest = search.search(enc.pooler_output[0].float(), int(enc.attention_mask[0].sum()))
            texts = [tok.decode(t, skip_special_tokens=True) for t, _ in nbest]
            rows.append({"id": ds.rows[i]["id"], "ref": ds.rows[i]["text"], "has_digits": ds.rows[i].get("has_digits", False),
                         "nbest": [{"text": t, "am": s, "n_tokens": len(k)} for t, (k, s) in zip(texts, nbest)]})
        wall = time.time() - t0
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        refs, digits = [r["ref"] for r in rows], [r["has_digits"] for r in rows]
        top1 = score(refs, [normalize(r["nbest"][0]["text"]) for r in rows], digits)
        oracle_hyps = [min((normalize(c["text"]) for c in r["nbest"]),
                           key=lambda h: _words_edits(r["ref"].split(), h.split())) for r in rows]
        oracle = score(refs, oracle_hyps, digits)
        key = "wer_nodigit" if "wer_nodigit" in top1 else "wer"
        summary[name] = {"utts": len(rows), "top1": top1[key], "oracle": oracle[key], "speed_x": ds.hours() * 3600 / wall}
        print(name, json.dumps(summary[name]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
