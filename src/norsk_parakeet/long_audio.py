"""Transcribe audio longer than ~30 s: cut at the quietest point in the last third of each chunk.

Whole-recording WER on 10 h of held-out audio: 11.0 with these cuts, 11.5 with fixed 30 s cuts, 14.0
with fixed 15 s cuts. Chunks much longer than the 25-30 s the model was trained on loop more often.
"""

import re

import numpy as np
import torch

SR = 16000


def split_at_pauses(wav: np.ndarray, max_s: float = 30.0) -> list[np.ndarray]:
    hop, n = SR // 20, int(max_s * SR)
    if len(wav) <= n:
        return [wav]
    frames = wav[:len(wav) // hop * hop].reshape(-1, hop).astype(np.float64)
    energy = np.convolve((frames**2).mean(1), np.ones(4) / 4, "same")  # over 200 ms
    chunks, a = [], 0
    while len(wav) - a > n:
        lo, hi = (a + n * 2 // 3) // hop, (a + n) // hop
        b = (lo + int(np.argmin(energy[lo:hi]))) * hop
        chunks.append(wav[a:b])
        a = b
    chunks.append(wav[a:])
    return chunks


def fix_number_spacing(text: str) -> str:
    """Put back the space before a number: "september2017" -> "september 2017". Words that end in a
    capital, such as CO2 and G20, keep their form."""
    return re.sub(r"(?<=[a-zæøåéèêóòôüäö])(?=\d)|(?<=[^\d\s][,;:])(?=\d)", " ", text)


@torch.no_grad()
def transcribe_long(processor, model, wav: np.ndarray, batch_size: int = 8) -> str:
    """wav: 16 kHz mono float32."""
    chunks = [c for c in split_at_pauses(wav) if len(c) >= SR // 5]
    out = []
    for i in range(0, len(chunks), batch_size):
        inputs = processor(chunks[i:i + batch_size], sampling_rate=SR, return_tensors="pt").to(model.device)
        out += processor.batch_decode(model.generate(**inputs).sequences, skip_special_tokens=True)
    return fix_number_spacing(" ".join(t.strip() for t in out))
