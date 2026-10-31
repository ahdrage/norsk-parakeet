# norsk-parakeet

Norwegian speech-to-text: [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra)
(a post-trained NVIDIA Parakeet TDT 0.6B v3) fine-tuned on open Norwegian data. It is a sibling of
[norsk-asr](https://github.com/ahdrage/norsk-asr) and uses the same training data, test sets, text
normalization and scoring, so the numbers compare one to one.

Parakeet v3 covers 25 European languages, including Danish and Swedish but not Norwegian. Its
tokenizer already has æ, ø, å and common Norwegian words, so no vocabulary change is needed, and
unlike norsk-asr it writes cased, punctuated text.

Weights: [ahdrage/norsk-parakeet](https://huggingface.co/ahdrage/norsk-parakeet) on Hugging Face
(CC-BY-4.0), trained on about 2,400 hours of Norwegian speech in two stages:

- **v1** (tag `v1`): one epoch over 1,679 h of open data, about 18 hours on one RTX PRO 4000
  Blackwell (24 GB).
- **v3** (current): v1 continued on additional Norwegian speech, about 2,400 h in total. At least
  as good as v2 on every test set, without [v2's runaway output](#runaway-output).
- **v2** (tag `v2`): an earlier continuation of v1, about 2,000 h in total. It sometimes keeps
  writing after the speech ends; use v3 instead.

## Results

WER (%) on digit-free utterances, scored with norsk-asr's `normalize()` (lowercase, no punctuation).

| Model | Training data | FLEURS | NPSC | NST | Stortinget |
|---|---|---|---|---|---|
| NB-Whisper large | ~66k h | **4.2** | 16.5 | **2.0** | **9.0** |
| norsk-asr `full_top1`, greedy | 1,680 h | 15.2 | 12.9 | 8.9 | 15.3 |
| norsk-asr `full_top1` + 4-gram LM | 1,680 h | 10.0 | 9.7 | 5.6 | 12.2 |
| parakeet-ultra, zero-shot | – | 64.3 | 71.8 | 67.1 | 74.7 |
| norsk-parakeet v1, greedy | 1,680 h | 8.1 | 7.6 | 4.6 | 11.2 |
| norsk-parakeet v2, greedy | ~2,000 h | 6.7 | 7.1 | 4.4 | 11.0 |
| **norsk-parakeet v3**, greedy | ~2,400 h | 6.2 | **7.1** | 4.2 | 10.9 |

All rows use the same test subsets: 1,000 utterances of NPSC and NST, all of FLEURS (357) and of
Stortinget validation (697). norsk-parakeet decodes greedily with no language model; v3's
character error rates are 2.1 / 2.7 / 1.1 / 7.2%.

It beats norsk-asr with its language model on all four sets, and NB-Whisper large on NPSC, the
set closest to spontaneous parliamentary speech, at 0.6B parameters against 1.55B. NB-Whisper,
trained on over 25 times as much audio, is still ahead on read speech (FLEURS, NST) and
Stortinget.

Zero-shot, the model hears Norwegian well but spells it as Swedish ("I följde sysselmanns
kontor", "för exempel"). A few hundred steps of fine-tuning fix most of that; the rest of the
gain comes slowly over the epoch (see [PROGRESS.md](PROGRESS.md) for the curve). Inference runs at
480–810× real time on this card.

On 2,126 held-out clips from recordings of the kind added after v1, scored against their
transcripts, WER is 18.1 for v1, 16.9 for v2 and 9.2 for v3.

### Runaway output

v2 sometimes keeps writing after the speech has ended, inventing plausible words and repeating a
short phrase many times. It came from training clips whose transcripts extend past the audio at
their edges, so v3 was retrained from v1 without such clips. On 3,571 held-out clips:

| | v1 | v2 | v3 |
|---|---|---|---|
| Clips repeating a phrase | 3 | 20 | **2** |
| Fastest output (words per second of audio) | 97 | 128 | **7** |

## Usage

```python
import torch
from transformers import AutoProcessor, ParakeetForTDT

processor = AutoProcessor.from_pretrained("ahdrage/norsk-parakeet")
model = ParakeetForTDT.from_pretrained("ahdrage/norsk-parakeet").eval()
# audio: 16 kHz mono float32 numpy array, up to ~30 s per clip
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs)
print(processor.batch_decode(out.sequences, skip_special_tokens=True)[0])
```

## How it is trained

- **Targets:** the original transcripts, cased and punctuated (NPSC's `<ee>`-style markers
  removed), since that is what Parakeet writes. Scoring normalizes both sides.
- **Data:** norsk-asr's packed shards (Stortinget, NST, NPSC Bokmål train), built with its
  `prep train` command: digit-free utterances with whisper WER ≤ 0.15. Clips over 30 s, and ~0.1%
  whose transcripts cannot fit their audio (over 30 or under 2 characters per second), are left out.
- **Everything trains:** all 0.6B parameters, AdamW, peak LR 5e-5, cosine decay, 64 s of audio per
  batch, bf16 autocast. BatchNorm keeps its pretrained statistics and layer-drop is off: with
  small batches, batch statistics blew the loss up from ~5 to ~3,000.
- **Memory-lean TDT loss** (`src/norsk_parakeet/loss.py`). transformers' TDT loss materializes
  the joint output (frames × tokens × 8,193) in float32 several times and runs out of memory on a
  24 GB card at useful batch sizes. The loss here computes the joint in time chunks under
  activation checkpointing and keeps only what the recursion needs: blank, next-label and duration
  log-probs. The backward pass uses alpha–beta posteriors instead of autograd through the
  recursion, and on the GPU both recursions are Triton kernels
  (`src/norsk_parakeet/tdt_triton.py`), one program per utterance walking all anti-diagonals.
  For a batch of two 30 s clips that is 1.8 ms against 2.2 s for the PyTorch loop, and training
  runs at 0.026 audio-h/s instead of 0.0135. `tests/test_loss.py` checks the gradients against
  autograd (equal to 1e-15) and the kernels against the PyTorch recursion.
- **Stage 2 (v2):** `norsk_parakeet.recordings` cuts long, timestamped recordings into clips of
  up to 25 s at sentence boundaries; `norsk_parakeet.filter` leaves out clips whose transcript
  disagrees strongly with v1's output; `scripts/run_v2.sh` runs packing, filtering and one epoch
  at LR 3e-5 from v1.
- **Stage 2 again (v3):** clips are cut in the middle of the pause between sentences, and clips
  with CER above 30% against v1's output, or whose first or last words v1 does not hear, are left
  out. `scripts/run_v3.sh` runs packing and one epoch at LR 3e-5 from v1 (79,323 steps).
- **Resumable:** a checkpoint (weights, optimizer, batch position) every 30 minutes; a restart
  continues with the same batch order.

## Training

```bash
python3 -m venv .venv
.venv/bin/pip install "torch>=2.14" --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -e .
# Data, with norsk-asr's prep (~190 GB for the full set):
.venv/bin/python -m norsk_asr.prep eval --out data/eval
.venv/bin/python -m norsk_asr.prep train --out data/train --stortinget-shards 10000 --npsc --workers 14
# Train one epoch; evaluates every 5,000 steps and on the full test sets at the end:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m norsk_parakeet.train --train-dir data/train --eval-dir data/eval --out runs/full --epochs 1
# Evaluate any checkpoint:
.venv/bin/python -m norsk_parakeet.evaluate --model runs/full/final --eval-dir data/eval --limit 1000 --out results/full
```

`scripts/run_full.sh` runs the full recipe as a systemd user service, with
`scripts/watchdog.sh` restarting it if it stops making progress and `scripts/status.sh` showing
where it is.

## Licenses

Code: Apache-2.0. The base model, parakeet-ultra, is CC-BY-4.0 (from NVIDIA's
parakeet-tdt-0.6b-v3, also CC-BY-4.0), so fine-tuned weights are too, with attribution to NVIDIA
and Moondream. The training data are the National Library of Norway's open releases of NPSC, NST
and Stortinget speech, as in norsk-asr.
