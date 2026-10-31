# Progress log

## 28 Oct 2026: zero-shot, pilot, full run started

Hardware: one NVIDIA RTX PRO 4000 Blackwell SFF (24 GB, 70 W), shared with a production
embedding service, so training is capped at 60% of GPU memory.

- **Zero-shot** (1,000-utterance subsets, digit-free WER): FLEURS 64.3, NPSC 71.8, NST 67.1,
  Stortinget 74.7. CER 19–28%. The errors are Swedish spellings of Norwegian words.
- **transformers works for training.** `ParakeetForTDT` (transformers 5.21) has a TDT loss, and
  the processor builds labels and decoder inputs from text. parakeet-ultra ships no processor or
  generation config; the base model's (`nvidia/parakeet-tdt-0.6b-v3`) are used. Only Moondream's
  extra `vad_head` is unused when loading.
- **Out of memory with the stock loss** at any useful batch: it keeps float32 copies of the full
  joint output. Replaced by a chunked, checkpointed joint that keeps only blank, label and
  duration log-probs (same loss to 7 digits on a test batch). Peak 13 GB at 80 s of audio per batch.
- **Loss of ~3,000 in training mode** on a 3-utterance batch, ~5 in eval mode: BatchNorm in the
  conformer convolutions uses batch statistics. Keeping BatchNorm in eval mode during training
  fixes it; layer-drop (0.1 in the config) is also off.
- **Pilot:** 155 h (30 Stortinget, 15 NST shards, all NPSC train), 80 s batches, LR 5e-5. Mid-run
  WER on 300-utterance subsets, FLEURS / NPSC / NST / Stortinget:
  - step 500: 25.2 / 27.4 / 24.3 / 35.3
  - step 1,000: 19.8 / 21.3 / 19.3 / 30.4
  - step 1,500: 18.1 / 19.2 / 17.9 / 28.5

  The run then crashed out of memory when the embedding service's cache grew, before its first
  checkpoint. Since then: periodic checkpoints, and a batch that runs out of memory is skipped.
- **Loss spikes** (78 at step 475, 1,008 in a later run) come from NPSC clips whose transcripts
  cannot fit their audio, such as a full sentence on a 1 s clip (238 characters per second). Clips
  outside 2–30 characters per second are now dropped (~0.1%).
- **Stortinget lags** the other sets, CER ~13% against ~5%. Its test clips average 26 s, and the
  pilot trained only on clips up to 20 s, which dropped 16% of the training audio. The full run
  uses clips up to 30 s.
- **Pilot speed** (clips up to 20 s): 0.024 audio-h/s with autograd through the TDT recursion,
  0.028 with the alpha–beta backward.
- **Full run started** 28 Oct, one epoch over 1,679 h (373,481 clips; 62 dropped as implausible),
  64 s batches, clips up to 30 s, LR 5e-5 with 500 warmup steps, evaluation every 5,000 steps.
  110,608 steps.
- **Long clips made the loss the bottleneck:** 0.0135 audio-h/s, 34 hours for the epoch, with the
  GPU at 41% and one CPU core saturated. The Python loop over anti-diagonals grows with clip
  length, and 30 s clips need ~500 diagonals per pass (2.2 s per batch). Triton kernels for both
  recursions take 1.8 ms; the run restarted with them at 23:12 and trains at 0.026 audio-h/s with
  the GPU at 100%, about 18 hours. The loss curve repeats the first start's (step 600: 0.755 vs 0.744).
- **Open files:** the service died at start with "Too many open files" (1,304 memory-mapped shards
  against systemd's default limit of 1,024); `LimitNOFILE=65536` in the unit.

## 29 Oct 2026: overnight

- Mid-run WER (300-utterance subsets), FLEURS / NPSC / NST / Stortinget:
  - step 5,000: 13.3 / 18.0 / 11.1 / 19.1
  - step 10,000: 11.5 / 14.4 / 9.3 / 16.5
  - step 15,000: 11.4 / 14.0 / 8.2 / 15.3
  - step 20,000: 11.0 / 12.7 / 7.3 / 14.4
  - step 25,000: 10.5 / 11.7 / 7.1 / 13.7
  - step 30,000: 9.8 / 11.2 / 6.5 / 13.4
  - step 35,000: 10.3 / 10.6 / 6.5 / 13.0
- **Loss spikes despite the speech-rate filter:** 27 at step 5,700 and 84 at step 35,000 (only
  every 100th batch is logged, so roughly 0.5% of batches). Clipping bounds each one, but each
  still moves the model a full step in a bad direction. Since step 37,514 (restart from the
  05:13 checkpoint) utterances above 15 loss per token are left out of the batch mean
  (`--max-token-loss`); normal ones are ~0.2–1 at this point. The count is logged as `outliers`.

## 29 Oct 2026: full run finished

- Mid-run WER (300-utterance subsets), FLEURS / NPSC / NST / Stortinget, continued:
  - step 40,000: 9.0 / 10.5 / 5.9 / 12.6
  - step 50,000: 9.1 / 9.9 / 5.4 / 12.0
  - step 60,000: 8.8 / 8.5 / 5.2 / 11.8
  - step 70,000: 8.6 / 7.9 / 5.1 / 11.4
  - step 80,000: 8.3 / 8.2 / 4.5 / 11.2
  - step 90,000: 8.0 / 7.6 / 4.3 / 11.1
  - step 100,000: 8.0 / 7.4 / 4.2 / 10.9
  - step 105,000: 8.1 / 7.2 / 4.2 / 11.0
- No utterance was left out by the loss cap after the 05:13 restart, and no batch ran out of
  memory. The run finished at 17:28, 110,608 steps.
- **Final** (1,000-utterance subsets as norsk-asr; all of FLEURS and Stortinget validation),
  greedy: FLEURS 8.1, NPSC 7.6, NST 4.6, Stortinget 11.2 WER; CER 3.5 / 2.8 / 1.3 / 7.6.
  Better than norsk-asr with its 4-gram LM on all four, and than NB-Whisper large on NPSC.
  The curve was flat from step 90,000, so more of the same data is unlikely to help much.

## 29 Oct 2026: language-model rescoring (not worth it)

- `norsk_parakeet.beam`: TDT beam search with n-best output (beam 1 matches greedy on 17 of 20
  clips; the rest differ in blank-versus-token ties). `norsk_parakeet.rescore`: rescoring with a
  causal LM, weights tuned on dev sets (FLEURS dev, NPSC eval split).
- Beam 8 on dev, WER: FLEURS 8.0, NPSC 8.3; **oracle** (best of the 8 candidates) 7.8 and 7.6. Even
  a perfect rescorer gains at most 0.2–0.7 points: the right transcript is rarely in the beam.
- Rescoring with NbAiLab/nb-llama-3.2-1B (best alpha 0.05, beta 0.5): 7.9 and 8.2, a 0.1 gain.
- Where the errors are (final test outputs): on Stortinget, mostly inserted and deleted small
  words (og, også, både, altså, jo, men) and formal-for-spoken swaps (slik → sånn). The references
  are edited proceedings, the model writes what was said. NPSC: dropped function words and
  dialect forms (blei → ble). FLEURS: names (Giza, Casablanca, Sacks), compounds and reference
  spellings ("i følge"). None of this is what a language model fixes.

## 30 Oct 2026: stage 2 (v2)

- v1 continued for one epoch on additional Norwegian speech mixed with part of the v1 data, about
  2,000 h of training data in total. The new recordings were cut into clips at sentence
  timestamps; clips where v1's output disagreed strongly with the transcript (audio shifted
  against its timestamps) were left out.
- Final WER, FLEURS / NPSC / NST / Stortinget: v1 8.1 / 7.6 / 4.6 / 11.2, v2 6.7 / 7.1 / 4.4 / 11.0.
  On a held-out test set of the new kind of speech, WER against its transcripts fell from 18.2
  to 10.0 (excluding two clips, below).
- **Regression:** v2 sometimes keeps writing after the speech ends, inventing words and then
  repeating a phrase (2,132 words for a 70-word clip). On 3,571 held-out clips, output far longer
  than the transcript: v1 9, v2 33.
- **Cause:** a share of the new clips have reference words in their first or last three that v1
  does not hear (the cut is too tight, or the timestamps are slightly off), so the model learns
  to write words with no audio under them. Next run (v3): cut in the middle of the pause between
  sentences, leave out clips whose edges v1 does not hear or whose CER is above 30%, start again
  from v1, and track the runaway count (v1: 9 of 3,571) as its own metric.

## 31 Oct 2026: v3

- Trained from v1, not v2, on new clips mixed with part of the v1 data, about 2,400 h of training
  data in total: 79,323 steps at LR 3e-5, with clips filtered as planned above.
- Final WER, FLEURS / NPSC / NST / Stortinget: v2 6.7 / 7.1 / 4.4 / 11.0, v3 6.2 / 7.1 / 4.2 / 10.9.
  On the aligned part of the held-out test set (2,126 clips, none excluded), WER against its
  transcripts: v1 18.1, v2 16.9, v3 9.2.
- **Runaway count, v1 / v2 / v3:** 9 / 33 / 19 of 3,571 clips. The metric turned out to be a poor
  one: it compares output length with the transcript, so it also fires on recordings shifted
  against their timestamps, where the model correctly writes other words than the transcript.
  v3 has none on the aligned clips (v1 0, v2 4), and the 19 read as ordinary speech.
- Better measures on the same clips: repeated phrases (a three-word sequence five or more times)
  3 / 20 / 2, where one of v3's two is a speaker really repeating a word; fastest output 97 / 128 / 7
  words per second. The regression is gone.
