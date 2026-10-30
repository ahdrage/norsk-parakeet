"""Training clips from long recordings: audio from the episode URL, transcripts with sentence timestamps.

An episode's transcript segments (sentences with start and end times) are merged into clips of up to
`max_s` seconds; a clip ends where the next segment would make it too long, at a speaker change,
or at a pause of more than `max_gap` seconds, so cuts fall between sentences.
"""

import io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import av
import numpy as np

SR = 16000


def _get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _hls_bytes(url: str, timeout: float) -> bytes:
    # HLS: a master playlist of audio variants, each a VOD list of MPEG-TS segments. The
    # lowest bitrate is plenty for 16 kHz speech. Segments are concatenated into one TS stream.
    lines = _get(url, timeout).decode().splitlines()
    uris = [l for l in lines if l and not l.startswith("#")]
    if any("#EXT-X-STREAM-INF" in l for l in lines):
        return _hls_bytes(urljoin(url, uris[0]), timeout)
    with ThreadPoolExecutor(8) as ex:
        return b"".join(ex.map(lambda u: _get(urljoin(url, u), timeout), uris))


def fetch_audio(url: str, timeout: float = 60) -> np.ndarray:
    """Download and decode (mp3, m4a, or HLS) to 16 kHz mono float32."""
    data = _hls_bytes(url, timeout) if ".m3u8" in url.split("?")[0] else _get(url, timeout)
    out = []
    with av.open(io.BytesIO(data)) as c:
        stream = c.streams.audio[0]
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for packet in c.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                # A corrupt packet (common in mp3 files): put silence of its length in its place
                # so the rest of the episode stays aligned with the transcript's timestamps.
                if packet.duration and packet.time_base:
                    out.append(np.zeros(int(packet.duration * packet.time_base * SR), np.float32))
                continue
            for frame in frames:
                for f in res.resample(frame):
                    out.append(f.to_ndarray()[0])
        for f in res.resample(None):
            out.append(f.to_ndarray()[0])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def make_clips(segments: list[dict], max_s: float = 25.0, max_gap: float = 1.5, min_s: float = 2.0,
               pad: float = 0.5) -> list[dict]:
    clips, cur = [], None
    for s in sorted(segments, key=lambda s: s["startTime"]):
        start, end, text = float(s["startTime"]), float(s["endTime"]), s["text"].strip()
        if not text or end <= start:
            continue
        if end - start > max_s:  # one long segment: usually music or a timing error
            if cur:
                clips.append(cur)
            cur = None
            continue
        if cur and (end - cur["start"] > max_s or start - cur["end"] > max_gap or s.get("speaker") != cur["speaker"]
                    or s.get("language") != cur["language"]):
            clips.append(cur)
            cur = None
        if cur is None:
            cur = {"start": start, "end": end, "text": text, "speaker": s.get("speaker"), "language": s.get("language")}
        else:
            cur["end"], cur["text"] = end, cur["text"] + " " + text
    if cur:
        clips.append(cur)
    clips = [c for c in clips if c["end"] - c["start"] >= min_s and c["language"] in (None, "no")]
    # Cut in the middle of the pause to the neighbouring speech, at most `pad` from this clip's
    # own speech: timestamps can be a little off, and a cut that clips a word teaches the model
    # to write words it cannot hear.
    times = sorted((float(s["startTime"]), float(s["endTime"])) for s in segments if s["text"].strip())
    for c in clips:
        prev_end = max((e for s, e in times if e <= c["start"]), default=None)
        next_start = min((s for s, e in times if s >= c["end"]), default=None)
        c["cut_start"] = c["start"] - pad if prev_end is None else max(c["start"] - pad, (prev_end + c["start"]) / 2)
        c["cut_end"] = c["end"] + pad if next_start is None else min(c["end"] + pad, (c["end"] + next_start) / 2)
    return clips


def cut(wav: np.ndarray, clip: dict) -> np.ndarray:
    a = max(0, int(clip.get("cut_start", clip["start"] - 0.2) * SR))
    b = min(len(wav), int(clip.get("cut_end", clip["end"] + 0.2) * SR))
    return wav[a:b]


# --- Packing into norsk-asr's shard format -------------------------------------------------------
#
#     python -m norsk_parakeet.broadcast pack --episodes data/bc/episodes.raw --out data/bc/train \
#         --test-out data/eval/broadcast_test --max-hours 450
#
# Episodes come from a JSON-lines export (one object per line: id, prov, show, dur, url, segs).
# A fixed set of episodes, one per show from different shows, is held out as a test set; the rest
# are downloaded in parallel, cut into clips and written to ~1 h shards. Restartable: finished
# episodes are listed in <out>/done.txt and skipped.

import argparse
import hashlib
import json
import signal
import time
from multiprocessing import Pool
from pathlib import Path

from norsk_asr.text import normalize


def _episode_clips(ep: dict):
    def alarm(*_):
        raise TimeoutError("episode took too long")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(900)
    try:
        for attempt in range(3):
            try:
                wav = fetch_audio(ep["url"], timeout=30)
                break
            except TimeoutError:
                raise
            except Exception as e:  # network hiccups: retry, then give up on the episode
                if attempt == 2:
                    return ep["id"], None, f"{type(e).__name__}: {e}"[:200]
                time.sleep(5)
        rows = []
        for c in make_clips(ep["segs"]):
            text = normalize(c["text"])
            if not text:
                continue
            pcm = np.clip(cut(wav, c) * 32768, -32768, 32767).astype(np.int16)
            if len(pcm) < SR or not 2 <= len(c["text"]) / (len(pcm) / SR) <= 30:  # past the audio's end, or misaligned
                continue
            rows.append(({"id": f"{ep['id']}_{c['start']:.2f}", "text": text, "orig": c["text"],
                          "has_digits": any(ch.isdigit() for ch in c["text"]), "episode": ep["id"],
                          "show": ep["show"], "prov": ep["prov"]}, pcm))
        return ep["id"], rows, None
    except Exception as e:
        return ep["id"], None, f"{type(e).__name__}: {e}"[:200]
    finally:
        signal.alarm(0)


class _ShardWriter:
    def __init__(self, out: Path, prefix: str, shard_hours: float = 1.0):
        self.out, self.prefix, self.limit = out, prefix, shard_hours * 3600 * SR
        self.n = len(list(out.glob(f"{prefix}_*.jsonl")))
        self.pcm, self.meta, self.offset = None, None, 0

    def add(self, row: dict, pcm: np.ndarray):
        if self.pcm is None:
            name = f"{self.prefix}_{self.n:04d}"
            self.pcm, self.meta = open(self.out / f"{name}.i16.part", "wb"), open(self.out / f"{name}.jsonl.part", "w")
            self.offset = 0
        self.pcm.write(pcm.tobytes())
        self.meta.write(json.dumps({**row, "offset": self.offset, "n": len(pcm)}, ensure_ascii=False) + "\n")
        self.offset += len(pcm)

    def maybe_close(self, force=False):
        # Only between episodes, so done.txt and the finished shards always agree.
        if self.pcm is not None and (force or self.offset >= self.limit):
            name = f"{self.prefix}_{self.n:04d}"
            self.pcm.close(), self.meta.close()
            (self.out / f"{name}.i16.part").rename(self.out / f"{name}.i16")
            (self.out / f"{name}.jsonl.part").rename(self.out / f"{name}.jsonl")
            self.pcm, self.n = None, self.n + 1
            return True
        return False


def _edge_misses(ref: str, hyp: str, k: int = 3) -> tuple[int, int]:
    """How many of the reference's first and last k words the model's output leaves out."""
    import jiwer

    miss = [False] * len(ref.split())
    for c in jiwer.process_words(ref, hyp or "-").alignments[0]:
        if c.type == "delete":
            for j in range(c.ref_start_idx, c.ref_end_idx):
                miss[j] = True
    return sum(miss[:k]), sum(miss[-k:])


class _Checker:
    """Transcribes an episode's clips with a reference model and keeps those whose transcript fits:
    CER at most max_cer, and no more than one of the first or last three words unheard. Clips
    whose text runs past their audio teach the model to write words it cannot hear."""

    def __init__(self, model_id: str, max_cer: float):
        import torch

        from .evaluate import load

        self.torch, self.dev, self.max_cer = torch, torch.device("cuda"), max_cer
        torch.cuda.set_per_process_memory_fraction(0.45)
        self.processor, self.model = load(model_id, self.dev)

    def __call__(self, rows):
        import jiwer

        from .evaluate import transcribe

        hyps, batch, secs = [], [], 0.0
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][1]))
        out = {}
        for i in order + [None]:
            if i is None or (batch and (len(batch) + 1) * len(rows[i][1]) / SR > 300):
                if batch:
                    out.update(zip(batch, transcribe(self.processor, self.model, [rows[j][1] for j in batch], self.dev)))
                batch = []
            if i is not None:
                batch.append(i)
        kept = []
        for i, (row, pcm) in enumerate(rows):
            hyp = normalize(out[i])
            cer = jiwer.cer(row["text"], hyp or "-")
            s, e = _edge_misses(row["text"], hyp)
            row = {**row, "cer": round(cer, 3), "edge_miss": [s, e]}
            if cer <= self.max_cer and s < 2 and e < 2:
                kept.append((row, pcm))
        return kept


def _test_episodes(eps: list[dict], n: int) -> set[str]:
    # One episode per show, from shows with at least 3 episodes, chosen by hash: deterministic.
    by_show = {}
    for e in eps:
        by_show.setdefault(e["show"], []).append(e)
    picks = []
    for show, es in by_show.items():
        if len(es) >= 3:
            es = sorted(es, key=lambda e: hashlib.md5(e["id"].encode()).hexdigest())
            ok = [e for e in es if 180 <= (e["dur"] or 0) <= 3600]
            if ok:
                picks.append(ok[0])
    picks.sort(key=lambda e: hashlib.md5(e["show"].encode()).hexdigest())
    return {e["id"] for e in picks[:n]}


def cmd_pack(a):
    eps = [json.loads(l) for l in open(a.episodes) if l.startswith("{")]
    if a.exclude_ids:  # e.g. an earlier run's test episodes
        excluded = set(Path(a.exclude_ids).read_text().split())
        eps = [e for e in eps if e["id"] not in excluded]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    test_ids = _test_episodes([e for e in eps if not a.test_prov or e["prov"] == a.test_prov], a.test_episodes)
    (out / "test_episodes.txt").write_text("\n".join(sorted(test_ids)) + "\n")
    done_file = out / "done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    for p in out.glob("*.part"):  # a shard cut off by a crash: its episodes are not in done.txt
        p.unlink()

    test_out = Path(a.test_out)
    if not test_out.with_suffix(".jsonl").exists():
        test_eps = [e for e in eps if e["id"] in test_ids]
        rows = []
        with Pool(a.workers) as pool:
            for eid, clips, err in pool.imap_unordered(_episode_clips, test_eps):
                if err:
                    print("test episode failed", eid, err, flush=True)
                rows += clips or []
        rows.sort(key=lambda r: r[0]["id"])
        tmp = test_out.parent / (test_out.name + ".tmp")
        with open(tmp.with_suffix(".i16"), "wb") as fp, open(tmp.with_suffix(".jsonl"), "w") as fm:
            off = 0
            for row, pcm in rows:
                fp.write(pcm.tobytes())
                fm.write(json.dumps({**row, "offset": off, "n": len(pcm)}, ensure_ascii=False) + "\n")
                off += len(pcm)
        tmp.with_suffix(".i16").rename(test_out.with_suffix(".i16"))
        tmp.with_suffix(".jsonl").rename(test_out.with_suffix(".jsonl"))
        print(f"TEST {len(rows)} clips, {sum(len(p) for _, p in rows) / SR / 3600:.1f} h from {len(test_ids)} episodes",
              flush=True)

    hours = sum(json.loads(l)["n"] for p in out.glob("bc_*.jsonl") for l in open(p)) / SR / 3600
    # Interleave sources and shows so a --max-hours cut still covers all of them.
    todo = [e for e in eps if e["id"] not in test_ids and e["id"] not in done]
    todo.sort(key=lambda e: hashlib.md5(e["id"].encode()).hexdigest())
    writer = _ShardWriter(out, "bc")
    pending, t0, failed, cut_h = [], time.time(), 0, 0.0
    with Pool(a.workers) as pool, open(done_file, "a") as fd:
        check = _Checker(a.filter_model, a.max_cer) if a.filter_model else None  # after forking the workers
        for eid, clips, err in pool.imap_unordered(_episode_clips, todo):
            if hours >= a.max_hours:
                break
            if err:
                failed += 1
                print("failed", eid, err, flush=True)
            clips = clips or []
            cut_h += sum(len(p) for _, p in clips) / SR / 3600
            if check and clips:
                clips = check(clips)
            for row, pcm in clips:
                writer.add(row, pcm)
                hours += len(pcm) / SR / 3600
            pending.append(eid)
            if writer.maybe_close():
                fd.write("\n".join(pending) + "\n")
                fd.flush()
                pending = []
                print(f"shard {writer.n - 1}: {hours:.1f} h kept of {cut_h:.1f} h cut, {failed} failed episodes, "
                      f"{time.time() - t0:.0f} s", flush=True)
        pool.terminate()
        if writer.maybe_close(force=True):
            fd.write("\n".join(pending) + "\n")
    print(f"TOTAL {hours:.1f} h of clips kept of {cut_h:.1f} h cut this run, {failed} failed episodes", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pack")
    s.add_argument("--episodes", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--test-out", required=True, help="path without suffix; .i16 and .jsonl are written")
    s.add_argument("--test-episodes", type=int, default=30)
    s.add_argument("--max-hours", type=float, default=450)
    s.add_argument("--exclude-ids", help="file of episode ids to leave out entirely")
    s.add_argument("--test-prov", help="pick test episodes only from this provider")
    s.add_argument("--filter-model", help="keep only clips whose transcript this model's output fits")
    s.add_argument("--max-cer", type=float, default=0.3)
    s.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    {"pack": cmd_pack}[a.cmd](a)


if __name__ == "__main__":
    main()
