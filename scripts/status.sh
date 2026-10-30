#!/usr/bin/env bash
# Where the full run is: services, last training line, evaluations so far.
cd "$(dirname "$0")/.."
for u in parakeet-pack parakeet-train parakeet-watchdog.timer; do printf '%-26s %s\n' "$u" "$(systemctl --user is-active $u)"; done
[ -f pack.log ] && echo "packed shards: $(grep -c '"shard"' pack.log) / 1304"
f=runs/full/log.jsonl
[ -f "$f" ] || exit 0
grep '"step"' "$f" | grep -v '"eval"' | tail -n 1
grep '"eval"' "$f" | python3 -c "$(cat <<'PY'
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    wer = {k: 100 * v.get("wer_nodigit", v["wer"]) for k, v in r["eval"].items()}
    print("WER at step", r["step"], " ".join(f"{k}={v:.1f}" for k, v in wer.items()))
PY
)"
[ -f runs/full/done ] && echo "finished $(cat runs/full/done)"
exit 0
