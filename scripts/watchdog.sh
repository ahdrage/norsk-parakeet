#!/usr/bin/env bash
# Run every 10 minutes by parakeet-watchdog.timer. Restarts a training service that is running
# but has not logged anything for 40 minutes (a hang rather than a crash; crashes restart on
# their own), and appends a one-line status to watchdog.log.
cd "$(dirname "$0")/.."
check() {
  local unit=$1 log=$2 state last
  state=$(systemctl --user is-active "$unit")
  [ "$state" = inactive ] && [ ! -f "$log" ] && return
  last=$(tail -n 1 "$log" 2>/dev/null | cut -c1-200)
  if [ "$state" = active ] && [ -f "$log" ] && [ -n "$(find "$log" -mmin +40)" ]; then
    echo "$(date -Is) $unit: no progress for 40 min, restarting" >> watchdog.log
    systemctl --user restart "$unit"
  fi
  echo "$(date -Is) $unit $state | $last" >> watchdog.log
}
check parakeet-train runs/full/log.jsonl
check parakeet-bc runs/bc1/log.jsonl
check parakeet-v3 runs/v3/log.jsonl
