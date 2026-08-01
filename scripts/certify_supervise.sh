#!/usr/bin/env bash
# Keep the six certification runs alive across container reboots.
#
# Liveness is judged by PROGRESS, not by process identity. Two reasons: this
# container's reboots recycle PIDs, so a recorded pid can come back attached to
# something else entirely; and magnum.np renames its own process to "magnumnp",
# so pgrep on the script name finds nothing while pgrep on a looser pattern
# matches the supervisor's own shell -- both failure modes have already cost
# this project runs. A checkpoint whose mtime has not moved in STALL seconds is
# dead regardless of what the process table says.
#
# Relaunching is safe: each seed resumes from its last checkpoint, features and
# magnetisation together, so a restart costs at most 25 frames.
set -u
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2 3 4 5}"
FRAMES="${FRAMES:-1200}"
STALL="${STALL:-420}"
POLL="${POLL:-120}"

frames_done () {          # frames in the checkpoint, 0 if there is none yet
    python - "$1" <<'PY' 2>/dev/null || echo 0
import sys, torch
try:
    o = torch.load(sys.argv[1], weights_only=False, map_location="cpu")
    print(len(o["feats"] if isinstance(o, dict) else o))
except Exception:
    print(0)
PY
}

while :; do
    alive=0
    for s in $SEEDS; do
        d="runs/certify/seed_$s"; ck="$d/features_multitone.pt"
        n=$(frames_done "$ck")
        [ "$n" -ge "$FRAMES" ] && continue
        alive=1
        # age of the most recent write; a fresh launch has none, so treat a
        # missing checkpoint as infinitely stale and let the log guard below
        # decide
        if [ -f "$ck" ]; then age=$(( $(date +%s) - $(stat -c %Y "$ck") ));
        else age=$(( $(date +%s) - $(stat -c %Y "$d/run.log" 2>/dev/null \
                                     || echo 0) )); fi
        if [ "$age" -gt "$STALL" ]; then
            echo "$(date -Is) seed $s stalled at $n/$FRAMES (${age}s) -- relaunching"
            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python \
                scripts/run_narma_modal.py --frames "$FRAMES" \
                --splits 150 600 150 --drive port --seed "$s" --outdir "$d" \
                >> "$d/run.log" 2>&1 &
            sleep 5      # stagger the demag-kernel builds
        fi
    done
    [ "$alive" -eq 0 ] && { echo "$(date -Is) all seeds complete"; break; }
    sleep "$POLL"
done
