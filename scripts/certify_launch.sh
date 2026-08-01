#!/usr/bin/env bash
# Six independent NARMA-10 realisations through the ported disk, one process
# each. Everything but the input draw is held fixed -- this is the data the
# certification in docs/narma10_certification.md is computed from.
#
# Deliberately oversubscribed: six single-threaded jobs on four cores finish in
# 6/4 of one job's time, where a strict pool of four would leave two cores idle
# during the second wave. Threading is worthless here anyway (26.1 ms/step at
# one thread, 27.6 at four -- the mesh is small enough to be launch-bound).
#
# Idempotent. Each seed checkpoints its features AND its magnetisation every 25
# frames, so re-running after a container reboot resumes rather than replays.
# Just run it again.
set -u
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0 1 2 3 4 5}"
FRAMES="${FRAMES:-1200}"

for s in $SEEDS; do
    d="runs/certify/seed_$s"
    mkdir -p "$d"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python scripts/run_narma_modal.py \
        --frames "$FRAMES" --splits 150 600 150 \
        --drive port --seed "$s" --outdir "$d" \
        >> "$d/run.log" 2>&1 &
    echo "$!" > "$d/pid"
    echo "seed $s -> pid $(cat "$d/pid")"
done
wait
