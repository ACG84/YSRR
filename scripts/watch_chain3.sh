#!/usr/bin/env bash
# Keep the 3-stage NARMA run alive for one watch window.
#
# Liveness is counted by argv[0], never by a substring of the process table:
# this script's own command line contains the script name, so a naive
# `grep -c run_narma_chain.py` never returns 0 and the relaunch never fires.
# That exact mistake cost nine minutes of a dead run tonight, and it is the
# third variant of the same error in this project after pgrep self-match and
# argv matching defeated by magnum.np's process rename.
set -u
cd "$(dirname "$0")/.."
runners () {
    local n=0 p cmd a0
    for p in $(ps -eo pid --no-headers); do
        cmd=$(tr "\0" "\n" < "/proc/$p/cmdline" 2>/dev/null) || continue
        case "$cmd" in *run_narma_chain.py*) ;; *) continue ;; esac
        a0=${cmd%%$'\n'*}
        case "${a0%% *}" in *python*|*magnumnp*) n=$((n+1)) ;; esac
    done
    echo "$n"
}
for i in $(seq 1 "${CYCLES:-9}"); do
    if [ "$(runners)" -eq 0 ] &&
       ! grep -qs 'frame 600/600' runs/chain3_narma.log; then
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python \
            scripts/run_narma_chain.py --n-disks 3 --frames 600 \
            --splits 150 300 75 --seed 0 --outdir runs/chain3_narma \
            >> runs/chain3_narma.log 2>&1 &
        echo "$(date -Is) relaunched"
    fi
    sleep 55
done
