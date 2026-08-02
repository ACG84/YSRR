#!/usr/bin/env bash
# Keep the damping-screen runs alive across container reboots.
#
# Same job as certify_supervise.sh and the same hard-won liveness rules, but
# driven by a list of (name, alpha) rather than seeds. See that script's header
# for why liveness is a recorded PID plus an argv[0] check and never pgrep,
# never a process-name match, and never checkpoint age.
set -u
cd "$(dirname "$0")/.." || exit 1
FRAMES="${FRAMES:-800}"
SPLITS="${SPLITS:-200 400 100}"
SEED="${SEED:-0}"
ROOT="${ROOT:-runs/screen}"
POLL="${POLL:-120}"
HANG="${HANG:-1200}"

# alpha values to screen. 0.008 is NOT here: the certification already measured
# it at 1200 frames and that run is the control, so re-simulating it would cost
# an hour to learn something already known to 2%.
ALPHAS="${ALPHAS:-0.006 0.004 0.002}"

mkdir -p "$ROOT"
echo $$ > "$ROOT/supervisor.pid"

ident_ok () {
    local cmd a0 head
    cmd=$(tr '\0' '\n' < "/proc/$1/cmdline" 2>/dev/null) || return 1
    a0=${cmd%%$'\n'*}
    head=${a0%% *}
    case "${head##*/}" in
        magnumnp*) return 0 ;;
        python*)   case "$cmd" in *run_narma_modal.py*) return 0 ;; esac ;;
    esac
    return 1
}

alive () {
    local pf="$1" pid
    [ -f "$pf" ] || return 1
    pid=$(cat "$pf" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    ident_ok "$pid"
}

launch () {              # launch <alpha> <dir>
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python \
        scripts/run_narma_modal.py --frames "$FRAMES" \
        --splits $SPLITS --drive port --seed "$SEED" \
        --alpha "$1" --outdir "$2" \
        >> "$2/run.log" 2>&1 &
    echo $! > "$2/pid"
    echo "$(date -Is) alpha $1 launched pid $!"
}

while :; do
    pending=0
    for a in $ALPHAS; do
        d="$ROOT/alpha_${a}"
        mkdir -p "$d"
        grep -q "frame ${FRAMES}/${FRAMES}" "$d/run.log" 2>/dev/null && continue
        pending=1
        if alive "$d/pid"; then
            ck="$d/features_multitone.pt"
            [ -f "$ck" ] || continue
            age=$(( $(date +%s) - $(stat -c %Y "$ck") ))
            [ "$age" -le "$HANG" ] && continue
            echo "$(date -Is) alpha $a wedged (${age}s) -- restarting"
            kill "$(cat "$d/pid")" 2>/dev/null
            sleep 2
        fi
        launch "$a" "$d"
        sleep 5
    done
    [ "$pending" -eq 0 ] && { echo "$(date -Is) all screen runs complete"; break; }
    sleep "$POLL"
done
rm -f "$ROOT/supervisor.pid"
