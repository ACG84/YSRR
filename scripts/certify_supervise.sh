#!/usr/bin/env bash
# Keep the six certification runs alive across container reboots.
#
# The container reboots roughly hourly and takes every process with it, so
# something has to notice and restart the runs. Restarting is cheap: each seed
# checkpoints its features AND its magnetisation every 25 frames, so a relaunch
# resumes rather than replays and costs at most 25 frames.
#
# Liveness is judged per seed by a RECORDED PID plus a /proc identity check, not
# by pgrep and not by checkpoint age. Both of the obvious alternatives have now
# failed in this project:
#
#   pgrep -f <pattern>   matches the shell running the pgrep whenever that
#                        shell's own command line contains the pattern. It has
#                        killed four runs here (exit 144) and, later, silently
#                        convinced a restart hook that a dead supervisor was
#                        alive. magnum.np also renames its processes to
#                        "magnumnp", so a pattern on the script name matches
#                        nothing once a run is past startup.
#
#   checkpoint mtime     a freshly relaunched seed writes nothing for ~410 s
#                        (relaxation, then 25 frames), so with a stall threshold
#                        below that the supervisor relaunches the seed it just
#                        started, once per poll. That produced ten processes for
#                        six seeds, two of them writing the same checkpoint
#                        file. The checkpoints survived; that was luck.
#
# So: a seed is alive if its recorded PID exists and /proc/PID/comm says it is
# ours. PID reuse after a reboot is caught by the comm check. Checkpoint age is
# still used, but only as a HANG detector, with a threshold far above the
# relaunch cost and applied only to processes that are alive.
set -u
cd "$(dirname "$0")/.." || exit 1
SEEDS="${SEEDS:-0 1 2 3 4 5}"
FRAMES="${FRAMES:-1200}"
HANG="${HANG:-1200}"     # alive but no checkpoint this long => wedged, restart
POLL="${POLL:-120}"
# magnum.np renames its process, and /proc/PID/comm is capped at 15 characters,
# so the value is the TRUNCATION "magnumnp script" rather than "magnumnp" or the
# full title. An equality test against either of those fails for every live
# process, which would make alive() always false and spawn a duplicate per poll
# -- the exact failure this rewrite exists to remove. Match the prefix.
COMM_PREFIX="magnumnp"

mkdir -p runs/certify
echo $$ > runs/certify/supervisor.pid

alive () {               # alive <pidfile>
    local pf="$1" pid
    [ -f "$pf" ] || return 1
    pid=$(cat "$pf" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    # guard against a recycled PID pointing at something else entirely
    case "$(cat /proc/$pid/comm 2>/dev/null)" in
        "$COMM_PREFIX"*) return 0 ;;
        *)               return 1 ;;
    esac
}

launch () {              # launch <seed> <dir>
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python \
        scripts/run_narma_modal.py --frames "$FRAMES" \
        --splits 150 600 150 --drive port --seed "$1" --outdir "$2" \
        >> "$2/run.log" 2>&1 &
    echo $! > "$2/pid"
    echo "$(date -Is) seed $1 launched pid $!"
}

while :; do
    pending=0
    for s in $SEEDS; do
        d="runs/certify/seed_$s"
        mkdir -p "$d"
        grep -q "frame ${FRAMES}/${FRAMES}" "$d/run.log" 2>/dev/null && continue
        pending=1

        if alive "$d/pid"; then
            ck="$d/features_multitone.pt"
            [ -f "$ck" ] || continue                  # still relaxing; let it
            age=$(( $(date +%s) - $(stat -c %Y "$ck") ))
            [ "$age" -le "$HANG" ] && continue        # healthy
            echo "$(date -Is) seed $s wedged (${age}s since checkpoint) -- restarting"
            kill "$(cat "$d/pid")" 2>/dev/null
            sleep 2
        fi
        launch "$s" "$d"
        sleep 5                                       # stagger demag builds
    done
    [ "$pending" -eq 0 ] && { echo "$(date -Is) all seeds complete"; break; }
    sleep "$POLL"
done
rm -f runs/certify/supervisor.pid
