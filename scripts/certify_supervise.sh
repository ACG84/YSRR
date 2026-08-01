#!/usr/bin/env bash
# Keep the six certification runs alive across container reboots.
#
# The container reboots roughly hourly and takes every process with it, so
# something has to notice and restart the runs. Restarting is cheap: each seed
# checkpoints its features AND its magnetisation every CKPT_EVERY frames (5), so
# a relaunch resumes rather than replays and costs at most 5 frames.
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
#
# Matching the prefix is necessary and NOT sufficient, because the rename does
# not happen at exec. A run has two identities:
#
#   phase 1, first ~100 s   comm "python", cmdline "python .../run_narma_modal.py
#                           --seed N"   (imports, demag kernel, relaxation)
#   phase 2, thereafter     comm "magnumnp script", cmdline "magnumnp script"
#
# Testing only for magnumnp declares every seed dead for its first ~100 s, and
# POLL is 120 s, so a startup that runs even slightly long -- six of them do
# start simultaneously -- is seen as dead and launched a SECOND time. That is
# not a wasted process, it is data loss: both copies resume from the same
# checkpoint and then write to it, so the slower one's saves overwrite the
# faster one's and frames go BACKWARDS. Seed 0 reached frame 900 and was rolled
# back to 845 exactly this way.
#
# So accept either identity, and read it from cmdline, which covers both phases
# with one test. Note phase 2 destroys argv, so the --seed argument is NOT
# recoverable from a running process; per-seed identity comes from the pid file
# and nowhere else.
#
# The test is on argv[0], not on the whole command line. Grepping the whole line
# for "run_narma_modal" matches any shell that merely MENTIONS the script -- the
# supervisor's own poll, the resume hook, a watchdog command -- so a recycled PID
# landing on one of those would be read as a live run. That self-match is the
# single most expensive bug in this project's history: it killed four runs
# through pkill and later told a restart hook that a dead supervisor was alive.
# A real runner has argv[0] of "python" (phase 1) or "magnumnp..." (phase 2); a
# shell has argv[0] of "bash". Checking argv[0] tells them apart.

mkdir -p runs/certify
echo $$ > runs/certify/supervisor.pid

ident_ok () {            # ident_ok <pid> -- is this PID one of OUR runners?
    local cmd a0 head
    cmd=$(tr '\0' '\n' < "/proc/$1/cmdline" 2>/dev/null) || return 1
    a0=${cmd%%$'\n'*}                       # argv[0] only
    # The rename collapses the whole title into argv[0] as ONE field:
    # "magnumnp scripts/run_narma_modal.py". So take the first word of argv[0]
    # before stripping a directory, or the basename of the last path in that
    # title is what gets tested and nothing matches.
    head=${a0%% *}
    case "${head##*/}" in
        magnumnp*) return 0 ;;              # phase 2: renamed
        python*)   case "$cmd" in *run_narma_modal.py*) return 0 ;; esac ;;
    esac
    return 1
}

alive () {               # alive <pidfile>
    local pf="$1" pid
    [ -f "$pf" ] || return 1
    pid=$(cat "$pf" 2>/dev/null) || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    # guard against a recycled PID pointing at something else entirely
    ident_ok "$pid"
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
