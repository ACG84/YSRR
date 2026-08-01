#!/usr/bin/env bash
# Restart the certification supervisor when a session starts, if the runs are
# unfinished.
#
# Why this exists. The container reboots roughly hourly and kills every process
# in it -- including the supervisor whose whole job is to restart the runs. The
# safety net dies with the thing it guards, silently: no traceback, no exit
# code, just processes that were there and then were not. Two such reboots cost
# this project about ninety minutes. Nothing inside the container outlives a
# reboot, so the restart has to hang off an event that fires afterwards, and
# SessionStart is that event.
#
# The liveness check reads ONE recorded PID rather than searching the process
# table. `pgrep -f certify_supervise` looks equivalent and is not: it matches
# any shell whose own command line contains the pattern, including the shell
# running the check. That is not hypothetical -- the first version of this file
# used pgrep, reported a healthy supervisor, and the supervisor was dead. The
# same self-match has cost this project four runs by way of pkill.
#
# Contract: fast, silent, idempotent, never fails a session start.
#   - no torch and no imports; completion comes from the run logs, so this costs
#     a few greps rather than six checkpoint loads
#   - prints one line ONLY when it actually restarts something
#   - exits 0 unconditionally; a hook that can fail can wedge the session it is
#     meant to help
#   - self-disables once every seed finishes, so it sits in the repo harmlessly
#     after the certification is done
set -u
FRAMES="${FRAMES:-1200}"
SEEDS="${SEEDS:-6}"

root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$root" 2>/dev/null || exit 0
[ -d runs/certify ] || exit 0

done_n=$(grep -l "frame ${FRAMES}/${FRAMES}" runs/certify/seed_*/run.log 2>/dev/null | wc -l)
[ "$done_n" -ge "$SEEDS" ] && exit 0

pf=runs/certify/supervisor.pid
if [ -f "$pf" ]; then
    pid=$(cat "$pf" 2>/dev/null)
    # /proc/PID/cmdline is NUL-separated; tr makes it greppable. The cmdline
    # check is what distinguishes our supervisor from whatever recycled that
    # PID after a reboot.
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null &&
       tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null |
       grep -q "certify_supervise.sh"; then
        exit 0
    fi
fi

nohup bash scripts/certify_supervise.sh >> runs/certify_supervise.log 2>&1 &
echo "certification incomplete ($done_n/$SEEDS seeds) -- supervisor restarted"
exit 0
