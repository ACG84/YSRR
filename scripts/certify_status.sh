#!/usr/bin/env bash
# Report certification health: how many seed runners are alive, whether the
# supervisor is alive, and how many frames are banked.
#
# Why this exists. Counting runners by process name is wrong, and wrong in a way
# that looks right most of the time. magnum.np renames its worker process a few
# minutes into startup, so a run has two distinct identities:
#
#   phase          /proc/PID/comm     /proc/PID/cmdline
#   first ~3 min   python             python scripts/run_narma_modal.py --seed N
#   after rename   magnumnp script    magnumnp script
#
# `ps -eo comm | grep '^magnumnp'` sees zero during phase 1; `pgrep -f
# run_narma_modal` sees zero during phase 2. Both are correct most of the time
# and both report an empty process table right after a restart -- exactly when a
# watchdog is deciding whether to relaunch. A watchdog that trusts either one
# relaunches six healthy runs on top of themselves.
#
# So liveness comes from the PID each runner recorded at launch, confirmed
# against its cmdline so a recycled PID cannot pose as a live run. That check
# does not care what the process calls itself.
set -u
FRAMES="${FRAMES:-1200}"
SEEDS="${SEEDS:-6}"

root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$root" || exit 2

cmdline_of() { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null; }

# Is this PID one of our runners? Tested on argv[0], never on the whole command
# line: any shell that merely mentions run_narma_modal -- this script's own
# grep, the watchdog, the resume hook -- would match a substring test, so a
# recycled PID landing on one would read as a live run. A runner has argv[0]
# "python" (phase 1, pre-rename) or "magnumnp..." (phase 2); a shell has "bash".
# The rename collapses the whole title into argv[0] as ONE field --
# "magnumnp scripts/run_narma_modal.py" -- so take the first word of argv[0]
# before stripping a directory, or what gets tested is the basename of the path
# inside that title and nothing matches.
ident_ok() {
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

alive=0 total=0 complete=0
for s in $(seq 0 $((SEEDS - 1))); do
    d=runs/certify/seed_$s
    pid=$(cat "$d/pid" 2>/dev/null)
    state=dead
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null && ident_ok "$pid"; then
        state=alive
        alive=$((alive + 1))
    fi
    n=$(grep -o "frame [0-9]*/${FRAMES}" "$d/run.log" 2>/dev/null |
        tail -1 | grep -o '^frame [0-9]*' | grep -o '[0-9]*')
    n=${n:-0}
    total=$((total + n))
    [ "$n" -ge "$FRAMES" ] && complete=$((complete + 1))
    printf 'seed%s %-5s %s/%s\n' "$s" "$state" "$n" "$FRAMES"
done

sup=dead
pf=runs/certify/supervisor.pid
if [ -f "$pf" ]; then
    spid=$(cat "$pf" 2>/dev/null)
    if [ -n "${spid:-}" ] && kill -0 "$spid" 2>/dev/null &&
       cmdline_of "$spid" | grep -q certify_supervise.sh; then
        sup=alive
    fi
fi

echo "supervisor $sup"
echo "runners $alive/$SEEDS alive, $complete/$SEEDS complete"
echo "banked $total/$((FRAMES * SEEDS)) frames"

# Exit 0 only when the sweep is finished or genuinely healthy, so a watchdog can
# branch on status rather than parsing this output.
[ "$complete" -ge "$SEEDS" ] && exit 0
{ [ "$sup" = alive ] && [ "$alive" -ge "$SEEDS" ]; } && exit 0
exit 1
