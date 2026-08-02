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

# A supervisor is alive if its recorded PID exists AND its cmdline is still the
# script we started. The cmdline check is what distinguishes our supervisor from
# whatever recycled that PID after a reboot.
sup_alive () {           # sup_alive <pidfile> <script name>
    local pid
    [ -f "$1" ] || return 1
    pid=$(cat "$1" 2>/dev/null)
    [ -n "${pid:-}" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$2"
}

done_n=$(grep -l "frame ${FRAMES}/${FRAMES}" runs/certify/seed_*/run.log 2>/dev/null | wc -l)
if [ "$done_n" -lt "$SEEDS" ] &&
   ! sup_alive runs/certify/supervisor.pid certify_supervise.sh; then
    nohup bash scripts/certify_supervise.sh >> runs/certify_supervise.log 2>&1 &
    echo "certification incomplete ($done_n/$SEEDS seeds) -- supervisor restarted"
fi

# The damping screen has the same problem and the same fix. It is a separate
# block rather than a loop because the two sweeps finish independently: the
# certification is done and its block is now a no-op, while the screen is not.
SCREEN_FRAMES="${SCREEN_FRAMES:-800}"
if [ -d runs/screen ]; then
    n_alpha=$(ls -d runs/screen/alpha_* 2>/dev/null | wc -l)
    s_done=$(grep -l "frame ${SCREEN_FRAMES}/${SCREEN_FRAMES}" \
             runs/screen/alpha_*/run.log 2>/dev/null | wc -l)
    if [ "$n_alpha" -gt 0 ] && [ "$s_done" -lt "$n_alpha" ] &&
       ! sup_alive runs/screen/supervisor.pid screen_supervise.sh; then
        nohup bash scripts/screen_supervise.sh >> runs/screen_supervise.log 2>&1 &
        echo "damping screen incomplete ($s_done/$n_alpha) -- supervisor restarted"
    fi
fi

# The readout probe is a single run rather than a sweep, so it gets no
# supervisor -- the hook relaunches it directly. Safe to do blindly: the runner
# resumes from its checkpoint, which now carries the waveform alongside the
# features, so a relaunch costs at most CKPT_EVERY frames.
PROBE_FRAMES="${PROBE_FRAMES:-600}"
probe=runs/statelock_probe
if [ -d "$probe" ] &&
   ! grep -qs "frame ${PROBE_FRAMES}/${PROBE_FRAMES}" \
        "$probe/run.log" runs/statelock_probe.log; then
    # Match the script name, not the process name: magnum.np renames the worker
    # to "magnumnp scripts/run_narma_modal.py" a few minutes into startup, so a
    # comm-based check reports zero runners while one is healthily running.
    if [ "$(ps -eo args --no-headers | grep -c '[r]un_narma_modal.py')" -eq 0 ]; then
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python \
            scripts/run_narma_modal.py --frames "$PROBE_FRAMES" \
            --splits 150 300 75 --drive port --seed 0 --save-state-lockin 6 \
            --outdir "$probe" >> runs/statelock_probe.log 2>&1 &
        echo "state probe unfinished -- relaunched (resumes from checkpoint)"
    fi
fi
# The cascade pair. Same reasoning as the probe above -- single runs, no
# supervisor, resume from checkpoint so a blind relaunch costs at most
# CKPT_EVERY frames. The relaxed ground state is cached per outdir, so a restart
# does not re-pay the multi-minute relax on a 248x108 mesh either.
CASCADE_FRAMES="${CASCADE_FRAMES:-600}"
for d in cascade_linked cascade_nolink; do
    [ -d "runs/$d" ] || continue
    grep -qs "frame ${CASCADE_FRAMES}/${CASCADE_FRAMES}" "runs/$d.log" && continue
    # Liveness by RECORDED PID, never by argv: the rename destroys --outdir, so
    # an argv match reports zero while a run is healthy and starts a second
    # writer on the same checkpoint. Confirm the PID is still one of ours.
    live=0
    if [ -f "runs/$d/pid" ]; then
        pid=$(cat "runs/$d/pid" 2>/dev/null)
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null &&
           tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null |
           grep -qE 'run_narma_coupled|magnumnp'; then
            live=1
        fi
    fi
    if [ "$live" -eq 0 ]; then
        extra=""
        [ "$d" = cascade_nolink ] && extra="--no-link"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python \
            scripts/run_narma_coupled.py --frames "$CASCADE_FRAMES" \
            --splits 150 300 75 --drive one --seed 0 $extra \
            --outdir "runs/$d" >> "runs/$d.log" 2>&1 &
        # Record the PID HERE, not inside the runner. The runner writes its own
        # pid only after importing torch and building the array, ~20-30 s in --
        # and a second hook invocation inside that window reads the PREVIOUS
        # boot's stale pid, calls the run dead, and starts a duplicate writer on
        # a live checkpoint. That is exactly what happened at 12:40, when the
        # SessionStart hook and a manual check landed seconds apart.
        echo $! > "runs/$d/pid"
        echo "$d unfinished -- relaunched (resumes from checkpoint)"
    fi
done
exit 0
