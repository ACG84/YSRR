"""Shared poly-tap probe: build, relax, burst, and time-of-arrival by xcorr.

Extracted so the single-point probe (`check_polytap_delay.py`) and the 2D sweep
(`sweep_polytap.py`) cannot drift apart. They previously would have carried two
copies of the delay estimator, which is exactly the code most worth having one
of -- three architectures in this project were judged on estimators that turned
out not to support the verdict, and the fix only counts if every caller gets it.
"""
from __future__ import annotations
import math, os
from pathlib import Path
import numpy as np, torch

from magnonic_nn.config import MU_0
from magnonic_nn.vortex import (PolyTapConfig, PolyTapArray,
                                DirCouplerConfig, DirCouplerArray)
from magnonic_nn._compat import get_device

FRAME_STEPS = 200


# Bus transport, per (carrier, bus width). One frame of delay is v_g times the
# frame duration, and v_g depends on BOTH -- so a tap spacing is only correct
# for the operating point it was measured at.
#
#   12 GHz / 80 nm    945 m/s   the as-built point. 189 nm per 200-step frame,
#                               measured. Note this is the BROADBAND velocity
#                               (948 m/s fitted over 12-20 GHz), not the local
#                               706 m/s, because a one-frame burst at 200 steps
#                               spans ~5 GHz and travels at the broadband one.
#    9 GHz / 160 nm   543 m/s   the proposed point, from the width sweep's
#                               phase-slope fit. A 1100-step frame spans ~0.9
#                               GHz, narrow enough that the LOCAL velocity is
#                               the right one to use.
BUS_V_G = {(12.0, 80.0): 945.0, (9.0, 160.0): 542.9}


def frame_nm_for(steps_per_frame, v_g=945.0, dt=1e-12):
    """Tap spacing that gives one frame of bus delay at `steps_per_frame`.

    The spacing has to track the frame length: a tap at `lag * frame_nm`
    delivers `lag` frames of delay only at the frame it was computed for.
    Changing --steps-per-frame alone would silently rescale every designed lag
    -- 400-step frames would turn a lag-5 tap into a lag-2.5 one -- and the run
    would report clean delays at the wrong values.

    It has to track the OPERATING POINT too, which is new: moving the carrier
    to 9 GHz and the bus to 160 nm takes v_g from 945 to 543 m/s, so the same
    frame is 597 nm of bus rather than 1039. Passing the wrong v_g here mislays
    every tap by a factor of 1.7 while every printed number still looks sane.
    """
    return v_g * steps_per_frame * dt


def make_array(n_taps, lags, gap_nm, coupler_len_nm, tap_alpha_mult,
               timesteps, dtype=torch.float32, bus_alpha_mult=1.0,
               bus_guide_width_nm=None, steps_per_frame=200,
               bus_width_nm=None, v_g=945.0, bus_guide_offset_nm=0.0,
               barrier=None):
    """Build the array for one point of the sweep.

    Returns (cfg, arr, geom_tag, run_tag). The two tags differ on purpose:
    `geom_tag` names the GEOMETRY and keys the cached ground state, `run_tag`
    also carries the damping. The ground state is an energy minimum and does not
    depend on damping, so every damping point reuses one relax -- which is what
    makes a 2D sweep affordable at all.
    """
    frame_kw = ({} if steps_per_frame == 200 and v_g == 945.0
                else {"frame_nm": frame_nm_for(steps_per_frame, v_g)})
    if bus_width_nm is not None:
        frame_kw["bus_width"] = bus_width_nm * 1e-9
    if coupler_len_nm > 0:
        cfg = DirCouplerConfig(n_taps=n_taps, tap_lags=tuple(lags),
                               coupler_len=coupler_len_nm * 1e-9,
                               coupler_gap=(gap_nm or 20.0) * 1e-9,
                               tap_alpha_mult=tap_alpha_mult,
                               bus_alpha_mult=bus_alpha_mult, **frame_kw)
        if cfg.coupler_len > cfg.max_coupler_len():
            raise ValueError(
                f"coupler_len {coupler_len_nm:.0f} nm exceeds the "
                f"{cfg.max_coupler_len()*1e9:.0f} nm tap spacing allows; "
                f"adjacent arms would merge into one waveguide")
        arr = DirCouplerArray(cfg, timesteps=timesteps, dtype=dtype)
        geom = f"n{n_taps}_cpl{int(coupler_len_nm)}"
    else:
        kw = ({} if bus_guide_width_nm is None
              else {"bus_guide_width": bus_guide_width_nm * 1e-9})
        if bus_guide_offset_nm:
            kw["bus_guide_offset"] = bus_guide_offset_nm * 1e-9
        if barrier:
            bl, ba, bk = barrier
            if bl > 0:
                kw.update(bus_barrier_len=bl * 1e-9, bus_barrier_alpha=ba,
                          bus_barrier_after=int(bk))
        cfg = PolyTapConfig(n_taps=n_taps, tap_lags=tuple(lags),
                            coupling_gap=gap_nm * 1e-9,
                            tap_alpha_mult=tap_alpha_mult,
                            bus_alpha_mult=bus_alpha_mult, **kw,
                            **frame_kw)
        arr = PolyTapArray(cfg, timesteps=timesteps, dtype=dtype)
        geom = f"n{n_taps}_gap{int(gap_nm)}"
        if bus_guide_width_nm is not None:
            geom = f"{geom}_bw{int(bus_guide_width_nm)}"
    # TAP LAGS BELONG IN THE TAG. Without them two arrays differing only in
    # tap POSITION share a cache key, and the sweep skips the second as
    # "cached" while reporting the first one's numbers under the second one's
    # name. Caught when lags 5,8,11,14 silently reused lags 3,5,7,9 and the
    # verdict announced "2 of 2 points" for a point that never ran. The m0
    # cache is keyed the same way, so a collision would also load a ground
    # state relaxed on a different mesh.
    geom = f"{geom}_L" + "-".join(f"{l:g}" for l in lags[:n_taps])
    if bus_width_nm is not None:
        geom = f"{geom}_bus{int(bus_width_nm)}"
    if steps_per_frame != 200:
        geom = f"{geom}_spf{steps_per_frame}"
    if v_g != 945.0:
        geom = f"{geom}_vg{int(v_g)}"
    # The offset moves the mask, so it moves the ground state too -- same
    # reason the tap lags are in here.
    if bus_guide_offset_nm:
        geom = f"{geom}_off{int(bus_guide_offset_nm)}"
    if barrier and barrier[0] > 0:
        geom = f"{geom}_bar{int(barrier[0])}a{barrier[1]:g}k{int(barrier[2])}"
    run = geom if tap_alpha_mult == 1.0 else f"{geom}_ta{tap_alpha_mult:g}"
    if bus_alpha_mult != 1.0:
        run = f"{run}_ba{bus_alpha_mult:g}"
    # The ground state does not depend on damping, so sharing one relax across a
    # damping column SHOULD be free -- and on CPU it is. On CUDA it is not:
    # points that reloaded a shared m0 returned four taps of identical amplitude
    # (spread 1.0, min amp 1.3e-02) while the same point with a fresh relax
    # reproduced the CPU numbers exactly (spacings 1.11, 1.05, -11.91; spread
    # 287x against 286x; min amp 2.78e-06 in both). Something about the restored
    # state is wrong on the device beyond the dtype/device fix already applied,
    # and it is not yet root-caused.
    #
    # So the m0 cache is keyed by RUN, not by geometry: every point relaxes its
    # own ground state. That is pure waste on CPU and ~100 s per point on a T4,
    # which is affordable precisely because the GPU is fast -- and a correct
    # sweep at 35 minutes beats a wrong one at 10.
    return cfg, arr, run, run


def ensure_m0(arr, outdir, geom_tag, relax_steps=8000, chunk=1000, dtype=torch.float32,
              log=print):
    """Relax to the ground state, checkpointing every `chunk` steps.

    Chunked because the relax is ~40 minutes on this mesh and both hosts this
    runs on reclaim idle machines -- the container here reboots roughly hourly
    and Colab recycles runtimes. An all-or-nothing relax on those odds does not
    merely risk the work, it can fail to ever complete.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    m0c, prog = outdir / f"m0_{geom_tag}.pt", outdir / f"m0_{geom_tag}.steps"
    done = 0
    if m0c.exists():
        arr.m0 = torch.load(m0c, weights_only=False).to(device=get_device(), dtype=dtype)
        done = int(prog.read_text().strip()) if prog.exists() else relax_steps
        log(f"[m0] {m0c.name} at {done}/{relax_steps}")
    while done < relax_steps:
        if arr.m0 is None:
            arr.relax(steps=chunk)
        else:
            arr.m0 = arr.rollout.relax(arr.m0, arr.h_zero, chunk, 0.5)
        done += chunk
        tmp = m0c.with_suffix(".pt.tmp")
        with open(tmp, "wb") as fh:
            torch.save(arr.m0.cpu(), fh); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, m0c)
        prog.write_text(str(done))
        log(f"  relax {done}/{relax_steps}")
    return arr.m0


@torch.no_grad()
def burst_response(arr, cfg, amp_mT, freq_ghz, n_burst, n_quiet,
                   bus=True, fresh=False, dtype=torch.float32, log=None):
    """Burst into the bus (and/or the disks), then silence. Returns (sig, drive).

    Uses the graph-capturable stepper: `rk4_step` takes a Python callable and
    breaks the CUDA graph every substep, which is why the array runs were
    previously measured at 1.0-1.5x on GPU and written off as dispatch-bound.
    """
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    if bus:
        unit[:, :, 0, 2] += arr.inject_mask
    if fresh:
        for k, s in enumerate(cfg.fresh_scale()):
            unit[:, :, 0, 0] += float(s) * arr.disk_masks[k]
    amp = amp_mT * 1e-3 / MU_0
    w = 2 * math.pi * freq_ghz * 1e9
    stepper, graphed = arr.rollout.graph_stepper()
    eager = arr.rollout.rk4_step_fields
    hz = arr.h_zero
    m = arr.m0.clone()
    out, drive = [], []
    # graph_stepper() already falls back if torch.compile refuses up front, but
    # the failure seen on the GPU happens on the first CALL, not at compile
    # request: a sweep builds several geometries in one process, and recompiling
    # for the second mesh shape raised "Unhandled FakeTensor Device Propagation"
    # inside dynamo. The same run with TORCHDYNAMO_DISABLE=1 completes and gives
    # correct numbers, so this is a tracing failure rather than a tensor that is
    # genuinely on the wrong device. Fall back for the rest of the rollout and
    # say so, rather than losing the sweep to it.
    for k in range(n_burst + n_quiet):
        tk = k * cfg.dt
        a0 = amp if k < n_burst else 0.0
        drive.append(1.0 if k < n_burst else 0.0)
        h0 = hz + unit * (a0 * math.sin(w * tk))
        hh = hz + unit * (a0 * math.sin(w * (tk + 0.5 * cfg.dt)))
        h1 = hz + unit * (a0 * math.sin(w * (tk + cfg.dt)))
        if graphed:
            try:
                torch.compiler.cudagraph_mark_step_begin()
                m = stepper(m, h0, hh, hh, h1).clone()
            except Exception as e:
                if log:
                    log(f"  compiled step failed ({type(e).__name__}); "
                        f"falling back to eager for this geometry")
                graphed, stepper = False, eager
                m = stepper(m, h0, hh, hh, h1)
        else:
            m = stepper(m, h0, hh, hh, h1)
        out.append(arr.port_signals(m).double().cpu().numpy().copy())
        if log and (k + 1) % 800 == 0:
            log(f"  step {k+1}/{n_burst+n_quiet}")
    return np.asarray(out), np.asarray(drive)


def _env(x, win):
    return np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode="same"))


def check_record_length(cfg, n_steps, v_g=706.0, response_frames=13.0):
    """Refuse a rollout that ends before the wave reaches the farthest tap.

    This is the error that invalidated the delay half of every poly-tap run in
    this project. The default record was burst 200 + quiet 2600 = 2.81 ns. At
    the MEASURED local group velocity at 12 GHz -- 706 m/s, from the phase slope
    of the loaded bus -- the wave needs 3.75 ns to reach the lag-14 tap and 6.42
    ns to reach the lag-24 tap. It never arrived. Every one of those runs was
    reporting the injection near field, which is instantaneous, and reporting it
    as though it were the delayed copy.

    Nothing in the output looked wrong: amplitudes were plausible, the estimator
    returned finite lags, and the lags were consistent from run to run. They
    were consistently the disk's own response time, which is the same at every
    tap because it has nothing to do with distance.

    v_g defaults to the LOCAL group velocity at the 12 GHz drive, not the 948
    m/s broadband fit -- an envelope travels at the local one, and the
    difference is 34%.

    CALLERS AT A DIFFERENT OPERATING POINT MUST PASS THEIR OWN. The default is
    706 m/s and the 9 GHz / 160 nm bus runs at 542.9, so the default overstates
    the velocity by 30% and therefore UNDERSTATES the transit -- leniency in
    the one direction this guard exists to prevent. Caught on the 9 GHz retune,
    where it asked for 10905 steps against the 13400 the real velocity needs.
    """
    far = max(cfg.tap_lags[:cfg.n_taps]) * cfg.frame_nm
    need = far / v_g / cfg.dt + response_frames * FRAME_STEPS
    if n_steps < need:
        raise SystemExit(
            f"record is {n_steps} steps ({n_steps*cfg.dt*1e9:.2f} ns) but the "
            f"farthest tap is\n{far*1e9:.0f} nm out, which the wave reaches "
            f"only after {far/v_g*1e9:.2f} ns. Allowing "
            f"{response_frames:.0f} frames\nfor the disk to respond, this needs "
            f"at least {need:.0f} steps. Raise --quiet.")
    return need


def delay_by_xcorr(sig, drive, n_taps, n_readout, freq_ghz,
                   steps_per_frame=FRAME_STEPS):
    """Per-tap arrival by envelope cross-correlation against the burst.

    Envelope rather than carrier: the 12 GHz period is 0.42 frames, so a
    carrier-level correlation is ambiguous modulo the period. Ring-up biases the
    absolute number late, but that bias is COMMON to every tap, which is why the
    caller should judge on differences between taps rather than on any single
    absolute lag.
    """
    win = max(4, int(round(1e3 / freq_ghz)))
    ref = drive - drive.mean()
    rows = []
    for d in range(n_taps):
        e = _env(np.sqrt((sig[:, d*n_readout:(d+1)*n_readout] ** 2).mean(axis=1)), win)
        ec = e - e.mean()
        c = np.correlate(ec, ref, mode="full")
        lags_steps = np.arange(-len(ref) + 1, len(ec))
        ok = lags_steps >= 0
        i = int(np.argmax(c[ok]))
        denom = math.sqrt(float((ec ** 2).sum()) * float((ref ** 2).sum()))
        rows.append({"tap": d + 1,
                     "xcorr_lag": lags_steps[ok][i] / steps_per_frame,
                     "corr": float(c[ok][i] / max(denom, 1e-30)),
                     "peak": float(e.max())})
    return rows


def score_point(rows, designed_lags, steps_per_frame=FRAME_STEPS, dt=1e-12):
    """Collapse one sweep point to the two numbers that decide it.

    A point is only useful if taps both RESOLVE (measured spacing tracks the
    designed spacing) and RECEIVE (the weakest tap is not lost in the spread).
    Every sweep in this project so far moved one of those and was defeated by
    the other, so both are reported and neither is reduced away.

    Errors are additionally reported in NANOSECONDS, because frames stopped
    being a fixed unit once the frame length became a variable. The same
    physical 0.60 ns tap separation is 3.00 frames at 200 steps and 1.50 at
    400, so a frame-count error silently halves when the frame doubles -- a
    comparison across frame lengths read in frames would show an improvement
    that is pure bookkeeping.
    """
    n = len(rows)
    d_designed = [designed_lags[k + 1] - designed_lags[k] for k in range(n - 1)]
    d_meas = [rows[k + 1]["xcorr_lag"] - rows[k]["xcorr_lag"] for k in range(n - 1)]
    err = [m - d for m, d in zip(d_meas, d_designed)]
    amps = [r["peak"] for r in rows]
    monotonic = sum(1 for m in d_meas if m > 0)
    return {
        "spacing_designed": d_designed,
        "spacing_measured": d_meas,
        "spacing_error": err,
        "n_monotonic": monotonic,
        "worst_spacing_error": max(abs(e) for e in err),
        "mean_abs_spacing_error": float(np.mean([abs(e) for e in err])),
        "spacing_error_ns": [e * steps_per_frame * dt * 1e9 for e in err],
        "mean_abs_spacing_error_ns": float(np.mean(
            [abs(e) for e in err])) * steps_per_frame * dt * 1e9,
        "amp_spread": float(max(amps) / max(min(amps), 1e-30)),
        "amp_min": float(min(amps)),
        "amps": amps,
    }
