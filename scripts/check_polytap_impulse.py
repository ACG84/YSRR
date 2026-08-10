#!/usr/bin/env python3
"""Does each tap receive the bus signal at the delay it was placed for?

Impulse before task, deliberately: a 600-frame NARMA run on this mesh is ~2
hours, and every previous architecture in this project was cheaper to falsify
with a burst than to disprove with a benchmark. Two things have to hold before
the expensive run is worth starting.

  delay      tap i sits at inject + lag_i * 189 nm, where 189 nm/frame is the
             MEASURED bus group delay (945 m/s). If the arrival times do not
             come back near the designed lags, the tap positions are wrong and
             every product separation is wrong with them.

  balance    the whole point of the bus over the chain. The delayed copy must
             arrive comparable to the fresh drive at the same disk, because a
             tap whose fresh sample is 100x larger stops being a delay tap --
             measured on the chain co-drive, where degree-1 capacity fell 8.91
             to 4.95 and the horizon pulled in from lag 12 to lag 7. Here the
             fresh drive is pre-scaled by exp(-d/951nm) to track the measured
             decay, so this checks the two arrive within a factor of a few.

Three arms, because the failure modes are different and separable:

    bus only      inject into the bus, no fresh drive. Gives arrival time and
                  delayed amplitude per tap.
    fresh only    drive the disks, nothing into the bus. Gives the fresh
                  amplitude per tap, and CROSSTALK: if disk 1's fresh drive
                  shows up at disk 4, the taps are not independent and the
                  products will be contaminated.
    both          what the device would actually run.

    python scripts/check_polytap_impulse.py
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import (PolyTapConfig, PolyTapArray,
                                DirCouplerConfig, DirCouplerArray)
from magnonic_nn._compat import get_device
from _polytap_probe import check_record_length


@torch.no_grad()
def pulse(arr, cfg, dtype, amp, freq, n_burst, n_quiet, bus=True, fresh=True,
          fresh_uniform=False):
    """Burst then silence; record every readout tap each step.

    `fresh_uniform` drives every disk at scale 1 instead of cfg.fresh_scale().
    That is what makes the fresh arm a clean per-tap MEASUREMENT: the ratio of
    what a tap reports from the bus to what it reports from a unit fresh drive
    is the number the co-drive scaling should be set from, and it cannot be read
    off a fresh arm that has already been pre-scaled by a guess.
    """
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    if bus:
        # into the bus, out of plane, at the injection segment
        unit[:, :, 0, 2] += arr.inject_mask
    if fresh:
        # into each disk body, in plane
        scales = ([1.0] * cfg.n_taps if fresh_uniform else cfg.fresh_scale())
        for k, s in enumerate(scales):
            unit[:, :, 0, 0] += float(s) * arr.disk_masks[k]
    stepper, graphed = arr.rollout.graph_stepper()
    hz = arr.h_zero
    m = arr.m0.clone()
    out = []
    for k in range(n_burst + n_quiet):
        tk = k * cfg.dt
        on = k < n_burst

        w = 2 * math.pi * freq
        a0 = amp if on else 0.0
        h0 = hz + unit * (a0 * math.sin(w * tk))
        hh = hz + unit * (a0 * math.sin(w * (tk + 0.5 * cfg.dt)))
        h1 = hz + unit * (a0 * math.sin(w * (tk + cfg.dt)))
        if graphed:
            torch.compiler.cudagraph_mark_step_begin()
            m = stepper(m, h0, hh, hh, h1).clone()
        else:
            m = stepper(m, h0, hh, hh, h1)
        out.append(arr.port_signals(m).double().cpu().numpy().copy())
    return np.asarray(out)


def envelope(x, win):
    return np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode="same"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-taps", type=int, default=4)
    p.add_argument("--lags", type=float, nargs="+", default=[5, 8, 11, 14])
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--burst", type=int, default=400)
    p.add_argument("--quiet", type=int, default=12000,
                   help="steps of silence after the burst. Was 2000, giving a\n"
                        "2.4 ns record when the wave needs 3.75 ns to reach the\n"
                        "lag-14 tap at the measured 706 m/s -- so the far taps\n"
                        "were reporting injection near field, not the delayed\n"
                        "copy, and the balance ratios computed from them were\n"
                        "ratios of the wrong quantity.")
    p.add_argument("--tap-alpha", type=float, default=1.0,
                   help="damping multiplier on the tap disk bodies. The balance\n"
                        "must be measured at the value the task run will use:\n"
                        "ta=10 is where the disk settles inside a frame and its\n"
                        "nonlinearity reaches the readout at all.")
    p.add_argument("--bus-alpha", type=float, default=1.0)
    p.add_argument("--fresh-uniform", action="store_true", default=True,
                   help="drive every disk at scale 1 in the fresh arm, so the\n"
                        "bus/fresh ratio is a measurement rather than a check on\n"
                        "a previous guess")
    p.add_argument("--gap", type=float, default=0.0,
                   help="nm; coupling gap between bus and each tap guide")
    p.add_argument("--coupler-len", type=float, default=0.0,
                   help="nm. >0 selects the DIRECTIONAL coupler geometry: the\n"
                        "tap's arm runs parallel to the bus over this length\n"
                        "instead of butting into it perpendicular. Coupling then\n"
                        "scales with length rather than proximity.")
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="runs/polytap_impulse")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if a.coupler_len > 0:
        cfg = DirCouplerConfig(n_taps=a.n_taps, tap_lags=tuple(a.lags),
                               coupler_len=a.coupler_len * 1e-9,
                               coupler_gap=(a.gap or 20.0) * 1e-9)
        arr = DirCouplerArray(cfg, timesteps=a.burst + a.quiet + 8, dtype=dtype)
        if cfg.coupler_len > cfg.max_coupler_len():
            raise SystemExit(
                f"coupler_len {cfg.coupler_len*1e9:.0f} nm exceeds the "
                f"{cfg.max_coupler_len()*1e9:.0f} nm that tap spacing allows; "
                f"adjacent arms would merge into one waveguide.")
    else:
        cfg = PolyTapConfig(n_taps=a.n_taps, tap_lags=tuple(a.lags),
                            coupling_gap=a.gap * 1e-9,
                            tap_alpha_mult=a.tap_alpha,
                            bus_alpha_mult=a.bus_alpha)
        arr = PolyTapArray(cfg, timesteps=a.burst + a.quiet + 8, dtype=dtype)
    check_record_length(cfg, a.burst + a.quiet)
    nx, ny = cfg.grid
    print(f"grid {nx}x{ny}, {cfg.n_taps} taps, designed lags "
          f"{list(cfg.tap_lags[:cfg.n_taps])}")

    # Relax in CHUNKS, checkpointing each one.
    #
    # This mesh takes ~36 minutes to relax and the container reboots roughly
    # hourly. An all-or-nothing relax on those odds does not merely risk the
    # work, it can fail to ever finish: the gap-30 build died mid-relax with
    # nothing saved and would have restarted from the ansatz every time. The
    # progress file records how many steps the saved state has had, so a
    # restart continues rather than repeating.
    tag = (f"n{cfg.n_taps}_cpl{int(a.coupler_len)}" if a.coupler_len > 0
           else f"n{cfg.n_taps}_gap{int(a.gap)}")
    # Damping does not move an energy minimum, so the ground state is shared in
    # principle -- but a shared m0 was already found to be silently wrong on
    # CUDA in this project, so key the cache by the full run instead.
    if a.tap_alpha != 1.0:
        tag = f"{tag}_ta{a.tap_alpha:g}"
    if a.bus_alpha != 1.0:
        tag = f"{tag}_ba{a.bus_alpha:g}"
    m0c = outdir / f"m0_{tag}.pt"
    prog = outdir / f"m0_{tag}.steps"
    done = 0
    if m0c.exists():
        arr.m0 = torch.load(m0c, weights_only=False).to(device=get_device(), dtype=dtype)
        done = int(prog.read_text().strip()) if prog.exists() else a.relax_steps
        print(f"[m0] restored from {m0c.name} at {done}/{a.relax_steps} steps")
    CHUNK = 1000
    while done < a.relax_steps:
        t0 = time.time()
        if arr.m0 is None:
            arr.relax(steps=CHUNK)              # builds the ansatz, then relaxes
        else:
            arr.m0 = arr.rollout.relax(arr.m0, arr.h_zero, CHUNK, 0.5)
        done += CHUNK
        tmp = m0c.with_suffix(".pt.tmp")
        with open(tmp, "wb") as fh:
            torch.save(arr.m0.cpu(), fh); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, m0c)
        prog.write_text(str(done))
        print(f"  relax {done}/{a.relax_steps} ({time.time()-t0:.0f}s)", flush=True)
    (outdir / f"pid_{tag}").write_text(str(os.getpid()))

    amp = a.amp_mT * 1e-3 / MU_0
    win = max(4, int(round(1e3 / a.freq)))
    dt_ns = cfg.dt * 1e9
    frame_ns = 200 * cfg.dt * 1e9
    npr = arr.n_readout

    arms = {"bus only": (True, False), "fresh only": (False, True),
            "both": (True, True)}
    res = {}
    for name, (bus, fresh) in arms.items():
        t0 = time.time()
        sig = pulse(arr, cfg, dtype, amp, a.freq * 1e9, a.burst, a.quiet,
                    bus=bus, fresh=fresh, fresh_uniform=a.fresh_uniform)
        rows = []
        print(f"\n--- {name} ({time.time()-t0:.0f}s) ---")
        print(f"{'tap':>4} {'designed':>9} {'arrive_ns':>10} {'meas_lag':>9} "
              f"{'peak amp':>11}")
        for d in range(cfg.n_taps):
            e = envelope(np.sqrt((sig[:, d*npr:(d+1)*npr] ** 2).mean(axis=1)), win)
            pk = float(e.max())
            thr = 0.1 * pk
            idx = int(np.argmax(e > thr)) if (e > thr).any() else -1
            t_arr = idx * dt_ns if idx >= 0 else float("nan")
            rows.append({"tap": d + 1, "designed_lag": cfg.tap_lags[d],
                         "arrive_ns": t_arr, "meas_lag": t_arr / frame_ns,
                         "peak": pk})
            print(f"{d+1:>4} {cfg.tap_lags[d]:>9.1f} {t_arr:>10.3f} "
                  f"{t_arr/frame_ns:>9.2f} {pk:>11.4e}", flush=True)
        res[name] = rows
        (outdir / f"results_{tag}.json").write_text(json.dumps(res, indent=2))

    print("\n=== balance: delayed copy vs fresh drive, per tap ===")
    print(f"{'tap':>4} {'bus amp':>11} {'fresh amp':>11} {'ratio':>8}")
    bal = []
    for d in range(cfg.n_taps):
        b = res["bus only"][d]["peak"]
        f = res["fresh only"][d]["peak"]
        bal.append(b / max(f, 1e-30))
        print(f"{d+1:>4} {b:>11.4e} {f:>11.4e} {b/max(f,1e-30):>8.3f}")
    spread = max(bal) / max(min(bal), 1e-30)
    print(f"\nbalance ratio spread across taps: {spread:.1f}x")

    # The number this script exists to produce.
    #
    # A tap's nonlinearity multiplies whatever is in its disk. The cross term is
    # bilinear in the two operands, so it is the ratio of their contributions AT
    # THE READOUT that has to be ~1 -- not the ratio of a drive amplitude to a
    # readout amplitude, which is what the NARMA runs were scaled by and which
    # compares two incommensurable quantities. Fresh response is linear in drive
    # scale over this range, so setting the scale to bus/fresh equalises them.
    print("\n=== recommended --fresh-scale for run_narma_polytap.py ===")
    print("  " + " ".join(f"{x:.6f}" for x in bal))
    print(f"\n(normalised to tap 1: "
          + " ".join(f"{x/bal[0]:.4f}" for x in bal) + ")")
    print("Measured at " + ("uniform unit" if a.fresh_uniform else "pre-scaled")
          + f" fresh drive and {a.amp_mT:.0f} mT, tap alpha x{a.tap_alpha:g}.\n"
          "Caveat: the disk compresses 24% between 10 and 30 mT, so this ratio\n"
          "is amplitude-dependent and is measured at the top of the range.")

    print("\n=== delay check ===")
    ok = True
    for d in range(cfg.n_taps):
        want = cfg.tap_lags[d]
        got = res["bus only"][d]["meas_lag"]
        good = np.isfinite(got) and abs(got - want) <= max(2.0, 0.3 * want)
        ok &= good
        print(f"  tap {d+1}: designed {want:.1f}, measured {got:.2f} "
              f"{'OK' if good else 'OFF'}")
    print()
    if ok and 0.1 < min(bal) and max(bal) < 10:
        print("Geometry behaves: taps receive the bus at their designed delays\n"
              "and the delayed copy is within an order of magnitude of the fresh\n"
              "drive at every tap. The NARMA run is worth its two hours.")
    elif not ok:
        print("Delays are WRONG. Tap positions follow from the measured 189 nm\n"
              "per frame; if arrivals disagree, the bus in this geometry is not\n"
              "the bus that was characterised -- the disks load it. Re-fit before\n"
              "spending the task run.")
    else:
        print(f"Delays are right but the BALANCE is off ({min(bal):.3f} to "
              f"{max(bal):.3f}).\nA tap whose fresh drive swamps its delayed copy "
              f"is an input disk, not a\ndelay tap -- that is exactly how the "
              f"chain co-drive failed. Re-scale\nfresh_scale() before the task run.")
    print(f"\nwrote {outdir / f'results_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
