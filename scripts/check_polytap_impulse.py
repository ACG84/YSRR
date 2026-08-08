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
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import PolyTapConfig, PolyTapArray


@torch.no_grad()
def pulse(arr, cfg, dtype, amp, freq, n_burst, n_quiet, bus=True, fresh=True):
    """Burst then silence; record every readout tap each step."""
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    if bus:
        # into the bus, out of plane, at the injection segment
        unit[:, :, 0, 2] += arr.inject_mask
    if fresh:
        # into each disk body, in plane, scaled to track the bus decay
        for k, s in enumerate(cfg.fresh_scale()):
            unit[:, :, 0, 0] += float(s) * arr.disk_masks[k]
    m = arr.m0.clone()
    out = []
    for k in range(n_burst + n_quiet):
        tk = k * cfg.dt
        on = k < n_burst

        def h(theta, tk=tk, on=on):
            if not on:
                return torch.zeros_like(unit)
            return unit * (amp * math.sin(2 * math.pi * freq * (tk + theta * cfg.dt)))

        m = arr.rollout.rk4_step(m, arr.h_zero, h)
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
    p.add_argument("--quiet", type=int, default=2000)
    p.add_argument("--gap", type=float, default=0.0,
                   help="nm; coupling gap between bus and each tap guide")
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--outdir", default="runs/polytap_impulse")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = PolyTapConfig(n_taps=a.n_taps, tap_lags=tuple(a.lags),
                        coupling_gap=a.gap * 1e-9)
    arr = PolyTapArray(cfg, timesteps=a.burst + a.quiet + 8, dtype=dtype)
    nx, ny = cfg.grid
    print(f"grid {nx}x{ny}, {cfg.n_taps} taps, designed lags "
          f"{list(cfg.tap_lags[:cfg.n_taps])}")

    m0c = outdir / f"m0_n{cfg.n_taps}_gap{int(a.gap)}.pt"
    if m0c.exists():
        arr.m0 = torch.load(m0c, weights_only=False).to(dtype)
        print(f"[m0] restored from {m0c.name}")
    else:
        t0 = time.time()
        arr.relax(steps=a.relax_steps)
        torch.save(arr.m0.cpu(), m0c)
        print(f"relaxed in {time.time()-t0:.0f}s")

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
                    bus=bus, fresh=fresh)
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
        (outdir / "results.json").write_text(json.dumps(res, indent=2))

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
    print(f"\nwrote {outdir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
