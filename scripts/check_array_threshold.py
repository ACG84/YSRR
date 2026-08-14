#!/usr/bin/env python3
"""Where does a tap disk go nonlinear IN THE ARRAY, not in isolation?

The 6.00 mT threshold at 9 GHz was measured on a lone PortedVortexDisk with six
readout guides. A tap disk is not that disk. It has five readout guides and a
sixth arm coupling it to the bus, it sits a coupling gap away from a 160 nm
waveguide, and the bus loads it. Any of that can move the mode the threshold
depends on -- and the whole 9 GHz operating point rests on the disk being 4.7x
closer to its mode there than at 12 GHz, so a shifted mode is not a detail.

The last NARMA run was configured against the isolated number and drove the
taps to 5.00 / 3.83 / 2.62 / 2.23 mT against a 6.00 mT threshold. Nothing was
ever nonlinear, and the run could not have produced a product whatever its
memory horizon was. This measures the number that run should have been
configured against.

Same lock-in as check_drive_nonlinearity, applied per tap: drive the disk
BODIES at a constant amplitude, settle, then integrate the port signals against
the carrier over whole cycles. Reports per tap:

    |A|/a        response per unit drive. Flat = linear.
    norm         that, relative to the lowest amplitude. The threshold is where
                 it departs from 1.
    core         the vortex has to survive, or the number describes a different
                 element than the one the array is built from.

The tell that separates a real threshold from the artifact that produced three
spurious 0.50 mT readings earlier in this project: |A| must keep SCALING with
drive while `norm` falls. A drive-independent |A| makes norm collapse on its
own and means the measurement, not the element, is saturating.

    python scripts/check_array_threshold.py --device cuda --bus-alpha 0.1 \
        --lags 3 5 7 9 --bus-width 160 --v-g 542.9 --freq 9.0
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from _polytap_probe import make_array, ensure_m0


@torch.no_grad()
def drive_taps(arr, cfg, amp_mT, freq_ghz, n_settle, n_meas, fresh_scale,
               dtype=torch.float32):
    """Drive the disk bodies at constant amplitude; lock in per port.

    Fresh drive only -- nothing into the bus. A tap's threshold is a property
    of the disk and its loading, and mixing in a delayed copy would measure the
    pair instead.
    """
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    for k, s in enumerate(fresh_scale):
        unit[:, :, 0, 0] += float(s) * arr.disk_masks[k]
    amp = amp_mT * 1e-3 / MU_0
    w = 2 * math.pi * freq_ghz * 1e9
    stepper, graphed = arr.rollout.graph_stepper()
    hz = arr.h_zero
    m = arr.m0.clone()
    n_sig = arr.port_signals(m).shape[0]
    accI = torch.zeros(n_sig, dtype=torch.float64)
    accQ = torch.zeros(n_sig, dtype=torch.float64)
    for k in range(n_settle + n_meas):
        tk = k * cfg.dt
        h0 = hz + unit * (amp * math.sin(w * tk))
        hh = hz + unit * (amp * math.sin(w * (tk + 0.5 * cfg.dt)))
        h1 = hz + unit * (amp * math.sin(w * (tk + cfg.dt)))
        if graphed:
            torch.compiler.cudagraph_mark_step_begin()
            m = stepper(m, h0, hh, hh, h1).clone()
        else:
            m = stepper(m, h0, hh, hh, h1)
        if k >= n_settle:
            p = arr.port_signals(m).double()
            accI += p * math.cos(w * tk)
            accQ += p * math.sin(w * tk)
    A = ((accI + 1j * accQ) / n_meas).cpu().numpy()
    return A, m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-taps", type=int, default=4)
    p.add_argument("--lags", type=float, nargs="+", default=[3, 5, 7, 9])
    p.add_argument("--gap", type=float, default=30.0)
    p.add_argument("--bus-guide-width", type=float, default=80.0)
    p.add_argument("--bus-width", type=float, default=160.0)
    p.add_argument("--bus-alpha", type=float, default=0.1)
    p.add_argument("--tap-alpha", type=float, default=1.0)
    p.add_argument("--v-g", type=float, default=542.9)
    p.add_argument("--freq", type=float, default=9.0)
    p.add_argument("--steps-per-frame", type=int, default=1200)
    p.add_argument("--amps-mT", type=float, nargs="+",
                   default=[0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30])
    p.add_argument("--cycles", type=float, default=7.2,
                   help="settle and measure windows in drive cycles, so the\n"
                        "lock-in integrates whole periods at any carrier")
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="runs/array_threshold")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    n_win = int(round(a.cycles / (a.freq * 1e9) / 1e-12))
    cfg, arr, run, _ = make_array(
        a.n_taps, a.lags, a.gap, 0.0, a.tap_alpha, 2 * n_win + 8, dtype,
        bus_alpha_mult=a.bus_alpha, bus_guide_width_nm=a.bus_guide_width,
        bus_width_nm=a.bus_width, steps_per_frame=a.steps_per_frame, v_g=a.v_g)
    print(f"{run}\nmesh {cfg.grid[0]}x{cfg.grid[1]}, {a.freq:g} GHz, "
          f"settle/meas {n_win} steps = {a.cycles:g} cycles", flush=True)
    ensure_m0(arr, outdir, run, relax_steps=a.relax_steps, dtype=dtype,
              log=lambda s: print(f"  {s}", flush=True))

    npr = arr.n_readout
    # Drive every disk equally. The per-tap fresh SCALE belongs to the balance
    # question, not this one: here each disk should see the amplitude on the
    # axis so its own threshold is what is read.
    scale = [1.0] * a.n_taps
    body = arr.mask[:, :, 0, 0].to(dtype)

    rows, ref = [], None
    print(f"\n{'amp_mT':>7} " + " ".join(f"{'tap'+str(k+1):>19}" for k in range(a.n_taps)))
    print(f"{'':>7} " + " ".join(f"{'|A|':>10}{'norm':>9}" for _ in range(a.n_taps)),
          flush=True)
    for amp_mT in a.amps_mT:
        t0 = time.time()
        A, m = drive_taps(arr, cfg, amp_mT, a.freq, n_win, n_win, scale, dtype)
        mags = [float(np.abs(A[k*npr:(k+1)*npr]).mean()) for k in range(a.n_taps)]
        per = [g / amp_mT for g in mags]
        if ref is None:
            ref = list(per)
        norm = [p_ / r for p_, r in zip(per, ref)]
        mz = float((m[:, :, 0, 2] * body).abs().max())
        rows.append({"amp_mT": amp_mT, "abs_A": mags, "per_mT": per,
                     "normalised": norm, "core_max_mz": mz,
                     "seconds": round(time.time() - t0, 1)})
        print(f"{amp_mT:>7.1f} " + " ".join(f"{g:>10.3e}{n:>9.3f}"
                                            for g, n in zip(mags, norm)),
              flush=True)
        (outdir / f"threshold_{run}.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'tap':>4} {'threshold':>10}  (first amplitude with |norm-1| > 0.05)")
    THR = 0.05
    ths = []
    for k in range(a.n_taps):
        hit = [r["amp_mT"] for r in rows if abs(r["normalised"][k] - 1.0) > THR]
        th = hit[0] if hit else None
        ths.append(th)
        print(f"{k+1:>4} {(f'{th:.1f} mT' if th else 'none in range'):>10}")

    # The artifact check, run rather than trusted.
    print("\nis |A| still scaling where norm falls? (a flat |A| means the")
    print("measurement saturated, not the element)")
    for k in range(a.n_taps):
        first, last = rows[0]["abs_A"][k], rows[-1]["abs_A"][k]
        da = rows[-1]["amp_mT"] / rows[0]["amp_mT"]
        print(f"  tap {k+1}: |A| x{last/max(first,1e-30):.1f} over a drive "
              f"x{da:.0f}  -> {'ok' if last > 2*first else 'FLAT, SUSPECT'}")
    print(f"\nwrote {outdir / f'threshold_{run}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
