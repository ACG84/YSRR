#!/usr/bin/env python3
"""How fast does a wavepacket actually travel down the guide?

This decides whether a deeper disk chain fixes the memory problem, and the two
plausible answers imply completely different machines.

The reservoir holds ~3 lags where NARMA-10 needs 10. A chain of disks is a
tapped delay line -- disk k sees the input delayed by k transit times -- so the
question is how much delay one hop buys. Two estimates disagree by 3.4x:

    v_g = 2 * f * lambda = 1968 m/s     naive, treating the mode as exchange-
                                        dominated where omega ~ k^2
    v_g = L_d * alpha * omega = 584 m/s inferred from the MEASURED decay length
                                        (968 nm) assuming it is damping-limited

At 584 m/s a 700 nm hop is 1.2 ns -- six frames -- and a three-disk chain
covers NARMA-10 with amplitude to spare. At 1968 m/s a hop is 0.35 ns, under
two frames, and ten frames needs six hops at 0.48 transmission each, leaving
1% of the signal. Worth measuring rather than assuming.

Method: a Gaussian-enveloped tone burst at one end, then track the ENVELOPE
peak past a series of gates down the guide. Group velocity is the slope of
arrival time against distance, which is what a delay line actually delivers --
phase velocity would give the wrong answer and is what the naive estimate
above is really measuring.

The first arrival is taken at a threshold crossing rather than at the envelope
maximum, because the magnetostatic field arrives instantaneously at these
scales and would otherwise be timed as an infinitely fast wave -- which is
exactly what the coupled run reported (17,500 m/s over 700 nm) through a link
that was below its own cutoff.

    python scripts/check_group_velocity.py
    python scripts/check_group_velocity.py --freqs 12 14 18
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0, MeshConfig, SolverConfig
from magnonic_nn.solver import LLGRollout

ALPHA_BULK, ALPHA_ABS = 0.008, 0.5


@torch.no_grad()
def time_of_flight(freq, width_nm=80.0, thickness_nm=20.0, length_nm=1600.0,
                   absorber_nm=300.0, steps=6000, amp_mT=3.0, dx=5e-9,
                   margin=6, burst_ps=300.0, dtype=torch.float32):
    """Return (gate positions nm, arrival times ns, fitted v_g m/s)."""
    nx = int(round(length_nm * 1e-9 / dx)) + 2 * margin
    ny = int(round(width_nm * 1e-9 / dx)) + 2 * margin
    x = (torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2) * dx
    y = (torch.arange(ny, dtype=torch.float64) - (ny - 1) / 2) * dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    L = length_nm * 1e-9
    mask = ((X.abs() <= L / 2) & (Y.abs() <= width_nm * 1e-9 / 2)).to(dtype)
    n_wide = int(mask[nx // 2].sum().item())
    mask = mask.reshape(nx, ny, 1, 1)

    mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=dx, dy=dx, dz=thickness_nm * 1e-9)
    solver = SolverConfig(dt=1e-12, timesteps=steps, checkpoint=False,
                          renormalize=True, demag=True)
    n_abs = int(round(absorber_nm * 1e-9 / dx))
    alpha = torch.full((nx, ny, 1, 1), ALPHA_BULK, dtype=dtype)
    if n_abs > 0:
        ramp = torch.linspace(0.0, 1.0, n_abs, dtype=dtype) ** 2
        a0 = nx - margin - n_abs
        alpha[a0:nx - margin, :, 0, 0] = (
            ALPHA_BULK + ramp[:, None] * (ALPHA_ABS - ALPHA_BULK))

    roll = LLGRollout(mesh, solver, A=1.3e-11, alpha=alpha, Ms_ref=800e3)
    roll.set_Ms(800e3 * mask)
    h0 = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    m = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    m[:, :, 0, 0] = 1.0
    m0 = roll.relax(m * mask, h0, 900, 0.5)

    src = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    src[margin:margin + 3, :, 0, 2] = mask[margin:margin + 3, :, 0, 0]
    amp = amp_mT * 1e-3 / MU_0
    t0_ps, sig_ps = burst_ps * 0.9, burst_ps / 4.0

    mm = m0.clone()
    prof = torch.zeros(steps, nx, dtype=torch.float64)
    for k in range(steps):
        tk = k * 1e-12

        def h(theta, tk=tk):
            t = (tk + theta * 1e-12) * 1e12
            env = math.exp(-0.5 * ((t - t0_ps) / sig_ps) ** 2)
            return src * (amp * env * math.sin(2 * math.pi * freq
                                               * (tk + theta * 1e-12)))

        mm = roll.rk4_step(mm, h0, h)
        prof[k] = (mm - m0)[:, :, 0, 2].double().sum(dim=1) / max(n_wide, 1)

    # Envelope via the analytic signal along time, per gate.
    P = prof.numpy()
    lo = margin + int(round(120e-9 / dx))
    hi = nx - margin - n_abs - int(round(40e-9 / dx))
    gates = np.arange(lo, hi, max(int(round(80e-9 / dx)), 1))
    from numpy.fft import fft, ifft
    xs, ts = [], []
    for g in gates:
        s = P[:, g]
        n = len(s)
        S = fft(s)
        Hf = np.zeros(n); Hf[0] = 1; Hf[1:n // 2] = 2; Hf[n // 2] = 1
        env = np.abs(ifft(S * Hf))
        pk = env.max()
        if pk < 1e-9:
            continue
        # threshold crossing, not the peak: the magnetostatic field arrives
        # instantaneously and inflates any peak-based estimate
        idx = np.argmax(env > 0.25 * pk)
        if idx == 0:
            continue
        xs.append((g - (margin + 1)) * dx * 1e9)
        ts.append(idx * 1e-3)                       # ps -> ns
    if len(xs) < 4:
        return np.array(xs), np.array(ts), float("nan")
    xs, ts = np.array(xs), np.array(ts)
    slope = np.polyfit(xs, ts, 1)[0]                # ns per nm
    vg = 1e-9 / (slope * 1e-9) if abs(slope) > 1e-12 else float("inf")
    return xs, ts, float(vg)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freqs", type=float, nargs="+", default=[12.0, 14.0, 18.0])
    p.add_argument("--width", type=float, default=80.0)
    p.add_argument("--thickness", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--frame-ns", type=float, default=0.2)
    p.add_argument("--hop-nm", type=float, default=700.0)
    p.add_argument("--outdir", default="runs/group_velocity")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"{'GHz':>5} {'v_g (m/s)':>11} {'hop delay':>11} {'frames/hop':>11}")
    rows = []
    for f in args.freqs:
        xs, ts, vg = time_of_flight(f * 1e9, args.width, args.thickness,
                                    steps=args.steps)
        hop_ns = (args.hop_nm * 1e-9 / vg) * 1e9 if np.isfinite(vg) and vg > 0 \
            else float("nan")
        frames = hop_ns / args.frame_ns
        rows.append({"ghz": f, "v_g": vg, "hop_ns": hop_ns,
                     "frames_per_hop": frames,
                     "gates_nm": xs.tolist(), "arrival_ns": ts.tolist()})
        print(f"{f:>5.0f} {vg:>11.0f} {hop_ns:>9.2f} ns {frames:>11.1f}",
              flush=True)
    (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if np.isfinite(r["frames_per_hop"])]
    print()
    if ok:
        best = max(ok, key=lambda r: r["frames_per_hop"])
        need = 10.0 / best["frames_per_hop"]
        print(f"At {best['ghz']:.0f} GHz one {args.hop_nm:.0f} nm hop is "
              f"{best['frames_per_hop']:.1f} frames.")
        print(f"NARMA-10 needs 10 lags, so {math.ceil(need)} hops "
              f"({math.ceil(need)+1} disks) would span it.")
        surv = 0.48 ** math.ceil(need)
        print(f"Amplitude surviving that many hops at 0.48 each: {surv:.3f}.")
        if surv < 0.02:
            print("That is below what a 300 K readout can be expected to")
            print("resolve -- the chain would deliver delay the ports cannot")
            print("see, so shorter hops or a lower-damping material comes first.")
    else:
        print("No gate produced a clean arrival; lengthen the guide or raise")
        print("the drive before drawing conclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
