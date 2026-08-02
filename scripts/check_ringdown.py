#!/usr/bin/env python3
"""What actually sets the disk's memory: damping, or radiation into the ports?

The damping sweep falsified the ring-down model. Predicted memory scaled as
1/(2 pi alpha N_cycles) and demanded a fourfold rise from alpha 0.008 -> 0.002;
measured Jaeger MC went 8.27 -> 8.61, a gain of 0.34 frames, with the r^2 cliff
pinned at lag 8 throughout. The parameter reached the physics -- banners record
ring-down 1.66 -> 6.63 ns and the damping field's minimum tracks alpha exactly
-- and the memory did not move.

The suspect is in the same field. Its MAXIMUM is 0.5 in every run: the absorbing
tapers, never swept. Above the guides' 11 GHz cutoff a driven disk radiates its
state into the ports and the tapers absorb it, a channel set by geometry rather
than by alpha. If that dominates, bulk Gilbert damping is a minority term and
scaling it does almost nothing -- which is exactly what was measured.

This tests it directly instead of inferring it from another 800-frame task run.
Drive, cut the drive, and watch the free decay of the deviation energy. The time
constant is the memory, measured rather than modelled.

The discriminating comparison:

  alpha 0.008 -> 0.002 at 6 ports   damping down 4x
  6 -> 2 -> 1 ports at alpha 0.008  radiating aperture down

Damping-limited predicts tau scales with 1/alpha and ignores port count.
Radiation-limited predicts tau barely moves with alpha and rises sharply as
ports are removed. The certification's whole diagnosis turns on which.

    python scripts/check_ringdown.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk


@torch.no_grad()
def energy_trace(disk, amp, freq, dtype, n_drive, n_free, record=4):
    """Drive, then let go. Returns deviation energy during the free decay."""
    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, 0, 0] = disk.disk_only[:, :, 0, 0]
    inside = disk.disk_only[:, :, 0, 0] > 0.5
    m = disk.m0.clone()

    for k in range(n_drive):
        t0 = k * cfg.dt

        def h_drive(theta, t0=t0):
            return unit * (amp * math.sin(2 * math.pi * freq
                                          * (t0 + theta * cfg.dt)))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)

    zero = (lambda theta: torch.zeros_like(unit))
    out = []
    for k in range(n_free):
        m = disk.rollout.rk4_step(m, disk.h_zero, zero)
        if k % record == 0:
            # Energy measured INSIDE the disk only. Deviation that has left into
            # the guides is gone from the reservoir whether or not a taper has
            # absorbed it yet, so counting the guides would hide the very loss
            # channel this script is here to weigh.
            d = (m - disk.m0)[:, :, 0, :][inside]
            out.append(float((d ** 2).sum()))
    return out


def fit_tau(E, dt_ns, floor_frac=0.02):
    """Exponential time constant of E(t), fitted while it is still decaying."""
    if not E or E[0] <= 0:
        return float("nan")
    e0 = E[0]
    pts = [(i * dt_ns, math.log(v)) for i, v in enumerate(E)
           if v > floor_frac * e0 and v > 0]
    if len(pts) < 8:
        return float("nan")
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    xm = sum(xs) / len(xs); ym = sum(ys) / len(ys)
    den = sum((x - xm) ** 2 for x in xs)
    if den <= 0:
        return float("nan")
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den
    return (-1.0 / slope) if slope < 0 else float("inf")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+",
                   default=["0.008:6", "0.002:6", "0.008:2", "0.008:1"],
                   help="alpha:n_ports pairs")
    p.add_argument("--amp-mT", type=float, default=20.0)
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--drive-steps", type=int, default=600)
    p.add_argument("--free-steps", type=int, default=2400)
    p.add_argument("--relax-steps", type=int, default=800)
    p.add_argument("--outdir", default="runs/ringdown")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    amp = a.amp_mT * 1e-3 / MU_0

    print(f"{a.amp_mT:.0f} mT at {a.freq:.1f} GHz, "
          f"{a.drive_steps} driven + {a.free_steps} free\n")
    print(f"{'alpha':>7} {'ports':>6} {'tau_ns':>9} {'tau_frames':>11} "
          f"{'model_ns':>9} {'meas/model':>11}")
    rows = []
    for spec in a.configs:
        al, npo = spec.split(":")
        al = float(al); npo = int(npo)
        t0 = time.time()
        cfg = PortedVortexConfig(alpha=al, n_ports=npo)
        disk = PortedVortexDisk(cfg, timesteps=a.drive_steps + a.free_steps + 8,
                                dtype=dtype)
        disk.relax(steps=a.relax_steps)
        E = energy_trace(disk, amp, a.freq * 1e9, dtype,
                         a.drive_steps, a.free_steps)
        dt_ns = 4 * cfg.dt * 1e9
        # Energy is amplitude squared, so its decay constant is half the
        # amplitude one the memory model is written in.
        tau = fit_tau(E, dt_ns) * 2.0
        frame_ns = 200 * cfg.dt * 1e9
        model = 1.0 / (al * 2 * math.pi * a.freq * 1e9) * 1e9
        rows.append({"alpha": al, "n_ports": npo, "tau_ns": tau,
                     "tau_frames": tau / frame_ns, "model_ns": model,
                     "ratio": tau / model, "E0": E[0] if E else None,
                     "seconds": round(time.time() - t0, 1)})
        print(f"{al:>7.4f} {npo:>6} {tau:>9.2f} {tau / frame_ns:>11.1f} "
              f"{model:>9.2f} {tau / model:>11.2f}", flush=True)
        (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    print("\ndamping-limited => tau tracks 1/alpha and ignores port count.\n"
          "radiation-limited => tau barely moves with alpha and rises as ports\n"
          "are removed. model_ns is the 1/(2 pi alpha f) prediction the damping\n"
          "sweep falsified; ratio << 1 means something other than alpha is\n"
          "setting the decay.")
    print(f"\nwrote {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
