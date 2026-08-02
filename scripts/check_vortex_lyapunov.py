#!/usr/bin/env python3
"""Maximal Lyapunov exponent of the ported vortex disk, measured properly.

Why this exists. check_aperture_stability.py estimates lambda from TWO points:
the twin-trajectory distance an eighth of the way in, and at the end,
log(dT/d0)/T. That is an exponent only while the perturbation is still growing
exponentially. Once the twins decorrelate, d saturates at a ceiling set by the
state norm and the formula reports log(ceiling/d0)/T -- a number that moves with
the transient rather than with any rate.

It is not a hypothetical failure. Re-running that script at its own defaults:

    n_ports 6   published -0.668   re-measured +0.862   (sign flip)
    n_ports 2   published +4.506   re-measured +2.774

both flagged saturated, and the 6-port case had no exponential window at all --
its perturbation grew 4x and plateaued, which is a stable system settling, read
as growth because d0 was sampled before the plateau. The aperture series that
conclusion rests on (1: +6.494, 2: +4.506, 6: -0.668) is not reproducible, and
a series measured this way cannot decide stability either way.

This project already knows the fix; check_lyapunov.py states it exactly --
"perturb by eps, integrate one frame, measure the growth factor, rescale back
to eps, repeat" -- but implements it for the Thiele model that the
micromagnetic disk replaced. This is that method for the vortex.

Renormalising every interval keeps the twins inside the linear regime, so what
is averaged is a growth RATE and saturation never enters. Run in float64:
the perturbation is deliberately tiny, and at float32 a 1e-6 separation sits
only ~10x above machine epsilon, so the growth factor would be measuring
rounding.

    python scripts/check_vortex_lyapunov.py --ports 6 4 2
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk


def unit_field(disk, dtype):
    n = disk.cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, 0, 0] = disk.disk_only[:, :, 0, 0]
    return unit


def renorm(m, mask):
    """Back onto the unit sphere, and zero outside the patterned element."""
    return m / m.norm(dim=-1, keepdim=True).clamp_min(1e-30) * mask


@torch.no_grad()
def step_n(disk, m, unit, amp, freq, k0, n_steps):
    cfg = disk.cfg
    for k in range(n_steps):
        t0 = (k0 + k) * cfg.dt

        def h_drive(theta, t0=t0):
            return unit * (amp * math.sin(2 * math.pi * freq
                                          * (t0 + theta * cfg.dt)))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
    return m


@torch.no_grad()
def lyapunov(disk, amp, freq, dtype, warmup=200, interval=100, n_int=14,
             eps_frac=1e-8, seed=0):
    """Benettin: grow for `interval` steps, log the factor, rescale, repeat."""
    cfg = disk.cfg
    unit = unit_field(disk, dtype)
    mask = disk.mask

    a = disk.m0.clone()
    a = step_n(disk, a, unit, amp, freq, 0, warmup)     # onto the attractor
    k = warmup

    eps = eps_frac * float(a.norm())
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(a.shape, generator=g, dtype=dtype) * mask
    p = p / p.norm().clamp_min(1e-30) * eps
    b = renorm(a + p, mask)

    logs, ds = [], []
    for _ in range(n_int):
        a = step_n(disk, a, unit, amp, freq, k, interval)
        b = step_n(disk, b, unit, amp, freq, k, interval)
        k += interval
        d = float((b - a).norm())
        if d <= 0:
            break
        logs.append(math.log(d / eps))
        ds.append(d / eps)
        b = renorm(a + (b - a) * (eps / d), mask)       # back to the linear regime

    T_ns = interval * cfg.dt * 1e9
    lam = (sum(logs) / len(logs) / T_ns) if logs else float("nan")
    # Second half only: if the two agree the estimate has converged, if they
    # disagree the warmup was too short and the number is a transient.
    h = len(logs) // 2
    lam_late = (sum(logs[h:]) / len(logs[h:]) / T_ns) if logs[h:] else float("nan")
    return lam, lam_late, ds


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ports", type=int, nargs="+", default=[6, 4, 2])
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=8.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--interval", type=int, default=100)
    p.add_argument("--intervals", type=int, default=14)
    p.add_argument("--relax-steps", type=int, default=800)
    p.add_argument("--eps-frac", type=float, default=1e-8)
    p.add_argument("--outdir", default="runs/vortex_lyapunov")
    a = p.parse_args()

    mnn.set_precision("float64"); mnn.set_device("cpu")
    dtype = torch.float64
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    amp = a.amp_mT * 1e-3 / MU_0
    total = a.warmup + a.interval * a.intervals

    print(f"{a.amp_mT:.0f} mT, {a.freq:.1f} GHz, float64, "
          f"{a.intervals} x {a.interval} steps after {a.warmup} warmup\n")
    print(f"{'n_ports':>8} {'lambda/ns':>10} {'late half':>10} "
          f"{'growth/interval':>16} {'regime':>9}")
    rows = []
    for npo in a.ports:
        t0 = time.time()
        cfg = PortedVortexConfig(n_ports=npo)
        disk = PortedVortexDisk(cfg, timesteps=total + 8, dtype=dtype)
        disk.relax(steps=a.relax_steps)
        lam, lam_late, ds = lyapunov(disk, amp, a.freq * 1e9, dtype,
                                     warmup=a.warmup, interval=a.interval,
                                     n_int=a.intervals, eps_frac=a.eps_frac)
        gm = (sum(ds) / len(ds)) if ds else float("nan")
        rows.append({"n_ports": npo, "lambda": lam, "lambda_late": lam_late,
                     "mean_growth": gm, "growths": ds,
                     "seconds": round(time.time() - t0, 1)})
        print(f"{npo:>8} {lam:>+10.3f} {lam_late:>+10.3f} {gm:>16.4f} "
              f"{'CHAOTIC' if lam > 0 else 'stable':>9}", flush=True)
        (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    print("\ngrowth factor per interval: >1 expands, <1 contracts. A converged\n"
          "estimate has lambda and its late half agreeing; if they do not, the\n"
          "warmup was too short and the exponent is still a transient.")
    print(f"\nwrote {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
