#!/usr/bin/env python3
"""Aperture is loss, and loss is what makes a driven vortex stable.

The result this exists to record. A single ported disk is stable at every
drive up to 30 mT (lambda -0.464/ns), while the coupled two-disk array was
violently unstable at the same drive (+5.55/ns) with no dependence on
separation at all. The geometries differed in a way that had gone unnoticed:
six ports on the single disk, one outward stub per disk in the array.

Isolating the variable resolves the whole confusion:

  n_ports 1    240 guide cells   lambda +6.494   chaotic
  n_ports 2    480 guide cells   lambda +4.506   chaotic
  n_ports 6   1440 guide cells   lambda -0.668   stable

A driven vortex needs enough open aperture to shed the energy being pumped
into it. That single fact explains why lambda looked separation-independent
(each disk had one stub whatever the distance), why wider apertures helped
(the link is another loss channel), and why two disks looked unstable where
one looked fine (the two-disk build simply gave each disk fewer guides).

Nothing here is about coupling. The instability was a property of a disk with
too little aperture, and the coupled array inherited it by construction.

    python scripts/check_aperture_stability.py
    python scripts/check_aperture_stability.py --ports 1 2 3 4 6 --amp-mT 30
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
def drive(disk, freq, n_steps, amp, dtype, perturb=None, record=8):
    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, 0, 0] = disk.disk_only[:, :, 0, 0]
    m = disk.m0.clone()
    if perturb is not None:
        m = (m + perturb)
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * disk.mask
    out = []
    for k in range(n_steps):
        t0 = k * cfg.dt
        def h_drive(theta, t0=t0):
            return unit * (amp * math.sin(2 * math.pi * freq * (t0 + theta * cfg.dt)))
        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        if k % record == 0:
            out.append(m.clone())
    return out


def record_dt(cfg, args, record=8):
    """ns between recorded states -- the x-axis the exponent is fitted on."""
    return record * cfg.dt * 1e9


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ports", type=int, nargs="+", default=[1, 2, 6])
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--relax-steps", type=int, default=800)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--outdir", default="runs/aperture_stability")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    amp = args.amp_mT * 1e-3 / MU_0
    rows = []

    print(f"{'n_ports':>8} {'guide cells':>12} {'lambda/ns':>10} {'regime':>10}")
    for npo in args.ports:
        cfg = PortedVortexConfig(n_ports=npo)
        disk = PortedVortexDisk(cfg, timesteps=args.steps, dtype=dtype)
        disk.relax(steps=args.relax_steps)
        guide_cells = int((disk.mask - disk.disk_only).sum())

        s1 = drive(disk, args.freq * 1e9, args.steps, amp, dtype)
        g = torch.Generator().manual_seed(0)
        pert = args.eps * torch.randn(disk.m0.shape, generator=g, dtype=dtype) * disk.mask
        s2 = drive(disk, args.freq * 1e9, args.steps, amp, dtype, perturb=pert)
        d = torch.stack([(a - b).norm() for a, b in zip(s1, s2)])
        d0 = float(d[len(d) // 8].clamp_min(1e-30)); dT = float(d[-1])
        T_ns = args.steps * cfg.dt * 1e9
        lam = math.log(max(dT, 1e-30) / d0) / T_ns

        # The two-point exponent above is only an exponent while the
        # perturbation is still growing exponentially. Once the two
        # trajectories decorrelate, d saturates at a ceiling set by the state
        # norm, and log(ceiling/d0)/T reports a number that depends on the
        # transient rather than on any rate -- which is how a lambda series can
        # come out non-monotonic in aperture and mean nothing. So: fit log d
        # over the unsaturated window as well, and record enough to tell the
        # two apart afterwards.
        dv = d.tolist()
        ceil = max(dv)
        lo = len(dv) // 8
        idx = [i for i in range(lo, len(dv)) if dv[i] < 0.2 * ceil and dv[i] > 0]
        lam_fit, n_fit = float("nan"), len(idx)
        if n_fit >= 8:
            xs = [i * record_dt(cfg, args) for i in idx]
            ys = [math.log(dv[i]) for i in idx]
            xm = sum(xs) / len(xs); ym = sum(ys) / len(ys)
            den = sum((x - xm) ** 2 for x in xs)
            if den > 0:
                lam_fit = sum((x - xm) * (y - ym)
                              for x, y in zip(xs, ys)) / den
        sat = dT >= 0.9 * ceil
        rows.append({"n_ports": npo, "guide_cells": guide_cells, "lambda": lam,
                     "lambda_fit": lam_fit, "d_start": d0, "d_end": dT,
                     "d_ceiling": ceil, "saturated": bool(sat),
                     "fit_points": n_fit})
        print(f"{npo:>8} {guide_cells:>12} {lam:>+10.3f} "
              f"{'CHAOTIC' if lam > 0 else 'stable':>10}"
              f"   fit {lam_fit:>+8.3f}  d {d0:.2e}->{dT:.2e}"
              f"{'  SATURATED' if sat else ''}", flush=True)

    (outdir / "results.json").write_text(json.dumps(rows, indent=2))
    stable = [r for r in rows if r["lambda"] < 0]
    if stable:
        need = min(r["guide_cells"] for r in stable)
        print(f"\nStability needs at least ~{need} guide cells of open aperture at "
              f"{args.amp_mT:g} mT.")
        print("Any array built from these disks must give EACH disk that much,")
        print("counting links and readout guides together -- otherwise the")
        print("instability is baked into the geometry before coupling is even")
        print("in question.")
    else:
        print("\nNo port count tested was stable; the drive itself is too high.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
