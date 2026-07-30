#!/usr/bin/env python3
"""Two ways out of the 3x layout mismatch, measured against each other.

The array spans 8200 nm; film1's twelve probes span 2750 nm. A ported disk is
500 nm across against a 250 nm probe pitch, so disks cannot sit one-per-probe.

OPTION 3 -- shorten the guides. Directly shrinks the footprint, but guide area
is what buys stability: 1/2/6 ports measured +6.49/+4.51/-0.67 per ns at 30 mT,
and shortening cuts area the same way removing guides does. So this sweeps
guide length and finds the shortest guide that still holds lambda < 0. The
answer is the minimum disk footprint, which sets how much fan-out is needed.

OPTION 2 -- fan out. Keep the guides and route each probe outward to its disk.
Costs a bend per channel, and a bent magnonic guide radiates: the question is
how much. Measured as transmission through an S-bend against a straight guide
of the SAME path length, so the comparison isolates the bend rather than the
extra distance.

    python scripts/check_layout_options.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0, MeshConfig, SolverConfig
from magnonic_nn.solver import LLGRollout
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk


# ------------------------------------------------------------------ option 3
@torch.no_grad()
def drive_disk(disk, freq, n_steps, amp, dtype, perturb=None, record=8):
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
        def h(theta, t0=t0):
            return unit * (amp * math.sin(2 * math.pi * freq * (t0 + theta * cfg.dt)))
        m = disk.rollout.rk4_step(m, disk.h_zero, h)
        if k % record == 0:
            out.append(m.clone())
    return out


def guide_length_sweep(lengths, amp_mT, freq, steps, dtype):
    rows = []
    print(f"{'guide nm':>9} {'footprint nm':>13} {'guide cells':>12} "
          f"{'lambda/ns':>10} {'verdict':>10}")
    for L in lengths:
        cfg = PortedVortexConfig(guide_length=L * 1e-9)
        disk = PortedVortexDisk(cfg, timesteps=steps, dtype=dtype)
        disk.relax(steps=1200)
        cells = int((disk.mask - disk.disk_only).sum())
        foot = 2 * (cfg.radius + cfg.guide_length) * 1e9
        amp = amp_mT * 1e-3 / MU_0
        s1 = drive_disk(disk, freq, steps, amp, dtype)
        g = torch.Generator().manual_seed(0)
        pert = 1e-5 * torch.randn(disk.m0.shape, generator=g, dtype=dtype) * disk.mask
        s2 = drive_disk(disk, freq, steps, amp, dtype, perturb=pert)
        d = torch.stack([(a - b).norm() for a, b in zip(s1, s2)])
        d0 = float(d[len(d) // 8].clamp_min(1e-30)); dT = float(d[-1])
        lam = math.log(max(dT, 1e-30) / d0) / (steps * cfg.dt * 1e9)
        rows.append({"guide_nm": L, "footprint_nm": foot, "cells": cells, "lambda": lam})
        print(f"{L:>9.0f} {foot:>13.0f} {cells:>12} {lam:>+10.3f} "
              f"{'stable' if lam < 0 else 'CHAOTIC':>10}", flush=True)
    return rows


# ------------------------------------------------------------------ option 2
def strip_geometry(offset_nm, length_nm, width_nm=40.0, dx=5e-9, margin=6,
                   thickness_nm=20.0):
    """A guide of fixed path length, straight or S-bent by ``offset_nm``."""
    nx = int(round(length_nm * 1e-9 / dx)) + 2 * margin
    span = abs(offset_nm) * 1e-9 + width_nm * 1e-9
    ny = int(round(span / dx)) + 2 * margin
    x = (torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2) * dx
    y = (torch.arange(ny, dtype=torch.float64) - (ny - 1) / 2) * dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    L = length_nm * 1e-9
    # centreline: raised cosine S-bend, zero offset gives a straight guide
    u = ((X + L / 2) / L).clamp(0.0, 1.0)
    centre = (offset_nm * 1e-9) * 0.5 * (1 - torch.cos(math.pi * u))
    mask = (X.abs() <= L / 2) & ((Y - centre).abs() <= width_nm * 1e-9 / 2)
    return mask.to(torch.float64).reshape(nx, ny, 1, 1), (nx, ny), centre


@torch.no_grad()
def bend_transmission(offset_nm, length_nm, freq, steps, dtype, amp_mT=5.0,
                      thickness_nm=20.0, width_nm=40.0):
    """Power at the far end of the guide, relative to the near end."""
    mask, (nx, ny), centre = strip_geometry(offset_nm, length_nm,
                                            width_nm=width_nm,
                                            thickness_nm=thickness_nm)
    mask = mask.to(dtype); dx = 5e-9
    mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=dx, dy=dx,
                      dz=thickness_nm * 1e-9)
    solver = SolverConfig(dt=1e-12, timesteps=steps, checkpoint=False,
                          renormalize=True, demag=True)
    alpha = torch.full((nx, ny, 1, 1), 0.008, dtype=dtype)
    roll = LLGRollout(mesh, solver, A=1.3e-11, alpha=alpha, Ms_ref=800e3)
    roll.set_Ms(800e3 * mask)
    h0 = torch.zeros(nx, ny, 1, 3, dtype=dtype)

    m = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    m[:, :, 0, 0] = 1.0                      # along the guide, shape anisotropy
    m = m * mask
    m0 = roll.relax(m, h0, 900, 0.5)

    src = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    src[6:9, :, 0, 2] = mask[6:9, :, 0, 0]   # transverse drive at the near end
    near, far = slice(12, 15), slice(nx - 15, nx - 12)
    amp = amp_mT * 1e-3 / MU_0
    pn = pf = 0.0
    mm = m0.clone()
    for k in range(steps):
        t0 = k * 1e-12
        def h(theta, t0=t0):
            return src * (amp * math.sin(2 * math.pi * freq * (t0 + theta * 1e-12)))
        mm = roll.rk4_step(mm, h0, h)
        if k > steps // 3:                    # after the wave has filled the guide
            d = (mm - m0)[:, :, 0, 2] ** 2
            pn += float(d[near].sum()); pf += float(d[far].sum())
    return pf / max(pn, 1e-30)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lengths", type=float, nargs="+", default=[150, 110, 80, 55])
    p.add_argument("--offsets", type=float, nargs="+", default=[0, 150, 300, 450])
    p.add_argument("--bend-length", type=float, default=700.0)
    p.add_argument("--outdir", default="runs/layout")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print("OPTION 3 -- shorten the guides\n")
    rows = guide_length_sweep(args.lengths, args.amp_mT, args.freq * 1e9,
                              args.steps, dtype)
    stable = [r for r in rows if r["lambda"] < 0]
    min_foot = min((r["footprint_nm"] for r in stable), default=None)

    print("\nOPTION 2 -- fan out with an S-bend "
          f"({args.bend_length:.0f} nm path, same length for every offset)\n")
    print(f"{'offset nm':>10} {'transmission':>13} {'vs straight':>12}")
    bends = []
    for off in args.offsets:
        t = bend_transmission(off, args.bend_length, args.freq * 1e9,
                              args.steps, dtype)
        bends.append({"offset_nm": off, "transmission": t})
        base = bends[0]["transmission"]
        print(f"{off:>10.0f} {t:>13.4f} {t / max(base, 1e-30):>11.2f}x", flush=True)

    (outdir / "results.json").write_text(json.dumps(
        {"guide_lengths": rows, "bends": bends}, indent=2))

    print()
    if min_foot:
        need = 250.0
        print(f"Option 3: smallest stable footprint {min_foot:.0f} nm against a "
              f"{need:.0f} nm probe pitch.")
        if min_foot <= need:
            print("  Fits one-per-probe. No fan-out needed.")
        else:
            print(f"  Still {min_foot / need:.1f}x too wide -- shortening alone "
                  f"does not close the gap.")
    else:
        print("Option 3: no guide length tested was stable; shortening is not viable.")
    if bends:
        worst = min(b["transmission"] for b in bends)
        base = bends[0]["transmission"]
        print(f"Option 2: worst bend passes {worst / max(base, 1e-30):.2f}x of straight.")
        print("  Fan-out is cheap if that ratio is near 1, and the dominant cost")
        print("  if it is not -- twelve channels each paying it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
