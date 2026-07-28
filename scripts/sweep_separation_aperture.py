#!/usr/bin/env python3
"""Separation and aperture: can either buy stability without killing the drive?

The coupled array is chaotic at the drive that makes a single disk useful
(lambda +2.716/ns at 30 mT). Two physical levers:

* SEPARATION. The instantaneous dipolar coupling -- which the guides never
  removed, and which is what B actually responds to at 40 ps -- falls as
  1/r^3. Doubling the distance cuts it eightfold.
* APERTURE. The link width is a controllable loss channel: energy leaving a
  disk through it is effective damping on the collective mode, so a wider
  aperture should stabilise without reducing the drive. That is the knob the
  earlier squeeze never had.

Aperture zero is included deliberately: with no link at all, anything disk B
still receives is dipolar, which is the differential control the previous
transfer measurement lacked.

Journalled per cell -- this container reclaims on idle.

    python scripts/sweep_separation_aperture.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import CoupledArrayConfig, CoupledDiskArray


@torch.no_grad()
def drive(arr, freq, n_steps, amp, dtype, perturb=None, record=8):
    cfg = arr.cfg
    nx, ny = cfg.grid
    unit = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    unit[:, :, 0, 0] = arr.disk_masks[0]
    m = arr.m0.clone()
    if perturb is not None:
        m = (m + perturb)
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * arr.mask
    energies, states = [], []
    for k in range(n_steps):
        t0 = k * cfg.dt
        def h_drive(theta, t0=t0):
            return unit * (amp * math.sin(2 * math.pi * freq * (t0 + theta * cfg.dt)))
        m = arr.rollout.rk4_step(m, arr.h_zero, h_drive)
        if k % record == 0:
            energies.append(arr.disk_energy(m)); states.append(m.clone())
    return torch.stack(energies), states


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freq", type=float, default=8.0)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--relax-steps", type=int, default=900)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--separations", type=float, nargs="+", default=[500, 800, 1200])
    p.add_argument("--apertures", type=float, nargs="+", default=[0, 20, 40, 80])
    p.add_argument("--outdir", default="runs/sep_aperture")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / "journal.jsonl"
    done = {}
    if jpath.exists():
        for line in jpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[(r["sep_nm"], r["ap_nm"])] = r

    print(f"{'sep nm':>7} {'ap nm':>6} {'lambda/ns':>10} {'B/A':>9} "
          f"{'cells':>7} {'verdict':>12}")
    for sep in args.separations:
        for ap in args.apertures:
            key = (sep, ap)
            if key in done:
                r = done[key]
                print(f"{sep:>7.0f} {ap:>6.0f} {r['lambda']:>+10.3f} {r['BA']:>9.5f} "
                      f"{r['cells']:>7} {'stable' if r['lambda']<0 else 'CHAOTIC':>12}"
                      "  (cached)", flush=True)
                continue
            cfg = CoupledArrayConfig(separation=sep * 1e-9, link_width=ap * 1e-9)
            arr = CoupledDiskArray(cfg, timesteps=args.steps, dtype=dtype)
            arr.relax(steps=args.relax_steps)
            amp = args.amp_mT * 1e-3 / MU_0
            _, s1 = drive(arr, args.freq * 1e9, args.steps, amp, dtype)
            g = torch.Generator().manual_seed(0)
            pert = args.eps * torch.randn(arr.m0.shape, generator=g, dtype=dtype) * arr.mask
            e2, s2 = drive(arr, args.freq * 1e9, args.steps, amp, dtype, perturb=pert)
            d = torch.stack([(a - b).norm() for a, b in zip(s1, s2)])
            d0 = float(d[len(d) // 8].clamp_min(1e-30)); dT = float(d[-1])
            span = args.steps * cfg.dt * 1e9
            lam = math.log(max(dT, 1e-30) / d0) / span
            ba = float(e2[-1, 1] / e2[-1, 0].clamp_min(1e-30))
            cells = int(arr.mask.sum())
            rec = {"sep_nm": sep, "ap_nm": ap, "lambda": lam, "BA": ba, "cells": cells}
            with jpath.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{sep:>7.0f} {ap:>6.0f} {lam:>+10.3f} {ba:>9.5f} {cells:>7} "
                  f"{'stable' if lam < 0 else 'CHAOTIC':>12}", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    stable = [r for r in rows if r["lambda"] < 0]
    print(f"\n{len(stable)}/{len(rows)} configurations stable at {args.amp_mT} mT")
    if stable:
        best = max(stable, key=lambda r: r["BA"])
        print(f"most strongly coupled stable point: {best['sep_nm']:.0f} nm "
              f"separation, {best['ap_nm']:.0f} nm aperture "
              f"(lambda {best['lambda']:+.3f}/ns, B/A {best['BA']:.5f})")
    else:
        print("No stable configuration -- neither separation nor aperture buys")
        print("stability at this drive, and the operating point has to move.")
    for sep in args.separations:
        r0 = next((r for r in rows if r["sep_nm"] == sep and r["ap_nm"] == 0), None)
        r40 = next((r for r in rows if r["sep_nm"] == sep and r["ap_nm"] == 40), None)
        if r0 and r40:
            print(f"  {sep:.0f} nm: dipolar-only B/A {r0['BA']:.5f} vs "
                  f"with 40 nm link {r40['BA']:.5f}  "
                  f"({r40['BA'] / max(r0['BA'], 1e-12):.1f}x from the link)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
