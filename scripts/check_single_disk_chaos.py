#!/usr/bin/env python3
"""At what drive does ONE vortex disk become chaotic?

The separation sweep refuted the coupling explanation: lambda was +5.55,
+5.49, +5.55 per ns at 500, 800 and 1200 nm with no link at all. A 2.4x
distance change moves an inter-disk dipolar effect by 14x and moved this by
nothing, so the instability is not between disks -- it is inside one.

That reframes everything measured on the array, including the apparent
transfer: with lambda ~ +5.5/ns, noise in the undriven disk amplifies by
e^11 over 2 ns, which is why "B/A" showed no distance trend and reached 1.3
in one cell (disk B holding more energy than the driven disk A).

So: sweep drive amplitude on a SINGLE ported disk and find where lambda
crosses zero. Below that crossing the disk is a usable reservoir; above it,
no readout can generalise however rich the spectra look. The AB/BA
discrimination was measured at 30 mT, so this also tells us whether that
result sits in a computable regime or a chaotic one.

    python scripts/check_single_disk_chaos.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
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
    states = []
    for k in range(n_steps):
        t0 = k * cfg.dt
        def h_drive(theta, t0=t0):
            return unit * (amp * math.sin(2 * math.pi * freq * (t0 + theta * cfg.dt)))
        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        if k % record == 0:
            states.append(m.clone())
    return states


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freq", type=float, default=8.0)
    p.add_argument("--amps-mT", type=float, nargs="+",
                   default=[1, 3, 6, 10, 15, 22, 30])
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--relax-steps", type=int, default=800)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--outdir", default="runs/single_chaos")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / "journal.jsonl"
    done = {}
    if jpath.exists():
        for line in jpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["amp_mT"]] = r

    cfg = PortedVortexConfig()
    disk = PortedVortexDisk(cfg, timesteps=args.steps, dtype=dtype)
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    print(f"relaxed {time.time()-t0:.0f}s, core mz "
          f"{float(disk.m0[:, :, 0, 2].max()):+.3f}\n", flush=True)
    print(f"{'drive mT':>9} {'lambda/ns':>10} {'|dm| start':>11} {'|dm| end':>10} "
          f"{'regime':>12}")

    span = args.steps * cfg.dt * 1e9
    for amp_mT in args.amps_mT:
        if amp_mT in done:
            r = done[amp_mT]
            print(f"{amp_mT:>9.0f} {r['lambda']:>+10.3f} {r['d0']:>11.2e} "
                  f"{r['dT']:>10.2e} {r['regime']:>12}  (cached)", flush=True)
            continue
        amp = amp_mT * 1e-3 / MU_0
        s1 = drive(disk, args.freq * 1e9, args.steps, amp, dtype)
        g = torch.Generator().manual_seed(0)
        pert = args.eps * torch.randn(disk.m0.shape, generator=g, dtype=dtype) * disk.mask
        s2 = drive(disk, args.freq * 1e9, args.steps, amp, dtype, perturb=pert)
        d = torch.stack([(a - b).norm() for a, b in zip(s1, s2)])
        d0 = float(d[len(d) // 8].clamp_min(1e-30)); dT = float(d[-1])
        lam = math.log(max(dT, 1e-30) / d0) / span
        regime = "chaotic" if lam > 0.05 else ("edge" if lam > -0.05 else "stable")
        rec = {"amp_mT": amp_mT, "lambda": lam, "d0": d0, "dT": dT, "regime": regime}
        with jpath.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{amp_mT:>9.0f} {lam:>+10.3f} {d0:>11.2e} {dT:>10.2e} {regime:>12}",
              flush=True)

    rows = sorted([json.loads(l) for l in jpath.read_text().splitlines() if l.strip()],
                  key=lambda r: r["amp_mT"])
    stable = [r for r in rows if r["lambda"] < 0]
    print()
    if stable and len(stable) < len(rows):
        cross_lo = max(r["amp_mT"] for r in stable)
        cross_hi = min(r["amp_mT"] for r in rows if r["lambda"] >= 0)
        print(f"lambda crosses zero between {cross_lo:.0f} and {cross_hi:.0f} mT.")
        print(f"The AB/BA discrimination was measured at 30 mT, "
              f"{'inside' if 30 <= cross_lo else 'ABOVE'} the stable window.")
    elif not stable:
        print("Chaotic at every drive tested, including the lowest -- the")
        print("instability is not a high-power effect and the operating point")
        print("is not the problem.")
    else:
        print("Stable at every drive tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
