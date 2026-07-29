#!/usr/bin/env python3
"""Coupling, measured on a geometry that is stable by construction.

Every earlier coupled measurement was taken on disks with one outward stub
each -- 240 guide cells, against the ~1440 that scripts/check_aperture_stability.py
showed a disk needs at 30 mT. Those runs were unstable before coupling was
even in question, so their transfer numbers meant nothing.

Here each disk keeps its six ports and the link is one of them. Same three
questions as before, now answerable:

1. STABILITY. Is lambda negative with the budget met? If not, nothing else
   in this script is worth reading.
2. TRANSFER. Drive disk A; does B respond, and when? The link is 700 nm of
   guide, so guided energy cannot arrive faster than the group velocity
   allows. An instantaneous response is the dipolar field, which no geometry
   removes.
3. LINK CONTRIBUTION. The same run with the link deleted is the differential
   control: whatever B still receives is dipolar, and the difference is what
   the guide actually carries.

    python scripts/check_coupled_ported.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import CoupledPortedConfig, CoupledPortedArray


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
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--outdir", default="runs/coupled_ported")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    amp = args.amp_mT * 1e-3 / MU_0
    results = {}

    for label, link_nm in (("linked", 40.0), ("no-link", 0.0)):
        cfg = CoupledPortedConfig(separation=args.separation * 1e-9,
                                  link_width=link_nm * 1e-9)
        arr = CoupledPortedArray(cfg, timesteps=args.steps, dtype=dtype)
        t0 = time.time(); arr.relax(steps=args.relax_steps)
        cores = [float((arr.m0[:, :, 0, 2] * arr.disk_masks[k]).max()) for k in (0, 1)]

        e, s1 = drive(arr, args.freq * 1e9, args.steps, amp, dtype)
        g = torch.Generator().manual_seed(0)
        pert = args.eps * torch.randn(arr.m0.shape, generator=g, dtype=dtype) * arr.mask
        _, s2 = drive(arr, args.freq * 1e9, args.steps, amp, dtype, perturb=pert)
        d = torch.stack([(a - b).norm() for a, b in zip(s1, s2)])
        d0 = float(d[len(d) // 8].clamp_min(1e-30)); dT = float(d[-1])
        lam = math.log(max(dT, 1e-30) / d0) / (args.steps * cfg.dt * 1e9)

        eb = e[:, 1]; floor = float(eb[:4].mean()) + 1e-20
        above = (eb > 20 * floor).nonzero()
        arrival = float(above[0]) * 8 * cfg.dt * 1e9 if len(above) else float("nan")
        ba = float(e[-1, 1] / e[-1, 0].clamp_min(1e-30))

        results[label] = {"lambda": lam, "BA": ba, "arrival_ns": arrival,
                          "guide_cells_per_disk": arr.guide_cells_per_disk,
                          "cores": cores, "seconds": round(time.time() - t0, 1)}
        print(f"{label:>8}: lambda {lam:+.3f}/ns  B/A {ba:.5f}  arrival "
              f"{arrival:.3f} ns  ({arr.guide_cells_per_disk} guide cells/disk)",
              flush=True)

    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    lk, nl = results["linked"], results["no-link"]
    print()
    if lk["lambda"] >= 0:
        print(f"Still unstable (lambda {lk['lambda']:+.3f}/ns) even with the aperture")
        print("budget met -- the transfer numbers below are not trustworthy.")
        return 0
    print(f"Stable with the budget met: lambda {lk['lambda']:+.3f}/ns.")
    gain = lk["BA"] / max(nl["BA"], 1e-12)
    print(f"B/A with link {lk['BA']:.5f} vs dipolar-only {nl['BA']:.5f} "
          f"-- the guide carries {gain:.1f}x")
    if not math.isnan(lk["arrival_ns"]) and lk["arrival_ns"] > 0:
        v = args.separation * 1e-9 / (lk["arrival_ns"] * 1e-9)
        print(f"first arrival {lk['arrival_ns']:.3f} ns over {args.separation:.0f} nm "
              f"implies {v:.0f} m/s")
        print("(a magnon group velocity is ~1e3 m/s; far above that is the dipolar")
        print(" field, which is instantaneous at these scales)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
