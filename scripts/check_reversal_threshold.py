#!/usr/bin/env python3
"""Is a spiking front end reachable before the disk goes chaotic?

The layered plan wants the input disk to fire -- core reversal as an
annihilating nonlinearity -- while the deep disk holds a long, stable memory.
That only works if there is a drive window where the core actually reverses and
lambda is still negative. Nothing guarantees one exists: aperture measurements
show a driven vortex needs open guides to shed pumped energy, and drive pushes
lambda positive, so "fires" and "stable" are pulling on the same rope.

The old film/Thiele architecture searched for exactly this corner
(check_operating_point.py: lambda < 0, noise/spread << 1, rate > 0) on a model
where a spike erased the only state variable there was. This is the same search
on the micromagnetic disk, where mode populations survive a reversal -- which is
the whole reason the layered idea is worth retrying.

Reports, per drive amplitude:

  core_mz    signed core polarity, +0.85-ish upright, -0.85-ish reversed
  flips      polarity sign changes during the drive: the spike count
  lambda     finite-time exponent, same probe as check_aperture_stability

The verdict wanted is an amplitude with flips > 0 and lambda < 0. If reversal
never appears at the 12 GHz carrier no matter the amplitude, that is informative
rather than a failure: core reversal is driven most efficiently near the
GYROTROPIC mode (sub-GHz), and 12 GHz drives magnons instead. In that case the
front end's nonlinearity has to be three-magnon splitting -- which this disk
already has above threshold, and which does not erase mode populations either.

    python scripts/check_reversal_threshold.py
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
    """Drive in-plane at `freq`; return recorded states and the core trace.

    Same driver as check_aperture_stability so the lambda numbers are
    comparable, plus the polarity trace this script exists to measure.
    """
    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, 0, 0] = disk.disk_only[:, :, 0, 0]
    m = disk.m0.clone()
    if perturb is not None:
        m = m + perturb
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * disk.mask
    states, core = [], []
    inside = disk.disk_only[:, :, 0, 0] > 0.5
    for k in range(n_steps):
        t0 = k * cfg.dt

        def h_drive(theta, t0=t0):
            return unit * (amp * math.sin(2 * math.pi * freq * (t0 + theta * cfg.dt)))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        if k % record == 0:
            states.append(m.clone())
            mz = m[:, :, 0, 2][inside]
            # Signed polarity: the core dominates both extremes, so hi+lo is
            # ~+0.85 upright and ~-0.85 reversed, and near zero only while the
            # core is mid-flip. A bare max() cannot see a reversal at all.
            core.append(float(mz.max()) + float(mz.min()))
    return states, core


def flips(trace, thresh=0.4):
    """Polarity sign changes, ignoring the ambiguous band around zero."""
    sign, n = 0, 0
    for v in trace:
        if abs(v) < thresh:
            continue
        s = 1 if v > 0 else -1
        if sign and s != sign:
            n += 1
        sign = s
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amps-mT", type=float, nargs="+",
                   default=[30, 50, 75, 110, 150])
    p.add_argument("--freq", type=float, default=12.0,
                   help="drive frequency in GHz; 12.0 is the reservoir carrier")
    p.add_argument("--n-ports", type=int, default=6)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--relax-steps", type=int, default=800)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--outdir", default="runs/reversal_threshold")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = PortedVortexConfig(n_ports=args.n_ports)
    rows = []
    print(f"drive {args.freq:.1f} GHz, {args.n_ports} ports, "
          f"{args.steps} steps\n")
    print(f"{'amp_mT':>8} {'core_mz':>9} {'|min|':>8} {'flips':>7} "
          f"{'lambda/ns':>10} {'verdict':>18}")
    for a_mT in args.amps_mT:
        t0 = time.time()
        amp = a_mT * 1e-3 / MU_0
        disk = PortedVortexDisk(cfg, timesteps=args.steps, dtype=dtype)
        disk.relax(steps=args.relax_steps)

        s1, core = drive(disk, args.freq * 1e9, args.steps, amp, dtype)
        g = torch.Generator().manual_seed(0)
        pert = args.eps * torch.randn(disk.m0.shape, generator=g,
                                      dtype=dtype) * disk.mask
        s2, _ = drive(disk, args.freq * 1e9, args.steps, amp, dtype, perturb=pert)
        d = torch.stack([(x - y).norm() for x, y in zip(s1, s2)])
        d0 = float(d[len(d) // 8].clamp_min(1e-30)); dT = float(d[-1])
        lam = math.log(max(dT, 1e-30) / d0) / (args.steps * cfg.dt * 1e9)

        nf = flips(core)
        cmin = min(core); cend = core[-1]
        fires = nf > 0
        stable = lam < 0
        verdict = ("FIRES+STABLE" if fires and stable else
                   "fires, chaotic" if fires else
                   "stable, no fire" if stable else "chaotic, no fire")
        rows.append({"amp_mT": a_mT, "core_end": cend, "core_min": cmin,
                     "flips": nf, "lambda": lam, "fires": fires,
                     "stable": stable, "seconds": round(time.time() - t0, 1)})
        print(f"{a_mT:>8.0f} {cend:>+9.3f} {cmin:>+8.3f} {nf:>7} "
              f"{lam:>+10.3f} {verdict:>18}", flush=True)
        (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if r["fires"] and r["stable"]]
    print()
    if ok:
        print("spiking front end is reachable at: "
              + ", ".join(f"{r['amp_mT']:.0f} mT" for r in ok))
    else:
        anyfire = any(r["fires"] for r in rows)
        print("no drive tested both fires and stays stable.")
        if not anyfire:
            print("  the core never reversed at this frequency. Core reversal is\n"
                  "  driven near the GYROTROPIC mode (sub-GHz), not at a 12 GHz\n"
                  "  magnon carrier -- so retry with --freq in that band before\n"
                  "  concluding the device cannot spike. The alternative front-end\n"
                  "  nonlinearity is three-magnon splitting, which this disk\n"
                  "  already has above threshold.")
    print(f"\nwrote {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
