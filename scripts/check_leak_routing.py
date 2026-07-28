#!/usr/bin/env python3
"""Which port leaks hardest, and does the answer depend on the input?

This is the routing criterion the whole architecture was built around,
measured directly for the first time. Earlier it was a property of a model
with two degrees of freedom per disk; here the disk carries dozens of magnon
modes and the ports physically decompose them (verified 3/3 in
scripts/verify_ports.py), so "which port leaks hardest" is a real observable.

Three questions, in order of how much they matter:

1. Is the leak pattern INPUT-DEPENDENT? Drive with "AB" and "BA" -- identical
   average input spectra -- and see whether the port power distribution
   differs. If it does not, routing carries no information and the criterion
   is useless however elegant.
2. Is it NONLINEAR? The same comparison at low power, where three-magnon
   scattering is off. A difference that survives there is linear filtering,
   not the mode redistribution the concept depends on.
3. Does it EVOLVE? Port powers per time window: a fixed ranking is a static
   splitter; a ranking that migrates is recruitment.

    python scripts/check_leak_routing.py
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
def drive_sequence(disk, order, f_a, f_b, pulse_steps, read_steps, amp, dtype):
    """Two pulses in the given order, then silence; return port signals."""
    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, :, 0] = disk.disk_only[:, :, :, 0]      # drive the disk, not the guides

    freqs = (f_a, f_b) if order == "AB" else (f_b, f_a)
    total = 2 * pulse_steps + read_steps
    m, out = disk.m0.clone(), []
    for k in range(total):
        seg, within = divmod(k, pulse_steps)
        if seg < 2:
            f = freqs[seg]
            env = math.sin(math.pi * within / pulse_steps) ** 2   # smooth on/off
            a = amp * env
        else:
            f, a = 0.0, 0.0
        t0 = k * cfg.dt

        def h_drive(theta, t0=t0, f=f, a=a):
            if a == 0.0:
                return unit * 0.0
            return unit * (a * math.sin(2 * math.pi * f * (t0 + theta * cfg.dt)))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        out.append(disk.port_signals(m))
    return torch.stack(out)                              # (T, n_ports)


def window_power(sig, n_windows=4):
    """Per-port power in successive time windows, normalised within each."""
    T = sig.shape[0]
    edges = np.linspace(0, T, n_windows + 1).astype(int)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        p = sig[a:b].pow(2).mean(dim=0)
        rows.append((p / p.sum().clamp_min(1e-30)).tolist())
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f-a", type=float, default=6.0)
    p.add_argument("--f-b", type=float, default=9.0)
    p.add_argument("--pulse-ns", type=float, default=1.5)
    p.add_argument("--read-ns", type=float, default=3.0)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--low-amp-mT", type=float, default=1.0)
    p.add_argument("--relax-steps", type=int, default=800)
    p.add_argument("--outdir", default="runs/leak_routing")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")   # validated vs float64
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = PortedVortexConfig()
    pulse_steps = int(args.pulse_ns * 1e-9 / cfg.dt)
    read_steps = int(args.read_ns * 1e-9 / cfg.dt)
    disk = PortedVortexDisk(cfg, timesteps=2 * pulse_steps + read_steps, dtype=dtype)
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    print(f"relaxed {time.time()-t0:.0f}s, core mz {float(disk.m0[:,:,0,2].max()):+.3f}",
          flush=True)

    results = {}
    for label, amp_mT in (("high", args.amp_mT), ("low", args.low_amp_mT)):
        per_order = {}
        for order in ("AB", "BA"):
            t0 = time.time()
            sig = drive_sequence(disk, order, args.f_a * 1e9, args.f_b * 1e9,
                                 pulse_steps, read_steps,
                                 amp_mT * 1e-3 / MU_0, dtype)
            tot = sig.pow(2).mean(dim=0)
            frac = (tot / tot.sum().clamp_min(1e-30))
            per_order[order] = {"port_fraction": [round(x, 4) for x in frac.tolist()],
                                "hardest": int(frac.argmax()),
                                "windows": window_power(sig),
                                "seconds": round(time.time() - t0, 1)}
            print(f"  {label} {order}: hardest port {int(frac.argmax())}  "
                  + " ".join(f"{x:.3f}" for x in frac.tolist())
                  + f"  ({time.time()-t0:.0f}s)", flush=True)

        a = torch.tensor(per_order["AB"]["port_fraction"])
        b = torch.tensor(per_order["BA"]["port_fraction"])
        l1 = float((a - b).abs().sum())
        cos = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
        per_order["l1_difference"] = l1
        per_order["cosine"] = cos
        results[label] = per_order
        print(f"{label} ({amp_mT} mT): AB/BA port-pattern L1 {l1:.4f}, cos {cos:.5f}",
              flush=True)

    ratio = results["high"]["l1_difference"] / max(results["low"]["l1_difference"], 1e-12)
    results["nonlinearity_ratio"] = ratio
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\ninput dependence of the leak pattern: L1 {results['high']['l1_difference']:.4f} "
          f"at high power vs {results['low']['l1_difference']:.4f} at low ({ratio:.1f}x)")
    win = results["high"]["AB"]["windows"]
    ranks = [int(np.argmax(w)) for w in win]
    print(f"hardest port per time window (AB, high): {ranks}")
    if ratio > 3 and results["high"]["l1_difference"] > 0.05:
        print("Routing is input-dependent AND nonlinear: which port leaks")
        print("hardest is a function of input history, which is the criterion")
        print("the architecture needs.")
    else:
        print("Routing is NOT usefully input-dependent at this operating point.")
        print("The ports split power by geometry alone, so 'leak hardest' carries")
        print("little information and cannot drive selection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
