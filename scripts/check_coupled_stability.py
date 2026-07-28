#!/usr/bin/env python3
"""Does guide-coupling two disks reintroduce the chaos that killed the array?

The single-disk results are all in isolation, and the Thiele array failed
exactly when disks were coupled: strong coupling near a threshold gave chaos
rather than computation. A guide carrying signal A->B also carries it back, so
the coupled geometry must be checked before any task result from it counts.

Two measurements:

1. TRANSFER. Drive disk A only; watch energy appear in disk B. It must arrive
   AFTER the propagation delay, not before -- an instantaneous response would
   mean the disks are dipole-coupled through the near field rather than
   through the guide, which is the failure mode the guides exist to remove.
2. STABILITY. Perturb the state slightly and track the divergence, at high and
   low drive. Growth means chaos; decay means a usable reservoir. Measured as
   a finite-time exponent so the verdict does not depend on where the
   trajectories eventually saturate.

    python scripts/check_coupled_stability.py
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
def drive(arr, freq, n_steps, amp, dtype, perturb=None, record=4):
    """Drive disk A only. Optional initial perturbation for the stability test."""
    cfg = arr.cfg
    nx, ny = cfg.grid
    unit = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    unit[:, :, 0, 0] = arr.disk_masks[0]              # in-plane, disk A only

    m = arr.m0.clone()
    if perturb is not None:
        m = m + perturb
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * arr.mask

    energies, states = [], []
    for k in range(n_steps):
        t0 = k * cfg.dt
        def h_drive(theta, t0=t0):
            return unit * (amp * math.sin(2 * math.pi * freq * (t0 + theta * cfg.dt)))
        m = arr.rollout.rk4_step(m, arr.h_zero, h_drive)
        if k % record == 0:
            energies.append(arr.disk_energy(m))
            states.append(m.clone())
    return torch.stack(energies), states


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freq", type=float, default=8.0)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--low-amp-mT", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--relax-steps", type=int, default=900)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--outdir", default="runs/coupled")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = CoupledArrayConfig()
    arr = CoupledDiskArray(cfg, timesteps=args.steps, dtype=dtype)
    t0 = time.time(); arr.relax(steps=args.relax_steps)
    cores = [float((arr.m0[:, :, 0, 2] * arr.disk_masks[k]).max()) for k in (0, 1)]
    print(f"relaxed {time.time()-t0:.0f}s, core mz per disk {cores[0]:+.3f} {cores[1]:+.3f}",
          flush=True)

    results = {}
    for label, amp_mT in (("high", args.amp_mT), ("low", args.low_amp_mT)):
        amp = amp_mT * 1e-3 / MU_0
        t0 = time.time()
        e, states = drive(arr, args.freq * 1e9, args.steps, amp, dtype)

        # --- transfer: when does B first rise above its own noise floor?
        eb = e[:, 1]
        floor = float(eb[:5].mean()) + 1e-12
        idx = int(torch.argmax((eb > 20 * floor).to(torch.int8)))
        arrival_ns = idx * 4 * cfg.dt * 1e9 if eb.max() > 20 * floor else float("nan")
        # ballistic lower bound: link length / group velocity ceiling
        link_nm = cfg.separation * 1e9
        ratio = float(e[-1, 1] / e[-1, 0].clamp_min(1e-30))

        # --- stability: perturbed twin, same drive
        g = torch.Generator().manual_seed(0)
        pert = args.eps * torch.randn(arr.m0.shape, generator=g, dtype=dtype) * arr.mask
        e2, states2 = drive(arr, args.freq * 1e9, args.steps, amp, dtype, perturb=pert)
        d = torch.stack([ (a - b).norm() for a, b in zip(states, states2) ])
        d0 = float(d[len(d)//8].clamp_min(1e-30))
        dT = float(d[-1])
        span_ns = args.steps * cfg.dt * 1e9
        lam = math.log(max(dT, 1e-30) / d0) / span_ns          # per ns
        results[label] = {"amp_mT": amp_mT, "arrival_ns": arrival_ns,
                          "B_over_A": ratio, "lambda_per_ns": lam,
                          "d_start": d0, "d_end": dT,
                          "seconds": round(time.time() - t0, 1)}
        print(f"{label} ({amp_mT} mT): B/A energy {ratio:.4f}, first arrival "
              f"{arrival_ns:.2f} ns, lambda {lam:+.3f}/ns "
              f"(|dm| {d0:.2e} -> {dT:.2e})", flush=True)

    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    hi = results["high"]
    print(f"\nlink is {cfg.separation*1e9:.0f} nm; arrival at "
          f"{hi['arrival_ns']:.2f} ns implies {cfg.separation/max(hi['arrival_ns'],1e-9)*1e9:.0f} m/s")
    if hi["lambda_per_ns"] < 0:
        print("Perturbations DECAY in the coupled array: guide coupling did not")
        print("reintroduce the instability that killed the Thiele array.")
    else:
        print("Perturbations GROW: the coupled array is chaotic at this drive,")
        print("and no task result from it should be trusted until the coupling")
        print("or the operating point is changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
