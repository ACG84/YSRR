#!/usr/bin/env python3
"""Is the link unstable because it is lossless? And is there a usable window?

Measured on the fully-ported array: no link gives lambda -0.413/ns, adding a
lossless link gives +0.422/ns. The proposed mechanism is that the link is the
only port not terminating in an absorber -- every outward guide is a drain,
the link is a feedback loop into another active disk.

Direct test: put loss inside the link and see whether stability returns. If it
does, feedback is confirmed as the cause. The sweep then asks the practical
question too -- whether any loss is enough to stabilise while the link still
carries far more than the dipolar background, which would be a usable
operating point rather than just a diagnosis.

Non-reciprocity is the fix that would avoid the trade entirely (forward path
intact, return path removed), but interfacial DMI only breaks reciprocity for
k perpendicular to M, and a link along x relaxes to M along x -- so it needs a
transverse bias that would also act on the vortices. Worth doing after this
establishes the mechanism, not before.

    python scripts/sweep_link_loss.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
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
    p.add_argument("--link-alphas", type=float, nargs="+",
                   default=[0.008, 0.02, 0.05, 0.12, 0.30])
    p.add_argument("--outdir", default="runs/link_loss")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / "journal.jsonl"
    done = {}
    if jpath.exists():
        for line in jpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["link_alpha"]] = r
    amp = args.amp_mT * 1e-3 / MU_0

    # dipolar-only reference, for judging whether the link still carries
    ref_path = outdir / "dipolar_ref.json"
    if ref_path.exists():
        ba_dip = json.loads(ref_path.read_text())["BA"]
    else:
        cfg = CoupledPortedConfig(link_width=0.0)
        arr = CoupledPortedArray(cfg, timesteps=args.steps, dtype=dtype)
        arr.relax(steps=args.relax_steps)
        e, _ = drive(arr, args.freq * 1e9, args.steps, amp, dtype)
        ba_dip = float(e[-1, 1] / e[-1, 0].clamp_min(1e-30))
        ref_path.write_text(json.dumps({"BA": ba_dip}))
    print(f"dipolar-only reference B/A = {ba_dip:.6f}\n", flush=True)
    print(f"{'link alpha':>11} {'lambda/ns':>10} {'B/A':>9} {'x dipolar':>10} "
          f"{'verdict':>14}")

    for la in args.link_alphas:
        if la in done:
            r = done[la]
            print(f"{la:>11.3f} {r['lambda']:>+10.3f} {r['BA']:>9.5f} "
                  f"{r['BA']/max(ba_dip,1e-12):>10.1f} "
                  f"{'stable' if r['lambda']<0 else 'CHAOTIC':>14}  (cached)", flush=True)
            continue
        cfg = CoupledPortedConfig(link_alpha=la)
        arr = CoupledPortedArray(cfg, timesteps=args.steps, dtype=dtype)
        arr.relax(steps=args.relax_steps)
        e, s1 = drive(arr, args.freq * 1e9, args.steps, amp, dtype)
        g = torch.Generator().manual_seed(0)
        pert = args.eps * torch.randn(arr.m0.shape, generator=g, dtype=dtype) * arr.mask
        _, s2 = drive(arr, args.freq * 1e9, args.steps, amp, dtype, perturb=pert)
        d = torch.stack([(a - b).norm() for a, b in zip(s1, s2)])
        d0 = float(d[len(d) // 8].clamp_min(1e-30)); dT = float(d[-1])
        lam = math.log(max(dT, 1e-30) / d0) / (args.steps * cfg.dt * 1e9)
        ba = float(e[-1, 1] / e[-1, 0].clamp_min(1e-30))
        rec = {"link_alpha": la, "lambda": lam, "BA": ba}
        with jpath.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{la:>11.3f} {lam:>+10.3f} {ba:>9.5f} {ba/max(ba_dip,1e-12):>10.1f} "
              f"{'stable' if lam < 0 else 'CHAOTIC':>14}", flush=True)

    rows = sorted([json.loads(l) for l in jpath.read_text().splitlines() if l.strip()],
                  key=lambda r: r["link_alpha"])
    usable = [r for r in rows if r["lambda"] < 0 and r["BA"] > 5 * ba_dip]
    print()
    if usable:
        best = max(usable, key=lambda r: r["BA"])
        print(f"Usable window exists: link alpha {best['link_alpha']:.3f} gives "
              f"lambda {best['lambda']:+.3f}/ns with the link still carrying "
              f"{best['BA']/max(ba_dip,1e-12):.1f}x the dipolar background.")
        print("Feedback confirmed as the instability: terminating the loop with")
        print("loss restores stability, and enough coupling survives to use.")
    elif any(r["lambda"] < 0 for r in rows):
        print("Loss does stabilise -- feedback confirmed as the cause -- but by")
        print("the time lambda goes negative the link no longer carries much")
        print("more than the dipolar background. Damping is a diagnosis, not a")
        print("fix; non-reciprocity is the way to keep the forward path.")
    else:
        print("Loss did not stabilise at any value tested, so the feedback")
        print("explanation is wrong and something else destabilises the linked")
        print("array.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
