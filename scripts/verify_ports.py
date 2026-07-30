#!/usr/bin/env python3
"""Do the waveguide ports actually perform the azimuthal mode decomposition?

The claim behind the ported geometry: a port at angle theta couples to
azimuthal mode n with weight exp(i n theta), so N evenly spaced ports sample
the disk's edge at N angles and their DFT across ports IS the mode
decomposition. If true, the modal readout is geometry rather than software.

Falsifiable version, run here: drive the disk with a field of KNOWN azimuthal
order -- h_z proportional to cos(n phi), applied inside the disk only, so the
guides carry only what propagates out -- and check that the port DFT peaks at
that same n. A geometry that scrambles modes, or ports that merely sample
total leakage, would show no such selectivity.

Also runs the whole thing at float32 against float64, since micromagnetics is
routinely done in single precision (mumax3 is float32) and halving the cost
would make every downstream experiment affordable.

    python scripts/verify_ports.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk, disk_mask


def azimuthal_drive(cfg, n_order, device="cpu", dtype=torch.float64):
    """h_z profile proportional to cos(n phi), confined to the disk."""
    n = cfg.n_cells
    idx = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    phi = torch.atan2(Y, X)
    prof = torch.cos(n_order * phi) * disk_mask(cfg, device, dtype)[:, :, 0, 0]
    h = torch.zeros(n, n, 1, 3, device=device, dtype=dtype)
    h[:, :, 0, 2] = prof
    return h


@torch.no_grad()
def drive_and_tap(disk, h_profile, freq, n_steps, amp):
    """Return per-port signals over time, (n_steps, n_ports)."""
    dt = disk.cfg.dt
    m, out = disk.m0.clone(), []
    for k in range(n_steps):
        t0, t1 = k * dt, (k + 1) * dt
        def h_drive(theta, t0=t0, t1=t1):
            t = t0 + theta * (t1 - t0)
            return h_profile * (amp * math.sin(2 * math.pi * freq * t))
        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        out.append(disk.port_signals(m))
    return torch.stack(out)


def port_dft(sig, freq, dt, skip_frac=0.5):
    """Mode weight per n, from the SIGNED port amplitudes.

    Lock-in detection at the drive frequency, not an RMS: a cos(n phi) drive
    makes alternate ports swing in antiphase, so any magnitude-only measure
    (std, RMS, power) discards precisely the sign pattern that distinguishes
    the modes and collapses every drive onto n=0. The first half of the record
    is skipped so the transient does not enter the average.
    """
    T = sig.shape[0]
    s = T // 2
    t = torch.arange(s, T, dtype=torch.float64) * dt
    ref = torch.exp(-2j * math.pi * freq * t).unsqueeze(-1)
    amps = (sig[s:].to(torch.complex128) * ref).sum(dim=0) / (T - s)
    spec = torch.fft.fft(amps).abs()
    return (spec / spec.sum().clamp_min(1e-30)).real


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freq", type=float, default=8.0, help="GHz")
    p.add_argument("--amp-mT", type=float, default=15.0)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--relax-steps", type=int, default=800)
    p.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--guide-width-nm", type=float, default=None,
                   help="wider ports subtend more perimeter, which is what\n"
                        "degrades angular selectivity -- so a width chosen\n"
                        "for propagation has to be re-checked here")
    p.add_argument("--guide-length-nm", type=float, default=None)
    p.add_argument("--outdir", default="runs/ports")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    results = {}

    for prec in ("float64", "float32"):
        mnn.set_precision(prec); mnn.set_device("cpu")
        dtype = torch.float64 if prec == "float64" else torch.float32
        over = {}
        if args.guide_width_nm is not None:
            over["guide_width"] = args.guide_width_nm * 1e-9
        if args.guide_length_nm is not None:
            over["guide_length"] = args.guide_length_nm * 1e-9
        cfg = PortedVortexConfig(**over)
        disk = PortedVortexDisk(cfg, timesteps=args.steps, dtype=dtype)
        t0 = time.time(); disk.relax(steps=args.relax_steps); relax_s = time.time() - t0
        core = float(disk.m0[:, :, 0, 2].max())
        arc = cfg.guide_width / cfg.radius * 180 / math.pi
        print(f"\n=== {prec}: relaxed {relax_s:.0f}s, core mz {core:+.3f}\n"
              f"    {cfg.n_ports} ports x {cfg.guide_width*1e9:.0f} nm = "
              f"{arc:.0f} deg each, {cfg.n_ports*arc:.0f} of 360 deg",
              flush=True)

        per_order = {}
        for n_order in args.orders:
            h = azimuthal_drive(cfg, n_order, dtype=dtype)
            t0 = time.time()
            sig = drive_and_tap(disk, h, args.freq * 1e9, args.steps,
                                args.amp_mT * 1e-3 / MU_0)
            w = port_dft(sig, args.freq * 1e9, cfg.dt).tolist()
            peak = int(np.argmax(w[:cfg.n_ports // 2 + 1]))
            per_order[n_order] = {"weights": [round(x, 4) for x in w],
                                  "peak_n": peak, "seconds": round(time.time() - t0, 1)}
            ok = "OK" if peak == n_order else "MISMATCH"
            print(f"  drive n={n_order}: port DFT peak at n={peak}  [{ok}]  "
                  + " ".join(f"{x:.3f}" for x in w[:cfg.n_ports // 2 + 1]), flush=True)
        results[prec] = {"core_mz": core, "relax_s": round(relax_s, 1),
                         "orders": per_order}

    # precision agreement on the quantity that matters
    agree = all(results["float64"]["orders"][n]["peak_n"]
                == results["float32"]["orders"][n]["peak_n"] for n in args.orders)
    dev = max(abs(a - b)
              for n in args.orders
              for a, b in zip(results["float64"]["orders"][n]["weights"],
                              results["float32"]["orders"][n]["weights"]))
    results["float32_matches_float64"] = agree
    results["max_weight_deviation"] = dev
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nfloat32 reproduces float64 mode assignment: {agree}")
    print(f"largest weight deviation between precisions: {dev:.4f}")
    hit = sum(results["float64"]["orders"][n]["peak_n"] == n for n in args.orders)
    print(f"port DFT recovered the driven mode in {hit}/{len(args.orders)} cases")
    if hit == len(args.orders):
        print("The ports perform the azimuthal decomposition physically.")
    else:
        print("Ports did NOT track the driven mode -- the geometry does not")
        print("implement the decomposition, and nothing downstream should")
        print("assume it does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
