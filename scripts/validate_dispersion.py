#!/usr/bin/env python3
"""Check the simulated film against the analytic spin-wave dispersion.

Run this first, before any training. It drives a strip of film at several
frequencies, measures the wavelength that actually propagates, and compares
against the Damon-Eshbach dipole-exchange formula. Agreement means the mesh,
the field terms, the timestep and the absorbing boundary are all doing what
they should. Disagreement localises the problem before you spend an hour
watching a loss that will not move:

* measured wavelength much *shorter* than analytic, at every frequency
  -> the demagnetisation field is off, leaving an exchange-only dispersion
* error growing with frequency -> the mesh is too coarse for the short
  wavelengths; refine ``dx`` or work lower in the band
* no propagating wave at all -> the drive is below the FMR

    python scripts/validate_dispersion.py
    python scripts/validate_dispersion.py --freqs 3.6 4.0 4.4 --nx 200
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch

import magnonic_nn as mnn


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="focus", choices=sorted(mnn.PRESETS))
    p.add_argument("--freqs", type=float, nargs="+", default=[3.6, 4.0, 4.5, 5.0],
                   help="frequencies in GHz")
    p.add_argument("--nx", type=int, default=160, help="strip length in cells")
    p.add_argument("--ny", type=int, default=8, help="strip width in cells")
    p.add_argument("--timesteps", type=int, default=600)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="fail if relative wavenumber error exceeds this")
    p.add_argument("--precision", default="float32", choices=["float32", "float64"])
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    mnn.set_precision(args.precision)

    cfg = mnn.get_preset(args.preset)
    cfg.solver.timesteps = args.timesteps
    cfg.device = str(mnn.set_device("cpu" if not torch.cuda.is_available() else cfg.device))

    fmr = mnn.kittel_fmr(cfg.fields, cfg.material)
    lo, hi = mnn.usable_band(cfg)
    print(f"film: {cfg.mesh.dz * 1e9:g} nm YIG, B0 = {cfg.fields.B0 * 1e3:g} mT, "
          f"dx = {cfg.mesh.dx * 1e9:g} nm")
    print(f"FMR = {fmr / 1e9:.3f} GHz, usable band {lo / 1e9:.2f} .. {hi / 1e9:.2f} GHz")
    print(f"strip: {args.nx} x {args.ny} cells, periodic along y, "
          f"{args.timesteps} steps ({cfg.duration * 1e9:.1f} ns)\n")

    results = mnn.measure_dispersion(
        cfg, freqs=[f * 1e9 for f in args.freqs], nx=args.nx, ny=args.ny
    )

    print(f"{'f (GHz)':>9}  {'lambda meas':>12}  {'lambda calc':>12}  {'k meas':>11}  "
          f"{'k calc':>11}  {'err':>7}")
    worst = 0.0
    for r in results:
        worst = max(worst, r["rel_error"])
        flag = "  <-- outside tolerance" if r["rel_error"] > args.tolerance else ""
        print(f"{r['freq'] / 1e9:9.2f}  {r['lambda_measured'] * 1e9:9.0f} nm  "
              f"{r['lambda_analytic'] * 1e9:9.0f} nm  {r['k_measured']:11.3e}  "
              f"{r['k_analytic']:11.3e}  {r['rel_error'] * 100:6.1f}%{flag}")

    print(f"\nworst relative error: {worst * 100:.1f}% (tolerance {args.tolerance * 100:.0f}%)")
    if worst > args.tolerance:
        print("\nNote: the largest errors normally sit at the band edges -- long "
              "wavelengths that barely fit inside the measurement window, and short "
              "ones approaching the mesh resolution. Errors in the middle of the "
              "band point at a real problem.")
        return 1
    print("dispersion matches theory across the tested band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
