#!/usr/bin/env python3
"""Does thinning the guide bring its band down to the disk's modes?

The guides measured a propagating band of 18-26 GHz at a 40 x 20 nm
cross-section, while the disk's azimuthal modes sit at 5-15 GHz -- so every
measurement so far was taken about 3x below cutoff and the guides were
evanescent stubs, not waveguides.

A narrow strip magnetised along its length resonates near
  f = (gamma/2pi) * Ms * sqrt((Ny - Nx)(Nz - Nx))
and for w >> t the demag factors go roughly as Ny ~ t/(t+w), Nz ~ w/(t+w). So
thinning at fixed width should push the band DOWN: 40x20 gives Ny ~ 0.33,
40x5 gives Ny ~ 0.11, which is a large change in the geometric mean.

Estimate is not measurement, so this sweeps thickness against frequency and
reports where each cross-section actually transmits.

    python scripts/check_thin_guides.py
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import magnonic_nn as mnn
from check_layout_options import bend_transmission

GAMMA_2PI = 28.0e9           # Hz/T
MU0_MS = 1.005               # tesla, permalloy


def predicted_fmr(width_nm, thickness_nm):
    """Shape-anisotropy resonance of a long strip magnetised along its axis."""
    w, t = width_nm, thickness_nm
    ny, nz = t / (t + w), w / (t + w)
    return GAMMA_2PI * MU0_MS * math.sqrt(ny * nz) / 1e9


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--thicknesses", type=float, nargs="+", default=[20, 10, 5])
    p.add_argument("--freqs", type=float, nargs="+", default=[6, 8, 10, 14, 18, 22])
    p.add_argument("--width", type=float, default=40.0)
    p.add_argument("--widths", type=float, nargs="+", default=None,
                   help="sweep width instead of thickness, at --thickness-fixed")
    p.add_argument("--thickness-fixed", type=float, default=5.0)
    p.add_argument("--length", type=float, default=700.0)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--outdir", default="runs/thin_guides")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / "journal.jsonl"
    done = {}
    if jpath.exists():
        for line in jpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[(r["thickness_nm"], r["ghz"])] = r["transmission"]

    header = "  ".join(f"{f:>5.0f}" for f in args.freqs)
    if args.widths:
        # Width sweep. The measured band edge sits ~4-5 GHz above the
        # demag-only prediction at BOTH thicknesses tested, an offset that does
        # not move with thickness -- the signature of transverse exchange
        # confinement, which depends on width and not thickness. Exchange
        # confinement goes as 1/w^2, so doubling the width should cut that
        # offset fourfold. This measures whether it does.
        print(f"transmission through {args.length:.0f} nm of "
              f"{args.thickness_fixed:.0f} nm-thick guide\n")
        print(f"{'width':>6} {'pred FMR':>9}   {header}")
        rows = []
        for w in args.widths:
            vals = []
            for f in args.freqs:
                key = (w, f)
                if key in done:
                    vals.append(done[key]); continue
                v = bend_transmission(0.0, args.length, f * 1e9, args.steps,
                                      torch.float32,
                                      thickness_nm=args.thickness_fixed,
                                      width_nm=w)
                with jpath.open("a") as fh:
                    fh.write(json.dumps({"thickness_nm": w, "ghz": f,
                                         "transmission": v}) + "\n")
                vals.append(v)
            rows.append({"width_nm": w, "values": vals})
            cells = "  ".join(f"{v:>5.2f}" for v in vals)
            print(f"{w:>6.0f} {predicted_fmr(w, args.thickness_fixed):>8.1f}G   "
                  f"{cells}", flush=True)
        (outdir / "results_width.json").write_text(json.dumps(rows, indent=2))
        low = [(r["width_nm"], f) for r in rows
               for f, v in zip(args.freqs, r["values"]) if v > 0.1]
        print()
        if low:
            best_w = min(w for w, f in low if f <= 10) if any(f <= 10 for _, f in low) else None
            if best_w:
                print(f"{best_w:.0f} nm width transmits at 10 GHz or below -- the guide "
                      f"finally covers the disk's low-order modes.")
                print("Cost: a wider guide admits more azimuthal orders, so mode")
                print("selectivity has to be re-measured, not assumed.")
            else:
                print("Widening helps but still does not reach 10 GHz; exchange is not")
                print("the whole story either.")
        return 0

    print(f"transmission through {args.length:.0f} nm of {args.width:.0f} nm-wide guide\n")
    print(f"{'thick':>6} {'pred FMR':>9}   {header}")
    rows = []
    for t in args.thicknesses:
        vals = []
        for f in args.freqs:
            key = (t, f)
            if key in done:
                vals.append(done[key])
                continue
            v = bend_transmission(0.0, args.length, f * 1e9, args.steps,
                                  torch.float32, thickness_nm=t,
                                  width_nm=args.width)
            with jpath.open("a") as fh:
                fh.write(json.dumps({"thickness_nm": t, "ghz": f,
                                     "transmission": v}) + "\n")
            vals.append(v)
        rows.append({"thickness_nm": t, "values": vals})
        cells = "  ".join(f"{v:>5.2f}" for v in vals)
        print(f"{t:>6.0f} {predicted_fmr(args.width, t):>8.1f}G   {cells}", flush=True)

    (outdir / "results.json").write_text(json.dumps(rows, indent=2))
    print()
    target = (5.0, 15.0)      # the disk's azimuthal modes
    best = None
    for r in rows:
        inband = [f for f, v in zip(args.freqs, r["values"])
                  if v > 0.1 and target[0] <= f <= target[1]]
        if inband and (best is None or len(inband) > best[1]):
            best = (r["thickness_nm"], len(inband), inband)
    if best:
        print(f"{best[0]:.0f} nm thickness transmits at {best[2]} GHz -- inside the "
              f"disk's 5-15 GHz azimuthal range.")
        print("The guides become actual waveguides at the operating point, and every")
        print("coupling measurement can be retaken on a channel that exists.")
    else:
        print("No thickness tested transmits inside 5-15 GHz. Thinning alone does not")
        print("bring the band down far enough; widening the guide, or biasing it, is")
        print("the next lever -- both at a cost to mode selectivity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
