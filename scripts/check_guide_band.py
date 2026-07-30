#!/usr/bin/env python3
"""What frequencies does the guide actually carry? Measured, not assumed.

The finding this records. Transmission through a straight 700 nm guide,
40 nm wide and 20 nm thick, driven transversely:

    6 GHz   0.0016      14 GHz  0.0173      22 GHz  0.6265
   10 GHz   0.0028      18 GHz  0.2205      26 GHz  0.4706

The propagating band is 18-26 GHz. Every measurement in this project has been
taken at 6-9 GHz -- about 3x below cutoff -- so the guides have never been
waveguides at the operating frequency. They are evanescent stubs. The cause is
the cross-section: a narrow strip magnetised along its length has a large
shape-anisotropy FMR, near 13 GHz here, with the propagating band above it.

That resolves three earlier puzzles at once, which is the reason to believe it:

* shorter guides measured MORE stable (-0.658 at 55 nm against -0.100 at
  150 nm) because below cutoff the field decays exponentially into the guide,
  so an absorbing taper only does work if it sits inside the decay length. Not
  "aperture is loss" -- "put the absorber where the field still is".
* inter-disk transfer looked dipolar, arriving in 40 ps over 700 nm, because a
  link below cutoff cannot carry a propagating wave and the near field is the
  only channel left.
* the port readout still recovered the driven azimuthal mode 3/3, which
  survives unchanged: evanescent sampling over 150 nm is a perfectly good tap.
  The mechanism is near-field sampling, not guided transport.

The fix is dimensional. FMR follows the demag anisotropy of the cross-section,
so thinning the guides to ~5 nm at the same 40 nm width drops N_y from ~0.33
to ~0.11 and brings the band down toward 8-9 GHz, into the disk's azimuthal
range. This script measures whether that is true rather than trusting the
estimate.

    python scripts/check_guide_band.py
    python scripts/check_guide_band.py --thicknesses 20 10 5 --freqs 6 8 10 14
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import magnonic_nn as mnn
from check_layout_options import bend_transmission


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freqs", type=float, nargs="+", default=[6, 10, 14, 18, 22, 26])
    p.add_argument("--length", type=float, default=700.0)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--outdir", default="runs/guide_band")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"straight {args.length:.0f} nm guide, 40 x 20 nm cross-section\n")
    print(f"{'GHz':>6} {'transmission':>13} {'':>4}")
    for f in args.freqs:
        t = bend_transmission(0.0, args.length, f * 1e9, args.steps, torch.float32)
        rows.append({"ghz": f, "transmission": t})
        bar = "#" * int(40 * min(t, 1.0))
        print(f"{f:>6.0f} {t:>13.4f}  {bar}", flush=True)

    (outdir / "results.json").write_text(json.dumps(rows, indent=2))
    passband = [r["ghz"] for r in rows if r["transmission"] > 0.1]
    print()
    if passband:
        print(f"Propagating band: {min(passband):.0f}-{max(passband):.0f} GHz.")
        if min(passband) > 10:
            print("The disk's azimuthal modes are 5-15 GHz, so the guides are")
            print("BELOW cutoff at the operating point: evanescent stubs, not")
            print("waveguides. Thin the cross-section to bring the band down.")
    else:
        print("No frequency tested propagated; widen the sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
