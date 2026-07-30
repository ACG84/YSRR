#!/usr/bin/env python3
"""Which frequencies carry the computation, not just the energy?

The guides pass a band. The disk computes across a spectrum. Whether a ported
device works depends on the overlap -- but the quantity to overlap is not
power. A band can hold a third of the ring-down and none of the information.

So this scores the AB/BA discrimination band by band, using the same honesty
check as the original test: the difference between the two pulse orders is
compared at HIGH drive against LOW drive. Both orders contain the same two
tones for the same durations, so a linear system must return identical average
spectra and any difference is nonlinear memory. A band where the high/low ratio
is ~1 shows a difference that is equally present without nonlinearity -- a
transient artefact, not computation.

That distinction turned out to matter. Nonlinear scattering raises the power
above 14 GHz from 3.8% to 11.7%, so on an energy criterion the guide band at
20 nm thickness looks usable. Scored on information it is worthless: ratio 1.0
at 14-26 GHz against 27 at 8-10 GHz. The upconverted harmonics carry energy the
guides can transport and nothing the readout can use.

Reads the spectra saved by run_modal_disk.py; runs no simulation.

    python scripts/check_band_information.py
    python scripts/check_band_information.py --bands 8,10 10,12 14,26
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch

DEFAULT_BANDS = [(1, 40), (5, 8), (8, 10), (9, 11), (10, 12), (11, 13),
                 (12, 14), (11, 14), (14, 18), (18, 26), (14, 26)]


def rel_diff(spec, mask):
    """Relative AB/BA separation inside a band; 0 means indistinguishable."""
    a, b = spec["AB"].numpy()[:, mask], spec["BA"].numpy()[:, mask]
    den = np.linalg.norm(a) + np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / max(den, 1e-30))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="runs/modal_disk")
    p.add_argument("--dt-rec", type=float, default=2e-12,
                   help="seconds between recorded snapshots in the source run")
    p.add_argument("--bands", nargs="*", default=None,
                   help="lo,hi pairs in GHz; default is a scan across the range")
    p.add_argument("--fmax", type=float, default=40.0)
    p.add_argument("--threshold", type=float, default=3.0,
                   help="high/low ratio above which a band is called nonlinear")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    indir = Path(args.indir)
    spec = {lab: torch.load(indir / f"spectra_{lab}.pt", weights_only=False)
            for lab in ("low", "high")}
    n_freq = spec["low"]["AB"].shape[1]
    freqs = np.fft.rfftfreq(2 * (n_freq - 1), d=args.dt_rec) / 1e9

    bands = DEFAULT_BANDS
    if args.bands:
        bands = [tuple(float(v) for v in b.split(",")) for b in args.bands]

    full = (freqs > 1.0) & (freqs <= args.fmax)
    total = spec["high"]["AB"].numpy()[:, full].sum()

    print("AB/BA separation by band. The ratio is the honest column: a band\n"
          "where high and low drive separate the two orders equally well is\n"
          "showing a transient, not nonlinear memory.\n")
    print(f"{'band GHz':>10} {'low diff':>9} {'high diff':>10} {'ratio':>7} "
          f"{'%power':>7}")
    rows = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs <= hi)
        if m.sum() < 2:
            continue
        l, h = rel_diff(spec["low"], m), rel_diff(spec["high"], m)
        ratio = h / max(l, 1e-12)
        pw = float(spec["high"]["AB"].numpy()[:, m].sum() / total * 100)
        rows.append({"lo_ghz": lo, "hi_ghz": hi, "low_diff": l, "high_diff": h,
                     "ratio": ratio, "percent_power": pw})
        print(f"{f'{lo:g}-{hi:g}':>10} {l:>9.4f} {h:>10.4f} {ratio:>7.1f} "
              f"{pw:>7.1f}")

    usable = [r for r in rows if r["ratio"] >= args.threshold
              and not (r["lo_ghz"] <= 1 and r["hi_ghz"] >= args.fmax)]
    print()
    if usable:
        best = max(usable, key=lambda r: r["ratio"])
        print(f"Discrimination is strongest at {best['lo_ghz']:g}-"
              f"{best['hi_ghz']:g} GHz (ratio {best['ratio']:.1f}, carrying "
              f"{best['percent_power']:.0f}% of the power).")
        print("The guides have to pass THAT band. A geometry that passes a band")
        print("with a ratio near 1 transports energy the readout cannot use.")
    else:
        print("No band shows nonlinear discrimination above the threshold.")

    out = Path(args.out) if args.out else indir / "band_information.json"
    out.write_text(json.dumps({"threshold": args.threshold, "bands": rows},
                              indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
