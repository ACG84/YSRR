#!/usr/bin/env python3
"""Combine ASVI spectra dumped to logs, and score distinguishability locally.

Six runs on this project have been lost to reclaimed machines. The last one
died after three of six states with every completed spectrum sitting in an npz
that went with the container, and the run before it after four of six. The
response is not another retry at the same size: it is to stop letting the
analysis depend on one machine surviving.

check_asvi_spectrum --dump-b64 prints each state's summed power spectrum to
stdout the moment it is computed. This reads those lines out of any number of
logs, merges them, and computes the verdict here. A run killed part-way
contributes everything it finished, and a six-state matrix can be assembled
from short runs that never coexisted.

    python scripts/combine_asvi_spectra.py runs/*.out --clean-ghz 1.0
"""
from __future__ import annotations
import argparse, base64, io, sys
import numpy as np


def read_dumps(paths):
    """{state: (f0, f1, power[n_freq, n_layers])} from every B64 line found."""
    out = {}
    for path in paths:
        with open(path, errors="replace") as fh:
            for line in fh:
                if not line.startswith("B64 "):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                spec, f0, f1, payload = parts[1], float(parts[2]), float(parts[3]), parts[4]
                try:
                    arr = np.load(io.BytesIO(base64.b64decode(payload)),
                                  allow_pickle=False)
                except Exception as e:
                    print(f"  skipping malformed dump for {spec}: {e}", file=sys.stderr)
                    continue
                if spec in out and out[spec][2].shape != arr.shape:
                    raise SystemExit(
                        f"{spec} appears twice with different shapes "
                        f"{out[spec][2].shape} and {arr.shape} -- these came from "
                        f"runs with different record lengths and their spectra are "
                        f"on different frequency grids. A correlation across them "
                        f"would be comparing different bins.")
                out[spec] = (f0, f1, arr)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--clean-ghz", type=float, default=1.0)
    a = p.parse_args()

    d = read_dumps(a.logs)
    if not d:
        raise SystemExit("no B64 dump lines found; was --dump-b64 passed?")
    names = sorted(d)
    shapes = {v[2].shape for v in d.values()}
    grids = {(round(v[0], 4), round(v[1], 4)) for v in d.values()}
    if len(shapes) > 1 or len(grids) > 1:
        raise SystemExit(f"mixed frequency grids {grids} / shapes {shapes}; "
                         f"refusing to correlate across them")
    f0, f1, _ = d[names[0]]
    n = d[names[0]][2].shape[0]
    f = np.linspace(f0, f1, n)
    print(f"{len(names)} states, {n} bins over {f0:.3f}-{f1:.2f} GHz "
          f"({(f1-f0)/max(n-1,1)*1e3:.0f} MHz)")
    for s in names:
        print(f"  {s}")

    def flat(P, band=None):
        v = (P if band is None else P[band]).reshape(-1)
        return v / max(np.linalg.norm(v), 1e-30)

    def matrix(band, label):
        print(f"\n{label}")
        print(f"{'':<24} " + " ".join(f"{s[:13]:>14}" for s in names))
        worst = (0.0, None, None)
        for i, x in enumerate(names):
            row = f"{x:<24} "
            for j, y in enumerate(names):
                c = float(np.dot(flat(d[x][2], band), flat(d[y][2], band)))
                row += f"{c:>14.3f}"
                if i < j and c > worst[0]:
                    worst = (c, x, y)
            print(row)
        return worst

    wf = matrix(None, f"FULL BAND ({f0:.2f}-{f1:.1f} GHz)")
    clean = f >= a.clean_ghz
    wc = matrix(clean, f"CLEAN BAND (>= {a.clean_ghz:g} GHz, {int(clean.sum())} bins)")
    print(f"\nworst pair full band  {wf[0]:.3f}  ({wf[1]} vs {wf[2]})")
    print(f"worst pair clean band {wc[0]:.3f}  ({wc[1]} vs {wc[2]})")
    print("the two bands " + ("AGREE" if (wf[0] > 0.99) == (wc[0] > 0.99)
                              else "DISAGREE -- believe the clean band"))
    if wc[0] > 0.99:
        print("\nAt least one pair is indistinguishable to a spectral readout.")
    elif wc[0] > 0.9:
        print("\nDistinguishable but not comfortably; check against a realistic\n"
              "readout bandwidth before counting these as independent states.")
    else:
        print("\nEvery microstate pair is spectrally distinct in the clean band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
