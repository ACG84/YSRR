#!/usr/bin/env python3
"""Does a saturated tap feed products to its linear neighbours downstream?

The above-threshold run drove the four taps to 20.1 / 15.9 / 10.6 / 8.9 mT
against in-array thresholds of 10 / 12 / none / 12. Taps 1 and 2 were
SATURATED; taps 3 and 4 were not. The readout mixed all four, and the verdict
was that nonlinearity arrived and the reservoir died -- capacity 12.26 -> 3.59,
memory horizon past lag 20 -> about lag 4, NARMA 1.088 -> 5.625.

But the disks couple to the bus BIDIRECTIONALLY. A saturated tap re-radiates
its distorted response into the bus, which carries it downstream to the taps
that are still linear. So the array may already contain the architecture the
chaos is hiding: a saturating feeder followed by a linear reservoir, with the
two summed together in the readout.

Splitting the readout tests that for nothing. If the sub-threshold taps carry
cross-lag products WITHOUT the rank explosion and memory collapse, then the
nonlinearity does propagate and the fix is to stop reading the saturated
element -- not to abandon the drive. If they carry no products, the
nonlinearity stays local to the disk that made it and the feeder has to be
coupled deliberately rather than incidentally.

Column layout, from check_nonlinear_capacity: tone-major, then disk, then port,
then quadrature -- index = tone*(n_disks*n_ports*2) + disk*(n_ports*2) + ...

    python scripts/split_readout_capacity.py --features data/narma_9ghz/...pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from magnonic_nn.reservoir import narma10
from certify_narma10 import SHIFT, standardise
from check_nonlinear_capacity import r2_of, shifted, targets


def analyse(F, label, splits, frames, max_lag, seed=0):
    u, s_ = narma10(frames, seed=seed)
    s = 4.0 * u - 1.0
    n_wash, n_train, n_val = splits
    tr = slice(n_wash, n_wash + n_train)
    va = slice(n_wash + n_train, n_wash + n_train + n_val)
    te = slice(n_wash + n_train + n_val, frames)

    Z = standardise(F, tr)
    sv = np.linalg.svd(F[tr] - F[tr].mean(0), compute_uv=False)
    rank = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), 0.99) + 1)

    fam = targets(s, max_lag)
    floor = {}
    for name, byk in fam.items():
        floor[name] = float(np.max([r2_of(Z, shifted(t, SHIFT), tr, va, te)
                                    for t in byk.values()]))
    fl = max(floor.values())

    curves, caps = {}, {}
    for name, byk in fam.items():
        c = {k: r2_of(Z, t, tr, va, te) for k, t in byk.items()}
        curves[name] = c
        caps[name] = float(sum(max(v - fl, 0.0) for v in c.values()))

    p1 = curves[[n for n in fam if n.startswith("deg1")][0]]
    horizon = max([k for k, v in p1.items() if v > 0.2], default=0)
    total = sum(caps.values())
    d1 = caps[[n for n in fam if n.startswith("deg1")][0]]
    return {"label": label, "cols": F.shape[1], "rank": rank, "floor": fl,
            "caps": caps, "total": total,
            "d1_share": d1 / total if total else float("nan"),
            "horizon": horizon, "p1": p1,
            "prod": caps[[n for n in fam if "s[n]*s[n-k]" in n][0]]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--n-disks", type=int, default=4)
    p.add_argument("--n-ports", type=int, default=5)
    p.add_argument("--n-tones", type=int, default=5)
    p.add_argument("--frames", type=int, default=260)
    p.add_argument("--splits", type=int, nargs=3, default=(40, 150, 40))
    p.add_argument("--max-lag", type=int, default=20)
    p.add_argument("--saturated", type=int, nargs="+", default=[0, 1],
                   help="0-indexed disks that were driven above threshold")
    a = p.parse_args()

    F = torch.load(a.features, weights_only=False)
    F = (F.numpy() if hasattr(F, "numpy") else np.asarray(F))[:a.frames]
    per_disk = a.n_ports * 2
    block = a.n_disks * per_disk
    assert F.shape[1] == a.n_tones * block, (F.shape, a.n_tones * block)

    def cols_for(disks):
        return [t * block + d * per_disk + j
                for t in range(a.n_tones) for d in disks for j in range(per_disk)]

    sat = sorted(set(a.saturated))
    lin = [d for d in range(a.n_disks) if d not in sat]
    views = [("all four taps", list(range(a.n_disks))),
             (f"saturated only (taps {[d+1 for d in sat]})", sat),
             (f"sub-threshold only (taps {[d+1 for d in lin]})", lin)]

    out = []
    for label, disks in views:
        out.append(analyse(F[:, cols_for(disks)], label, tuple(a.splits),
                           a.frames, a.max_lag))

    print(f"{'readout':<34} {'cols':>5} {'rank':>5} {'floor':>6} {'deg1':>7} "
          f"{'prod':>6} {'total':>7} {'d1%':>5} {'horizon':>8}")
    for r in out:
        d1 = r["caps"][[n for n in r["caps"] if n.startswith("deg1")][0]]
        print(f"{r['label']:<34} {r['cols']:>5} {r['rank']:>5} {r['floor']:>6.3f} "
              f"{d1:>7.2f} {r['prod']:>6.2f} {r['total']:>7.2f} "
              f"{r['d1_share']*100:>4.0f}% {r['horizon']:>8}")

    print(f"\nlinear memory P1(s[n-k]) by lag")
    print(f"{'lag':>4} " + " ".join(f"{r['label'][:16]:>17}" for r in out))
    for k in range(0, a.max_lag + 1, 2):
        print(f"{k:>4} " + " ".join(f"{r['p1'].get(k, float('nan')):>17.3f}"
                                    for r in out))

    sat_r, lin_r = out[1], out[2]
    print()
    # Read product and memory SEPARATELY. An earlier version required the
    # sub-threshold view to win on both and printed "carries no more product"
    # over numbers showing it carried twice as much -- a verdict contradicting
    # its own table.
    more_prod = lin_r["prod"] > 1.5 * sat_r["prod"]
    kept_mem = lin_r["horizon"] >= 8          # the sub-threshold run reached 20+
    print(f"product: sub-threshold {lin_r['prod']:.2f} vs saturated "
          f"{sat_r['prod']:.2f}  ({'more' if more_prod else 'comparable'})")
    print(f"memory:  horizon {lin_r['horizon']} vs {sat_r['horizon']} frames, "
          f"against 20+ when the whole array ran sub-threshold "
          f"({'kept' if kept_mem else 'LOST'})")
    print(f"floors:  {lin_r['floor']:.3f} vs {sat_r['floor']:.3f} -- a higher "
          f"floor makes the product estimate less trustworthy, not more")
    print()
    if more_prod and kept_mem:
        print("Separating the readout works: the unsaturated taps carry the\n"
              "products and keep the memory. Stop reading the saturated disk.")
    elif not kept_mem:
        print("The unsaturated taps lost their memory too. They are individually\n"
              "linear -- taps 3 and 4 saw 10.6 and 8.9 mT against thresholds of\n"
              "none-in-range and 12 -- so the corruption arrived through the BUS,\n"
              "which every tap shares. A saturating element in line with the\n"
              "delay line pollutes the delay line. Separating the READOUT is not\n"
              "enough; the feeder needs a path that does not carry the memory.")
    else:
        print("The unsaturated taps carry no more product than the saturated\n"
              "ones: the nonlinearity stays local to the disk that made it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
