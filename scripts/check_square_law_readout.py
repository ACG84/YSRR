#!/usr/bin/env python3
"""The device's state is linear in u. Can a square-law readout supply the products?

The capacity decomposition found the chain spending 99% of its measured capacity
on degree-1 targets, with the cross-lag product at lag 9 -- the one term
NARMA-10 cannot get from any linear filter -- sitting at exactly 0.000. Priced
in software, those products are worth a great deal: a 15-lag filter given its
own products scores 0.0460 against 0.1243 for the best pure linear filter and
0.2120 for the device.

Before concluding the magnetics need changing, check the readout. The current
one is a coherent lock-in returning (real, imaginary) per mode -- a LINEAR
functional of the magnetisation. If the state is linear in u, a linear readout
of it is linear in u by construction, and no amount of depth will change that.
The nonlinearity has to enter somewhere, and the lock-in is where it currently
cannot.

A square-law detector is the obvious candidate and it is not a compromise: |A|^2
is what a diode, a bolometer, or any power detector returns, and measuring power
is EASIER in hardware than maintaining a phase reference for coherent I/Q. So
this asks whether the cheaper detector is also the better one.

The mechanism is worth stating because it predicts the result. If the mode
amplitude is a linear functional of the input history,

    A[n] = sum_k c_k u[n-k]        then      |A[n]|^2 = sum_ij c_i c_j* u[n-i] u[n-j]

which contains every cross-lag product the task wants, at every pair of lags the
state reaches. A square-law readout of a linear state is a product generator.
The question is whether the products survive at usable amplitude, and whether
they are the right pairs.

Three readouts on the SAME cached run, so the state is held fixed and only the
detector changes:

    coherent    (re, im) per mode              what every number so far used
    power       re^2 + im^2 per mode           square-law, half the columns
    both        coherent and power together    what a real instrument could do

Runs on cached features. No simulation.

    python scripts/check_square_law_readout.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from magnonic_nn.reservoir import narma10
from certify_narma10 import SHIFT, lags, poly2, score, standardise
from check_nonlinear_capacity import r2_of, shifted, targets


def eff_rank(X, tr, frac=0.99):
    sv = np.linalg.svd(X[tr] - X[tr].mean(0), compute_uv=False)
    return int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), frac) + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="runs/chain3_narma/features_n3_linked.pt")
    p.add_argument("--n-disks", type=int, default=3)
    p.add_argument("--n-ports", type=int, default=6)
    p.add_argument("--n-tones", type=int, default=5)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", type=int, nargs=3, default=(150, 300, 75))
    p.add_argument("--max-lag", type=int, default=20)
    p.add_argument("--outdir", default=None)
    a = p.parse_args()

    obj = torch.load(a.features, weights_only=False)
    F = np.asarray(obj["feats"] if isinstance(obj, dict) else obj, dtype=np.float64)
    if len(F) < a.frames:
        print(f"INCOMPLETE: {len(F)}/{a.frames} frames")
        return 1
    F = F[:a.frames]
    outdir = Path(a.outdir or Path(a.features).parent)

    u, y = narma10(a.frames, seed=a.seed)
    s = 4.0 * u - 1.0
    n_wash, n_train, n_val = a.splits
    tr = slice(n_wash, n_wash + n_train)
    va = slice(n_wash + n_train, n_wash + n_train + n_val)
    te = slice(n_wash + n_train + n_val, a.frames)

    # Columns are (re, im) pairs in order, so power is a stride-2 fold.
    assert F.shape[1] % 2 == 0
    re, im = F[:, 0::2], F[:, 1::2]
    variants = {
        "coherent (re, im)": F,
        "power (re^2+im^2)": re ** 2 + im ** 2,
        "both": np.hstack([F, re ** 2 + im ** 2]),
    }

    fam = targets(s, a.max_lag)
    rows = {}
    print(f"{'readout':<20} {'dim':>5} {'rank':>5} {'deg1':>7} {'deg2 P2':>8} "
          f"{'prod':>7} {'prod@9':>7} {'NMSE':>8}")
    for name, X in variants.items():
        Z = standardise(X, tr)
        fl = max(max(r2_of(Z, shifted(t, SHIFT), tr, va, te) for t in byk.values())
                 for byk in fam.values())
        cap, at9 = {}, 0.0
        for fname, byk in fam.items():
            tot = 0.0
            for k, t in byk.items():
                v = r2_of(Z, t, tr, va, te)
                tot += max(v - fl, 0.0)
                if fname == "deg2  s[n]*s[n-k]" and k == 9:
                    at9 = v
            cap[fname] = tot
        nm = score(X, y, a.splits)
        rows[name] = {"dim": int(X.shape[1]), "rank": eff_rank(X, tr),
                      "floor": fl, "capacity": cap, "prod_at_lag9": at9,
                      "narma_nmse": nm}
        print(f"{name:<20} {X.shape[1]:>5} {rows[name]['rank']:>5} "
              f"{cap['deg1  P1(s[n-k])']:>7.2f} {cap['deg2  P2(s[n-k])']:>8.2f} "
              f"{cap['deg2  s[n]*s[n-k]']:>7.2f} {at9:>7.3f} {nm:>8.4f}",
              flush=True)

    lin_best = min(score(lags(u, k), y, a.splits) for k in (10, 11, 15, 20, 26, 40))
    p2 = score(poly2(u, 15), y, a.splits)
    print(f"\nbaselines: best linear filter {lin_best:.4f}, "
          f"linear+products (15 lags) {p2:.4f}")

    coh, pw = rows["coherent (re, im)"], rows["power (re^2+im^2)"]
    best = min(rows.values(), key=lambda r: r["narma_nmse"])
    print()
    if pw["capacity"]["deg2  s[n]*s[n-k]"] > 5 * coh["capacity"]["deg2  s[n]*s[n-k]"] \
            and pw["capacity"]["deg2  s[n]*s[n-k]"] > 0.5:
        print("The square-law readout GENERATES the products, as the algebra says\n"
              "it must. The missing nonlinearity was in the detector, not in the\n"
              "magnetics -- and the detector that supplies it is the cheaper one.")
    else:
        print("The square-law readout does NOT supply usable products. Either the\n"
              "state is not linear enough in u for |A|^2 to factor cleanly, or the\n"
              "products it makes are swamped by the squared linear terms. The\n"
              "nonlinearity has to come from the drive or the magnetics.")
    if best["narma_nmse"] < lin_best:
        print(f"\nAnd it beats the best linear filter: {best['narma_nmse']:.4f} "
              f"vs {lin_best:.4f}.")
    else:
        print(f"\nStill short of the best linear filter: "
              f"{best['narma_nmse']:.4f} vs {lin_best:.4f}.")

    out = outdir / "square_law_readout.json"
    out.write_text(json.dumps({"variants": rows, "linear_best": lin_best,
                               "poly2_15": p2}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
