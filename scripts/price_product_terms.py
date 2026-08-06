#!/usr/bin/env python3
"""How much would the co-drive's product term actually be worth, if it works?

The co-drive arm is predicted to give one nonlinear element both a ~9.5-frame
delayed copy and a fresh sample, manufacturing `u[n]*u[n-9]` -- the term
NARMA-10 needs and the chain currently has at exactly 0.000 capacity.

Before spending three hours confirming whether it appears, price what it is
worth if it does. A full `linear+products` filter at 15 lags scores 0.0460
against 0.1243 for the best pure linear filter, but that filter has 105 products
and the co-drive would supply a handful near one lag. Those are very different
purchases and only the second one is on offer.

So add products to a linear filter ONE FAMILY AT A TIME, cheapest first, and see
where the curve crosses what the device has to beat. This is arithmetic on the
task, not a device measurement -- it bounds what any mechanism delivering these
terms could buy, which is the number that decides whether the arm is worth its
compute.

The device's own readout penalty is applied as a separate, measured factor: the
chain reaches 0.2008 while holding linear memory comparable in span to a 20-lag
filter that scores 0.1243, so its linear path costs ~1.6x over an ideal tap.
Any product capacity it gains presumably pays that same tax.

    python scripts/price_product_terms.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from magnonic_nn.reservoir import narma10
from certify_narma10 import lags, poly2, score, LAG_CHOICES


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", type=int, nargs=3, default=(150, 300, 75))
    p.add_argument("--base-lags", type=int, default=20)
    p.add_argument("--device-nmse", type=float, default=0.1905,
                   help="best device number so far (coherent+power readout)")
    p.add_argument("--readout-penalty", type=float, default=None,
                   help="default: measured as device/linear at matched span")
    p.add_argument("--outdir", default="runs/product_pricing")
    a = p.parse_args()

    u, y = narma10(a.frames, seed=a.seed)
    splits = tuple(a.splits)
    L = lags(u, a.base_lags)

    lin_best = min(score(lags(u, k), y, splits) for k in LAG_CHOICES)
    pen = a.readout_penalty or (a.device_nmse / lin_best)

    def with_cols(extra):
        X = L if not extra else np.concatenate([L] + extra, axis=1)
        return X.shape[1], score(X, y, splits)

    def prod(i, j):
        return (np.roll(u, i) * np.roll(u, j))[:, None]

    # Families chosen to separate two very different physical asks. Products
    # spanning a LONG delay need a nonlinear element fed by both a delayed and a
    # fresh copy -- the co-drive. Products between NEARBY samples need only a
    # nonlinearity with a short memory, which the driven disk already has: it
    # makes s[n-5]*s[n-6] at r^2 0.208 today.
    K = 15
    arms = {
        f"linear {a.base_lags} lags (baseline)": [],
        "+ u[n]*u[n-9] only": [prod(0, 9)],
        "+ u[n]*u[n-k], k=8,9,10": [prod(0, k) for k in (8, 9, 10)],
        "+ u[n]*u[n-k], k=1..15": [prod(0, k) for k in range(1, 16)],
        "+ squares u[n-i]^2 only": [prod(i, i) for i in range(K)],
        "+ ADJACENT u[n-i]*u[n-i-1]": [prod(i, i + 1) for i in range(K)],
        "+ near pairs |i-j|<=2": [prod(i, j) for i in range(K)
                                  for j in range(i, min(i + 3, K))],
        "+ near pairs |i-j|<=4": [prod(i, j) for i in range(K)
                                  for j in range(i, min(i + 5, K))],
        "+ far pairs |i-j|>=5": [prod(i, j) for i in range(K)
                                 for j in range(i + 5, K)],
    }
    print(f"{'readout':<32} {'dim':>5} {'NMSE':>8} {'x device':>9} {'/1.6 tax':>9}")
    rows = {}
    for name, extra in arms.items():
        d, v = with_cols(extra)
        rows[name] = {"dim": d, "nmse": v, "taxed": v * pen}
        print(f"{name:<32} {d:>5} {v:>8.4f} {a.device_nmse/v:>9.2f} "
              f"{v*pen:>9.4f}")

    full = poly2(u, 15)
    rows["linear+ALL products, 15 lags"] = {"dim": full.shape[1],
                                            "nmse": score(full, y, splits)}
    rows["linear+ALL products, 15 lags"]["taxed"] = \
        rows["linear+ALL products, 15 lags"]["nmse"] * pen
    r = rows["linear+ALL products, 15 lags"]
    print(f"{'linear+ALL products, 15 lags':<32} {r['dim']:>5} {r['nmse']:>8.4f} "
          f"{a.device_nmse/r['nmse']:>9.2f} {r['taxed']:>9.4f}")

    print(f"\nbest pure linear filter (the bar): {lin_best:.4f}")
    print(f"device, best readout so far:       {a.device_nmse:.4f}")
    print(f"device's linear-path readout tax:  {pen:.2f}x")
    print("\n'/1.6 tax' applies that measured tax to each software number -- a")
    print("rough estimate of where a DEVICE supplying those same terms would land.")

    near = rows["+ near pairs |i-j|<=2"]
    far = rows["+ far pairs |i-j|>=5"]
    sq = rows["+ squares u[n-i]^2 only"]
    print(f"\nwhere the value in the 105 products actually is:")
    print(f"  squares alone          {sq['nmse']:.4f}")
    print(f"  near pairs |i-j|<=2    {near['nmse']:.4f}   (short-memory mixing)")
    print(f"  far pairs  |i-j|>=5    {far['nmse']:.4f}   (long-delay mixing)")
    one = rows["+ u[n]*u[n-9] only"]["taxed"]
    band = rows["+ u[n]*u[n-k], k=8,9,10"]["taxed"]
    print()
    if one < lin_best:
        print(f"The single product term alone would be enough: taxed {one:.4f} "
              f"vs the {lin_best:.4f} bar.")
    elif band < lin_best:
        print(f"One product term is NOT enough after tax ({one:.4f} vs "
              f"{lin_best:.4f}),\nbut a small band around lag 9 would be "
              f"({band:.4f}). The co-drive has to\nsupply products at several "
              f"neighbouring lags, not just one.")
    else:
        print(f"Even a band of products around lag 9 would not clear the bar "
              f"after tax\n({band:.4f} vs {lin_best:.4f}). Products at one delay "
              f"are not what is\nmissing -- the readout tax is, and that is a "
              f"different problem.")

    # The rank ceiling, which is the constraint nobody costed. The readout
    # delivers 21 independent directions no matter how many columns it has --
    # measured, and identical for 180 and 240 features. The useful product
    # family has 75 terms. So project each design matrix onto its own top 21
    # principal components (fitted on TRAIN only) and re-score: that bounds what
    # ANY rank-21 readout could extract from these terms, however perfectly the
    # physics produced them.
    n_wash, n_train, n_val = splits
    trs = slice(n_wash, n_wash + n_train)
    print(f"\n{'design matrix':<32} {'dim':>5} {'full':>8} {'rank-21':>9} "
          f"{'taxed':>8}")
    rank_rows = {}
    for name, extra in (("linear 20 lags", []),
                        ("linear + far pairs |i-j|>=5",
                         [prod(i, j) for i in range(K) for j in range(i + 5, K)]),
                        ("linear + ALL products",
                         [poly2(u, 15)[:, a.base_lags:]])):
        X = L if not extra else np.concatenate([L] + extra, axis=1)
        mu, sd = X[trs].mean(0), X[trs].std(0)
        Zx = (X - mu) / np.where(sd < 1e-12, 1.0, sd)
        _, _, Vt = np.linalg.svd(Zx[trs], full_matrices=False)
        P = Zx @ Vt[:21].T
        full_v, r21 = score(X, y, splits), score(P, y, splits)
        rank_rows[name] = {"dim": int(X.shape[1]), "full": full_v,
                           "rank21": r21, "taxed": r21 * pen}
        print(f"{name:<32} {X.shape[1]:>5} {full_v:>8.4f} {r21:>9.4f} "
              f"{r21*pen:>8.4f}")
    best21 = min(r["taxed"] for r in rank_rows.values())
    print(f"\nbest achievable through a rank-21 readout, after the device's "
          f"measured\n1.53x tax: {best21:.4f}, against the {lin_best:.4f} bar. "
          + ("FEASIBLE." if best21 < lin_best else "NOT feasible this way."))

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(
        {"linear_best": lin_best, "device_nmse": a.device_nmse,
         "readout_penalty": pen, "arms": rows,
         "rank_limited": rank_rows}, indent=2))
    print(f"\nwrote {outdir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
