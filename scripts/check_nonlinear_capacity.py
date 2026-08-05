#!/usr/bin/env python3
"""Does the device compute anything a linear filter cannot? Decompose its capacity.

The depth ladder ended at a specific, uncomfortable number: three stages score
0.2120 on NARMA-10 against 0.1243 for a 20-lag LINEAR filter. The chain has the
memory -- r^2 > 0.5 recall of the input out to lag 15, MC 15.7 -- and still
loses to a filter that does nothing but weight past inputs. A reservoir that
holds fifteen lags and is beaten by a linear tap on twenty is not short of
memory. It is failing to compute.

NARMA-10 says exactly what computing would mean here:

    y[n+1] = 0.3 y[n] + 0.05 y[n] * sum(y[n-9..n]) + 1.5 * u[n-9] * u[n] + 0.1

The term `u[n-9] * u[n]` is a product of two inputs nine steps apart. No linear
filter can produce it at any lag count. A reservoir is supposed to supply
precisely that -- it is the textbook justification for the architecture -- and
if the device carried it, the linear baseline could not keep up.

So measure it directly, per Dambre et al.'s information processing capacity. For
each target function of the input history, ridge-regress it from the device
state and record the test r^2. Summed over an ORTHOGONAL family of targets those
r^2 are additive and bounded by the state's rank, so the split between degrees
says where the device's finite capacity is being spent.

Orthogonality matters and is easy to get wrong. u ~ U[0, 0.5] is non-negative,
so u^2 correlates with u at ~0.97 and a naive "u^2 capacity" is mostly linear
memory wearing a hat. The fix is the standard one: rescale to s = 4u - 1 on
[-1, 1], where the Legendre polynomials are orthogonal for uniform input.

    degree 1   P1(s[n-k]) = s[n-k]              linear memory
    degree 2   P2(s[n-k]) = (3 s^2 - 1) / 2     same-lag nonlinearity
    degree 2   s[n] * s[n-k]                    CROSS-LAG product -- the one
                                                NARMA-10 needs, at k = 9

Two controls, because a capacity estimate with 300 training rows and 180
features will report something for any target at all:

  floor     the same targets shifted 37 frames out of reach -- the certification
            protocol's leakage guard, reused. Anything at or below this is what
            the estimator returns for information the device cannot have.
  worth     poly2 baselines: a linear filter given the products in software. If
            the products are cheap to buy and worth a lot, then the device not
            having them is the whole story.

Runs on cached features. No simulation.

    python scripts/check_nonlinear_capacity.py                    # 3-chain
    python scripts/check_nonlinear_capacity.py \
        --features runs/chain4_narma/features_n4_linked.pt --n-disks 4
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from magnonic_nn.reservoir import narma10
from certify_narma10 import (LAMS, SHIFT, lags, poly2, ridge_fit, ridge_predict,
                             score, standardise, LAG_CHOICES)

TONES = ("12.0 (carrier)", "9.9", "10.3", "13.7", "24.0 (2f)")


def r2_of(Z, t, tr, va, te):
    """Test r^2 of reconstructing target t, lambda chosen on validation."""
    best, w = None, None
    for lam in LAMS:
        ww = ridge_fit(Z[tr], t[tr], lam)
        e = float(np.mean((t[va] - ridge_predict(Z[va], ww)) ** 2))
        if best is None or e < best:
            best, w = e, ww
    p = ridge_predict(Z[te], w)
    if p.std() <= 1e-12:
        return 0.0
    c = np.corrcoef(t[te], p)[0, 1]
    return float(max(c, 0.0) ** 2) if np.isfinite(c) else 0.0


def shifted(x, k):
    """x[n-k], zero-padded at the front."""
    out = np.zeros_like(x)
    out[k:] = x[:len(x) - k] if k else x
    return out


def targets(s, max_lag):
    """The orthogonal family, as {family: {lag: vector}}.

    Degree 3 and the delayed cross-product are here so that "the device is
    linear" is a measurement rather than a consequence of only having looked at
    degree 2. Total measured capacity still falls short of the state's rank, and
    it must: this is a handful of families out of an infinite basis. What the
    accounting supports is the SHARE -- of the capacity that was found, how much
    is degree 1 -- not a claim that the unmeasured remainder is empty.
    """
    p1 = s
    p2 = (3 * s ** 2 - 1) / 2
    p3 = (5 * s ** 3 - 3 * s) / 2
    fam = {
        "deg1  P1(s[n-k])": {k: shifted(p1, k) for k in range(max_lag + 1)},
        "deg2  P2(s[n-k])": {k: shifted(p2, k) for k in range(max_lag + 1)},
        "deg2  s[n]*s[n-k]": {k: p1 * shifted(p1, k)
                              for k in range(1, max_lag + 1)},
        # A cross product that does NOT involve the newest sample. If the device
        # only ever mixes the present with the past, this family stays empty
        # while the one above fills -- a distinction the task cares about.
        "deg2  s[n-5]*s[n-k]": {k: shifted(p1, 5) * shifted(p1, k)
                                for k in range(6, max_lag + 1)},
        "deg3  P3(s[n-k])": {k: shifted(p3, k) for k in range(max_lag + 1)},
    }
    return fam


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
    s = 4.0 * u - 1.0                     # U[0, 0.5] -> [-1, 1]
    n_wash, n_train, n_val = a.splits
    tr = slice(n_wash, n_wash + n_train)
    va = slice(n_wash + n_train, n_wash + n_train + n_val)
    te = slice(n_wash + n_train + n_val, a.frames)

    per_disk = a.n_ports * 2
    block = a.n_disks * per_disk
    assert F.shape[1] == a.n_tones * block, (F.shape, a.n_tones * block)

    Z = standardise(F, tr)
    sv = np.linalg.svd(F[tr] - F[tr].mean(0), compute_uv=False)
    rank = int(np.searchsorted(np.cumsum(sv ** 2) / np.sum(sv ** 2), 0.99) + 1)
    print(f"features {F.shape}, effective rank {rank} "
          f"(99% of variance) -- the ceiling on total capacity\n")

    fam = targets(s, a.max_lag)

    # The floor. Same estimator, same shapes, on targets built from an input the
    # device never saw at that alignment.
    floor = {}
    for name, byk in fam.items():
        vals = [r2_of(Z, shifted(t, SHIFT), tr, va, te) for t in byk.values()]
        floor[name] = float(np.max(vals))
    fl = max(floor.values())
    print("noise floor (targets shifted %d frames out of reach):" % SHIFT)
    for name, v in floor.items():
        print(f"  {name:<20} max r^2 {v:.3f}")
    print(f"  -> treating r^2 <= {fl:.3f} as nothing\n")

    print(f"{'lag':>4} " + " ".join(f"{n.split()[-1]:>18}" for n in fam))
    curves = {n: {} for n in fam}
    for k in range(a.max_lag + 1):
        cells = []
        for name, byk in fam.items():
            if k not in byk:
                cells.append(f"{'-':>18}")
                continue
            v = r2_of(Z, byk[k], tr, va, te)
            curves[name][k] = v
            mark = "*" if k == 9 else " "
            cells.append(f"{v:>17.3f}{mark}")
        print(f"{k:>4} " + " ".join(cells), flush=True)
    print("   (* = lag 9, the pairing NARMA-10's product term needs)")

    print(f"\n{'family':<22} {'capacity (sum r^2 above floor)':>32}")
    cap = {}
    for name, c in curves.items():
        cap[name] = float(sum(max(v - fl, 0.0) for v in c.values()))
        print(f"{name:<22} {cap[name]:>32.2f}")
    tot = sum(cap.values())
    lin = cap["deg1  P1(s[n-k])"]
    print(f"{'TOTAL measured':<22} {tot:>32.2f}   of rank {rank}")
    print(f"{'degree-1 share':<22} {100*lin/max(tot,1e-9):>31.0f}% of measured")

    # Per-tone: is any of the nonlinearity showing up at 2f, where a nonlinear
    # response would put it? The readout already locks in at 24 GHz for exactly
    # this reason, so the answer is a column slice rather than a new run.
    print(f"\n{'tone (GHz)':<16} {'deg1':>8} {'deg2 P2':>9} {'deg2 prod':>11} {'deg3':>7}")
    per_tone = {}
    for t in range(a.n_tones):
        Zt = Z[:, t * block:(t + 1) * block]
        row = {}
        for name, byk in fam.items():
            row[name] = float(sum(max(r2_of(Zt, v, tr, va, te) - fl, 0.0)
                                  for v in byk.values()))
        per_tone[TONES[t]] = row
        print(f"{TONES[t]:<16} {row['deg1  P1(s[n-k])']:>8.2f} "
              f"{row['deg2  P2(s[n-k])']:>9.2f} "
              f"{row['deg2  s[n]*s[n-k]']:>11.2f} "
              f"{row['deg3  P3(s[n-k])']:>7.2f}", flush=True)

    # Is the 2f tone a genuine second harmonic, or the carrier leaking?
    #
    # It matters for the diagnosis. Real second-harmonic generation would be a
    # nonlinearity in the carrier dynamics -- something to build on. Leakage
    # from a 200-step lock-in window is a redundant copy of the 12 GHz block and
    # explains why 180 features carry rank 21.
    #
    # Canonical correlation separates them: a leaked copy is a linear image of
    # the carrier block and its canonical correlations sit near 1, while an
    # independently generated harmonic carries its own directions.
    def top_cc(A, B, k=3, eps=1e-6):
        Qa, _ = np.linalg.qr(A - A.mean(0) + eps * np.random.default_rng(0)
                             .standard_normal(A.shape))
        Qb, _ = np.linalg.qr(B - B.mean(0) + eps * np.random.default_rng(1)
                             .standard_normal(B.shape))
        sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
        return [float(min(x, 1.0)) for x in sv[:k]]

    carrier = Z[tr, 0:block]
    print(f"\n{'tone vs carrier':<16} {'canonical correlations (top 3)':>34}")
    ccs = {}
    for t in range(1, a.n_tones):
        c = top_cc(carrier, Z[tr, t * block:(t + 1) * block])
        ccs[TONES[t]] = c
        print(f"{TONES[t]:<16} {' '.join(f'{x:>10.4f}' for x in c):>34}")

    # What the missing products are worth, priced in software.
    print(f"\n{'baseline':<28} {'dim':>5} {'NARMA-10 NMSE':>15}")
    bl = {}
    for k in LAG_CHOICES:
        bl[f"linear {k} lags"] = (k, score(lags(u, k), y, a.splits))
    for k in (10, 11, 15):
        X = poly2(u, k)
        bl[f"linear+products {k} lags"] = (X.shape[1], score(X, y, a.splits))
    for name, (d, v) in bl.items():
        print(f"{name:<28} {d:>5} {v:>15.4f}")

    out = outdir / "nonlinear_capacity.json"
    out.write_text(json.dumps(
        {"rank": rank, "floor": floor, "curves": curves, "capacity": cap,
         "per_tone": per_tone, "tone_vs_carrier_cc": ccs,
         "baselines": {k: {"dim": d, "nmse": v} for k, (d, v) in bl.items()}},
        indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
