#!/usr/bin/env python3
"""Certify -- or fail to certify -- the ported disk on NARMA-10.

The protocol, the arms, and the decision rule are fixed in
docs/narma10_certification.md and were written before any device data for this
protocol existed. This script only executes them.

Three tiers, all decided by the upper bound of a 95% t-interval on a paired
per-seed difference, all requiring the leakage guard to pass:

    tier 1   device < linear 10-lag     the conventional baseline. Reported for
                                        comparability only: measuring the
                                        baselines on the target alone shows a
                                        10-lag filter loses 0.71 where a 20-lag
                                        filter gets 0.17, because NARMA-10's
                                        y-recursion carries u's influence far
                                        past ten steps. Beating it certifies
                                        nothing.
    tier 2   device < best linear       THE certification. Lag count chosen on
                                        validation, so the linear filter gets
                                        every advantage the data allows.
    tier 3   linear+device < linear     residual value. Does the device add
                                        anything a linear filter has not
                                        already extracted? Stays meaningful
                                        when tier 2 fails.

A favourable mean with an interval straddling zero is not a pass. That is
already the state of the single-draw number this exists to resolve.

Runs on cached features -- no simulation.

    python scripts/certify_narma10.py --root runs/certify
    python scripts/certify_narma10.py --root runs/certify --seeds 0 1 2
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from scipy import stats
from magnonic_nn.reservoir import narma10, ridge_fit, ridge_predict, nmse

SHIFT = 37          # the leakage guard's misalignment: longer than any memory
                    # the device has, so nothing legitimate can survive it
LAMS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
LAG_CHOICES = (10, 20, 40)


def narma_from(u: np.ndarray, order: int) -> np.ndarray:
    """NARMA-`order` target for an input the device has ALREADY been driven with.

    The family table has to reuse each seed's own `u` -- that is the sequence the
    magnetisation actually saw -- and vary only the target. Regenerating `u` per
    order would score the device against a history it never experienced.
    """
    y = np.zeros(len(u))
    for k in range(order, len(u) - 1):
        y[k + 1] = (0.3 * y[k] + 0.05 * y[k] * y[k - order + 1:k + 1].sum()
                    + 1.5 * u[k - order + 1] * u[k] + 0.1)
        if not np.isfinite(y[k + 1]) or abs(y[k + 1]) > 1e3:
            y[k + 1] = 0.0
    return y


def load_features(path: Path) -> np.ndarray:
    obj = torch.load(path, weights_only=False, map_location="cpu")
    t = obj["feats"] if isinstance(obj, dict) else obj
    return t.cpu().numpy()


def lags(u: np.ndarray, k: int) -> np.ndarray:
    """u_n ... u_{n-k+1}, zero-padded at the start (inside the washout)."""
    out = np.zeros((len(u), k))
    for j in range(k):
        out[j:, j] = u[:len(u) - j]
    return out


def poly2(u: np.ndarray, k: int) -> np.ndarray:
    """The k lags and every product of them -- the nonlinearity in software."""
    L = lags(u, k)
    prods = [L[:, i] * L[:, j] for i in range(k) for j in range(i, k)]
    return np.concatenate([L, np.stack(prods, axis=1)], axis=1)


def standardise(X: np.ndarray, tr: slice) -> np.ndarray:
    """Centre and scale on the TRAINING block only.

    The rest of this project standardises on the whole series, which quietly
    lets the test block set its own mean and variance. A small leak, but a leak,
    and a certification is the one place it cannot be waved off.
    """
    if X.shape[1] == 0:
        return X
    mu, sd = X[tr].mean(0), X[tr].std(0)
    return (X - mu) / np.where(sd < 1e-12, 1.0, sd)


def score(X: np.ndarray, y: np.ndarray, splits):
    """Test NMSE with lambda chosen on validation. Test is never fitted on."""
    n_wash, n_train, n_val = splits
    tr = slice(n_wash, n_wash + n_train)
    va = slice(n_wash + n_train, n_wash + n_train + n_val)
    te = slice(n_wash + n_train + n_val, len(y))
    Z = standardise(X, tr)
    best = None
    for lam in LAMS:
        w = ridge_fit(Z[tr], y[tr], lam)
        e = nmse(y[va], ridge_predict(Z[va], w))
        if best is None or e < best[0]:
            best = (e, lam, w)
    return nmse(y[te], ridge_predict(Z[te], best[2]))


def best_lag_count(u, y, splits) -> int:
    """Pick the linear baseline's depth on VALIDATION, so the filter the device
    must beat is the best one the data supports rather than a convention."""
    n_wash, n_train, n_val = splits
    tr = slice(n_wash, n_wash + n_train)
    va = slice(n_wash + n_train, n_wash + n_train + n_val)
    best = None
    for k in LAG_CHOICES:
        X = standardise(lags(u, k), tr)
        for lam in LAMS:
            w = ridge_fit(X[tr], y[tr], lam)
            e = nmse(y[va], ridge_predict(X[va], w))
            if best is None or e < best[0]:
                best = (e, k)
    return best[1]


def paired(per_seed, a, b, ks):
    """Paired t-interval on NMSE(a) - NMSE(b). Negative favours a."""
    d = np.array([per_seed[k][a] - per_seed[k][b] for k in ks])
    K = len(d)
    mean = float(d.mean())
    if K < 2:
        return {"d": d, "mean": mean, "lo": float("nan"), "hi": float("nan"),
                "t": float("nan"), "p": float("nan"), "wins": int((d < 0).sum()),
                "p_sign": float("nan"), "K": K}
    se = float(d.std(ddof=1) / np.sqrt(K))
    tc = float(stats.t.ppf(0.975, K - 1))
    t = mean / se if se > 0 else float("nan")
    return {"d": d, "mean": mean, "lo": mean - tc * se, "hi": mean + tc * se,
            "t": t, "p": float(stats.t.cdf(t, K - 1)) if np.isfinite(t) else float("nan"),
            "wins": int((d < 0).sum()),
            "p_sign": float(stats.binomtest(int((d < 0).sum()), K, 0.5,
                                            alternative="greater").pvalue),
            "K": K}


def show(label, st, guard_ok):
    passed = bool(st["hi"] < 0 and guard_ok)
    print(f"  {label}")
    print("    " + "  ".join(f"{v:+.4f}" for v in st["d"]))
    print(f"    mean {st['mean']:+.4f}   95% CI [{st['lo']:+.4f}, {st['hi']:+.4f}]"
          f"   t({st['K']-1}) = {st['t']:.2f}, p = {st['p']:.4f}"
          f"   sign {st['wins']}/{st['K']}"
          f"   -> {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/certify")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--splits", type=int, nargs=3, default=(150, 600, 150),
                   help="washout / train / val; test is the remainder")
    p.add_argument("--frames", type=int, default=None,
                   help="truncate every seed to this many frames. Scores in "
                        "this project move with series length, so the arms must "
                        "be matched on it; defaults to the shortest seed found.")
    p.add_argument("--family", type=int, nargs="*", default=[2, 3, 5],
                   help="extra NARMA orders for the supplementary table, "
                        "scored on each seed's own u. Pass nothing to skip.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = Path(args.root)
    found = {}
    for d in sorted(root.glob("seed_*")):
        f = d / "features_multitone.pt"
        if not f.exists():
            continue
        k = int(d.name.split("_")[1])
        if args.seeds is None or k in args.seeds:
            found[k] = load_features(f)
    if not found:
        print(f"no features under {root}/seed_*/features_multitone.pt")
        return 1

    n = args.frames or min(len(F) for F in found.values())
    splits = tuple(args.splits)
    n_test = n - sum(splits)
    if n_test < 50:
        print(f"only {n_test} test frames at {n} frames with splits {splits}; "
              f"refusing to certify on that")
        return 1

    ks = sorted(found)
    print(f"seeds {ks}   frames {n} (matched)   splits {splits} + {n_test} test"
          f"   device dim {found[ks[0]].shape[1]}")
    short = [k for k in ks if len(found[k]) < n]
    if short:
        print(f"  truncated to the shortest complete seed; still running: {short}")
    print()

    per_seed, chosen = {}, {}
    for k in ks:
        F = found[k][:n]
        u, y = narma10(n, seed=k)
        nl = best_lag_count(u, y, splits)
        chosen[k] = nl
        L = lags(u, nl)
        A = {
            "constant":       np.zeros((n, 0)),
            "input_only":     u[:, None],
            "linear_10lag":   lags(u, 10),
            "linear_20lag":   lags(u, 20),
            "linear_best":    L,
            "poly2_10lag":    poly2(u, 10),
            "device":         F,
            "linear+device":  np.concatenate([L, F], axis=1),
            "device_shifted": np.roll(F, SHIFT, axis=0),
        }
        per_seed[k] = {name: score(X, y, splits) for name, X in A.items()}

    order = ["constant", "input_only", "linear_10lag", "linear_20lag",
             "linear_best", "poly2_10lag", "device", "linear+device",
             "device_shifted"]
    print("test NMSE by seed")
    print(f"  {'arm':<15}" + "".join(f"{k:>9d}" for k in ks) + f"{'mean':>9}")
    for name in order:
        v = [per_seed[k][name] for k in ks]
        print(f"  {name:<15}" + "".join(f"{x:>9.4f}" for x in v)
              + f"{np.mean(v):>9.4f}")
    print(f"  {'(best lags)':<15}" + "".join(f"{chosen[k]:>9d}" for k in ks))
    print()

    guard = [per_seed[k]["device_shifted"] for k in ks]
    guard_ok = all(v > 0.9 for v in guard)
    print(f"leakage guard   device features shifted {SHIFT} frames against the "
          f"target: NMSE {min(guard):.3f}-{max(guard):.3f}")
    print("  " + ("PASS -- misaligned features predict nothing, as they must"
                  if guard_ok else
                  "FAIL -- something is scoring alignment that is not there; "
                  "nothing below is interpretable"))
    print()

    t1 = paired(per_seed, "device", "linear_10lag", ks)
    t2 = paired(per_seed, "device", "linear_best", ks)
    t3 = paired(per_seed, "linear+device", "linear_best", ks)
    print("paired differences (negative favours the device)")
    p1 = show("tier 1   device - linear_10lag   [conventional, not decisive]",
              t1, guard_ok)
    p2 = show("tier 2   device - linear_best    [THE certification]", t2, guard_ok)
    p3 = show("tier 3   (linear+device) - linear_best   [residual value]",
              t3, guard_ok)
    print()

    dev = float(np.mean([per_seed[k]["device"] for k in ks]))
    lin = float(np.mean([per_seed[k]["linear_best"] for k in ks]))
    if p2:
        print(f"CERTIFIED for NARMA-10. Across {len(ks)} independent input "
              f"realisations the device\nbeats the best linear filter on the "
              f"same input history ({dev:.4f} vs {lin:.4f}), and\nthe interval "
              f"excludes zero.")
    else:
        print(f"NOT CERTIFIED for NARMA-10. The device scores {dev:.4f} against "
              f"{lin:.4f} for the best\nlinear filter on the same input.", end=" ")
        if p1:
            print("It does beat the conventional 10-lag\nbaseline "
                  f"({per_seed[ks[0]]['linear_10lag']:.4f}-class), which is what "
                  "this project reported before -- but that\nbaseline is "
                  "under-powered for NARMA-10 and beating it is not a result.")
        else:
            print("It does not beat the conventional\n10-lag baseline either.")
        if p3:
            print(f"\nThe device does have residual value: appended to the "
                  f"linear filter it improves on\nit by {-t3['mean']:.4f} NMSE, "
                  f"significantly. The nonlinearity is real and additive; there\n"
                  f"is just not enough memory behind it for this task.")
        else:
            print("\nNor does it add anything on top of the linear filter "
                  "(tier 3 fails), so on this\ntask the ports are not "
                  "contributing computation a linear readout lacks.")

    print(f"\ncalibration   device NRMSE "
          f"{np.mean([np.sqrt(per_seed[k]['device']) for k in ks]):.3f}   "
          f"best linear NRMSE {np.sqrt(lin):.3f}   "
          f"(few-hundred-node echo-state networks reach ~0.2)")

    # ---- supplementary: the rest of the NARMA family ---------------------
    # EXPLORATORY, not part of the certification. The pre-registered claim is
    # NARMA-10; these orders are scored on the same features and the same seeds
    # afterwards, so a pass here is one of a dozen comparisons and carries the
    # multiplicity that implies. Reported because NARMA-2 is where this project
    # last claimed a win, and it is the order the device comes closest on.
    fam = {}
    if args.family:
        print(f"\nsupplementary -- NARMA-{{{','.join(map(str, args.family))}}} on the "
              f"same features and seeds.\nEXPLORATORY: not pre-registered, and "
              f"unadjusted for {3 * len(args.family)} extra comparisons.")
        print(f"  {'order':>5} {'conv n-lag':>11} {'best lin':>9} {'device':>8} "
              f"{'lin+dev':>8} {'tier2 CI':>22} {'tier3 CI':>22}")
        for order in args.family:
            ps = {}
            for k in ks:
                u, _ = narma10(n, seed=k)
                y = narma_from(u, order)
                nl = best_lag_count(u, y, splits)
                L = lags(u, nl)
                ps[k] = {"conv": score(lags(u, order), y, splits),
                         "linear_best": score(L, y, splits),
                         "device": score(found[k][:n], y, splits),
                         "linear+device": score(
                             np.concatenate([L, found[k][:n]], axis=1), y, splits)}
            a = paired(ps, "device", "linear_best", ks)
            b = paired(ps, "linear+device", "linear_best", ks)
            mean = lambda nm: float(np.mean([ps[k][nm] for k in ks]))
            fam[order] = {"per_seed": ps, "tier2": {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in a.items()},
                          "tier3": {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in b.items()}}
            print(f"  {order:>5} {mean('conv'):>11.4f} {mean('linear_best'):>9.4f} "
                  f"{mean('device'):>8.4f} {mean('linear+device'):>8.4f} "
                  f"{'[%+.4f,%+.4f]%s' % (a['lo'], a['hi'], '*' if a['hi'] < 0 else ' '):>22} "
                  f"{'[%+.4f,%+.4f]%s' % (b['lo'], b['hi'], '*' if b['hi'] < 0 else ' '):>22}")
        print("  * interval excludes zero in the device's favour")

    out = Path(args.out) if args.out else root / "certification.json"
    out.write_text(json.dumps({
        "claim": "device beats the best linear filter on NARMA-10, across "
                 "independent input realisations",
        "certified": p2, "tier1_beats_10lag": p1, "tier3_residual_value": p3,
        "guard_ok": guard_ok, "frames": n, "splits": list(splits),
        "n_test": n_test, "seeds": ks, "best_lags": chosen,
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "tiers": {name: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                         for kk, vv in st.items()}
                  for name, st in (("tier1", t1), ("tier2", t2), ("tier3", t3))},
        "family_exploratory": {str(o): {"tier2": v["tier2"], "tier3": v["tier3"],
                                        "per_seed": {str(k): pv for k, pv
                                                     in v["per_seed"].items()}}
                               for o, v in fam.items()},
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
