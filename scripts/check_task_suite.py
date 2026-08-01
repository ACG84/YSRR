#!/usr/bin/env python3
"""What CAN this reservoir compute? Capacity by nonlinearity degree and delay.

NARMA-10 needs ten lags and the disk holds three, so it loses to a shift
register. That says what the device cannot do; it says nothing about what it
can. Rather than guess at a friendlier benchmark and risk picking one that
flatters the hardware, this maps the capability directly, in the frame of
Dambre et al.'s information processing capacity: how well the reservoir state
reconstructs a family of target functions of the recent input, indexed by how
far back they reach and how nonlinear they are.

    degree 1   u_{n-k}                 pure memory, no nonlinearity
    degree 2   u_{n-i} u_{n-j}         products -- needs BOTH memory and mixing
    degree 2   u_{n-k}^2 - <u^2>       pure nonlinearity at one delay
    degree 3   u_{n-i} u_{n-j} u_{n-l}

Degree-1 capacity is what a delay line gives for free, so it is the baseline to
beat rather than a result. Everything at degree >= 2 is work a linear filter
cannot do at any length -- that is the part the disk's three-magnon scattering
is supposed to supply, and the part worth building a device for.

Also scores the NARMA family at several orders, since NARMA-n's memory
requirement is n and the device's ceiling should show as a knee.

Runs entirely on cached features -- no simulation. Every readout uses the same
washout/train/val/test protocol as the main experiments, and every target is
scored on data never fitted.

    python scripts/check_task_suite.py
    python scripts/check_task_suite.py --features runs/narma_lowabs/features_multitone.pt
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from magnonic_nn.reservoir import ridge_fit, ridge_predict, nmse


def load_features(path):
    """Accept both checkpoint formats.

    Checkpoints now carry {"feats", "m"} so a restart can reload the reservoir
    state instead of replaying; older files are a bare tensor.
    """
    obj = torch.load(path, weights_only=False)
    return obj["feats"] if isinstance(obj, dict) else obj


def narma(n_order, length, seed=0):
    """NARMA-n. Memory requirement is n, so the family sweeps exactly the axis
    this device is short on."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, length)
    y = np.zeros(length)
    for k in range(n_order, length - 1):
        y[k + 1] = (0.3 * y[k]
                    + 0.05 * y[k] * y[k - n_order + 1:k + 1].sum()
                    + 1.5 * u[k - n_order + 1] * u[k] + 0.1)
        if not np.isfinite(y[k + 1]) or abs(y[k + 1]) > 1e3:
            y[k + 1] = 0.0                      # NARMA diverges for some draws
    return u, y


def score(X, t, splits, lams=(1e-8, 1e-6, 1e-4, 1e-2, 1.0)):
    """r^2 on held-out data, chosen by validation. 0 if the target is constant."""
    n_wash, n_train, n_val = splits
    tr = slice(n_wash, n_wash + n_train)
    va = slice(n_wash + n_train, n_wash + n_train + n_val)
    te = slice(n_wash + n_train + n_val, len(t))
    if t[te].std() < 1e-12:
        return 0.0
    best, wb = None, None
    for lam in lams:
        w = ridge_fit(X[tr], t[tr], lam)
        e = float(np.mean((t[va] - ridge_predict(X[va], w)) ** 2))
        if best is None or e < best:
            best, wb = e, w
    p = ridge_predict(X[te], wb)
    if p.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(t[te], p)[0, 1] ** 2)


def shift(u, k):
    out = np.zeros_like(u)
    if k > 0:
        out[k:] = u[:len(u) - k]
    else:
        out[:] = u
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="runs/narma_modal/features.pt")
    p.add_argument("--splits", type=int, nargs=3, default=(150, 650, 150))
    p.add_argument("--max-lag", type=int, default=8)
    p.add_argument("--narma-orders", type=int, nargs="+", default=[2, 3, 5, 10])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    F = load_features(args.features).numpy()
    n = len(F)
    X = (F - F.mean(0)) / F.std(0).clip(1e-12)
    splits = tuple(args.splits)
    rng = np.random.default_rng(0)
    u = rng.uniform(0.0, 0.5, n)                 # same draw narma10() uses
    uc = u - u.mean()

    print(f"features {Path(args.features).name}  dim {X.shape[1]}  frames {n}\n")

    # ---- degree 1: pure memory ------------------------------------------
    lin = [score(X, shift(uc, k), splits) for k in range(args.max_lag + 1)]
    print("degree 1 -- pure memory, u_(n-k).  A delay line gives this free.")
    print("  lag   " + " ".join(f"{k:>5d}" for k in range(args.max_lag + 1)))
    print("  r^2   " + " ".join(f"{v:>5.2f}" for v in lin))
    print(f"  linear capacity {sum(lin):.2f}\n")

    # ---- degree 2: products ---------------------------------------------
    print("degree 2 -- u_(n-i) u_(n-j).  NO linear filter can do this at any")
    print("length; it is what the nonlinearity is for.")
    quad = {}
    hdr = "  i\\j  " + " ".join(f"{j:>5d}" for j in range(args.max_lag + 1))
    print(hdr)
    for i in range(args.max_lag + 1):
        row = []
        for j in range(args.max_lag + 1):
            if j < i:
                row.append("     ")
                continue
            t = shift(uc, i) * shift(uc, j)
            v = score(X, t - t.mean(), splits)
            quad[(i, j)] = v
            row.append(f"{v:>5.2f}")
        print(f"  {i:>3d}  " + " ".join(row))
    qcap = sum(quad.values())
    print(f"  quadratic capacity {qcap:.2f}\n")

    # ---- degree 3 --------------------------------------------------------
    cub = {}
    for i, j, k in itertools.combinations_with_replacement(range(4), 3):
        t = shift(uc, i) * shift(uc, j) * shift(uc, k)
        cub[(i, j, k)] = score(X, t - t.mean(), splits)
    ccap = sum(cub.values())
    best_c = max(cub.items(), key=lambda kv: kv[1])
    print(f"degree 3 -- capacity {ccap:.2f} over lags 0-3, best "
          f"u_(n-{best_c[0][0]}) u_(n-{best_c[0][1]}) u_(n-{best_c[0][2]}) "
          f"= {best_c[1]:.2f}\n")

    # ---- NARMA family ----------------------------------------------------
    print("NARMA-n (memory requirement n), test NMSE against a linear n-lag filter")
    print(f"  {'order':>6} {'reservoir':>11} {'linear':>9} {'verdict':>10}")
    narma_rows = []
    for order in args.narma_orders:
        uu, yy = narma(order, n, seed=0)
        L = np.stack([shift(uu, k) for k in range(order)], axis=1)
        n_wash, n_train, n_val = splits
        te = slice(n_wash + n_train + n_val, n)

        def nm(M):
            best, wb = None, None
            for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
                w = ridge_fit(M[slice(n_wash, n_wash + n_train)],
                              yy[slice(n_wash, n_wash + n_train)], lam)
                e = nmse(yy[slice(n_wash + n_train, n_wash + n_train + n_val)],
                         ridge_predict(M[slice(n_wash + n_train,
                                               n_wash + n_train + n_val)], w))
                if best is None or e < best:
                    best, wb = e, w
            return float(nmse(yy[te], ridge_predict(M[te], wb)))

        r, l = nm(X), nm(L)
        narma_rows.append({"order": order, "reservoir": r, "linear": l})
        mark = "WINS" if r < l * 0.95 else ("ties" if r < l * 1.1 else "loses")
        print(f"  {order:>6} {r:>11.4f} {l:>9.4f} {mark:>10}")

    out = Path(args.out) if args.out else \
        Path(args.features).parent / "task_suite.json"
    out.write_text(json.dumps(
        {"linear_capacity": sum(lin), "quadratic_capacity": qcap,
         "cubic_capacity": ccap, "linear_by_lag": lin,
         "quadratic": {f"{i},{j}": v for (i, j), v in quad.items()},
         "narma": narma_rows}, indent=2))

    print()
    reach = [(i, j) for (i, j), v in quad.items() if v > 0.15]
    if reach:
        deepest = max(max(i, j) for i, j in reach)
        print(f"Nonlinear products are recoverable out to lag {deepest}, with "
              f"quadratic capacity {qcap:.2f}.")
        print("That is the honest envelope: tasks whose nonlinearity spans no")
        print(f"more than ~{deepest} steps of history are within reach; ones")
        print("needing ten are not, and no readout fixes that.")
    else:
        print("No quadratic term is recoverable above noise. The device is")
        print("behaving as a linear filter, and the nonlinearity measured in")
        print("the AB/BA test is not reaching the readout in usable form.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
