#!/usr/bin/env python3
"""The NARMA-10 certification, as three panels.

    left    test NMSE against linear-filter depth, six seeds, and where the
            device lands on that axis. The cliff at eleven lags is the u[n-10]
            column entering -- one of the two inputs to the task's own product
            term, which a "10-lag" design matrix excludes.
    middle  per-seed paired comparison, device against the best linear filter,
            with the mean difference and its 95% interval
    right   one test window: truth, device, best linear filter

    python scripts/render_certification.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from magnonic_nn.reservoir import narma10, ridge_fit, ridge_predict, nmse

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb", "#e4e2dd"
LAMS = tuple(10.0 ** e for e in range(-10, 7))


def load(path):
    o = torch.load(path, weights_only=False, map_location="cpu")
    return (o["feats"] if isinstance(o, dict) else o).cpu().numpy()


def lags(u, k):
    out = np.zeros((len(u), k))
    for j in range(k):
        out[j:, j] = u[:len(u) - j]
    return out


def fit(X, y, splits, want_pred=False):
    w0, tr_n, va_n = splits
    tr, va = slice(w0, w0 + tr_n), slice(w0 + tr_n, w0 + tr_n + va_n)
    te = slice(w0 + tr_n + va_n, len(y))
    if X.shape[1]:
        mu, sd = X[tr].mean(0), X[tr].std(0)
        X = (X - mu) / np.where(sd < 1e-12, 1.0, sd)
    best = None
    for lam in LAMS:
        w = ridge_fit(X[tr], y[tr], lam)
        e = nmse(y[va], ridge_predict(X[va], w))
        if best is None or e < best[0]:
            best = (e, w)
    p = ridge_predict(X[te], best[1])
    return (nmse(y[te], p), p) if want_pred else nmse(y[te], p)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/certify")
    p.add_argument("--splits", type=int, nargs=3, default=(150, 600, 150))
    p.add_argument("--max-lag", type=int, default=32)
    p.add_argument("--out", default="runs/certification.png")
    args = p.parse_args()

    root = Path(args.root)
    cert = json.loads((root / "certification.json").read_text())
    ks = cert["seeds"]
    n, splits = cert["frames"], tuple(cert["splits"])
    feats = {k: load(root / f"seed_{k}" / "features_multitone.pt")[:n] for k in ks}

    curves, dev, lin = [], [], []
    for k in ks:
        u, y = narma10(n, seed=k)
        curves.append([fit(lags(u, j), y, splits) for j in range(1, args.max_lag + 1)])
        dev.append(cert["per_seed"][str(k)]["device"])
        lin.append(cert["per_seed"][str(k)]["linear_best"])
    C = np.array(curves)
    x = np.arange(1, args.max_lag + 1)

    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(14.4, 4.5), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.15, 0.8, 1.25], "wspace": 0.27,
                     "left": 0.055, "right": 0.985, "top": 0.87, "bottom": 0.145})

    # ---- left: the memory cliff -----------------------------------------
    a1.fill_between(x, C.min(0), C.max(0), color=BLUE, alpha=0.16, lw=0)
    a1.plot(x, C.mean(0), color=BLUE, lw=2.1, label="linear filter on u")
    a1.axhline(np.mean(dev), color=ORANGE, lw=2.1, ls=(0, (5, 3)), label="device")
    a1.axvline(11, color=MUTED, lw=1.0, ls=(0, (2, 3)))
    a1.annotate("u[n-10] enters", xy=(11, C.mean(0)[10]), xytext=(13.5, 0.62),
                fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    a1.set_xlabel("lags of u", fontsize=9, color=INK2)
    a1.set_ylabel("test NMSE", fontsize=9, color=INK2)
    a1.set_title("what memory buys", fontsize=10.5, color=INK, loc="left", pad=7)
    a1.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")

    # ---- middle: paired ---------------------------------------------------
    for i, k in enumerate(ks):
        a2.plot([0, 1], [lin[i], dev[i]], color=MUTED, lw=0.9, zorder=1)
    a2.scatter(np.zeros(len(ks)), lin, s=44, color=BLUE, zorder=2, edgecolor=SURFACE)
    a2.scatter(np.ones(len(ks)), dev, s=44, color=ORANGE, zorder=2, edgecolor=SURFACE)
    a2.set_xlim(-0.42, 1.42); a2.set_xticks([0, 1])
    a2.set_xticklabels(["best linear", "device"], fontsize=9, color=INK2)
    a2.set_ylabel("test NMSE", fontsize=9, color=INK2)
    lo, hi = cert["tiers"]["tier2"]["lo"], cert["tiers"]["tier2"]["hi"]
    a2.set_title(f"{len(ks)} seeds · Δ 95% CI [{lo:+.3f}, {hi:+.3f}]",
                 fontsize=10.5, color=INK, loc="left", pad=7)

    # ---- right: one test window ------------------------------------------
    k0 = ks[0]
    u, y = narma10(n, seed=k0)
    nl = cert["best_lags"][str(k0)] if isinstance(cert["best_lags"], dict) \
        else cert["best_lags"][0]
    _, pd_ = fit(feats[k0], y, splits, want_pred=True)
    _, pl_ = fit(lags(u, int(nl)), y, splits, want_pred=True)
    t0 = sum(splits)
    m = slice(0, min(160, n - t0))
    xs = np.arange(t0, t0 + len(y[t0:][m]))
    a3.plot(xs, y[t0:][m], color=INK, lw=2.0, label="truth")
    a3.plot(xs, pl_[m], color=BLUE, lw=1.5, label=f"linear ({nl} lags)")
    a3.plot(xs, pd_[m], color=ORANGE, lw=1.5, label="device")
    a3.set_xlabel("frame", fontsize=9, color=INK2)
    a3.set_title(f"held-out window · seed {k0}", fontsize=10.5, color=INK,
                 loc="left", pad=7)
    a3.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3,
              loc="upper center")

    for ax in (a1, a2, a3):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)

    verdict = "CERTIFIED" if cert["certified"] else "NOT CERTIFIED"
    fig.suptitle(f"NARMA-10 · {verdict}", fontsize=13.5, fontweight="bold",
                 color=INK, x=0.055, ha="left", y=0.965)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
