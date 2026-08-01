#!/usr/bin/env python3
"""The NARMA-10 task through the ported disk, and an honest scoreboard.

Five panels. The one that carries the argument is E: the memory function,
measured rather than asserted. Fitting the reservoir's features to u_{n-k} for
each lag k shows how many steps of the past are actually recoverable, and
summing it gives the Jaeger memory capacity. That is what decides whether a tie
with the linear tap-delay is a memory problem or a nonlinearity problem -- and
they need different fixes, so guessing between them is expensive.

Panel D is the counterpart: predictions against truth on the held-out test
window, where a readout that tracks the slow structure and misses the sharp
excursions looks quite different from one that is simply noisy.

    python scripts/render_narma.py
    python scripts/render_narma.py --indir runs/narma_multitone --tag multitone
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
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
GRID = "#e4e2dd"


def load_features(path):
    """Accept both checkpoint formats.

    Checkpoints now carry {"feats", "m"} so a restart can reload the reservoir
    state instead of replaying; older files are a bare tensor.
    """
    obj = torch.load(path, weights_only=False, map_location="cpu")
    t = obj["feats"] if isinstance(obj, dict) else obj
    return t.cpu()          # a run on CUDA saves device tensors


def lag_matrix(u, n_lags):
    out = np.zeros((len(u), n_lags))
    for k in range(n_lags):
        out[k:, k] = u[:len(u) - k]
    return out


def memory_function(X, u, splits, max_lag=25, lams=(1e-6, 1e-4, 1e-2, 1.0)):
    """r^2 of reconstructing u_{n-k} from the reservoir state at n.

    The standard Jaeger measure. Its sum is the memory capacity, bounded above
    by the number of features, and its shape says WHERE the memory stops --
    which a single lag-1 autocorrelation cannot.
    """
    n_wash, n_train, n_val = splits
    tr = slice(n_wash, n_wash + n_train)
    te = slice(n_wash + n_train + n_val, len(u))
    out = []
    for k in range(max_lag + 1):
        tgt = np.zeros_like(u)
        tgt[k:] = u[:len(u) - k]
        best = -1.0
        for lam in lams:
            w = ridge_fit(X[tr], tgt[tr], lam)
            p = ridge_predict(X[te], w)
            a, b = tgt[te], p
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            best = max(best, float(np.corrcoef(a, b)[0, 1] ** 2))
        out.append(max(best, 0.0))
    return np.array(out)


def fit_predict(X, y, splits, lams=(1e-8, 1e-6, 1e-4, 1e-2, 1.0)):
    n_wash, n_train, n_val = splits
    tr = slice(n_wash, n_wash + n_train)
    va = slice(n_wash + n_train, n_wash + n_train + n_val)
    te = slice(n_wash + n_train + n_val, len(y))
    best = None
    for lam in lams:
        w = ridge_fit(X[tr], y[tr], lam)
        e = nmse(y[va], ridge_predict(X[va], w))
        if best is None or e < best[0]:
            best = (e, w)
    w = best[1]
    return ridge_predict(X[te], w), te


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="runs/narma_modal")
    p.add_argument("--features", default=None)
    p.add_argument("--frames", type=int, default=1200)
    p.add_argument("--splits", type=int, nargs=3, default=(150, 650, 150))
    p.add_argument("--tag", default="singletone")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    indir = Path(args.indir)
    fpath = Path(args.features) if args.features else None
    if fpath is None:
        cands = sorted(indir.glob("features*.pt"))
        if not cands:
            print(f"no features in {indir}"); return 2
        fpath = cands[0]
    F = load_features(fpath).numpy()
    n = len(F)
    u, y = narma10(args.frames, seed=0)
    u, y = u[:n], y[:n]
    X = (F - F.mean(0)) / F.std(0).clip(1e-12)
    splits = tuple(args.splits)

    rows = {
        "input only": lag_matrix(u, 1),
        "linear 10-lag": lag_matrix(u, 10),
        f"ports modal ({X.shape[1]}d)": X,
    }
    preds, scores = {}, {}
    for name, M in rows.items():
        pr, te = fit_predict(M, y, splits)
        preds[name] = pr
        scores[name] = float(nmse(y[te], pr))
    mf = memory_function(X, u, splits)
    mc = float(mf.sum())

    fig = plt.figure(figsize=(15.0, 9.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.85, 1.0, 1.0], hspace=0.52,
                          wspace=0.28, left=0.06, right=0.975, top=0.875,
                          bottom=0.07)

    # ---- A: the task ------------------------------------------------------
    axA = fig.add_subplot(gs[0, :2], facecolor=SURFACE)
    sl = slice(200, 380)
    axA.plot(np.arange(sl.start, sl.stop), u[sl], color=MUTED, lw=1.6)
    axA.plot(np.arange(sl.start, sl.stop), y[sl], color=INK, lw=2.2)
    axA.annotate("input u", (sl.start + 4, u[sl][:12].max()),
                 color=MUTED, fontsize=10, fontweight="bold")
    axA.annotate("target y  (NARMA-10)", (sl.start + 4, y[sl][:12].max() + 0.04),
                 color=INK, fontsize=10, fontweight="bold")
    axA.set_xlabel("frame", fontsize=9.5, color=INK2)
    axA.set_title("A · the task — y depends on the last 10 inputs, nonlinearly",
                  fontsize=12, fontweight="bold", color=INK, loc="left", pad=8)
    for s in axA.spines.values():
        s.set_color(GRID)
    axA.grid(color=GRID, lw=0.8); axA.set_axisbelow(True)
    axA.tick_params(colors=MUTED, labelsize=8.5)

    # ---- B: the scoreboard ------------------------------------------------
    axB = fig.add_subplot(gs[0, 2], facecolor=SURFACE)
    names = list(scores)
    vals = [scores[k] for k in names]
    cols = [MUTED, ORANGE, BLUE]
    bars = axB.barh(range(len(names)), vals, color=cols, edgecolor=SURFACE, lw=2)
    for i, v in enumerate(vals):
        axB.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=10,
                 color=INK, fontweight="bold")
    axB.set_yticks(range(len(names)))
    axB.set_yticklabels(names, fontsize=9.5, color=INK2)
    axB.invert_yaxis()
    axB.axvline(1.0, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    axB.text(1.02, len(names) - 0.4, "predicting\nthe mean", fontsize=8.5,
             color=MUTED)
    axB.set_xlim(0, max(vals) * 1.32)
    axB.set_xlabel("test NMSE  (lower is better)", fontsize=9.5, color=INK2)
    axB.set_title("B · scoreboard", fontsize=12, fontweight="bold", color=INK,
                  loc="left", pad=8)
    for s in axB.spines.values():
        s.set_color(GRID)
    axB.grid(color=GRID, lw=0.8, axis="x"); axB.set_axisbelow(True)
    axB.tick_params(colors=MUTED, labelsize=8.5)

    # ---- C: features ------------------------------------------------------
    axC = fig.add_subplot(gs[1, :], facecolor=SURFACE)
    v = np.percentile(np.abs(X), 99)
    im = axC.imshow(X[sl].T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                    extent=[sl.start, sl.stop, X.shape[1] - 0.5, -0.5],
                    interpolation="nearest")
    axC.set_ylabel("feature", fontsize=9.5, color=INK2)
    axC.set_xlabel("frame", fontsize=9.5, color=INK2)
    axC.set_title(f"C · what the ports deliver — {X.shape[1]} features per frame, "
                  "standardised", fontsize=12, fontweight="bold", color=INK,
                  loc="left", pad=8)
    cb = fig.colorbar(im, ax=axC, fraction=0.02, pad=0.012)
    cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_edgecolor(GRID)
    axC.tick_params(colors=MUTED, labelsize=8.5)
    for s in axC.spines.values():
        s.set_color(GRID)

    # ---- D: prediction on held-out test -----------------------------------
    axD = fig.add_subplot(gs[2, :2], facecolor=SURFACE)
    _, te = fit_predict(X, y, splits)
    tx = np.arange(te.start, te.stop)[:180]
    axD.plot(tx, y[te][:180], color=INK, lw=2.4)
    key = f"ports modal ({X.shape[1]}d)"
    axD.plot(tx, preds[key][:180], color=BLUE, lw=1.9)
    axD.plot(tx, preds["linear 10-lag"][:180], color=ORANGE, lw=1.5,
             ls=(0, (5, 2)))
    axD.annotate("truth", (tx[3], y[te][:8].max()), color=INK, fontsize=10,
                 fontweight="bold")
    axD.annotate("ports", (tx[3], preds[key][:8].max() + 0.03), color=BLUE,
                 fontsize=10, fontweight="bold")
    axD.annotate("linear 10-lag", (tx[3], preds["linear 10-lag"][:8].min() - 0.06),
                 color=ORANGE, fontsize=10, fontweight="bold")
    axD.set_xlabel("frame (held-out test)", fontsize=9.5, color=INK2)
    axD.set_title("D · prediction on data never fitted", fontsize=12,
                  fontweight="bold", color=INK, loc="left", pad=8)
    axD.grid(color=GRID, lw=0.8); axD.set_axisbelow(True)
    axD.tick_params(colors=MUTED, labelsize=8.5)
    for s in axD.spines.values():
        s.set_color(GRID)

    # ---- E: the memory function -------------------------------------------
    axE = fig.add_subplot(gs[2, 2], facecolor=SURFACE)
    axE.bar(range(len(mf)), mf, color=AQUA, edgecolor=SURFACE, lw=1.4)
    axE.axvspan(0, 10, color=ORANGE, alpha=0.10, zorder=0)
    axE.text(10.4, max(mf) * 0.86, "NARMA-10\nneeds these", fontsize=8.5,
             color=ORANGE, fontweight="bold")
    axE.set_xlabel("lag k (frames)", fontsize=9.5, color=INK2)
    axE.set_ylabel("r²  recovering u$_{n-k}$", fontsize=9.5, color=INK2)
    axE.set_title(f"E · memory function — capacity {mc:.1f}", fontsize=12,
                  fontweight="bold", color=INK, loc="left", pad=8)
    axE.grid(color=GRID, lw=0.8, axis="y"); axE.set_axisbelow(True)
    axE.tick_params(colors=MUTED, labelsize=8.5)
    for s in axE.spines.values():
        s.set_color(GRID)

    lin = scores["linear 10-lag"]
    res = scores[key]
    verdict = ("beats the linear tap-delay: the nonlinear mixing is doing work"
               if res < lin * 0.9 else
               "ties the linear tap-delay: memory is real, nonlinear mixing is not "
               "yet contributing")
    fig.suptitle("NARMA-10 through the ported vortex disk, read out at the ports",
                 fontsize=18, fontweight="bold", color=INK, x=0.06, ha="left",
                 y=0.958)
    fig.text(0.06, 0.912,
             f"{X.shape[1]} features · memory capacity {mc:.1f} of "
             f"{X.shape[1]} possible · {verdict}.",
             fontsize=11, color=INK2, ha="left")

    out = Path(args.out) if args.out else Path(f"runs/narma_{args.tag}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"memory capacity {mc:.2f}; scores " +
          ", ".join(f"{k} {v:.4f}" for k, v in scores.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
