#!/usr/bin/env python
"""Is the 8-lag memory ceiling in the disk, or in the lock-in that reads it?

Every memory number this project has produced comes from the same 60 features:
six ports x five lock-in tones x (re, im), one vector per frame. Jaeger MC
measures memory visible IN THOSE FEATURES, which equals the memory in the
magnetisation only if the projection is faithful.

Three things say it may not be. MC sat at 8.27 / 8.29 / 8.41 / 8.61 across a
fourfold change in Gilbert damping, and at 6.5 / 4.8 / 8.1 frames of free-decay
tau across a sixfold change in port count -- an instrument reading the same
number while the sample changes underneath it. Certification tier 3 found the 60
features add nothing to a linear filter. And the features carry only ~20
dimensions of variance, with two tone pairs at 0.993 and 0.999 canonical
correlation. Meanwhile the renormalised Lyapunov exponent at the operating point
is ~0, the edge of chaos, where a state should hold information for a long time.

So: compare MC computed from the SAME run through different readouts.

  lockin_60     the shipped readout, five tones
  wavePCA_60    top 60 principal components of the raw port waveform
  wave_full     every waveform sample kept

wavePCA_60 is the comparison that decides. It has exactly the dimension of the
shipped readout, so a difference cannot be explained by "more features fit
better" -- the confound that would make wave_full alone unconvincing. The PCA
basis is fitted on the TRAINING slice only; fitting it on all frames would let
the test block inform its own representation.

If wavePCA_60 >> lockin_60, the ceiling belongs to the lock-in and is fixable
without touching the device. If they agree, the disk genuinely forgets at 8
frames and no architecture built from it reaches NARMA-10.

    python scripts/compare_readout_memory.py runs/readout_probe
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from certify_narma10 import memory_function                        # noqa: E402
from magnonic_nn.reservoir import narma10                          # noqa: E402


def pca_project(W, tr, k):
    """Top-k PCs of W, basis fitted on the training slice only."""
    mu = W[tr].mean(0)
    Wc = W - mu
    # economy SVD of the training block; V columns are the component directions
    _, _, Vt = np.linalg.svd(Wc[tr], full_matrices=False)
    return Wc @ Vt[:k].T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run directory containing features_multitone.pt")
    p.add_argument("--max-lag", type=int, default=28)
    p.add_argument("--dims", type=int, default=60,
                   help="PCA width; defaults to the lock-in's own 60")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ck = torch.load(Path(a.run) / "features_multitone.pt", weights_only=False)
    F = np.asarray(ck["feats"], dtype=np.float64)
    if ck.get("waves") is None:
        sys.exit(f"{a.run} has no waveform; rerun with --save-waveform")
    W = np.asarray(ck["waves"], dtype=np.float64)
    n = min(len(F), len(W))
    F, W = F[:n], W[:n]
    u, _ = narma10(n, seed=0)

    w = int(0.25 * n); tr_n = int(0.50 * n); va = int(0.125 * n)
    splits = (w, tr_n, va)
    tr = slice(w, w + tr_n)

    arms = {
        "lockin_60": F,
        f"wavePCA_{a.dims}": pca_project(W, tr, a.dims),
        "wave_full": W,
    }

    print(f"{n} frames, splits {splits}, waveform {W.shape[1]} dims/frame\n")
    print(f"{'readout':<16} {'dims':>6} {'MC':>7} {'cliff':>7} "
          f"{'r2@8':>7} {'r2@10':>7} {'r2@12':>7}")
    rows = {}
    for name, X in arms.items():
        r2, mc, hor = memory_function(X, u, splits, max_lag=a.max_lag)
        rows[name] = {"dims": int(X.shape[1]), "mc": mc, "cliff": hor,
                      "r2": r2}
        print(f"{name:<16} {X.shape[1]:>6} {mc:>7.2f} {hor:>7} "
              f"{r2[8]:>7.2f} {r2[10]:>7.2f} {r2[12]:>7.2f}", flush=True)

    print("\nr^2 of reconstructing u[n-k]")
    print("lag           " + "  ".join(f"{k:>4}" for k in range(a.max_lag + 1)))
    for name, r in rows.items():
        print(f"{name:<14}" + "  ".join(f"{v:>4.2f}" for v in r["r2"]))

    lock = rows["lockin_60"]["mc"]
    pca = rows[f"wavePCA_{a.dims}"]["mc"]
    print(f"\nequal-dimension comparison: lock-in {lock:.2f} vs "
          f"waveform PCA {pca:.2f}  ({pca / lock:.2f}x)")
    if pca > 1.3 * lock:
        print("  The ceiling is in the READOUT. The disk holds more memory than\n"
              "  five lock-in tones expose, and the fix is a better projection\n"
              "  rather than different physics.")
    elif pca < 1.15 * lock:
        print("  The ceiling is in the DISK. A richer projection of the same\n"
              "  waveform sees no more memory, so ~8 frames is what the\n"
              "  magnetisation actually retains, and no readout recovers lag 10.")
    else:
        print("  Ambiguous: a real but modest gain. Neither story is clean;\n"
              "  report the numbers rather than a verdict.")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
