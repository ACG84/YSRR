#!/usr/bin/env python
"""Is the 8-lag memory ceiling in the disk, or in the lock-in that reads it?

Every memory number this project has produced comes from the same 60 features:
six ports x five lock-in tones x (re, im). Jaeger MC measures memory visible IN
THOSE FEATURES, which equals the memory in the magnetisation only if the
projection is faithful. MC sat at 8.27/8.29/8.41/8.61 across a fourfold change
in damping and stayed put across a sixfold change in port count -- an instrument
reading the same number while the sample changes underneath it.

WHY THE OBVIOUS TEST DOES NOT WORK. The first version of this script compared
the lock-in against the RAW port waveform, on the reasoning that the raw signal
is a superset of any projection of it. The raw arm scored r^2 = 0.02 at lag ZERO
-- it could not recover the current input, which is impossible for a superset --
and the reason is a property of the encoding worth stating plainly.

The drive is phase-continuous across frames and one frame is 2.4 carrier cycles,
so the carrier phase at frame start advances 0.4 cycle per frame and realigns
every five. Measured on the stored waveform, the within-frame pattern has cosine
similarity +0.957 at frame separation 5 and -0.616 at separation 1. Indexing
features by sample position WITHIN a frame therefore throws away the absolute
phase, and no single fixed linear map can demodulate five different phases at
once. The lock-in never had that problem because it demodulates against absolute
time.

So the honest comparison demodulates the stored waveform the same way, offline,
using absolute time reconstructed from (frame, sample) -- and then asks whether
MORE TONES see more memory than five. That is the real question anyway: not
"waveform versus lock-in" but "is five tones enough".

Arms:

  lockin_60         the shipped readout, five tones, straight from the run
  rebuilt_60        the same five tones re-derived here from the waveform.
                    A CHECK, not a result: it must reproduce lockin_60, or the
                    offline demodulation is wrong and nothing below is valid.
  bank_PCA_60       a 40-tone bank projected to exactly 60 dimensions
  bank_full         the whole 40-tone bank

bank_PCA_60 is what decides. It has the shipped readout's exact dimension, so a
gap cannot be dismissed as more features fitting better, and its PCA basis is
fitted on the TRAINING slice alone so the test block cannot shape its own
representation.

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


def demodulate(W, tones_hz, n_ports, steps_per_frame, wave_stride, dt):
    """Lock-in the stored waveform at arbitrary tones, using ABSOLUTE time.

    W is (frames, n_samples * n_ports). Absolute time is recoverable because the
    sample grid is known: frame j, stored sample k -> (j*steps_per_frame +
    k*wave_stride)*dt. Getting this from the frame-local index instead is exactly
    the mistake that made the raw-waveform arm score zero.
    """
    n_frames = W.shape[0]
    ns = W.shape[1] // n_ports
    S = W.reshape(n_frames, ns, n_ports)
    k = np.arange(ns) * wave_stride
    out = []
    for j in range(n_frames):
        t = (j * steps_per_frame + k) * dt
        row = []
        for f in tones_hz:
            w = 2 * np.pi * f
            I = (S[j] * np.cos(w * t)[:, None]).sum(0) / ns
            Q = (S[j] * np.sin(w * t)[:, None]).sum(0) / ns
            # same cross-port DFT the six guides physically implement
            M = np.fft.fft(I + 1j * Q)
            for q in range(n_ports):
                row += [M[q].real, M[q].imag]
        out.append(row)
    return np.asarray(out, dtype=np.float64)


def pca_project(X, tr, k):
    """Top-k PCs of X, basis fitted on the training slice only."""
    mu = X[tr].mean(0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc[tr], full_matrices=False)
    return Xc @ Vt[:k].T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--max-lag", type=int, default=28)
    p.add_argument("--dims", type=int, default=60)
    p.add_argument("--n-tones", type=int, default=40)
    p.add_argument("--tone-lo", type=float, default=1.0)
    p.add_argument("--tone-hi", type=float, default=30.0)
    p.add_argument("--n-ports", type=int, default=6)
    p.add_argument("--steps-per-frame", type=int, default=200)
    p.add_argument("--wave-stride", type=int, default=2)
    p.add_argument("--dt-ps", type=float, default=1.0)
    p.add_argument("--shipped-tones", type=float, nargs="*",
                   default=[12.0, 9.9, 10.3, 13.7, 24.0])
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ck = torch.load(Path(a.run) / "features_multitone.pt", weights_only=False)
    F = np.asarray(ck["feats"], dtype=np.float64)
    if ck.get("waves") is None:
        sys.exit(f"{a.run} has no waveform; rerun with --save-waveform")
    W = np.asarray(ck["waves"], dtype=np.float64)
    n = min(len(F), len(W)); F, W = F[:n], W[:n]
    u, _ = narma10(n, seed=0)
    dt = a.dt_ps * 1e-12

    w0 = int(0.25 * n); tr_n = int(0.50 * n); va = int(0.125 * n)
    splits = (w0, tr_n, va); tr = slice(w0, w0 + tr_n)

    rebuilt = demodulate(W, [f * 1e9 for f in a.shipped_tones],
                         a.n_ports, a.steps_per_frame, a.wave_stride, dt)
    # The bank must CONTAIN the shipped tones, not merely bracket them. A
    # uniform grid misses them -- its nearest point to the 12 GHz carrier is
    # 12.154 GHz -- and that is not a rounding detail: the shipped carrier tone
    # is phase-locked to the drive, so its demodulated value is stable frame to
    # frame, while an off-resonance tone beats against the drive and its I/Q
    # rotate until any linear correlation with u is gone. Measured: a pure grid
    # reaches |corr(column, u)| = 0.125 where the shipped tones reach 0.930.
    # Union-ing them in makes the bank a true superset, so bank_full can only
    # match or beat lockin_60 and a shortfall means a fitting failure rather
    # than a result.
    grid = np.linspace(a.tone_lo, a.tone_hi, a.n_tones)
    bank_ghz = np.unique(np.concatenate([grid, np.asarray(a.shipped_tones)]))
    bank_hz = bank_ghz * 1e9
    print(f"tone bank: {len(bank_ghz)} tones, "
          f"{bank_ghz.min():.2f}-{bank_ghz.max():.2f} GHz, "
          f"shipped five included")
    bank = demodulate(W, bank_hz, a.n_ports, a.steps_per_frame,
                      a.wave_stride, dt)

    # Validation gate. Column signs/scales differ (the stride halves the sum),
    # so compare per-column |correlation| rather than values.
    cc = []
    for j in range(F.shape[1]):
        if F[:, j].std() > 1e-30 and rebuilt[:, j].std() > 1e-30:
            cc.append(abs(np.corrcoef(F[:, j], rebuilt[:, j])[0, 1]))
    cc = float(np.median(cc)) if cc else 0.0
    print(f"{n} frames, splits {splits}, waveform {W.shape[1]} dims/frame")
    print(f"offline demodulation check: median |corr| between shipped and "
          f"rebuilt five-tone features = {cc:.4f}")
    if cc < 0.9:
        print("  FAILED -- the offline demodulation does not reproduce the\n"
              "  shipped readout, so the tone-bank arms below mean nothing.")
        sys.exit(1)
    print("  passed\n")

    arms = {
        "lockin_60": F,
        "rebuilt_60": rebuilt,
        f"bank_PCA_{a.dims}": pca_project(bank, tr, a.dims),
        "bank_full": bank,
    }

    print(f"{'readout':<16} {'dims':>6} {'MC':>7} {'cliff':>7} "
          f"{'r2@0':>6} {'r2@8':>6} {'r2@10':>6} {'r2@12':>6}")
    rows = {}
    for name, X in arms.items():
        r2, mc, hor = memory_function(X, u, splits, max_lag=a.max_lag)
        rows[name] = {"dims": int(X.shape[1]), "mc": mc, "cliff": hor, "r2": r2}
        print(f"{name:<16} {X.shape[1]:>6} {mc:>7.2f} {hor:>7} "
              f"{r2[0]:>6.2f} {r2[8]:>6.2f} {r2[10]:>6.2f} {r2[12]:>6.2f}",
              flush=True)

    print("\nr^2 of reconstructing u[n-k]")
    print("lag           " + "  ".join(f"{k:>4}" for k in range(a.max_lag + 1)))
    for name, r in rows.items():
        print(f"{name:<14}" + "  ".join(f"{v:>4.2f}" for v in r["r2"]))

    lock = rows["lockin_60"]["mc"]; bank_pca = rows[f"bank_PCA_{a.dims}"]["mc"]
    full = rows["bank_full"]["mc"]

    # VALIDITY GATE. bank_full contains the shipped tones as an exact subset, so
    # it cannot legitimately carry LESS memory. If it does, the limit being
    # measured is the estimator, not the device: ridge applies ONE global lambda
    # to every standardised column, so a few hundred uninformative-but-
    # high-variance dimensions dilute the shrinkage and drown the signal, and
    # PCA cannot rescue it because it ranks directions by variance while the
    # high-variance directions here are the uninformative ones. Reporting a
    # "ceiling is in the disk" verdict off that would be reading a fitting
    # failure as physics.
    if full < 0.95 * lock:
        print(f"\nINVALID COMPARISON: bank_full ({full:.2f}) is a strict "
              f"superset of lockin_60 ({lock:.2f}) yet scores lower.")
        print("  A superset cannot hold less information, so this measures the\n"
              "  ridge, not the disk -- one global lambda cannot shrink "
              "uninformative\n  columns while sparing informative ones. No "
              "verdict is available from\n  this comparison; the question needs "
              "a readout that is richer WITHOUT\n  being noisier -- read the "
              "magnetisation directly rather than adding tones.")
        if a.out:
            Path(a.out).write_text(json.dumps(rows, indent=2))
            print(f"\nwrote {a.out}")
        return

    print(f"\nequal-dimension comparison: shipped 5-tone {lock:.2f} vs "
          f"{a.n_tones}-tone bank at the same width {bank_pca:.2f} "
          f"({bank_pca / lock:.2f}x)")
    if bank_pca > 1.3 * lock:
        print("  The ceiling is in the READOUT: five tones under-sample what the\n"
              "  ports carry, and a wider bank of the same width sees more.")
    elif bank_pca < 1.15 * lock:
        print("  The ceiling is in the DISK: eight times as many tones, projected\n"
              "  to the same width, see no more memory. ~8 frames is what the\n"
              "  magnetisation retains, and no readout recovers lag 10.")
    else:
        print("  Ambiguous: a real but modest gain. Report the numbers, not a\n"
              "  verdict.")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
