#!/usr/bin/env python3
"""Does temporal averaging recover SNR the spike jitter destroyed?

The operating-point sweep found noise/spread ~1.0-1.4 across a 3x drive range
and 4x coupling range, while lambda stayed negative everywhere -- as low as
-1.11. Contractive analog dynamics cannot produce a macroscopic noise floor,
so the floor is spike-SEQUENCE divergence between noise realisations, not
thermal orbit jitter. Individual spike times are irrecoverable in principle
once the threshold is stochastic.

That leaves rate coding. Spike-timing jitter is independent frame to frame,
while the input-driven component is correlated over the ~10-frame memory
horizon, so a boxcar of width k should suppress the first as ~1/sqrt(k) and
leave the second. If noise/spread crosses 0.5 at modest k, the spiking regime
is usable through rates rather than timings -- and a downstream denoiser,
whose main mechanism is exactly this temporal filtering, is well motivated.

Reported per width: noise/spread, and the surviving signal bandwidth as the
lag-1 autocorrelation of the smoothed features (averaging that destroys the
signal too is no good).

    python scripts/check_rate_coding.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from magnonic_nn.reservoir import FilmResponse
from magnonic_nn.thiele import ThieleConfig, ThieleDisks

COLS = [0, 1, 2, 3, 6]      # analog features; polarity/spike columns excluded


def features(P, cfg, steps, noise_seed):
    d = ThieleDisks(cfg, noise_seed=noise_seed)
    drive = torch.zeros(cfg.n_disks, dtype=torch.float64)
    out = []
    for n in range(len(P)):
        drive.copy_(torch.from_numpy(P[n]))
        out.append(d.run_frame(drive, steps).numpy())
    return np.stack(out)


def boxcar(F, k):
    if k <= 1:
        return F
    kern = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, kern, mode="valid"),
                               0, F)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--response", default="runs/film_response/response.pt")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--steps-per-frame", type=int, default=300)
    p.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = p.parse_args()

    film = FilmResponse.load(args.response)
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, 1.0, args.frames)
    P = film(u)[:, :12]

    points = [("drive 0.12, g 0.08", 0.12, 0.08),
              ("drive 0.20, g 0.08", 0.20, 0.08)]

    for name, ds, g in points:
        cfg = ThieleConfig(coupling=g, spike_kick=0.0, phase_capture=0.5,
                           drive_scale=ds, temperature=300.0, alpha_spread=0.5)
        A = features(P, cfg, args.steps_per_frame, noise_seed=0)
        B = features(P, cfg, args.steps_per_frame, noise_seed=7)
        print(f"\n{name}   ({A[:, :, 5].mean():.3f} spk/disk/frame)")
        print(f"{'width':>6} {'noise/spread':>13} {'lag-1 autocorr':>15} {'verdict':>12}")
        for k in args.widths:
            a, b = boxcar(A[:, :, COLS], k), boxcar(B[:, :, COLS], k)
            fa = a.reshape(len(a), -1)
            diff = np.linalg.norm((a - b).reshape(len(a), -1), axis=1)
            spread = np.linalg.norm(fa - fa.mean(0), axis=1)
            ns = float(diff.mean() / max(spread.mean(), 1e-12))
            z = (fa - fa.mean(0)) / fa.std(0).clip(1e-12)
            ac = float(np.mean([np.corrcoef(z[:-1, i], z[1:, i])[0, 1]
                                for i in range(z.shape[1])
                                if np.isfinite(np.corrcoef(z[:-1, i], z[1:, i])[0, 1])]))
            verdict = "USABLE" if ns < 0.5 else ""
            print(f"{k:>6} {ns:>13.3f} {ac:>15.3f} {verdict:>12}")

    print("\nnoise/spread < 0.5 means signal dominates thermal spike jitter.")
    print("lag-1 autocorrelation rising toward 1 means the smoothing is")
    print("eating the signal's own bandwidth -- the cost side of the trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
