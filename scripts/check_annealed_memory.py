#!/usr/bin/env python3
"""Does a spike write memory that outlives the gyration, or erase it?

Twin-trajectory probe of the spiking disk array, at the trial operating
point, driven by the measured film response to a fixed random input series.

Three comparisons against a reference run A:

  B  polarity twin   at frame 40, disk 5's polarity is flipped by hand --
                     the state a reversal writes, without the reversal
  C  analog twin     at frame 40, disk 5's orbit is contracted to 0.3 r --
                     the state a reversal erases, without the flip
  D  echo twin       different random initial orbits, same everything else --
                     the classic echo-state (fading memory) test

Prediction if the polarity latch is real annealed memory: C's divergence
fades on the analog horizon (~10 frames), B's persists until disk 5 next
fires -- and D tells us whether the reservoir as a whole forgets its initial
condition (echo state) or carries permanent state (a latching machine).
Feature distances are reported with polarity columns excluded, so persistence
in B means the latch is *dynamically* readable, not just visible as the p
feature itself.

    python scripts/check_annealed_memory.py
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


def run(P, cfg, steps, n_frames, perturb=None, init_seed=None):
    """Integrate; returns (features per frame, spike times of each disk)."""
    disks = ThieleDisks(cfg)
    if init_seed is not None:
        g = torch.Generator().manual_seed(init_seed)
        disks.X = 0.2 * cfg.R * torch.randn(cfg.n_disks, 2, generator=g,
                                            dtype=torch.float64)
    feats, spike_log = [], [[] for _ in range(cfg.n_disks)]
    drive = torch.zeros(cfg.n_disks, dtype=torch.float64)
    for n in range(n_frames):
        if perturb is not None and n == perturb[0]:
            perturb[1](disks)
        drive.copy_(torch.from_numpy(P[n]))
        f = disks.run_frame(drive, steps)
        for i in range(cfg.n_disks):
            if f[i, 5] > 0:
                spike_log[i].append(n)
        feats.append(f.numpy())
    return np.stack(feats), spike_log


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--response", default="runs/film_response/response.pt")
    p.add_argument("--frames", type=int, default=140)
    p.add_argument("--steps-per-frame", type=int, default=300)
    p.add_argument("--perturb-frame", type=int, default=40)
    p.add_argument("--disk", type=int, default=5)
    args = p.parse_args()

    film = FilmResponse.load(args.response)
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, 1.0, args.frames)
    P = film(u)[:, :12]
    cfg = ThieleConfig(coupling=0.08, spike_kick=0.05)   # the spiking arm
    T, k, d5 = args.frames, args.perturb_frame, args.disk

    A, spikes_A = run(P, cfg, args.steps_per_frame, T)
    B, spikes_B = run(P, cfg, args.steps_per_frame, T,
                      perturb=(k, lambda dk: dk.p.__setitem__(d5, -dk.p[d5])))
    C, _ = run(P, cfg, args.steps_per_frame, T,
               perturb=(k, lambda dk: dk.X.__setitem__(d5, 0.3 * dk.X[d5])))
    D, _ = run(P, cfg, args.steps_per_frame, T, init_seed=123)

    analog = [0, 1, 2, 3, 6]         # feature columns excluding polarity, spikes

    def dist(Z, ref, frame):
        return float(np.linalg.norm(Z[frame][:, analog] - ref[frame][:, analog]))

    print(f"perturbation at frame {k}, disk {d5}; distances exclude polarity "
          f"and spike columns\n")
    print(f"{'frame':>6} {'B: p-flip':>10} {'C: contract':>12} {'D: echo':>10}")
    base = max(dist(D, A, 0), 1e-12)
    for n in list(range(k, min(k + 12, T))) + list(range(k + 15, T, 10)):
        print(f"{n:>6} {dist(B, A, n):>10.4f} {dist(C, A, n):>12.4f} "
              f"{dist(D, A, n) / base:>10.4f}")

    sA = [t for t in spikes_A[d5] if t >= k]
    sB = [t for t in spikes_B[d5] if t >= k]
    print(f"\ndisk {d5} spikes after frame {k}:  A {sA[:8]}  B {sB[:8]}")
    tail = slice(T - 20, T)
    pA, pB = A[tail][:, :, 4], B[tail][:, :, 4]
    print(f"polarity word equal over last 20 frames: {bool((pA == pB).all())}")
    dD = np.mean([dist(D, A, n) for n in range(T - 20, T)])
    d0 = np.mean([dist(D, A, n) for n in range(0, 20)])
    print(f"echo twin distance, first 20 vs last 20 frames: {d0:.4f} -> {dD:.4f}"
          f"  ({'fading -- echo state holds' if dD < 0.1 * d0 else 'NOT fading'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
