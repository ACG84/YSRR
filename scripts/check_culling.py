#!/usr/bin/env python3
"""Does the culling spike stabilise the array and run the annealing cycle?

The twin-trajectory probe showed the kick-only spiking array is chaotic: the
echo twin never converges, so nothing fades and the reservoir cannot
generalise. Phase capture -- each reversal burst pulling the cluster toward a
common phase -- is contractive, and therefore the candidate cure.

Two measurements, over a grid of capture strengths:

* **Echo restoration.** Twin runs from different initial orbits under the
  same input. If capture restores fading memory the twin distance decays;
  its decay ratio (last 20 / first 20 frames) is the number to watch.
* **The annealing cycle itself.** Kuramoto order of the envelope phases,
  R = |mean exp(i phi_j)| with phi = atan2(Q, I), event-triggered around
  spikes: culling predicts R jumps UP at a spike and relaxes back DOWN as
  heterogeneous inputs rebuild the decorrelated state.

    python scripts/check_culling.py
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


def run(P, cfg, steps, init_seed=None):
    disks = ThieleDisks(cfg)
    if init_seed is not None:
        g = torch.Generator().manual_seed(init_seed)
        disks.X = 0.2 * cfg.R * torch.randn(cfg.n_disks, 2, generator=g,
                                            dtype=torch.float64)
    feats = []
    drive = torch.zeros(cfg.n_disks, dtype=torch.float64)
    for n in range(len(P)):
        drive.copy_(torch.from_numpy(P[n]))
        feats.append(disks.run_frame(drive, steps).numpy())
    return np.stack(feats)              # (frames, disks, 7)


def order_parameter(F):
    """Kuramoto order of envelope phases, orbit-weighted, per frame."""
    I, Q, r = F[:, :, 0], F[:, :, 1], np.sqrt(F[:, :, 6])
    phase = np.arctan2(Q, I)
    w = r / r.sum(axis=1, keepdims=True).clip(1e-12)
    return np.abs((w * np.exp(1j * phase)).sum(axis=1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--response", default="runs/film_response/response.pt")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--steps-per-frame", type=int, default=300)
    p.add_argument("--captures", type=float, nargs="+",
                   default=[0.0, 0.4, 0.8])
    args = p.parse_args()

    film = FilmResponse.load(args.response)
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, 1.0, args.frames)
    P = film(u)[:, :12]

    print(f"{'condition':>16} {'echo decay':>11} {'spk/disk/frm':>13} "
          f"{'<R>':>6} {'dR at spike':>12} {'verdict':>24}")
    # Control first: the smooth nonlinear chain with firing disabled. If THIS
    # is already non-fading, chaos precedes spiking and no spike-time
    # mechanism can cure it -- the fix would be upstream, in coupling or
    # stiffening.
    conditions = [("no-spike ctrl", ThieleConfig(coupling=0.08, spike_kick=0.0,
                                                 v_crit=float("inf"),
                                                 drive_scale=0.06,
                                                 alpha_spread=0.5))]
    conditions += [
        (f"capture={lam:.1f}", ThieleConfig(coupling=0.08, spike_kick=0.0,
                                            phase_capture=lam,
                                            drive_scale=0.06,
                                            alpha_spread=0.5))
        for lam in args.captures
    ]
    for name, cfg in conditions:
        A = run(P, cfg, args.steps_per_frame)
        B = run(P, cfg, args.steps_per_frame, init_seed=123)
        analog = [0, 1, 2, 3, 6]
        d = np.linalg.norm((A - B)[:, :, analog].reshape(len(A), -1), axis=1)
        decay = float(d[-20:].mean() / d[:20].mean().clip(1e-12))

        R = order_parameter(A)
        spikes = A[:, :, 5].sum(axis=1)
        rate = float(A[:, :, 5].mean())
        # event-triggered: R change across frames containing >=1 spike,
        # against the change across quiet frames as the null
        ev = [t for t in range(1, len(R) - 1) if spikes[t] > 0]
        qu = [t for t in range(1, len(R) - 1) if spikes[t] == 0]
        dR_ev = float(np.mean([R[t + 1] - R[t - 1] for t in ev])) if ev else float("nan")
        dR_qu = float(np.mean([R[t + 1] - R[t - 1] for t in qu])) if qu else float("nan")

        verdict = ("fading restored" if decay < 0.2 else
                   "partial" if decay < 0.7 else "still chaotic")
        print(f"{name:>16} {decay:>11.3f} {rate:>13.3f} {R.mean():>6.3f} "
              f"{dR_ev - dR_qu:>+12.4f} {verdict:>24}")

    print("\necho decay: twin distance, last-20/first-20 frames (<1 fades)")
    print("dR at spike: order-parameter change through spike frames minus the")
    print("same through quiet frames -- positive means spikes RAISE coherence")
    print("(the cull), and the quiet-frame drift is the rebuild.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
