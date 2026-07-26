#!/usr/bin/env python3
"""Is the disk array chaotic? Measure amplification, not saturation.

The culling probe's echo-decay ratio cannot answer this once noise exists.
Both a chaotic system and a fading one end with twin distance equal to the
noise floor -- chaos because every trajectory saturates at the attractor
diameter, fading because the perturbation relaxed away. The ratio is ~1 in
both cases; it is degenerate.

What separates them is whether a SMALL perturbation grows. This measures a
finite-time maximal Lyapunov exponent by the standard renormalised two-
trajectory method: perturb by eps, integrate one frame, measure the growth
factor, rescale back to eps, repeat. lambda > 0 is chaos regardless of where
trajectories eventually saturate.

Reported alongside it: the noise floor as a fraction of the feature spread.
If thermal noise at 1.5% of R is amplified into full-scale feature
divergence, the array is a noise amplifier and no readout can generalise; if
it stays microscopic, the dynamics are contractive and usable.

    python scripts/check_lyapunov.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from magnonic_nn.reservoir import FilmResponse
from magnonic_nn.thiele import ThieleConfig, ThieleDisks


def clone_state(src, dst):
    dst.X = src.X.clone()
    dst.p = src.p.clone()
    dst.since_switch = src.since_switch.clone()
    dst.t = src.t


def lyapunov(P, cfg, steps, eps_frac=1e-6, warmup=20, noise_seed=0):
    """Finite-time maximal Lyapunov exponent, per frame.

    With noise on, both copies share a noise seed so the divergence measured
    is the response to the perturbation, not to different noise draws.
    """
    eps = eps_frac * cfg.R
    a = ThieleDisks(cfg, noise_seed=noise_seed)
    b = ThieleDisks(cfg, noise_seed=noise_seed)
    drive = torch.zeros(cfg.n_disks, dtype=torch.float64)

    for n in range(warmup):
        drive.copy_(torch.from_numpy(P[n]))
        a.run_frame(drive, steps)

    clone_state(a, b)
    b.X[0, 0] += eps
    logs = []
    for n in range(warmup, len(P)):
        drive.copy_(torch.from_numpy(P[n]))
        # identical noise realisation in both copies
        b.noise_gen = torch.Generator().manual_seed(noise_seed + 1000 * n)
        a.noise_gen = torch.Generator().manual_seed(noise_seed + 1000 * n)
        a.run_frame(drive, steps)
        b.run_frame(drive, steps)
        d = float((b.X - a.X).norm())
        if d <= 0:
            continue
        logs.append(math.log(d / eps))
        # renormalise b back to eps along the separation direction
        b.X = a.X + (b.X - a.X) * (eps / d)
        clone_p = a.p.clone()
        b.p = clone_p          # polarity is discrete: keep copies on the same
        b.since_switch = a.since_switch.clone()   # branch to isolate analog growth
    return float(np.mean(logs)) if logs else float("nan")


def noise_floor(P, cfg, steps):
    """Feature divergence from noise alone, relative to feature spread."""
    outs = []
    for seed in (0, 7):
        d = ThieleDisks(cfg, noise_seed=seed)
        drive = torch.zeros(cfg.n_disks, dtype=torch.float64)
        f = []
        for n in range(len(P)):
            drive.copy_(torch.from_numpy(P[n]))
            f.append(d.run_frame(drive, steps).numpy())
        outs.append(np.stack(f))
    A, B = outs
    cols = [0, 1, 2, 3, 6]
    diff = np.linalg.norm((A - B)[:, :, cols].reshape(len(A), -1), axis=1)
    spread = np.linalg.norm(A[:, :, cols].reshape(len(A), -1)
                            - A[:, :, cols].reshape(len(A), -1).mean(0), axis=1)
    return float(diff[-20:].mean() / max(spread[-20:].mean(), 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--response", default="runs/film_response/response.pt")
    p.add_argument("--frames", type=int, default=70)
    p.add_argument("--steps-per-frame", type=int, default=300)
    args = p.parse_args()

    film = FilmResponse.load(args.response)
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, 1.0, args.frames)
    P = film(u)[:, :12]

    base = dict(coupling=0.08, spike_kick=0.0, drive_scale=0.06, alpha_spread=0.5)
    conditions = [
        ("no-spike ctrl", ThieleConfig(v_crit=float("inf"), **base)),
        ("spiking T=0", ThieleConfig(**base)),
        ("spiking T=300", ThieleConfig(temperature=300.0, **base)),
        ("spk cap=0.5 T=300", ThieleConfig(temperature=300.0, phase_capture=0.5, **base)),
        ("weak coupling", ThieleConfig(temperature=300.0, phase_capture=0.5,
                                       **{**base, "coupling": 0.02})),
    ]
    print(f"{'condition':>20} {'lambda/frame':>13} {'noise/spread':>13} {'regime':>16}")
    for name, cfg in conditions:
        lam = lyapunov(P, cfg, args.steps_per_frame)
        nf = noise_floor(P, cfg, args.steps_per_frame) if cfg.temperature > 0 else 0.0
        regime = ("CHAOTIC" if lam > 0.05 else
                  "edge" if lam > -0.05 else "contractive")
        print(f"{name:>20} {lam:>+13.4f} {nf:>13.3f} {regime:>16}")
    print("\nlambda > 0: small perturbations grow -- chaos, whatever the")
    print("saturation level. noise/spread: thermal divergence as a fraction of")
    print("the feature spread; ~1 means noise is amplified to full scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
