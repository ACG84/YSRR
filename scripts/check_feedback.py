#!/usr/bin/env python3
"""How good a gradient can a feedback side channel give?

Backpropagation needs the transpose of the forward operator. A physical device
does not have one lying around, so the practical question is what a *side
channel* can deliver instead -- and at what cost.

The cheapest conceivable channel carries a single number: the loss. That is
enough for SPSA (simultaneous perturbation stochastic approximation): perturb
every design variable at once by a random sign pattern, measure the loss
change, and correlate. Two rollouts per sample, no transpose, no adjoint, no
access to internal state.

It is unbiased, and its accuracy grows as ``sqrt(n / N)`` for ``n`` samples and
``N`` design variables -- so matching one backward pass costs ``O(N)`` rollouts.
This script measures that against the exact autograd gradient and checks the
scaling holds.

**The conclusion flips between simulation and hardware.** In simulation a
rollout is seconds, so O(N) rollouts is hopeless and backprop wins by orders of
magnitude. In a device a rollout is the physical duration of the experiment --
tens of nanoseconds -- so O(N) rollouts is microseconds and the side channel
wins outright, because the device never had a transpose to offer.

    python scripts/check_feedback.py
    python scripts/check_feedback.py --nx 32 --samples 24
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import magnonic_nn as mnn


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nx", type=int, default=24)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--eps", type=float, default=1e-3, help="perturbation size")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    mnn.set_precision("float64")
    mnn.set_device("cpu")

    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = args.nx
    cfg.material.abc_width = max(args.nx // 6, 2)
    cfg.solver.timesteps = args.timesteps
    cfg.solver.relax_steps = 80
    cfg.fields.Bt = 1e-3

    nx, ny, _ = cfg.mesh.n
    src = mnn.LineSource(cfg.mesh, cfg.fields, cfg.material.abc_width + 1, 0,
                         cfg.material.abc_width + 1, ny - 1)
    probes = mnn.linear_probe_array(cfg.mesh, 3, x=nx - cfg.material.abc_width - 2,
                                    r=1.0, margin=cfg.material.abc_width)

    torch.manual_seed(args.seed)
    model = mnn.SpinWaveNetwork(cfg, [src], probes)
    rho = model.geometry.rho
    with torch.no_grad():
        rho.copy_(torch.randn_like(rho) * 0.2)

    signal = mnn.tone(cfg, 4.0e9)
    target = torch.tensor([1])
    mask = model.geometry.design_mask > 0
    n_params = int(mask.sum())

    # exact gradient, for reference
    model.zero_grad(set_to_none=True)
    mnn.intensity_cross_entropy(model(signal).unsqueeze(0), target).backward()
    g_true = rho.grad.clone()
    base = rho.detach().clone()

    def loss_at(params):
        with torch.no_grad():
            rho.copy_(params)
            model.equilibrium(force=True)
            return float(mnn.intensity_cross_entropy(model(signal).unsqueeze(0), target))

    print(f"mesh {nx}x{ny}, {args.timesteps} steps, {n_params} design variables\n")
    print(f"{'samples':>8} {'rollouts':>9} {'cos to true':>12} {'sqrt(n/N)':>11}")

    torch.manual_seed(args.seed + 1)
    accum = torch.zeros_like(rho)
    for i in range(1, args.samples + 1):
        # Rademacher perturbation of every variable at once
        direction = (torch.randint(0, 2, rho.shape, dtype=rho.dtype) * 2 - 1) * mask
        plus = loss_at(base + args.eps * direction)
        minus = loss_at(base - args.eps * direction)
        accum += (plus - minus) / (2 * args.eps) * direction

        estimate = accum / i
        a, b = g_true[mask], estimate[mask]
        cos = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
        if i in (1, 2, 4, 8, 16, 32, args.samples):
            print(f"{i:>8} {2 * i:>9} {cos:>+12.4f} {math.sqrt(i / n_params):>11.3f}")

    with torch.no_grad():
        rho.copy_(base)

    rollout_ns = cfg.duration * 1e9
    print()
    print(f"Matching one backward pass needs O(N) = {n_params} rollouts.")
    print(f"In simulation that is hopeless. In hardware one rollout is the")
    print(f"experiment's own duration, {rollout_ns:.0f} ns, so {2 * n_params} rollouts")
    print(f"is {2 * n_params * rollout_ns * 1e-3:.0f} us per gradient -- and the device never")
    print("had a transpose to offer in the first place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
