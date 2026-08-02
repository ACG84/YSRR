#!/usr/bin/env python
"""Measure how much memory a run actually has, against what its damping predicts.

The NARMA-10 certification failed for one identified reason: the device holds
8.12 frames of history (Jaeger MC) and the task needs 11, because NARMA-10's
product term reaches u[n-10] and a linear filter's NMSE falls off a cliff --
0.71 to 0.37 -- exactly between 10 and 11 lags. Memory in frames is

    MC ~= tau / T_frame = 1 / (2 pi alpha N_cycles)

and at the shipped alpha = 0.008 with 2.40 carrier cycles per frame that
predicts 8.29 against 8.12 measured, so the model is good to 2%. This script
exists to find out whether it STAYS good when alpha is lowered to buy memory,
which is the whole premise of the fix.

Cheap on purpose: it reads features a run already wrote and refits a ridge, so
screening three damping values costs three short simulations and no analysis
time worth counting. Deliberately reports the r^2 curve and not just MC -- MC
sums r^2 over lags, so a device with a long shallow tail and one with a sharp
cliff at the same MC are not the same device, and only the cliff position says
whether u[n-10] is reachable.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from certify_narma10 import load_features, memory_function          # noqa: E402
from magnonic_nn.reservoir import narma10                           # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="run directories to screen")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", type=int, nargs=3, default=(150, 600, 150))
    # 14 is the certification's default and it truncates the moment memory
    # improves -- the point of lowering alpha is to push the cliff past 11, and
    # a curve that stops at 14 cannot show a cliff at 16.
    p.add_argument("--max-lag", type=int, default=28)
    p.add_argument("--carrier-ghz", type=float, default=12.0)
    p.add_argument("--steps-per-frame", type=int, default=200)
    p.add_argument("--dt-ps", type=float, default=1.0,
                   help="integrator step; frame = steps_per_frame * dt")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    rows = []
    for d in a.dirs:
        d = Path(d)
        F = load_features(d / "features_multitone.pt")
        n = len(F)
        u, _ = narma10(n, seed=a.seed)
        # splits must fit the frames actually completed
        w, tr, va = a.splits
        if w + tr + va >= n:
            tr = max(1, int(0.55 * n)); w = int(0.15 * n); va = int(0.15 * n)
        r2, mc, hor = memory_function(F, u, (w, tr, va), max_lag=a.max_lag)

        alpha = None
        log = d / "run.log"
        if log.exists():
            for line in reversed(log.read_text(errors="ignore").splitlines()):
                if line.startswith("alpha "):
                    alpha = float(line.split()[1].rstrip(",")); break
        frame_ns = a.steps_per_frame * a.dt_ps * 1e-3
        pred = None
        if alpha:
            tau_ns = 1.0 / (alpha * 2 * math.pi * a.carrier_ghz * 1e9) * 1e9
            pred = tau_ns / frame_ns

        rows.append({"dir": str(d), "frames": n, "alpha": alpha,
                     "mc": mc, "horizon": hor, "predicted_frames": pred,
                     "r2": r2})

    print(f"{'run':<34} {'alpha':>7} {'frames':>7} {'MC':>7} "
          f"{'cliff':>7} {'pred':>7} {'meas/pred':>10}")
    for r in rows:
        ratio = (r["mc"] / r["predicted_frames"]) if r["predicted_frames"] else float("nan")
        print(f"{Path(r['dir']).name:<34} "
              f"{(r['alpha'] if r['alpha'] else float('nan')):>7.4f} "
              f"{r['frames']:>7} {r['mc']:>7.2f} {r['horizon']:>7} "
              f"{(r['predicted_frames'] or float('nan')):>7.2f} {ratio:>10.2f}")

    print("\nr^2 of reconstructing u[n-k]  (cliff = first lag below 0.5)")
    hdr = "  ".join(f"{k:>4}" for k in range(0, a.max_lag + 1))
    print(f"{'lag':<14}{hdr}")
    for r in rows:
        cells = "  ".join(f"{v:>4.2f}" for v in r["r2"])
        print(f"{Path(r['dir']).name[:13]:<14}{cells}")

    # The number that decides whether this fix can work at all. NARMA-10 needs
    # u[n-10], so a cliff at or below 10 means the term is unreachable no matter
    # what the readout does.
    print("\nreaches u[n-10]?")
    for r in rows:
        ok = r["horizon"] > 10
        print(f"  {Path(r['dir']).name:<32} cliff at lag {r['horizon']:>3}  "
              f"{'YES' if ok else 'no'}")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
