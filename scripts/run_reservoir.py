#!/usr/bin/env python3
"""Trial the spiking vortex-disk reservoir on time-series prediction.

Nested comparisons, each stage having to earn its keep:

    input-only      ridge on u_n alone           no memory, no nonlinearity
    film-only       + memoryless film features   nonlinearity, still no memory
    linear disks    + disks, spiking lobotomised gyration memory, no spikes
    spiking         full: reversal + refractory + coupling + spike kicks

Scored on NARMA-10 (NMSE; needs 10-step memory and input products), Jaeger's
linear memory capacity, and one-step Mackey-Glass. If 'spiking' does not beat
'linear disks', the spiking added nothing on this task and the honest result
is exactly that.

    python scripts/run_reservoir.py
    python scripts/run_reservoir.py --n 2200 --steps-per-frame 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from magnonic_nn.reservoir import (
    FilmResponse, ReservoirRunner, fit_eval, mackey_glass, memory_capacity, narma10,
)
from magnonic_nn.thiele import ThieleConfig


def disk_cfg(spiking: bool) -> ThieleConfig:
    if spiking:
        return ThieleConfig(coupling=0.08, spike_kick=0.05)
    # Lobotomy: no reversal (infinite threshold), no coupling, no stiffening --
    # what remains is 12 independent damped linear gyrators.
    return ThieleConfig(v_crit=float("inf"), coupling=0.0, spike_kick=0.0,
                        beta_nl=0.0, kappa_nl=0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--response", default="runs/film_response/response.pt")
    p.add_argument("--n", type=int, default=3200)
    p.add_argument("--splits", type=int, nargs=3, default=(200, 2000, 400),
                   metavar=("WASH", "TRAIN", "VAL"),
                   help="the remainder is the test set")
    p.add_argument("--steps-per-frame", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="runs/reservoir_trial")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    film = FilmResponse.load(args.response)
    splits = tuple(args.splits)

    u, y = narma10(args.n, seed=args.seed)
    u_norm = u / 0.5                       # film sweep is calibrated on [0, 1]

    results = {}

    # -- baselines without disks ---------------------------------------------
    results["input_only"] = {"narma10": fit_eval(u[:, None], y, splits)}
    film_feats = np.concatenate([film(u_norm), u[:, None]], axis=1)
    results["film_only"] = {"narma10": fit_eval(film_feats, y, splits)}

    # -- disk reservoirs ------------------------------------------------------
    for name, spiking in (("linear_disks", False), ("spiking", True)):
        t0 = time.time()
        runner = ReservoirRunner(film, disk_cfg(spiking),
                                 steps_per_frame=args.steps_per_frame)
        X = runner.features(u_norm, progress_every=800)
        elapsed = time.time() - t0

        entry = {"narma10": fit_eval(X, y, splits), "feature_time_s": round(elapsed, 1)}
        mc_curve, mc = memory_capacity(X, u, splits)
        entry["memory_capacity"] = round(mc, 3)
        entry["mc_curve"] = [round(float(v), 4) for v in mc_curve]
        if spiking:
            total_spikes = float(X[:, 5::7][:, :12].sum()) if X.shape[1] >= 84 else 0.0
            entry["total_spikes"] = total_spikes
        results[name] = entry
        print(f"{name}: NARMA-10 test NMSE {entry['narma10']['nmse_test']:.4f}, "
              f"MC {mc:.2f}  ({elapsed:.0f} s)")

    # -- Mackey-Glass, full stack only ----------------------------------------
    mg = mackey_glass(args.n, seed=args.seed)
    mg_u = (mg - mg.min()) / (mg.max() - mg.min())
    mg_y = np.roll(mg, -1)                 # one-step prediction
    for name, spiking in (("linear_disks", False), ("spiking", True)):
        runner = ReservoirRunner(film, disk_cfg(spiking),
                                 steps_per_frame=args.steps_per_frame)
        X = runner.features(mg_u)
        results[name]["mackey_glass"] = fit_eval(X[:-1], mg_y[:-1], splits)

    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'condition':<14} {'NARMA-10':>10} {'MG-1step':>10} {'MC':>7}")
    for name, r in results.items():
        n10 = r["narma10"]["nmse_test"]
        mg1 = r.get("mackey_glass", {}).get("nmse_test")
        mc = r.get("memory_capacity")
        print(f"{name:<14} {n10:>10.4f} "
              f"{mg1 if mg1 is None else f'{mg1:.4f}':>10} "
              f"{mc if mc is None else f'{mc:.2f}':>7}")
    print(f"\nresults in {outdir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
