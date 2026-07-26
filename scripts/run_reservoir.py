#!/usr/bin/env python3
"""Trial the spiking vortex-disk reservoir on time-series prediction.

Nested comparisons, each stage having to earn its keep:

    input-only      ridge on u_n alone            no memory, no nonlinearity
    film-only       + memoryless film features    nonlinearity, still no memory
    linear disks    + disks: pure damped gyrators  memory, no disk nonlinearity
    nonlinear disks + stiffening/damping/coupling  everything except firing
    spiking         + reversal, refractory, kicks  the one-knob contrast

The last pair differ ONLY in the firing mechanism (v_crit and spike_kick), so
"did the spikes help" is a one-knob comparison rather than a five-knob
confound. Every disk condition also carries the film features, keeping the
ladder strictly nested. Scored on NARMA-10 (NMSE; needs 10-step memory and
input products), Jaeger's linear memory capacity, and one-step Mackey-Glass
(reported with its trivial baselines -- lag-1 autocorrelation is 0.99, so a
small NMSE there is table stakes, not an achievement). If 'spiking' does not
beat 'nonlinear disks', the honest result is that firing added nothing here.

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


DISK_ARMS = {
    # strictly widening ladder; 'spiking' vs 'nonlinear_disks' is one knob
    "linear_disks": ThieleConfig(v_crit=float("inf"), coupling=0.0,
                                 spike_kick=0.0, beta_nl=0.0, kappa_nl=0.0),
    "nonlinear_disks": ThieleConfig(v_crit=float("inf"), coupling=0.08,
                                    spike_kick=0.0),
    "spiking": ThieleConfig(coupling=0.08, spike_kick=0.05),
}


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

    # -- Mackey-Glass series and its trivial baselines ------------------------
    mg = mackey_glass(args.n, seed=args.seed)
    mg_u = (mg - mg.min()) / (mg.max() - mg.min())
    mg_y = np.roll(mg, -1)[:-1]            # one-step prediction, wrap dropped
    results["input_only"]["mackey_glass"] = fit_eval(mg_u[:-1, None], mg_y, splits)
    results["film_only"]["mackey_glass"] = fit_eval(
        np.concatenate([film(mg_u), mg_u[:, None]], axis=1)[:-1], mg_y, splits)

    # -- disk reservoirs ------------------------------------------------------
    for name, cfg in DISK_ARMS.items():
        t0 = time.time()
        runner = ReservoirRunner(film, cfg, steps_per_frame=args.steps_per_frame)
        X = runner.features(u_norm, progress_every=800)
        entry = {"narma10": fit_eval(X, y, splits),
                 "total_spikes": runner.total_spikes}
        mc_curve, mc = memory_capacity(X, u, splits)
        entry["memory_capacity"] = round(mc, 3)
        entry["mc_curve"] = [round(float(v), 4) for v in mc_curve]

        runner_mg = ReservoirRunner(film, cfg, steps_per_frame=args.steps_per_frame)
        X_mg = runner_mg.features(mg_u)
        entry["mackey_glass"] = fit_eval(X_mg[:-1], mg_y, splits)
        entry["feature_time_s"] = round(time.time() - t0, 1)
        results[name] = entry
        print(f"{name}: NARMA-10 test NMSE {entry['narma10']['nmse_test']:.4f}, "
              f"MC {mc:.2f}, spikes {runner.total_spikes:.0f}")

    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'condition':<16} {'NARMA-10':>10} {'MG-1step':>10} {'MC':>7} {'spikes':>8}")
    for name, r in results.items():
        n10 = f"{r['narma10']['nmse_test']:.4f}"
        mg1 = r.get("mackey_glass")
        mg_s = "-" if mg1 is None else f"{mg1['nmse_test']:.4f}"
        mc = r.get("memory_capacity")
        mc_s = "-" if mc is None else f"{mc:.2f}"
        sp = r.get("total_spikes")
        sp_s = "-" if sp is None else f"{sp:.0f}"
        print(f"{name:<16} {n10:>10} {mg_s:>10} {mc_s:>7} {sp_s:>8}")
    print(f"\nresults in {outdir}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
