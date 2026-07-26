#!/usr/bin/env python3
"""Measure the film's probe-power response P_i(u) over an amplitude grid.

The reservoir trial treats the film as a fixed nonlinear projection: input
sample u -> drive amplitude -> probe powers ("what leaks where"). With no
disk-to-film feedback the film is memoryless frame to frame, so its response
can be measured once and interpolated. This script does the measuring, on the
real micromagnetic film -- 12 probes, random frozen scatterer, two-tone drive
whose amplitude spans the linear-to-nonlinear range (peak Bt at u = 1 lands in
the 20-30 mT regime the vowel runs established as solidly non-linear).

Amplitude points are independent forward rollouts, so this is also the clean
test of the pending parallelism hypothesis: if the solver is dispatch-bound
rather than compute-bound, N single-threaded processes should beat one
N-threaded process by nearly N. ``--bench`` measures exactly that before
committing the full sweep.

    python scripts/measure_film_response.py --bench
    python scripts/measure_film_response.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BT_MAX = 25e-3          # tesla at u = 1; solidly non-linear per the vowel runs
FREQS = (3.3e9, 3.9e9)  # two-tone drive, both inside the film's usable band
RHO_SEED = 7


def build_model(threads: int | None):
    import torch
    if threads:
        torch.set_num_threads(threads)
    import magnonic_nn as mnn

    mnn.set_precision("float32")
    mnn.set_device("cpu")

    cfg = mnn.get_preset("vowels")
    cfg.mesh.nx = cfg.mesh.ny = 80
    cfg.solver.timesteps = 1000
    cfg.fields.Bt = BT_MAX

    nx, ny, _ = cfg.mesh.n
    src = mnn.LineSource(cfg.mesh, cfg.fields, cfg.material.abc_width + 1, 0,
                         cfg.material.abc_width + 1, ny - 1)
    probes = mnn.linear_probe_array(cfg.mesh, 12, x=nx - 15, r=1.0,
                                    margin=cfg.material.abc_width)
    torch.manual_seed(RHO_SEED)
    model = mnn.SpinWaveNetwork(cfg, [src], probes)
    with torch.no_grad():
        model.geometry.rho.copy_(torch.randn_like(model.geometry.rho) * 0.3)
    signal = mnn.multitone(cfg, FREQS)
    return model, signal, cfg


def measure_points(u_values, threads=1):
    """Forward rollouts at each amplitude; returns (len(u), 12) powers."""
    import torch
    model, signal, _ = build_model(threads)
    out = []
    for u in u_values:
        with torch.no_grad():
            out.append(model(signal * float(u)).tolist())
    return out


def _worker(args):
    u_chunk, threads = args
    return measure_points(u_chunk, threads)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-points", type=int, default=33)
    p.add_argument("--u-min", type=float, default=0.02,
                   help="lowest amplitude; zero drive tells us nothing")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--outdir", default="runs/film_response")
    p.add_argument("--bench", action="store_true",
                   help="time 4 procs x 1 thread against 1 proc x 4 threads "
                        "on 4 points, then exit")
    args = p.parse_args()

    if args.bench:
        u4 = [0.25, 0.5, 0.75, 1.0]
        t0 = time.time()
        measure_points(u4, threads=4)
        serial = time.time() - t0
        ctx = get_context("spawn")
        t0 = time.time()
        with ctx.Pool(4) as pool:
            pool.map(_worker, [([u], 1) for u in u4])
        par = time.time() - t0
        print(f"1 proc x 4 threads: {serial:6.1f} s for 4 rollouts")
        print(f"4 procs x 1 thread: {par:6.1f} s for 4 rollouts   "
              f"({serial / par:.2f}x)")
        return 0

    import numpy as np
    import torch

    u_grid = np.linspace(args.u_min, 1.0, args.n_points)
    chunks = np.array_split(u_grid, args.workers)
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        parts = pool.map(_worker, [(list(c), 1) for c in chunks])
    elapsed = time.time() - t0

    P = torch.tensor([row for part in parts for row in part], dtype=torch.float64)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"u_grid": torch.tensor(u_grid), "P": P,
                "Bt_max": BT_MAX, "freqs": FREQS, "rho_seed": RHO_SEED},
               outdir / "response.pt")
    meta = {"n_points": args.n_points, "workers": args.workers,
            "elapsed_s": round(elapsed, 1),
            "P_ranges": [[float(P[:, i].min()), float(P[:, i].max())]
                         for i in range(P.shape[1])]}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{args.n_points} points, {args.workers} workers: {elapsed:.0f} s")
    print(f"saved {outdir}/response.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
