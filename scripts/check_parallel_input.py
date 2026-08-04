#!/usr/bin/env python3
"""Does a phase-diverse parallel input layer enrich the deep stage?

Two disks driven identically are exactly redundant -- measured, every canonical
correlation 1.000, and twelve features carrying the five dimensions six already
carried. Offsetting the feed phase by 90 degrees breaks that: 5 -> 8 effective
dimensions, from nothing but feed line length.

That was measured on two ISOLATED disks. The architectural question is different
and one step further on: if two phase-diverse inputs feed a DEEP stage, does the
deep stage inherit the enrichment, or does the link average it away before it
arrives?

The three-disk chain expresses this without new geometry. Drive the two ENDS
(stages 0 and 2) as the parallel input layer; read the MIDDLE (stage 1) as the
deep stage fed by both. Arms:

  single      drive stage 0 only            -- the cascade as already measured
  in-phase    drive stages 0 and 2 together -- redundant parallel inputs
  90 deg      drive stages 0 and 2 offset   -- phase-diverse parallel inputs

An impulse, not a task run: ~2 minutes an arm against 3.4 hours, because the
question here is whether the deep stage's state gets richer, which effective
rank answers directly.

Effective rank is necessary, not sufficient. Redundant features cannot help;
new dimensions are not automatically useful ones. A task run decides whether the
enrichment carries lag information, and this decides whether that run is worth
its compute.

    python scripts/check_parallel_input.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import ChainPortedConfig, ChainPortedArray


def eff_rank(X, frac=0.99):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    return int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), frac) + 1)


@torch.no_grad()
def run(arr, cfg, dtype, amp, freq, n_burst, n_quiet, stages, phases):
    units = []
    for s in stages:
        u = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
        u[:, :, 0, 0] = arr.disk_masks[s].to(dtype)
        units.append(u)
    m = arr.m0.clone()
    out = []
    for k in range(n_burst + n_quiet):
        tk = k * cfg.dt
        on = k < n_burst

        def h(theta, tk=tk, on=on):
            t = tk + theta * cfg.dt
            f = torch.zeros_like(units[0])
            if on:
                for u, ph in zip(units, phases):
                    f = f + u * (amp * math.sin(2 * math.pi * freq * t + ph))
            return f

        m = arr.rollout.rk4_step(m, arr.h_zero, h)
        out.append(arr.port_signals(m).double().cpu().numpy().copy())
    return np.asarray(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-disks", type=int, default=3)
    p.add_argument("--deep-stage", type=int, default=1)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--burst", type=int, default=400)
    p.add_argument("--quiet", type=int, default=1600)
    p.add_argument("--m0", default="runs/chain3_narma/m0_n3_linked.pt")
    p.add_argument("--outdir", default="runs/parallel_input")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = ChainPortedConfig(n_disks=a.n_disks, separation=700e-9,
                            link_width=80e-9)
    arr = ChainPortedArray(cfg, timesteps=a.burst + a.quiet + 8, dtype=dtype)
    m0p = Path(a.m0)
    if m0p.exists():
        arr.m0 = torch.load(m0p, weights_only=False).to(dtype)
        print(f"[m0] restored from {m0p}")
    else:
        t0 = time.time(); arr.relax(steps=10000, require_tol=2e-3)
        torch.save(arr.m0.cpu(), m0p); print(f"relaxed {time.time()-t0:.0f}s")

    amp = a.amp_mT * 1e-3 / MU_0
    ends = [0, a.n_disks - 1]
    arms = {
        "single (stage 0)":   ([0], [0.0]),
        "in-phase ends":      (ends, [0.0, 0.0]),
        "90 deg ends":        (ends, [0.0, math.pi / 2]),
    }
    npb = cfg.n_ports
    d = a.deep_stage
    print(f"\ndeep stage = {d}, inputs = {ends}\n")
    print(f"{'arm':<18} {'deep rms':>11} {'deep rank':>10} {'all rank':>9} "
          f"{'deep/input':>11}")
    rows = {}
    for name, (stages, phases) in arms.items():
        t0 = time.time()
        S = run(arr, cfg, dtype, amp, a.freq * 1e9, a.burst, a.quiet,
                stages, phases)
        D = S[:, d * npb:(d + 1) * npb]
        I = np.hstack([S[:, s * npb:(s + 1) * npb] for s in stages])
        drms = float(np.sqrt((D ** 2).mean()))
        irms = float(np.sqrt((I ** 2).mean()))
        rows[name] = {"deep_rms": drms, "deep_rank": eff_rank(D),
                      "all_rank": eff_rank(S), "ratio": drms / max(irms, 1e-30),
                      "seconds": round(time.time() - t0, 1)}
        print(f"{name:<18} {drms:>11.4e} {rows[name]['deep_rank']:>10} "
              f"{rows[name]['all_rank']:>9} {rows[name]['ratio']:>11.4f}",
              flush=True)
        (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    single = rows["single (stage 0)"]["deep_rank"]
    base = rows["in-phase ends"]["deep_rank"]
    div = rows["90 deg ends"]["deep_rank"]
    print(f"\ndeep-stage effective rank: single {single}, in-phase {base}, "
          f"90 deg {div}")
    # Compare against the SINGLE-input control, not just against the redundant
    # pair. Beating in-phase only shows phase diversity undoes the damage a
    # redundant pair does; the question is whether a parallel layer beats one
    # input at all, and an earlier version of this check declared victory on
    # 5 > 4 while the single-input arm also scored 5.
    if div > single:
        print("  Phase diversity SURVIVES the link: the deep stage inherits\n"
              "  dimensions the redundant pair does not give it, so a parallel\n"
              "  input layer is worth its hardware -- and a task run is worth\n"
              "  its compute.")
    else:
        print("  The enrichment does NOT reach the deep stage. Phase diversity\n"
              "  recovers what a redundant pair loses but does not beat a SINGLE\n"
              "  input, so the extra dimensions stay local to the input layer.\n"
              "  A parallel input layer delivers more POWER to the deep stage\n"
              "  without delivering more STATE, and power was already the wrong\n"
              "  currency for depth.")
    print(f"\nwrote {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
