#!/usr/bin/env python
"""Do two coupled stages compose their memory, or is stage B a delayed copy?

Depth is the last mechanism standing. Damping, radiation and readout are each
measured dead ends for extending the single disk's ~8-frame memory, and
NARMA-10's recursion needs u[n-10]. If stage B integrates stage A's output and
each holds ~8 frames, B's dependence on u should reach further than A's.

Only disk A is driven, so B reaches u solely through the link. That is the
cascade; driving both disks would give two parallel reservoirs and answer a
different question.

Arms per run, all scored on the same memory function used everywhere else:

  A       stage 1, six ports, 60 features
  B       stage 2, six ports, 60 features -- the one under test
  A+B     both, 120 features

and both runs: linked, and the no-link control with the link material deleted.

Decision rule, fixed in docs/two_stage_cascade.md before any data existed:

  1. linked B has no lag with r^2 > 0.5   -> INCONCLUSIVE. The link delivers
     0.68% of the driven stage's amplitude, so this means the measurement had no
     signal, NOT that composition fails.
  2. no-link B shows memory of u          -> INVALID. The coupling would be
     stray dipolar field rather than guided transport, and the linked result
     would be measuring the wrong mechanism.
  3. cliff(B) >= 11 or cliff(A+B) >= 11   -> COMPOSITION HOLDS.
  4. otherwise                            -> composition fails on this geometry.

Rule 1 exists because three readout designs in this project produced confident
verdicts from arms that turned out to carry no signal at all.

    python scripts/compare_cascade_memory.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from certify_narma10 import memory_function                        # noqa: E402
from magnonic_nn.reservoir import narma10                          # noqa: E402


def split_disks(F, n_tones=5, n_ports=6):
    """(frames, 120) -> per-disk views.

    Layout is written by run_narma_coupled: for each tone, disk A's six ports as
    (re, im) then disk B's six. So tone i occupies 2*n_ports*2 columns, disk A
    the first half of that block and disk B the second.
    """
    per_disk = n_ports * 2
    blk = per_disk * 2
    A = np.concatenate([F[:, i * blk: i * blk + per_disk]
                        for i in range(n_tones)], axis=1)
    B = np.concatenate([F[:, i * blk + per_disk: (i + 1) * blk]
                        for i in range(n_tones)], axis=1)
    return A, B


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--linked", default="runs/cascade_linked/features_linked.pt")
    p.add_argument("--nolink", default="runs/cascade_nolink/features_nolink.pt")
    p.add_argument("--splits", type=int, nargs=3, default=(150, 300, 75))
    p.add_argument("--max-lag", type=int, default=28)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    runs = {}
    for tag, path in (("linked", a.linked), ("nolink", a.nolink)):
        ck = torch.load(Path(path), weights_only=False)
        F = np.asarray(ck["feats"] if isinstance(ck, dict) else ck,
                       dtype=np.float64)
        runs[tag] = F
        print(f"{tag:<8} {F.shape}")
    n = min(len(F) for F in runs.values())
    u, _ = narma10(n, seed=0)
    splits = tuple(a.splits)
    print(f"\n{n} frames, splits {splits}, max_lag {a.max_lag}\n")

    rows = {}
    print(f"{'run':<8} {'arm':<5} {'dims':>5} {'MC':>7} {'cliff':>6} "
          f"{'r2@0':>6} {'r2@5':>6} {'r2@8':>6} {'r2@10':>6} {'r2@12':>6}")
    for tag, F in runs.items():
        F = F[:n]
        A, B = split_disks(F)
        for arm, X in (("A", A), ("B", B), ("A+B", F)):
            r2, mc, hor = memory_function(X, u, splits, max_lag=a.max_lag)
            rows[f"{tag}/{arm}"] = {"dims": int(X.shape[1]), "mc": mc,
                                    "cliff": hor, "r2": r2,
                                    "peak_r2": float(max(r2))}
            print(f"{tag:<8} {arm:<5} {X.shape[1]:>5} {mc:>7.2f} {hor:>6} "
                  f"{r2[0]:>6.2f} {r2[5]:>6.2f} {r2[8]:>6.2f} "
                  f"{r2[10]:>6.2f} {r2[12]:>6.2f}", flush=True)

    print("\nr^2 of reconstructing u[n-k]")
    print("arm            " + "  ".join(f"{k:>4}" for k in range(a.max_lag + 1)))
    for k, r in rows.items():
        print(f"{k:<15}" + "  ".join(f"{v:>4.2f}" for v in r["r2"]))

    lb, nb = rows["linked/B"], rows["nolink/B"]
    la, lab = rows["linked/A"], rows["linked/A+B"]

    print(f"\nlinked B peak r^2 {lb['peak_r2']:.2f} | "
          f"no-link B peak r^2 {nb['peak_r2']:.2f}")
    print(f"cliffs: A {la['cliff']}, B {lb['cliff']}, A+B {lab['cliff']}   "
          f"(single disk reference: MC 8.04, cliff 8)")

    if lb["peak_r2"] <= 0.5:
        print("\nINCONCLUSIVE. Stage B never reconstructs u above r^2 0.5 at any\n"
              "lag, so the cascade delivered too little signal to measure. This\n"
              "is NOT evidence against composition -- it says the link is too\n"
              "weak to test it. A stronger link or a larger drive is the fix.")
    elif nb["peak_r2"] > 0.5:
        print("\nINVALID. Stage B shows memory of u with the link REMOVED, so the\n"
              "coupling is stray dipolar field rather than guided transport and\n"
              "the linked result measures the wrong mechanism.")
    elif lb["cliff"] >= 11 or lab["cliff"] >= 11:
        print("\nCOMPOSITION HOLDS. The cascade reaches past the single disk's\n"
              "lag-8 cliff, so depth extends memory where damping, radiation and\n"
              "readout could not. Next step is the six-seed certification.")
    else:
        print("\nCOMPOSITION FAILS. Stage B carries signal but its memory does not\n"
              "reach past the single disk's cliff -- B is a delayed copy of A\n"
              "rather than an integrator of it, so depth buys nothing here.")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
