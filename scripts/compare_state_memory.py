#!/usr/bin/env python
"""Does the magnetisation hold memory the ports do not show?

The one question left after the damping sweep and the free-decay test both came
back negative. Measured MC has sat at ~8.3 through a fourfold change in Gilbert
damping and a sixfold change in port count, which is what a saturated INSTRUMENT
looks like -- so before concluding the disk forgets at 8 frames, check whether
the disk is what is being measured.

Two earlier attempts at this failed, both for reasons that shaped this one:

  raw waveform   scored r^2 = 0.02 at lag ZERO. The drive is phase-continuous
                 and a frame is 2.4 carrier cycles, so carrier phase advances
                 0.4 cycle per frame; indexing features by position WITHIN a
                 frame discards absolute phase and no fixed linear map
                 demodulates five phases at once.
  45-tone bank   a strict superset of the shipped five tones scored MC 2.07
                 against their 8.04. Ridge applies ONE global lambda to every
                 standardised column, so a few hundred uninformative-but-
                 high-variance columns dilute the shrinkage; PCA cannot rescue
                 it because it ranks by variance and here the high-variance
                 directions are the uninformative ones.

Both failed the same way: the readout got richer AND noisier at once. A fixed
random projection of the magnetisation avoids that. It is unbiased -- no tone,
no phase reference, no geometry preference -- and at the shipped readout's own
width of 60 it is a like-for-like comparison where neither arm can win on size.

  lockin_60   six ports x five tones x (re, im), what the device actually emits
  state_60    60 fixed random projections of (m - m0) inside the disk

Read it as: state_60 >> lockin_60 means the state holds memory the PORTS do not
expose, so the ceiling is in the readout and is fixable without changing the
physics. state_60 ~= lockin_60 means the magnetisation itself forgets at ~8
frames, and no readout and no cascade of these disks reaches NARMA-10's u[n-10].

    python scripts/compare_state_memory.py runs/state_probe
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from certify_narma10 import memory_function                        # noqa: E402
from magnonic_nn.reservoir import narma10                          # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--max-lag", type=int, default=28)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ck = torch.load(Path(a.run) / "features_multitone.pt", weights_only=False)
    if ck.get("state") is None:
        sys.exit(f"{a.run} has no state projections; rerun with --save-state N")
    F = np.asarray(ck["feats"], dtype=np.float64)
    S = np.asarray(ck["state"], dtype=np.float64)
    n = min(len(F), len(S)); F, S = F[:n], S[:n]
    u, _ = narma10(n, seed=0)

    w = int(0.25 * n); tr = int(0.50 * n); va = int(0.125 * n)
    splits = (w, tr, va)

    print(f"{n} frames, splits {splits}\n")
    print(f"{'readout':<12} {'dims':>5} {'MC':>7} {'cliff':>6} "
          f"{'r2@0':>6} {'r2@5':>6} {'r2@8':>6} {'r2@10':>6} {'r2@12':>6}")
    rows = {}
    for name, X in (("lockin_60", F), ("state_60", S)):
        r2, mc, hor = memory_function(X, u, splits, max_lag=a.max_lag)
        rows[name] = {"dims": int(X.shape[1]), "mc": mc, "cliff": hor, "r2": r2}
        print(f"{name:<12} {X.shape[1]:>5} {mc:>7.2f} {hor:>6} "
              f"{r2[0]:>6.2f} {r2[5]:>6.2f} {r2[8]:>6.2f} {r2[10]:>6.2f} "
              f"{r2[12]:>6.2f}", flush=True)

    print("\nr^2 of reconstructing u[n-k]")
    print("lag         " + "  ".join(f"{k:>4}" for k in range(a.max_lag + 1)))
    for name, r in rows.items():
        print(f"{name:<12}" + "  ".join(f"{v:>4.2f}" for v in r["r2"]))

    # Sanity gate. An arm that cannot recover the CURRENT input is broken, not
    # informative -- that is exactly how the two failed designs announced
    # themselves, and reading a verdict off a broken arm is how a method bug
    # becomes a physics claim.
    bad = [k for k, r in rows.items() if r["r2"][0] < 0.8]
    if bad:
        print(f"\nINVALID: {', '.join(bad)} cannot recover u[n] at lag 0.")
        print("  A readout that cannot see the present carries no verdict about\n"
              "  the past. If state_60 is the failing arm, the random projection\n"
              "  is too narrow -- rerun with --save-state 200 -- rather than\n"
              "  concluding anything about the disk.")
        if a.out:
            Path(a.out).write_text(json.dumps(rows, indent=2))
        return

    lock, st = rows["lockin_60"], rows["state_60"]
    print(f"\nequal-dimension comparison: ports {lock['mc']:.2f} vs "
          f"state {st['mc']:.2f}  ({st['mc'] / lock['mc']:.2f}x), "
          f"cliff {lock['cliff']} vs {st['cliff']}")
    if st["mc"] > 1.3 * lock["mc"] or st["cliff"] > lock["cliff"] + 2:
        print("  The ceiling is in the READOUT. The magnetisation holds memory\n"
              "  the ports do not expose, so a better physical readout -- not\n"
              "  different physics -- is the route to NARMA-10.")
    elif st["mc"] < 1.15 * lock["mc"]:
        print("  The ceiling is in the DISK, not the readout. An unbiased read\n"
              "  of the state at the same width sees no more memory than the\n"
              "  ports do, so ~8 frames is what the magnetisation retains and a\n"
              "  better readout will not extend it.\n"
              "  This does NOT close the cascade route. Two stages each holding\n"
              "  ~8 frames compose: the second integrates the first's output, so\n"
              "  depth is the remaining way to reach lag 10 -- unlike damping and\n"
              "  unlike readout, both of which are now measured dead ends.")
    else:
        print("  Ambiguous: a real but modest gain. Report the numbers, not a\n"
              "  verdict.")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
