#!/usr/bin/env python3
"""Does a THIRD stage extend memory further than two? Score each stage's ports.

The two-stage cascade answered the architectural question in principle: driving
only stage A and reading A+B moved the memory window from lags 0-7 (single disk,
MC 8.04) out to lags 0-11 (MC 10.58), because the link's transit delay puts
stage B's response several frames behind the drive. Depth is the one mechanism
of four that actually bought memory -- damping, radiation and readout width all
failed.

The impulse sweep then predicted where that stops. Guided transfer fits 0.092
per hop in a chain (interior stages leak through four free radiating ports),
against a fixed 0.0043 dipolar floor, so the useful ceiling is four stages; and
stage 3's impulse peak lands at lag 9.48, which is where NARMA-10's u[n-10]
product term lives.

This scores the prediction on the task rather than on an impulse. One 600-frame
NARMA-10 run through a 3-chain driven at stage 1 gives every stage's features
simultaneously, so the arms are free:

    S1          stage 1 ports only      the input stage -- what one disk gives
    S2          stage 2 ports only      one hop of delay
    S3          stage 3 ports only      two hops: does lag ~9 actually appear?
    S1+S2       first two stages        the measured two-stage cascade
    all three   the full chain          does the third stage ADD, or dilute?

Report the WINDOW of lags where r^2 > 0.5 and the PEAK lag, not a "cliff". The
cliff metric assumes memory decays monotonically from lag 0, which is true of a
single disk and false of a delayed stage: linked stage B spans lags 3-12 and the
cliff dutifully reported 1, describing the delay as a failure.

    python scripts/check_chain3_memory.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from magnonic_nn.reservoir import narma10
from certify_narma10 import memory_function, score, lags, LAG_CHOICES


def window(r2, thr=0.5):
    """Lags whose r^2 clears the threshold, as (lo, hi) of the contiguous run
    containing the peak -- a delayed stage's support does not start at lag 0."""
    ok = [k for k, v in enumerate(r2) if v > thr]
    if not ok:
        return None
    pk = int(np.argmax(r2))
    lo = hi = pk
    while lo - 1 in ok:
        lo -= 1
    while hi + 1 in ok:
        hi += 1
    return (lo, hi) if pk in ok else None


def best_linear(u, y, splits):
    """Best linear filter with its lag count chosen on validation."""
    n_wash, n_train, n_val = splits
    va = slice(n_wash + n_train, n_wash + n_train + n_val)
    best, bk = None, None
    for k in LAG_CHOICES:
        X = lags(u, k)
        e = score(X, y, splits)
        if best is None or e < best:
            best, bk = e, k
    return best, bk


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="runs/chain3_narma/features_n3_linked.pt")
    p.add_argument("--n-disks", type=int, default=3)
    p.add_argument("--n-ports", type=int, default=6)
    p.add_argument("--n-tones", type=int, default=5)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", type=int, nargs=3, default=(150, 300, 75))
    p.add_argument("--max-lag", type=int, default=28)
    p.add_argument("--outdir", default="runs/chain3_narma")
    a = p.parse_args()

    obj = torch.load(a.features, weights_only=False)
    F = (obj["feats"] if isinstance(obj, dict) else obj)
    F = np.asarray(F, dtype=np.float64)
    print(f"features {F.shape} from {a.features}")
    if len(F) < a.frames:
        print(f"INCOMPLETE: {len(F)}/{a.frames} frames -- not scoring")
        return 1
    F = F[:a.frames]

    u, y = narma10(a.frames, seed=a.seed)
    splits = tuple(a.splits)

    # Column layout, set by run_narma_chain.run_reservoir: for each tone, for
    # each disk, for each of the six cross-port modes, (real, imag). So a tone
    # block is n_disks * n_ports * 2 columns wide and each disk owns a
    # contiguous n_ports * 2 slice inside it.
    per_disk = a.n_ports * 2
    block = a.n_disks * per_disk
    assert F.shape[1] == a.n_tones * block, (F.shape, a.n_tones * block)

    def cols(stages):
        idx = []
        for t in range(a.n_tones):
            for d in stages:
                s = t * block + d * per_disk
                idx += list(range(s, s + per_disk))
        return np.array(idx)

    # Per-stage arms isolate where the memory sits; cumulative arms say whether
    # adding the next stage buys anything the previous ones did not already
    # have. Built from n_disks so the single disk and the two-stage cascade can
    # be scored by this exact code path -- they were run at the same 600 frames,
    # same seed, same splits, and a comparison against numbers produced by a
    # different scorer is not a comparison.
    arms = {f"S{d+1}": [d] for d in range(a.n_disks)}
    for k in range(2, a.n_disks + 1):
        arms["+".join(f"S{d+1}" for d in range(k))] = list(range(k))

    # The matched-budget control, and the reason the cumulative arms alone would
    # not settle anything: depth also multiplies the feature count, 60 -> 120 ->
    # 180, and a wider readout fits better whether or not the extra stages hold
    # anything. So spend the SAME 60 columns spread across every stage -- the
    # first per_disk/n_disks modes of each -- against all 60 concentrated in the
    # input stage. Equal budget, different geometry; only the geometry differs.
    # Do it for every depth 2..N inside THIS run, not by comparing against the
    # two-disk cascade: that was a different mesh, a different link topology and
    # its own ground state, so the depth-2 and depth-3 numbers there differ by
    # more than depth. Within one run the geometry is held fixed and only the
    # number of stages sharing the 60 columns changes.
    spread = {}
    for k in range(2, a.n_disks + 1):
        ke = per_disk // k
        spread[f"spread60 over S1-S{k}"] = np.array(
            [t * block + d * per_disk + c
             for t in range(a.n_tones) for d in range(k) for c in range(ke)])

    lin_best, lin_k = best_linear(u, y, splits)
    lin10 = score(lags(u, 10), y, splits)
    print(f"\nbaselines: linear_10lag {lin10:.4f}   "
          f"linear_best {lin_best:.4f} (k={lin_k})\n")

    print(f"{'arm':<20} {'dim':>4} {'MC':>7} {'window':>10} {'peak':>5} "
          f"{'r2@peak':>8} {'NARMA NMSE':>11}")
    rows = {}
    todo = [(n, cols(s), s) for n, s in arms.items()]
    todo += [(n, idx, None) for n, idx in spread.items()]
    for name, idx, stages in todo:
        X = F[:, idx]
        r2, mc, _ = memory_function(X, u, splits, max_lag=a.max_lag)
        w = window(r2)
        pk = int(np.argmax(r2))
        nm = score(X, y, splits)
        rows[name] = {"stages": stages, "dim": int(X.shape[1]), "MC": mc,
                      "window": list(w) if w else None, "peak_lag": pk,
                      "r2_peak": r2[pk], "narma_nmse": nm, "r2": r2}
        ws = f"{w[0]}-{w[1]}" if w else "none"
        print(f"{name:<20} {X.shape[1]:>4} {mc:>7.2f} {ws:>10} {pk:>5} "
              f"{r2[pk]:>8.3f} {nm:>11.4f}", flush=True)

    out = Path(a.outdir) / f"memory_by_stage_n{a.n_disks}.json"
    out.write_text(json.dumps(
        {"baselines": {"linear_10lag": lin10, "linear_best": lin_best,
                       "linear_best_k": lin_k},
         "arms": rows}, indent=2))

    print("\nreference: single disk MC 8.04, window 0-7, peak 0, NMSE 0.7913;")
    print("           two stages 0.2451, three stages 0.2008 (same conditions);")
    print(f"           linear_best {lin_best:.4f}.")
    print()
    if a.n_disks < 3:
        print(f"wrote {out}")
        return 0

    # Judge the DEEPEST stage this run actually has, and judge it on memory and
    # on the task separately -- they are different questions and at depth 4 they
    # give different answers. An earlier version hardcoded the depth-3
    # comparison and so reported "the third stage ADDS" for a four-stage run,
    # which was true and not what the run was asked.
    n = a.n_disks
    prev = rows["+".join(f"S{d+1}" for d in range(n - 1))]
    full = rows["+".join(f"S{d+1}" for d in range(n))]
    d_mc = full["MC"] - prev["MC"]
    d_nm = full["narma_nmse"] - prev["narma_nmse"]
    print(f"stage {n}: memory {d_mc:+.2f} MC, task {d_nm:+.4f} NMSE "
          f"({prev['narma_nmse']:.4f} -> {full['narma_nmse']:.4f})")
    if d_mc > 0.5 and d_nm < -0.01:
        print(f"  Stage {n} ADDS on both counts: it extends the memory AND the\n"
              f"  task can use what it extended.")
    elif d_mc > 0.5:
        print(f"  Stage {n} extends MEMORY but not PERFORMANCE. It holds lags\n"
              f"  the task does not need, and its columns cost more in readout\n"
              f"  variance than the lags return. Depth has saturated for this\n"
              f"  task at {n-1} stages -- which is a statement about NARMA-10's\n"
              f"  10-lag horizon, not about the chain.")
    else:
        print(f"  Stage {n} adds nothing: the per-hop transfer has taken its\n"
              f"  signal below what the readout can use.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
