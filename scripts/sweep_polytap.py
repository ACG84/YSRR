#!/usr/bin/env python3
"""Sweep coupling against tap damping, because the two are not independent.

Every sweep so far moved one knob and was defeated by the other:

    gap sweep         fixed the bus draining at tap 1 (through-loss 0.431 ->
                      0.912) and halved what the tap received
    directional
    coupler           landed on the same 157x spread by a different route
    damping sweep     produced the first monotonic tap ordering in the project
                      at 10x, then got WORSE at 30x as taps 3 and 4 fell to
                      6.1e-6 and 2.7e-6, too weak to place

The reason is that the two knobs govern different halves of the same
requirement. Damping decides whether a tap can RESOLVE a delay -- it must ring
for less than the 0.60 ns that separates neighbouring taps. Coupling decides
whether the tap RECEIVES enough to be measured at all. The useful region needs
both, and no one-dimensional cut through it can find that.

So this is the two-dimensional cut. It reports resolution and reception
separately and reduces neither away, because a point that resolves perfectly on
signal too weak to read is not a result.

Ground states are cached per GEOMETRY, not per point: the relax is the
expensive part and damping does not move an energy minimum, so a whole damping
column costs one relax. Results are written after every point, so a reclaimed
Colab runtime or a rebooted container loses at most the point in flight.

    python scripts/sweep_polytap.py --device cuda
    python scripts/sweep_polytap.py --quick        # 4 points, for smoke-testing
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import magnonic_nn as mnn
from _polytap_probe import (make_array, ensure_m0, burst_response,
                            delay_by_xcorr, score_point)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-taps", type=int, default=4)
    p.add_argument("--lags", type=float, nargs="+", default=[5, 8, 11, 14])
    p.add_argument("--gaps", type=float, nargs="*", default=[0, 15, 30],
                   help="nm; coupling gaps for the stub geometry")
    p.add_argument("--couplers", type=float, nargs="*", default=[300, 450],
                   help="nm; directional-coupler lengths (gap fixed at 20 nm)")
    p.add_argument("--tap-alphas", type=float, nargs="+", default=[1, 3, 10, 30],
                   help="tap damping multipliers")
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--burst", type=int, default=200)
    p.add_argument("--quiet", type=int, default=2600)
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true",
                   help="tiny grid and short rollout; checks the plumbing, not "
                        "the physics")
    p.add_argument("--outdir", default="runs/polytap_sweep")
    a = p.parse_args()

    if a.quick:
        a.gaps, a.couplers, a.tap_alphas = [0], [], [1, 10]
        a.burst, a.quiet, a.relax_steps = 20, 60, 1000

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "sweep.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    # (gap_nm, coupler_len_nm) pairs. A coupler length of 0 selects the stub.
    geometries = [(g, 0.0) for g in a.gaps] + [(20.0, c) for c in a.couplers]
    steps = a.burst + a.quiet + 8
    total = len(geometries) * len(a.tap_alphas)
    print(f"{total} points: {len(geometries)} geometries x "
          f"{len(a.tap_alphas)} damping values, device={a.device}\n")

    n = 0
    for gap_nm, cpl_nm in geometries:
        for ta in a.tap_alphas:
            n += 1
            try:
                cfg, arr, geom, run = make_array(
                    a.n_taps, a.lags, gap_nm, cpl_nm, ta, steps, dtype)
            except ValueError as e:
                print(f"[{n}/{total}] skipped: {e}")
                continue
            if run in results:
                print(f"[{n}/{total}] {run}: cached")
                continue
            t0 = time.time()
            ensure_m0(arr, outdir, geom, relax_steps=a.relax_steps, dtype=dtype,
                      log=lambda s: print(f"    {s}", flush=True))
            sig, drive = burst_response(arr, cfg, a.amp_mT, a.freq,
                                        a.burst, a.quiet, bus=True, fresh=False,
                                        dtype=dtype)
            rows = delay_by_xcorr(sig, drive, cfg.n_taps, arr.n_readout, a.freq)
            sc = score_point(rows, list(cfg.tap_lags[:cfg.n_taps]))
            results[run] = {"gap_nm": gap_nm, "coupler_len_nm": cpl_nm,
                            "tap_alpha_mult": ta, "taps": rows, **sc,
                            "seconds": round(time.time() - t0, 1)}
            results_path.write_text(json.dumps(results, indent=2))
            print(f"[{n}/{total}] {run}: "
                  f"spacings {[round(float(x),2) for x in sc['spacing_measured']]} "
                  f"(designed {sc['spacing_designed']}), "
                  f"mono {sc['n_monotonic']}/{cfg.n_taps-1}, "
                  f"spread {sc['amp_spread']:.0f}x, "
                  f"min amp {sc['amp_min']:.2e}  [{results[run]['seconds']:.0f}s]",
                  flush=True)

    if not results:
        print("no points completed")
        return 1

    print(f"\n{'point':<22} {'mono':>5} {'mean|err|':>10} {'spread':>9} {'min amp':>10}")
    for k, v in sorted(results.items(),
                       key=lambda kv: (-kv[1]["n_monotonic"],
                                       kv[1]["mean_abs_spacing_error"])):
        print(f"{k:<22} {v['n_monotonic']:>5} {v['mean_abs_spacing_error']:>10.2f} "
              f"{v['amp_spread']:>9.0f} {v['amp_min']:>10.2e}")

    print("\nWhat a usable point looks like: every consecutive pair monotonic, "
          "mean\nspacing error well under the 3.0 frames being resolved, and "
          "the weakest tap\nstill above ~1e-5 where the estimator can place "
          "it.\n\nCPU reference points, measured through this same code path:\n"
          "  gap 0 /  1x   mono 1/3, mean|err| 3.15, spread 255x, min 4.9e-06\n"
          "  gap 0 / 10x   mono 2/3, mean|err| 6.25, spread 286x, min 2.8e-06\n"
          "The 10x point is the one that first ordered three taps correctly;\n"
          "its mean|err| is worse only because tap 4 drops out entirely, which\n"
          "is the reception half of the problem this sweep exists to separate.\n"
          "(The 314x quoted elsewhere is the IMPULSE run at burst 400; these\n"
          "are the delay run at burst 200, so amplitudes differ.)")
    print(f"\nwrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
