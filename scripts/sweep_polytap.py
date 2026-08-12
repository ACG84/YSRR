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
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import magnonic_nn as mnn
from _polytap_probe import (make_array, ensure_m0, burst_response,
                            delay_by_xcorr, score_point, check_record_length)


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
    p.add_argument("--bus-guide-widths", type=float, nargs="*", default=[None],
                   help="nm; width of the ONE guide coupling each disk to the\n"
                        "bus, measured along the bus. It is a receiving\n"
                        "aperture: at the 12 GHz drive the wave is 82.3 nm long\n"
                        "and this guide is 80 nm, so it spans 0.97 of a\n"
                        "wavelength and integrates the wave away (sinc 0.029).\n"
                        "Coupled amplitude goes as W*sinc(kW/2), peaking near\n"
                        "W = lambda/2, so 40-50 nm should beat 80 by ~11x.")
    p.add_argument("--bus-alphas", type=float, nargs="+", default=[1.0],
                   help="bus damping multipliers; <1 is a lower-loss film,\n"
                        "which lengthens the delay line rather than\n"
                        "sharpening the taps")
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--steps-per-frame", type=int, default=200,
                   help="frame length in steps. Tap spacing scales with it so\n"
                        "designed lags stay in frames, and the burst is one\n"
                        "frame wide -- so this also sets packet bandwidth. A\n"
                        "0.2 ns burst spans ~5 GHz over which v_g runs 706-1267\n"
                        "m/s and arrivals smear ~3 frames; a 0.6 ns burst spans\n"
                        "~1.7 GHz and should smear ~3x less.")
    p.add_argument("--burst", type=int, default=None,
                   help="burst width in steps; defaults to one frame")
    p.add_argument("--quiet", type=int, default=12000,
                   help="steps of silence after the burst. Was 2600, which is\n"
                        "2.6 ns -- SHORTER THAN THE TRANSIT. At the measured\n"
                        "706 m/s the lag-14 tap is 3.75 ns away and the lag-24\n"
                        "tap 6.42 ns, so the wave never arrived and every delay\n"
                        "measured here was the disk's own response time to the\n"
                        "injection near field. check_record_length() now\n"
                        "refuses a record that ends before the wave lands.")
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true",
                   help="tiny grid and short rollout; checks the plumbing, not "
                        "the physics")
    p.add_argument("--outdir", default="runs/polytap_sweep")
    p.add_argument("--worker", type=float, nargs=5, default=None,
                   metavar=("GAP", "COUPLER", "TAP_ALPHA", "BUS_ALPHA", "BUSW"),
                   help="internal: run exactly ONE point and exit. BUSW < 0 "
                        "means 'leave the bus guide at its default width'")
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
    if a.burst is None:
        a.burst = a.steps_per_frame
    steps = a.burst + a.quiet + 8
    total = (len(geometries) * len(a.tap_alphas) * len(a.bus_alphas)
             * len(a.bus_guide_widths))
    # Only the PARENT has a grid. A worker is handed one point on the command
    # line and never sees --gaps/--tap-alphas, so it re-parses their defaults
    # and would announce "20 points: 5 geometries x 4 damping values" while
    # running exactly one -- harmless until the worker's output is streamed,
    # at which point it is the first line a reader sees.
    if a.worker is None:
        print(f"{total} points: {len(geometries)} geometries x "
              f"{len(a.tap_alphas)} damping values, device={a.device}\n")

    # ONE POINT PER PROCESS.
    #
    # magnum.np keeps process-global state, so constructing a second array in
    # the same interpreter leaves tensors from the two geometries on different
    # devices: the sweep died at the second point with "Expected all tensors to
    # be on the same device" in h_eff, in BOTH the compiled and the eager path,
    # while the identical geometry run alone in a fresh process completed in
    # 14 s. That is a property of the library, not something this script can
    # tidy around, so the parent spawns a worker per point and each worker
    # writes its own result file. Per-point files also mean no read-modify-write
    # on a shared json, so a killed worker cannot corrupt the sweep.
    pts = outdir / "points"; pts.mkdir(exist_ok=True)

    if a.worker is not None:
        gap_nm, cpl_nm, ta, ba, bw = a.worker
        cfg, arr, geom, run = make_array(
            a.n_taps, a.lags, gap_nm, cpl_nm, ta, steps, dtype,
            bus_alpha_mult=ba, bus_guide_width_nm=None if bw < 0 else bw,
            steps_per_frame=a.steps_per_frame)
        check_record_length(cfg, steps)
        t0 = time.time()
        ensure_m0(arr, outdir, geom, relax_steps=a.relax_steps, dtype=dtype,
                  log=lambda s: print(f"    {s}", flush=True))
        sig, drive = burst_response(arr, cfg, a.amp_mT, a.freq, a.burst,
                                    a.quiet, bus=True, fresh=False, dtype=dtype)
        rows = delay_by_xcorr(sig, drive, cfg.n_taps, arr.n_readout, a.freq,
                              steps_per_frame=a.steps_per_frame)
        sc = score_point(rows, list(cfg.tap_lags[:cfg.n_taps]))
        (pts / f"{run}.json").write_text(json.dumps(
            {"gap_nm": gap_nm, "coupler_len_nm": cpl_nm, "tap_alpha_mult": ta,
             "bus_alpha_mult": ba, "bus_guide_width_nm": None if bw < 0 else bw,
             "taps": rows, **sc, "seconds": round(time.time() - t0, 1)}, indent=2))
        print(f"{run}: spacings "
              f"{[round(float(x),2) for x in sc['spacing_measured']]}, "
              f"mono {sc['n_monotonic']}/{cfg.n_taps-1}, "
              f"spread {sc['amp_spread']:.0f}x, min amp {sc['amp_min']:.2e}")
        return 0

    n = 0
    for gap_nm, cpl_nm in geometries:
      for bw in a.bus_guide_widths:
       for ba in a.bus_alphas:
        for ta in a.tap_alphas:
            n += 1
            run = ((f"n{a.n_taps}_cpl{int(cpl_nm)}" if cpl_nm > 0
                    else f"n{a.n_taps}_gap{int(gap_nm)}")
                   + ("" if bw is None else f"_bw{int(bw)}")
                   + ("" if bw is None else "")
                   + ("" if a.steps_per_frame == 200
                      else f"_spf{a.steps_per_frame}")
                   + ("" if ta == 1.0 else f"_ta{ta:g}")
                   + ("" if ba == 1.0 else f"_ba{ba:g}"))
            if (pts / f"{run}.json").exists():
                print(f"[{n}/{total}] {run}: cached"); continue
            cmd = [sys.executable, __file__, "--worker", str(gap_nm),
                   str(cpl_nm), str(ta), str(ba),
                   str(-1.0 if bw is None else bw), "--device", a.device,
                   "--outdir", str(outdir), "--n-taps", str(a.n_taps),
                   "--lags", *[str(x) for x in a.lags],
                   "--amp-mT", str(a.amp_mT), "--freq", str(a.freq),
                   "--burst", str(a.burst), "--quiet", str(a.quiet),
                   "--steps-per-frame", str(a.steps_per_frame),
                   "--relax-steps", str(a.relax_steps)]
            # STREAM the worker rather than capturing it. A point is a relax
            # plus a rollout -- tens of minutes on the longer buses -- and
            # capture_output means the parent prints nothing for that whole
            # span, so a relax in progress is indistinguishable from a hang.
            # That matters on a hosted runtime whose CLI tears the session down
            # after a fixed gap between OUTPUT lines, not between results.
            #
            # The last line is still what gets summarised, so the digest at the
            # end is unchanged; only the silence is gone.
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
            lines = []
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                lines.append(line)
                print(f"    | {line}", flush=True)
            if proc.wait() != 0:
                print(f"[{n}/{total}] {run}: FAILED -- "
                      + " | ".join(lines[-3:]), flush=True)
                continue
            print(f"[{n}/{total}] " + (lines[-1] if lines else "(no output)"),
                  flush=True)

    results = {f.stem: json.loads(f.read_text()) for f in sorted(pts.glob("*.json"))}
    results_path.write_text(json.dumps(results, indent=2))

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
