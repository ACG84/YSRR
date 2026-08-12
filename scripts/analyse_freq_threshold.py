#!/usr/bin/env python3
"""Does the nonlinear threshold scale with drive frequency, as 1/gamma demands?

The threshold-memory invariant says that for a resonantly driven mode

    h_th ~ alpha * omega / gamma        tau ~ 1 / (alpha * omega)

so their product is 1/gamma = 5.7 ps*T and neither alpha nor omega survives it.
That explains why the element survey closed five routes -- Ms, radius, bias and
frequency are all variables the product is blind to -- and it makes one sharp,
cheap prediction: at fixed alpha the threshold is PROPORTIONAL TO FREQUENCY.

This reads the per-frequency sweeps and tests exactly that, three ways:

    ratio       h_th / (alpha*omega/gamma). Constant across frequency means the
                scaling holds, whatever the constant is. The CONSTANT is the
                off-resonance penalty: permalloy measures 15.00 mT at 12 GHz
                against 3.43 predicted, so a flat ratio of ~4.4 confirms the
                scaling while saying the disk is driven off its own mode.
    slope       a log-log fit of h_th against f. The invariant wants 1.0.
    dip         a ratio that FALLS toward 1 at some frequency locates the
                element's actual resonance, which is where the useful operating
                point is. That is worth more than the scaling itself: it is the
                gate on whether the frame-length experiment can reach threshold.

A flat h_th, slope ~0, would refute the resonant-cone picture outright and take
the frame argument with it.

    python scripts/analyse_freq_threshold.py --dir runs/freq
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from magnonic_nn.config import MU_0

GAMMA = 2.212761569e5 / MU_0     # rad/(s*T); magnum.np's gamma is per A/m


THRESH_COMP, THRESH_PHASE = 0.05, 10.0


def threshold_from(rows):
    """Lowest STABLE amplitude that is measurably nonlinear.

    Deliberately the same rule check_drive_nonlinearity.py applies, on the same
    keys -- compression past 5% or phase past 10 degrees, on a row whose vortex
    survived. Recomputed here rather than scraped from stdout because two
    sessions lost their output to a `| tail` in the launcher, and the results
    files survived both.

    Returns None when nothing crosses. That is a real outcome and must not be
    quietly rendered as the largest amplitude tested -- an element that never
    goes nonlinear would otherwise enter the fit as though it had.
    """
    for r in rows:
        if not r.get("vortex_ok", True):
            continue
        if (abs(r["normalised"] - 1.0) > THRESH_COMP
                or abs(r["d_phase_deg"]) > THRESH_PHASE):
            return r["amp_mT"]
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="runs/freq", help="directory of results_*.json")
    p.add_argument("--alpha", type=float, default=0.008)
    a = p.parse_args()

    pts = []
    for f in sorted(Path(a.dir).glob("results_*.json")):
        m = re.search(r"_f([0-9.]+)\.json$", f.name)
        if not m:
            continue
        rows = json.loads(f.read_text())
        h = threshold_from(rows)
        pts.append((float(m.group(1)), h, len(rows)))
    if not pts:
        raise SystemExit(f"no results_*_f*.json in {a.dir}")
    pts.sort()

    print(f"alpha {a.alpha}, gamma {GAMMA:.4g} rad/(s*T), "
          f"1/gamma {1e12/GAMMA:.2f} ps*T\n")
    print(f"{'f (GHz)':>8} {'h_th (mT)':>10} {'predicted':>10} {'ratio':>7} "
          f"{'tau (ns)':>9} {'amps':>5}")
    fs, hs = [], []
    for f_ghz, h, n in pts:
        w = 2 * math.pi * f_ghz * 1e9
        pred = a.alpha * w / GAMMA * 1e3
        tau = 1.0 / (a.alpha * w) * 1e9
        if h is None:
            print(f"{f_ghz:>8.2f} {'none':>10} {pred:>10.2f} {'--':>7} "
                  f"{tau:>9.2f} {n:>5}")
            continue
        print(f"{f_ghz:>8.2f} {h:>10.2f} {pred:>10.2f} {h/pred:>7.2f} "
              f"{tau:>9.2f} {n:>5}")
        fs.append(f_ghz); hs.append(h)

    if len(fs) < 3:
        print("\nfewer than three frequencies crossed; no scaling to fit.")
        return 0
    # Log-log slope. The invariant wants 1.0; a flat threshold gives 0.
    sl, ic = np.polyfit(np.log(fs), np.log(hs), 1)
    pred_h = np.exp(ic) * np.array(fs) ** sl
    ss_res = float(((np.array(hs) - pred_h) ** 2).sum())
    ss_tot = float(((np.array(hs) - np.mean(hs)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\nlog-log slope {sl:+.2f} (R^2 = {r2:.3f}); "
          f"the invariant predicts +1.00, a flat threshold 0.00")

    ratios = [h / (a.alpha * 2 * math.pi * f * 1e9 / GAMMA * 1e3)
              for f, h in zip(fs, hs)]
    lo = int(np.argmin(ratios))
    print(f"off-resonance penalty runs {min(ratios):.2f}x to {max(ratios):.2f}x; "
          f"lowest at {fs[lo]:g} GHz")
    # A resonance is an INTERIOR minimum. Requiring that is not fussiness: a
    # perfectly flat threshold makes the ratio fall as 1/f by construction, so
    # its minimum always lands on the highest frequency tested, and a naive
    # min() test reported "a dip at 12 GHz" on synthetic flat data -- announcing
    # a resonance in exactly the case that refutes the model.
    interior = 0 < lo < len(fs) - 1
    if interior and min(ratios) < 0.5 * max(ratios):
        print(f"\nA DIP at {fs[lo]:g} GHz: the threshold approaches its "
              f"on-resonance floor there,\nwhich is where the element's own mode "
              f"is and where a tap should be driven.")
    elif not interior:
        print("  (minimum sits at an endpoint, so this is a trend across the "
              "band rather\n  than a resonance in it; widen the sweep to "
              "bracket one.)")
    if abs(sl - 1.0) < 0.35:
        print("\nSCALING HOLDS. h_th tracks frequency, so the invariant's "
              "threshold factor is\nconfirmed and the frame-length argument "
              "stands on measured ground.")
    elif abs(sl) < 0.35:
        print("\nFLAT. The threshold does not track frequency, which refutes "
              "the resonant-cone\npicture and the frame argument built on it. "
              "The invariant would need\nre-deriving before any frame-length "
              "run is worth its hours.")
    else:
        print(f"\nSlope {sl:+.2f} is neither 1 nor 0. Structure in the response "
              f"-- most likely a\nmode crossing the band -- so read the ratio "
              f"column rather than the fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
