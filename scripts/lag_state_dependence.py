#!/usr/bin/env python3
"""Is the response lag a measure of MEMORY, or a restatement of the DRIVE?

The hodograph's middle panel shows the angular lag between the drive direction
and the array's response, and at seed 4 the long-memory point (62 mT) spread
over 83 degrees against the short-memory point's (80 mT) 49. That is the
ordering a memory measure should produce. It is also exactly the ordering a
measure of nothing but drive amplitude would produce, because a weaker field
turns the state less far per input and so leaves the response further from the
drive direction. One comparison at one seed cannot tell those apart.

THE SEPARATION. Memory horizon varies enormously between input draws at FIXED
amplitude -- 7 to 19 at 70 mT in the 20-input runs -- while the drive along
that axis is identical by construction. So:

    measures memory   lag spread tracks horizon WITHIN an amplitude, across
                      seeds. The amplitude cannot produce this, being constant.
    measures drive    lag spread is flat across seeds within an amplitude and
                      moves only between amplitudes.

The within-amplitude correlation is the test. The between-amplitude one is
confounded and is reported only because its absence would be informative.

WHY IT WOULD MATTER. Horizon costs a 40-input run with two initial conditions
and a convergence threshold, and the threshold brought censoring and
window-length artefacts that consumed four rounds of the drive sweep. Lag
spread is computed from ONE trajectory, per input, with no second start and no
threshold. If it tracks horizon it is the cheaper probe; if it does not, the
hodograph is a picture and not an instrument.

    python scripts/lag_state_dependence.py
"""
from __future__ import annotations
import argparse, glob, json, math, os, re, collections
import numpy as np

WRAP = lambda d: (d + 180.0) % 360.0 - 180.0


def net(parts):
    return (sum(p["mx"] for p in parts) / len(parts),
            sum(p["my"] for p in parts) / len(parts))


def stats(run):
    """Horizon and lag statistics for one (amplitude, seed) run."""
    h = run["history"]
    if "theta_deg" not in h[0] or "parts" not in h[0]:
        return None
    hit = next((x["n"] for x in h if x["rel"] <= 0.05), None)
    dr = np.array([x["theta_deg"] for x in h])
    lag = np.array([WRAP(math.degrees(math.atan2(*net(x["parts"][0])[::-1]))
                         - t) for x, t in zip(h, dr)])
    # Residual after a smooth trend in drive angle. Whatever survives cannot be
    # explained by the input, because the drive angle IS the input.
    res = lag - np.polyval(np.polyfit(dr, lag, 5), dr)
    return {"horizon": hit, "span": float(lag.max() - lag.min()),
            "resid": float(res.std()), "n": len(h)}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="runs/drive_sweep/*_n40.json")
    a = p.parse_args()

    rows = []
    for f in sorted(glob.glob(a.glob)):
        m = re.search(r"_d(\d+)_n40$", os.path.basename(f)[:-5])
        if not m:
            continue
        for r in json.load(open(f)):
            s = stats(r)
            if s:
                rows.append({"amp": r["amp_mT"], "seed": int(m.group(1)), **s})
    if not rows:
        raise SystemExit("no runs carry per-part vectors")

    print(f"{'mT':>5}{'seed':>6}{'horizon':>9}{'lag span':>10}{'residual':>10}")
    for r in sorted(rows, key=lambda q: (q["amp"], q["seed"])):
        hz = str(r["horizon"]) if r["horizon"] else "never"
        print(f"{r['amp']:>5.0f}{r['seed']:>6}{hz:>9}"
              f"{r['span']:>10.1f}{r['resid']:>10.2f}")

    # ---- the test: within an amplitude, across seeds ---------------------
    print("\nWITHIN amplitude (drive held constant -- this is the real test)")
    print(f"{'mT':>5}{'runs':>6}{'horizons':>18}{'r(horizon,span)':>17}"
          f"{'r(horizon,resid)':>18}")
    allp = []
    for amp in sorted({r["amp"] for r in rows}):
        e = [r for r in rows if r["amp"] == amp and r["horizon"]]
        if len(e) < 3:
            print(f"{amp:>5.0f}{len(e):>6}{'(need 3+ seeds)':>18}")
            continue
        hz = [r["horizon"] for r in e]
        rs, rr = pearson(hz, [r["span"] for r in e]), pearson(hz, [r["resid"] for r in e])
        allp.append((amp, rs, rr))
        print(f"{amp:>5.0f}{len(e):>6}{','.join(map(str, sorted(hz))):>18}"
              f"{rs:>17.2f}{rr:>18.2f}")

    print("\nBETWEEN amplitudes (confounded by drive -- context only)")
    for k in ("span", "resid"):
        byamp = collections.defaultdict(list)
        for r in rows:
            byamp[r["amp"]].append(r[k])
        line = "  ".join(f"{a:g} mT {np.mean(v):.1f}" for a, v in sorted(byamp.items()))
        print(f"  mean {k}: {line}")

    # ---- partial correlation, controlling for amplitude ------------------
    # Stratifying into one group per amplitude throws away power: three groups
    # of three, two of which have no horizon variance to correlate against.
    # Regressing BOTH variables on amplitude and correlating what is left uses
    # every run at once and removes the confound explicitly rather than by
    # slicing. With n runs and one covariate the test has n-3 degrees of
    # freedom, which is stated alongside the number so it cannot be read as
    # more than it is.
    e = [r for r in rows if r["horizon"]]
    if len(e) >= 6:
        A = np.array([r["amp"] for r in e], float)
        H = np.array([r["horizon"] for r in e], float)
        print(f"\nPARTIAL correlation with amplitude regressed out (n={len(e)}, "
              f"df={len(e)-3})")
        for k in ("span", "resid"):
            Y = np.array([r[k] for r in e], float)
            hr = H - np.polyval(np.polyfit(A, H, 1), A)
            yr = Y - np.polyval(np.polyfit(A, Y, 1), A)
            r_ = pearson(hr, yr)
            # two-sided t on n-3 df, normal approximation for the tail
            if np.isfinite(r_) and abs(r_) < 1:
                t = r_ * math.sqrt((len(e) - 3) / max(1 - r_ * r_, 1e-12))
                pv = math.erfc(abs(t) / math.sqrt(2))
            else:
                t, pv = float("nan"), float("nan")
            print(f"  r(horizon, {k} | amplitude) = {r_:+.2f}  "
                  f"t={t:+.2f}  p~{pv:.2f}")
        print("  amplitude alone explains: "
              f"r(amp,horizon)={pearson(A, H):+.2f}, "
              f"r(amp,resid)={pearson(A, [r['resid'] for r in e]):+.2f}")

    print("\nverdict")
    if not allp:
        print("  not enough seeds per amplitude to run the within-amplitude test")
    else:
        good = [x for x in allp if x[1] > 0.5 or x[2] > 0.5]
        if len(good) >= max(2, len(allp) - 1):
            print("  lag spread tracks horizon at CONSTANT drive, so it is not a\n"
                  "  restatement of the amplitude. It is the cheaper probe: one\n"
                  "  trajectory, no second start, no convergence threshold.")
        elif any(abs(x[1]) > 0.5 or abs(x[2]) > 0.5 for x in allp):
            print("  mixed. Correlation appears at some amplitudes and not others,\n"
                  "  or changes sign. Not usable as a memory measure without\n"
                  "  understanding which regime it holds in.")
        else:
            print("  lag spread does NOT track horizon at constant drive. The\n"
                  "  seed-4 ordering was the amplitude, not the memory, and the\n"
                  "  hodograph is a picture rather than an instrument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
