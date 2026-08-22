#!/usr/bin/env python3
"""Summarise the drive sweep, using statistics that survive a short run.

Two metrics in this project's ESP checker turned out to be unreliable in the
regime the sweep was optimising into, and both failures point the same way:
a statistic computed from a handful of samples at the end of a run measures the
run's length, not the device.

  horizon         the convergence step. RIGHT-CENSORED at n_steps: a run of 20
                  inputs reporting horizon 19 has two post-convergence samples
                  and a true horizon that could be 25 or 40. Values near
                  n_steps are floors, not measurements, and must not be ranked
                  against each other.
  last-5 liveness the USEFUL/FREEZES split asks whether the final five
                  post-convergence inputs visit more than one state. At a
                  measured switch rate near 0.4 a run of five non-switches has
                  probability ~0.13, so across thirty amplitude-seed
                  combinations several FREEZE by chance. The verdict was
                  reading the input draw.

SWITCH RATE replaces the second. It pools every post-convergence transition
rather than the last five, so its error falls with run length instead of
staying fixed, and it does not care where in the run the switches happened.

It also settles a question the individual trajectories got wrong. Reading two
runs by eye suggested a clean amplitude threshold -- "only |u| above about 0.9
switches". Pooled, the smallest |u| that DID switch is 0.08 and the largest
that did NOT is 0.93, at the same amplitude. There is no threshold: the same
input magnitude goes either way depending on the state it lands on, which is
the history dependence a reservoir needs and a fixed threshold would preclude.
"""
from __future__ import annotations
import argparse, collections, glob, json, os, re, statistics as st


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        tag = os.path.basename(f)[:-5]
        m = re.match(r"s(\d+)_a([\d.]+)_m([\d\-]+)_d(\d+)(?:_n(\d+))?$", tag)
        if not m:
            continue
        settle, alpha, _, seed, nst = m.groups()
        for r in json.load(open(f)):
            out.append({"settle": int(settle), "alpha": float(alpha),
                        "seed": int(seed), "n_steps": int(nst or 20),
                        "amp": r["amp_mT"], "history": r["history"]})
    return out


def summarise(r):
    h = r["history"]
    hit = next((x["n"] for x in h if x["rel"] <= 0.05), None)
    if hit is None:
        return {"horizon": None, "censored": False, "switch": None,
                "n_post": 0, "states": 0}
    post = [x for x in h if x["n"] >= hit]
    ch = sum(1 for a, b in zip(post, post[1:]) if a["labels"][0] != b["labels"][0])
    return {"horizon": hit,
            # A horizon within 3 of the end has too few post-convergence
            # samples to be anything but a lower bound.
            "censored": hit >= r["n_steps"] - 2,
            "switch": ch / max(len(post) - 1, 1) if len(post) > 1 else None,
            "n_post": len(post), "states": len(set(x["labels"][0] for x in post))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="runs/drive_sweep/*.json")
    p.add_argument("--settle", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.5)
    a = p.parse_args()

    rows = [r for r in load(a.glob)
            if r["settle"] == a.settle and r["alpha"] == a.alpha]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["amp"]].append((r["seed"], r["n_steps"], summarise(r)))

    print(f"settle {a.settle}, alpha {a.alpha}\n")
    print(f"{'mT':>4}{'seeds':>7}{'fail':>6}{'horizon (per seed)':>26}"
          f"{'censored':>10}{'switch rate':>13}{'states':>8}")
    for amp in sorted(by):
        e = by[amp]
        hs = [s["horizon"] for _, _, s in e if s["horizon"]]
        cen = sum(1 for _, _, s in e if s["censored"])
        fail = sum(1 for _, _, s in e if s["horizon"] is None)
        sws = [s["switch"] for _, _, s in e if s["switch"] is not None]
        sts = [s["states"] for _, _, s in e if s["horizon"]]
        hstr = ",".join(str(x) for x in sorted(hs)) if hs else "-"
        sw = f"{st.mean(sws):.2f}" if sws else "-"
        stt = f"{st.mean(sts):.1f}" if sts else "-"
        print(f"{amp:>4.0f}{len(e):>7}{fail:>6}{hstr:>26}{cen:>10}{sw:>13}{stt:>8}")

    print("\ncensored = horizon within 2 of n_steps, so a floor not a value.")
    print("An amplitude with any censored seed cannot be ranked on horizon.")

    # ------------------------------------------------------- paired by seed
    # The seed IS the input sequence: one draw of u is shared by every
    # amplitude in a run and by every run with that seed. So the seed is a
    # nuisance factor COMMON to the amplitudes being compared, and it cancels
    # in a per-seed difference.
    #
    # This matters because the seed effect is larger than the amplitude effect.
    # Unpaired, 62 mT spans horizon 9-19 and 70 mT spans 7-16: the ranges
    # overlap so heavily they look like one distribution, and four rounds of
    # this sweep were spent adding seeds to average the variance down. Paired,
    # 62 mT wins on every seed. The variance was never irreducible noise --
    # it was removable by construction, by differencing instead of averaging.
    hz = collections.defaultdict(dict)
    for r in rows:
        hz[r["amp"]][r["seed"]] = summarise(r)["horizon"]
    amps = sorted(hz)
    pairs = [(a, b) for i, a in enumerate(amps) for b in amps[i + 1:]
             if len(set(hz[a]) & set(hz[b])) >= 2]
    if pairs:
        print("\npaired by seed (same input sequence at both amplitudes)")
        print(f"{'pair':>12}{'per-seed difference':>26}{'mean':>8}  consistent?")
        for a, b in pairs:
            sd = sorted(set(hz[a]) & set(hz[b]))
            d = [hz[a][s] - hz[b][s] for s in sd
                 if hz[a][s] and hz[b][s]]
            if not d:
                continue
            sign = ("all favour %g mT" % a if all(x > 0 for x in d)
                    else "all favour %g mT" % b if all(x < 0 for x in d)
                    else "MIXED -- no consistent winner")
            ds = ",".join(f"{x:+d}" for x in d)
            print(f"{f'{a:g} vs {b:g}':>12}{ds:>26}{sum(d)/len(d):>+8.1f}  {sign}")

    # Pooled switch statistics: is there an amplitude threshold on |u|?
    lo, hi = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        h = r["history"]
        hit = next((x["n"] for x in h if x["rel"] <= 0.05), None)
        if hit is None:
            continue
        post = [x for x in h if x["n"] >= hit]
        for x, y in zip(post, post[1:]):
            (lo if x["labels"][0] != y["labels"][0] else hi)[r["amp"]].append(abs(y["u"]))
    print(f"\n{'mT':>4}{'min |u| switched':>18}{'max |u| held':>14}  overlap?")
    for amp in sorted(lo):
        if not lo[amp] or not hi[amp]:
            continue
        a_, b_ = min(lo[amp]), max(hi[amp])
        print(f"{amp:>4.0f}{a_:>18.2f}{b_:>14.2f}  "
              f"{'YES -- state dependent' if a_ < b_ else 'no -- threshold'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
