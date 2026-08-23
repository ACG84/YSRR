#!/usr/bin/env python3
"""Hodographs for an angle-driven ASVI reservoir.

A meteorological hodograph traces the tip of the wind vector against HEIGHT,
so that shear -- how the vector turns as you climb -- becomes a shape rather
than a table of numbers. The adaptation here traces the tip of the net
magnetisation against INPUT INDEX, and it fits this device unusually well for
one specific reason: the drive is angle-encoded. The input u[n] is delivered as
a field DIRECTION, 45 + 90*u degrees at fixed magnitude, so input and state
live in the same plane and can be drawn on the same axes. The question "how far
does the state lag the drive" is then a visible angle rather than a statistic.

What each panel is for.

WHERE THE LITERAL VERSION FAILS. A meteorological hodograph is readable
because its ordering parameter is SMOOTH: consecutive heights give
neighbouring wind vectors, so the connecting line means something. Our input
is i.i.d. uniform, so consecutive states jump across the arc and joining them
in input order draws a tangle whose line segments carry no information. That
is not a plotting defect, it is the wrong ordering for this data, and the
first version of this script produced exactly that spaghetti. Two adaptations
keep the geometry and drop the false ordering.

  difference hodograph   the DIFFERENCE between the two ESP starts, traced
                         against n. This is the one quantity that does evolve
                         smoothly whatever the input does, because contraction
                         is a property of the map rather than of the draw. The
                         echo state property becomes a spiral into the origin,
                         and the horizon is where the spiral arrives -- drawn
                         rather than thresholded, and carrying the geometry of
                         the approach that the scalar `rel` discards.
  transfer map           response angle against DRIVE angle, one point per
                         input, coloured by n. A memoryless device would lie
                         on a single curve: one drive angle, one response. The
                         measured device does not -- the same drive angle
                         reaches different responses depending on the state it
                         arrives at, which is the state dependence the pooled
                         switch statistics inferred (smallest |u| that switched
                         0.03, largest that held 0.94) shown directly instead
                         of inferred.
  per island-layer       small multiples in each island's OWN frame
                         (longitudinal, transverse), as scatter rather than a
                         joined path for the reason above. A square lattice
                         has two sublattices at 90 degrees, so lab-frame
                         traces differ for a trivial reason; the island frame
                         removes that and makes them comparable.

WHY THIS AND NOT A TIME SERIES. The label the ESP checker records is one
character per island-layer, and five inputs of the 4x4 run carried identical
labels while the relative distance sat at 0.21 -- the state was moving in a way
the readout could not express. A hodograph plots the continuous vector that
character was quantising, so that motion has somewhere to show up.

    python scripts/plot_hodograph.py runs/hodograph/asvi_esp.json -o hodo.png
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def net(parts):
    """Array net in-plane moment, lab frame."""
    return (sum(p["mx"] for p in parts) / len(parts),
            sum(p["my"] for p in parts) / len(parts))


def coloured_path(ax, xs, ys, cmap, lw=1.8, alpha=1.0):
    """A path whose colour advances with input index, so direction of travel
    is readable without arrowheads cluttering a small panel.

    add_collection does NOT extend the axes data limits, so an axis holding
    only LineCollections autoscales to nothing and renders blank -- which is
    exactly what the first smoke test produced. update_datalim is what makes
    the limits follow the data.
    """
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, linewidth=lw, alpha=alpha)
    lc.set_array(np.linspace(0, 1, max(len(seg), 1)))
    ax.add_collection(lc)
    ax.update_datalim(np.column_stack([xs, ys]))
    return lc


def square_limits(ax, pad=0.12, floor=0.25):
    """Equal-aspect panels need a square window or the trace is distorted.
    A floor keeps a nearly-stationary trace from being blown up into noise."""
    x0, x1 = ax.dataLim.x0, ax.dataLim.x1
    y0, y1 = ax.dataLim.y0, ax.dataLim.y1
    if not all(map(np.isfinite, (x0, x1, y0, y1))):
        return
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max((x1 - x0) / 2, (y1 - y0) / 2, floor) * (1 + pad)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json", nargs="+", help="asvi_esp.json files")
    p.add_argument("-o", "--out", default="hodograph.png")
    p.add_argument("--conv", type=float, default=0.05)
    p.add_argument("--max-parts", type=int, default=8)
    a = p.parse_args()

    runs = []
    for f in a.json:
        for r in json.load(open(f)):
            if r["history"] and "parts" in r["history"][0]:
                runs.append(r)
    if not runs:
        raise SystemExit("no run carries per-part vectors; re-run "
                         "check_asvi_esp.py with the vector readout")

    n = len(runs)
    fig = plt.figure(figsize=(5.2 * n, 10.4), constrained_layout=True)
    gs = fig.add_gridspec(3, n, height_ratios=[1.25, 1.25, 1.0])

    for c, r in enumerate(runs):
        h = r["history"]
        hit = next((x["n"] for x in h if x["rel"] <= a.conv), None)
        amp = r["amp_mT"]

        # ---- panel 1: difference hodograph, the ESP contraction ----------
        ax = fig.add_subplot(gs[0, c])
        D = [(net(x["parts"][0])[0] - net(x["parts"][1])[0],
              net(x["parts"][0])[1] - net(x["parts"][1])[1]) for x in h]
        dx, dy = zip(*D)
        lc = coloured_path(ax, dx, dy, "viridis")
        ax.scatter(dx, dy, c=np.linspace(0, 1, len(dx)), cmap="viridis",
                   s=18, zorder=3, edgecolor="white", linewidth=0.4)
        ax.plot(0, 0, "k+", ms=13, mew=2, zorder=4)
        if hit:
            ax.plot(dx[hit - 1], dy[hit - 1], "o", ms=13, mfc="none",
                    mec="crimson", mew=2, zorder=5,
                    label=f"converged n={hit}")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal")
        ax.set_title(f"{amp:g} mT — start separation, spiralling in",
                     fontsize=10)
        ax.set_xlabel(r"$\Delta M_x$"); ax.set_ylabel(r"$\Delta M_y$")
        ax.axhline(0, color="0.9", lw=.6); ax.axvline(0, color="0.9", lw=.6)
        square_limits(ax)
        fig.colorbar(lc, ax=ax, label="input n", fraction=0.046)

        # ---- panel 2: transfer map, response angle vs drive angle --------
        ax2 = fig.add_subplot(gs[1, c])
        dr = [x["theta_deg"] for x in h if "theta_deg" in x]
        if dr:
            # LAG, not raw response angle. atan2 returns (-180, 180], so a
            # response just past +180 is drawn at the far bottom of the axis
            # and reads as an outlier when it is a neighbour -- the synthetic
            # test produced exactly that at drive 105 deg. Wrapping the
            # DIFFERENCE removes the seam, and puts "no memory" on a
            # horizontal line where a spread is easy to see.
            wrap = lambda d: (d + 180.0) % 360.0 - 180.0
            resp = []
            for x, t in zip(h, dr):
                mx, my = net(x["parts"][0])
                resp.append(wrap(math.degrees(math.atan2(my, mx)) - t))
            sc = ax2.scatter(dr, resp, c=range(1, len(resp) + 1),
                             cmap="plasma", s=34, edgecolor="white", lw=.4)
            lo, hi_ = min(dr) - 6, max(dr) + 6
            ax2.axhline(0, ls="--", color="0.65", lw=1,
                        label="zero lag (no memory)")
            ax2.set_xlim(lo, hi_)
            ax2.legend(fontsize=8, loc="best")
            fig.colorbar(sc, ax=ax2, label="input n", fraction=0.046)
        ax2.set_title("transfer map: same drive, different response",
                      fontsize=10)
        ax2.set_xlabel("drive angle (deg)"); ax2.set_ylabel("response lag (deg)")
        ax2.grid(alpha=.25)

        # ---- panel 3: per island-layer, island frame ---------------------
        ax3 = fig.add_subplot(gs[2, c])
        npart = min(len(h[0]["parts"][0]), a.max_parts)
        cols = min(npart, 4)
        rows = math.ceil(npart / cols)
        ax3.axis("off")
        inner = ax3.get_subplotspec().subgridspec(rows, cols, wspace=.05, hspace=.05)
        for j in range(npart):
            axx = fig.add_subplot(inner[j // cols, j % cols])
            L = [x["parts"][0][j]["ml"] for x in h]
            T = [x["parts"][0][j]["mt"] for x in h]
            axx.scatter(L, T, c=range(len(L)), cmap="magma", s=9,
                        linewidth=0)
            axx.set_aspect("equal")
            axx.set_xlim(-1.15, 1.15); axx.set_ylim(-1.15, 1.15)
            axx.set_xticks([]); axx.set_yticks([])
            axx.axhline(0, color="0.9", lw=.5); axx.axvline(0, color="0.9", lw=.5)
            axx.text(.04, .90, f"L{j}", transform=axx.transAxes, fontsize=7)
        ax3.set_title("per island-layer (longitudinal, transverse)",
                      fontsize=10, pad=16)

    # ---- the number behind the middle panel ------------------------------
    # A picture of scatter is an argument only if the scatter is bigger than
    # the trend. For a memoryless device lag would be a single-valued function
    # of drive angle, so two nearly equal drives would give nearly equal lags.
    # Reporting the closest pair is fit-free: it assumes no model of the trend
    # and cannot be inflated by choosing a poor one.
    print("\nstate dependence in the lag (memoryless => lag is a function of "
          "drive angle alone)")
    for r in runs:
        h = r["history"]
        if "theta_deg" not in h[0]:
            continue
        wrap = lambda d: (d + 180.0) % 360.0 - 180.0
        dr = np.array([x["theta_deg"] for x in h])
        lag = np.array([wrap(math.degrees(math.atan2(*net(x["parts"][0])[::-1]))
                             - x["theta_deg"]) for x in h])
        res = lag - np.polyval(np.polyfit(dr, lag, 5), dr)
        o = np.argsort(dr); ds, ls = dr[o], lag[o]
        near = [(abs(ls[i + 1] - ls[i]), ds[i], ds[i + 1])
                for i in range(len(ds) - 1) if ds[i + 1] - ds[i] < 5.0]
        best = max(near) if near else None
        print(f"  {r['amp_mT']:g} mT: lag spans {lag.min():+.1f} to "
              f"{lag.max():+.1f} deg; residual after a smooth fit "
              f"{res.std():.1f} deg std")
        if best:
            print(f"           drives {best[1]:.1f} and {best[2]:.1f} deg "
                  f"({best[2]-best[1]:.1f} apart) differ in lag by "
                  f"{best[0]:.1f} deg")

    fig.suptitle("ASVI reservoir — contraction geometry and input/response lag",
                 fontsize=13)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"wrote {a.out}  ({n} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
