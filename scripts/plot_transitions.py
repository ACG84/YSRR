#!/usr/bin/env python3
"""Time-domain view of the ASVI reservoir: WHICH island switches, and WHEN.

The hodograph collapses the array to one net vector and plots its geometry.
That is the right object for asking whether the state contracts, and the wrong
one for asking what the array is doing internally -- twenty-four islands
summing to the same net moment can be in very different configurations, which
is precisely the degeneracy that let five inputs of the 4x4 run carry identical
labels while the relative distance sat at 0.21.

Putting time on the x axis and the island-layers on the y axis undoes that
collapse. Three panels, each answering a question the net vector cannot:

  raster          per island-layer orientation angle, atan2(m_trans, m_long) in
                  the island's OWN frame, as colour against input index. A
                  switch is a colour change. Reading down a column shows
                  whether islands switch TOGETHER -- an avalanche -- or
                  independently, which is the difference between an array with
                  many effective degrees of freedom and one behaving as a
                  single big spin.
  transition size |d(m_long, m_trans)| per island-layer per input, summed over
                  the array, against the drive angle that produced it. This is
                  where state dependence becomes legible as an event rather
                  than a statistic: the same drive angle appears twice with a
                  large excursion once and none the other time.
  dwell times     histogram of how many inputs an island-layer holds an
                  orientation before changing. A reservoir needs a spread: all
                  short dwells is a device that forgets immediately, all long
                  is one that has latched, and the distribution's tail is the
                  memory the readout can actually use.

The raster is the one that answers "what are the state transitions", and it
needs the per-part vectors that check_asvi_esp.py and run_narma_asvi.py now
record. Runs from before that change carry only the one-character label and
cannot be read this way.

    python scripts/plot_transitions.py runs/drive_sweep/s500_a0.5_m62_d4_n40.json
"""
from __future__ import annotations
import argparse, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WRAP = lambda d: (d + 180.0) % 360.0 - 180.0


def series(run, start=0):
    """(n_inputs, n_parts) arrays of orientation angle and step-to-step jump."""
    h = run["history"]
    P = np.array([[[p["ml"], p["mt"]] for p in x["parts"][start]] for x in h])
    ang = np.degrees(np.arctan2(P[:, :, 1], P[:, :, 0]))
    jump = np.zeros_like(ang)
    jump[1:] = np.linalg.norm(P[1:] - P[:-1], axis=2)
    return ang, jump, P


def dwells(ang, thresh):
    """Runs of inputs during which an island-layer holds its orientation."""
    out = []
    for j in range(ang.shape[1]):
        run_len = 1
        for n in range(1, len(ang)):
            if abs(WRAP(ang[n, j] - ang[n - 1, j])) > thresh:
                out.append(run_len); run_len = 1
            else:
                run_len += 1
        out.append(run_len)
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json", nargs="+")
    p.add_argument("-o", "--out", default="transitions.png")
    p.add_argument("--switch-deg", type=float, default=25.0,
                   help="orientation change counted as a switch")
    a = p.parse_args()

    runs = []
    for f in a.json:
        # A NARMA checkpoint carries the same per-island-layer vectors as an
        # ESP json, flattened four-per-part. Reading both means a long CPU
        # NARMA run doubles as the data source for this view instead of
        # needing its own GPU run -- which matters when there is no GPU.
        if f.endswith(".pt"):
            import torch
            ck = torch.load(f, map_location="cpu", weights_only=False)
            F = np.asarray(ck["feats"], dtype=float)
            J = F.shape[1] // 4
            th = ck.get("thetas", [])
            hist = [{"parts": [[{"ml": row[4*j], "mt": row[4*j+1],
                                 "circ": row[4*j+2], "pk": row[4*j+3]}
                                for j in range(J)]],
                     **({"theta_deg": th[i]} if i < len(th) else {})}
                    for i, row in enumerate(F)]
            runs.append({"amp_mT": ck["fp"].get("amp", float("nan")),
                         "history": hist})
            print(f"{f}: {len(hist)} inputs, {J} island-layers, "
                  f"drive angle {'stored' if th else 'MISSING (middle panel empty)'}")
            continue
        for r in json.load(open(f)):
            if r["history"] and "parts" in r["history"][0]:
                runs.append(r)
    if not runs:
        raise SystemExit("no run carries per-part vectors")

    n = len(runs)
    fig, axes = plt.subplots(3, n, figsize=(6.4 * n, 10.5), squeeze=False,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.3, 1.0, 0.8]})

    for c, r in enumerate(runs):
        h = r["history"]
        ang, jump, P = series(r)
        amp = r["amp_mT"]
        drive = np.array([x.get("theta_deg", np.nan) for x in h])
        N, J = ang.shape

        # ---- raster: who switches, when --------------------------------
        ax = axes[0][c]
        im = ax.imshow(ang.T, aspect="auto", origin="lower", cmap="twilight",
                       vmin=-180, vmax=180,
                       extent=[0.5, N + 0.5, -0.5, J - 0.5], interpolation="nearest")
        ax.set_yticks(range(J))
        ax.set_yticks(range(J), [f"L{j}" for j in range(J)], fontsize=7)
        ax.set_xlabel("input n"); ax.set_title(
            f"{amp:g} mT — orientation per island-layer (island frame)", fontsize=10)
        fig.colorbar(im, ax=ax, label="angle (deg)", fraction=0.03)

        # ---- transition size against the drive that caused it ----------
        ax2 = axes[1][c]
        tot = jump.sum(1)
        sc = ax2.scatter(drive, tot, c=np.arange(1, N + 1), cmap="plasma",
                         s=32, edgecolor="white", lw=.4)
        ax2.set_xlabel("drive angle (deg)")
        ax2.set_ylabel(r"total $|\Delta m|$ over array")
        ax2.set_title("same drive, different excursion", fontsize=10)
        ax2.grid(alpha=.25)
        fig.colorbar(sc, ax=ax2, label="input n", fraction=0.03)
        # Pair up near-identical drives and label the largest disagreement:
        # a single event carries the state dependence more plainly than a
        # correlation does.
        o = np.argsort(drive)
        best = None
        for i in range(len(o) - 1):
            if drive[o[i + 1]] - drive[o[i]] < 5.0:
                d = abs(tot[o[i + 1]] - tot[o[i]])
                if best is None or d > best[0]:
                    best = (d, o[i], o[i + 1])
        if best:
            _, i1, i2 = best
            ax2.plot(drive[[i1, i2]], tot[[i1, i2]], "k--", lw=1, zorder=1)
            ax2.annotate(f"{drive[i1]:.0f}° vs {drive[i2]:.0f}°\n"
                         f"Δ={abs(tot[i1]-tot[i2]):.2f}",
                         xy=(drive[i2], tot[i2]), fontsize=8,
                         xytext=(6, 6), textcoords="offset points")

        # ---- dwell times -----------------------------------------------
        ax3 = axes[2][c]
        d = dwells(ang, a.switch_deg)
        ax3.hist(d, bins=range(1, int(d.max()) + 2), align="left",
                 color="#4878a8", edgecolor="white")
        ax3.set_xlabel(f"inputs held (switch > {a.switch_deg:g}°)")
        ax3.set_ylabel("count")
        ax3.set_title(f"dwell times — median {np.median(d):.0f}, "
                      f"max {d.max()}", fontsize=10)
        nsw = sum(1 for j in range(J) for k in range(1, N)
                  if abs(WRAP(ang[k, j] - ang[k-1, j])) > a.switch_deg)
        print(f"{amp:g} mT: {nsw} switches over {N} inputs x {J} island-layers "
              f"= {nsw/(N*J):.2f} per island per input; dwell median "
              f"{np.median(d):.0f}, max {d.max()}")

    fig.suptitle("ASVI reservoir — state transitions in the time domain",
                 fontsize=13)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
