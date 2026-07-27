#!/usr/bin/env python3
"""Render what encasing the disks in waveguides changes.

Three things, each answering a measured failure rather than a preference:

* coupling stops being the near-field dipolar term that the 250 nm probe
  pitch forces on us at a 50 nm edge gap -- the measured chaos source -- and
  becomes a guided coupling whose strength and delay are set by geometry;
* the delay holds history OUTSIDE the disk, where a spike's reset cannot
  reach it, which is aimed squarely at the measured lag-1 autocorrelation of
  -0.006;
* the guide is multi-port, and the ports share one emission budget, so
  routing power to the readout costs recruitment strength.

Guide lengths are computed from the dipole-exchange dispersion at the bias
that puts FMR below the disks' gyrotropic frequency, not asserted.

    python scripts/render_waveguides.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import magnonic_nn as mnn
from magnonic_nn.dispersion import frequency_of_k, kittel_fmr
from magnonic_nn.thiele import ThieleConfig

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET, MAGENTA = "#4a3aa7", "#e87ba4"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"


def group_velocity(freq, fields, material, thickness, dk_frac=1e-3):
    """dω/dk by central difference on the dipole-exchange dispersion."""
    from magnonic_nn.dispersion import k_of_frequency
    k = k_of_frequency(freq, fields, material, thickness)
    if not np.isfinite(k) or k <= 0:
        return float("nan")
    dk = k * dk_frac
    f_hi = frequency_of_k(k + dk, fields, material, thickness)
    f_lo = frequency_of_k(k - dk, fields, material, thickness)
    return 2 * np.pi * (f_hi - f_lo) / (2 * dk)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/waveguides.png")
    args = p.parse_args()

    mnn.set_precision("float64")
    mnn.set_device("cpu")

    tc = ThieleConfig(coupling=0.01, phase_capture=0.5, drive_scale=0.12,
                      temperature=300.0, alpha_spread=0.5,
                      wg_coupling=0.10, wg_feedback=0.06,
                      port_neighbour=0.6, port_readout=0.3, port_return=0.1)
    td = tc.derived()
    frame_ns = 300 * td["dt"] * 1e9
    f0_ghz = td["f0"] / 1e9

    # A guide that carries the disks' 1 GHz needs FMR below it: find the bias.
    cfg = mnn.get_preset("vowels")
    bias_mT, fmr_ghz, vg = None, None, None
    for b in np.arange(1.0, 40.0, 0.5):
        cfg.fields.B0 = b * 1e-3
        f = kittel_fmr(cfg.fields, cfg.material) / 1e9
        if f >= f0_ghz:
            break
        bias_mT, fmr_ghz = b, f
    cfg.fields.B0 = (bias_mT or 5.0) * 1e-3
    vg = group_velocity(f0_ghz * 1e9, cfg.fields, cfg.material, cfg.mesh.dz)

    delays = [("neighbour", 3, AQUA), ("return (adjoint)", 4, MAGENTA),
              ("self-feedback", 7, VIOLET)]

    fig = plt.figure(figsize=(15.0, 10.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.40,
                          wspace=0.22, left=0.115, right=0.972, top=0.835,
                          bottom=0.065)

    # ---------------- panel A: before vs after ------------------------------
    axA = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    axA.set_xlim(0, 10); axA.set_ylim(0, 6.2); axA.axis("off")
    axA.text(0, 5.85, "before — near-field dipolar", fontsize=11.5,
             color=INK, fontweight="bold")
    for j in range(4):
        cx = 1.4 + j * 2.1
        axA.add_patch(Circle((cx, 4.5), 0.72, fc=AQUA, alpha=0.16, ec=AQUA, lw=1.8))
        if j < 3:
            for r, a in ((0.30, 0.55), (0.55, 0.30), (0.85, 0.16)):
                axA.add_patch(Circle(((cx + cx + 2.1) / 2, 4.5), r * 2.1 / 2,
                                     fc="none", ec=ORANGE, lw=1.1, alpha=a))
    axA.text(5.0, 3.32, "instantaneous · isotropic · 50 nm gap forced by the\n"
                        "250 nm probe pitch  →  measured chaos source",
             fontsize=9.4, color=ORANGE, ha="center")

    axA.text(0, 2.55, "after — guided", fontsize=11.5, color=INK,
             fontweight="bold")
    for j in range(4):
        cx = 1.4 + j * 2.1
        axA.add_patch(FancyBboxPatch((cx - 0.95, 0.55), 1.9, 1.55,
                                     boxstyle="round,pad=0.04",
                                     fc=BLUE, alpha=0.10, ec=BLUE, lw=1.7))
        axA.add_patch(Circle((cx, 1.32), 0.62, fc=AQUA, alpha=0.20, ec=AQUA, lw=1.7))
        if j < 3:
            axA.add_patch(FancyArrowPatch((cx + 0.95, 1.32), (cx + 1.15, 1.32),
                                          arrowstyle="-|>", mutation_scale=13,
                                          color=BLUE, lw=2.2))
    axA.text(5.0, 0.02, "delayed · directed · strength AND phase set by geometry",
             fontsize=9.4, color=BLUE, ha="center")

    # ---------------- panel B: the port budget ------------------------------
    axB = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    axB.set_xlim(-0.4, 11.2); axB.set_ylim(0, 6.6); axB.axis("off")
    axB.add_patch(FancyBboxPatch((3.4, 2.3), 3.2, 2.2,
                                 boxstyle="round,pad=0.06",
                                 fc=BLUE, alpha=0.09, ec=BLUE, lw=2.0))
    axB.add_patch(Circle((5.0, 3.4), 0.85, fc=AQUA, alpha=0.22, ec=AQUA, lw=2.0))
    axB.text(5.0, 3.4, "disk", ha="center", va="center", fontsize=10.5,
             color=INK, fontweight="bold")
    ports = [
        (0.6, "neighbour", AQUA, (0.7, 3.4), "recruitment"),
        (0.3, "readout", YELLOW, (9.4, 5.0), "→ film₂ decoder"),
        (0.1, "return", MAGENTA, (9.4, 1.7), "→ adjoint channel"),
    ]
    for frac, name, col, (tx, ty), note in ports:
        sx = 3.4 if tx < 5 else 6.6
        axB.add_patch(FancyArrowPatch((sx, 3.4), (tx, ty), arrowstyle="-|>",
                                      mutation_scale=15, color=col,
                                      lw=1.2 + 5.0 * frac,
                                      connectionstyle="arc3,rad=0.13"))
        ha = "right" if tx > 5 else "left"
        axB.text(tx, ty + 0.42, f"{name}  {frac:.0%}", fontsize=10, color=col,
                 fontweight="bold", ha=ha)
        axB.text(tx, ty + 0.06, note, fontsize=8.8, color=INK2, ha=ha)
    axB.text(5.0, 1.45, "one emission budget, enforced:  0.6 + 0.3 + 0.1 ≤ 1",
             ha="center", fontsize=10.2, color=INK, fontweight="bold")
    axB.text(5.0, 0.75, "power routed to the readout is recruitment strength the\n"
                        "array no longer has — the tradeoff is physical, not a knob",
             ha="center", fontsize=9.3, color=INK2)
    axB.set_title("multi-port guide", fontsize=11.5, color=INK, loc="left", pad=6)

    # ---------------- panel C: the delay budget -----------------------------
    axC = fig.add_subplot(gs[1, 0], facecolor=SURFACE)
    for i, (name, nfr, col) in enumerate(delays):
        tau = nfr * frame_ns
        L = vg * tau * 1e-9 * 1e6 if np.isfinite(vg) else float("nan")
        axC.barh(i, nfr, height=0.55, color=col, alpha=0.85)
        axC.text(nfr + 0.18, i, f"{tau:.1f} ns   ≈ {L:.1f} µm of guide",
                 va="center", fontsize=9.6, color=INK2)
    axC.axvline(10, color=ORANGE, lw=2.0, ls="--")
    axC.text(10.15, len(delays) - 0.35, "NARMA-10\nneeds 10", color=ORANGE,
             fontsize=9.4, va="top")
    axC.set_yticks(range(len(delays)))
    axC.set_yticklabels([d[0] for d in delays], fontsize=10)
    axC.set_xlabel(f"delay (frames of {frame_ns:.2f} ns)", fontsize=9.8, color=INK2)
    axC.set_xlim(0, 15)
    axC.tick_params(labelsize=9, colors=INK2)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axC.spines[s].set_color(MUTED)
    axC.set_title("delay is the memory the disks cannot hold",
                  fontsize=11.5, color=INK, loc="left", pad=8)

    # ---------------- panel D: the failure it targets ------------------------
    axD = fig.add_subplot(gs[1, 1], facecolor=SURFACE)
    widths = [1, 2, 4, 8, 16, 32]
    ns_measured = [1.221, 1.255, 1.293, 1.351, 1.396, 1.485]
    axD.plot(widths, ns_measured, "-o", color=ORANGE, lw=2.2, ms=7,
             label="measured, no guides")
    axD.axhline(1.0, color=MUTED, lw=1.3, ls=":")
    axD.axhline(0.5, color=AQUA, lw=1.8, ls="--")
    axD.text(1.15, 0.54, "usable", fontsize=9.2, color=AQUA)
    axD.text(1.15, 1.04, "noise = signal", fontsize=9.2, color=MUTED)
    axD.set_xscale("log", base=2)
    axD.set_xticks(widths)
    axD.set_xticklabels(widths)
    axD.set_ylim(0, 1.65)
    axD.set_xlabel("boxcar width (frames)", fontsize=9.8, color=INK2)
    axD.set_ylabel("noise / signal spread", fontsize=9.8, color=INK2)
    axD.tick_params(labelsize=9, colors=INK2)
    axD.legend(fontsize=9, frameon=False, loc="lower right")
    for s in ("top", "right"):
        axD.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axD.spines[s].set_color(MUTED)
    axD.set_title("why: averaging made it worse, because there was\n"
                  "no memory to average over (lag-1 autocorr −0.006)",
                  fontsize=11, color=INK, loc="left", pad=8)

    fig.suptitle("What encasing the disks in waveguides changes",
                 fontsize=15.5, color=INK, x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.900,
             f"Disk gyration {f0_ghz:.2f} GHz.  A guide that carries it needs "
             f"FMR below it: {bias_mT:.1f} mT bias gives {fmr_ghz:.2f} GHz, "
             f"group velocity {vg:.0f} m/s.\nGuide lengths below are computed "
             f"from that dispersion, not assumed — every delay is a few microns "
             f"of on-chip waveguide.",
             fontsize=9.9, color=INK2, ha="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"  f0 = {f0_ghz:.3f} GHz, guide bias {bias_mT} mT -> FMR {fmr_ghz:.3f} GHz, "
          f"v_g = {vg:.1f} m/s, frame = {frame_ns:.2f} ns")
    for name, nfr, _ in delays:
        print(f"  {name:18s} {nfr} frames = {nfr * frame_ns:5.1f} ns = "
              f"{vg * nfr * frame_ns * 1e-9 * 1e6:5.2f} um")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
