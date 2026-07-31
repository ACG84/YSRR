#!/usr/bin/env python3
"""The whole proposed machine, to scale, with verified and unverified separated.

film1 (routing) -> ported disk array (reservoir) -> film2 (denoising readout).

The separation is the point of the figure. A schematic that draws a measured
result and an untested intention in the same ink is how a project talks itself
into believing its own plan, and this one has already had to retire a 42x link
transfer, a separation sweep and a link-loss sweep that were all measured on a
channel that did not exist at the frequency used. So each element is drawn in
one of three states and the legend says which:

    VERIFIED    measured on this geometry, with the number quoted
    STABLE      simulated and converged, but its usefulness not yet shown
    PROPOSED    never simulated at this scale

Scale is honest too: every element is drawn at its true size in nanometres, in
one coordinate system, which is how the 3x layout mismatch between the disk
footprint and the film probe pitch was found in the first place.

    python scripts/render_full_architecture.py
"""
from __future__ import annotations
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Wedge
import magnonic_nn as mnn
from magnonic_nn.vortex import PortedVortexConfig

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET = "#4a3aa7"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
GRID = "#e4e2dd"

VERIFIED, STABLE, PROPOSED = AQUA, BLUE, ORANGE

pc = PortedVortexConfig()
R = pc.radius * 1e9                      # 100 nm
GL = pc.guide_length * 1e9               # 150 nm
GW = pc.guide_width * 1e9                # 80 nm
UNIT = 2 * (R + GL)                      # 500 nm footprint per disk
ARC = GW / R * 180 / math.pi             # 46 deg


def ported_disk(ax, cx, cy, scale=1.0, face=BLUE, alpha=0.9, label=None,
                lw=1.4):
    """One ported disk at true relative size."""
    r, gl, gw = R * scale, GL * scale, GW * scale
    for k in range(pc.n_ports):
        th = 2 * math.pi * k / pc.n_ports
        dx, dy = math.cos(th), math.sin(th)
        ax.add_patch(Rectangle((cx + dx * r * 0.7 - gw / 2 * abs(dy),
                                cy + dy * r * 0.7 - gw / 2 * abs(dx)),
                               0, 0))  # placeholder for bbox
        ax.plot([cx + dx * r * 0.75, cx + dx * (r + gl)],
                [cy + dy * r * 0.75, cy + dy * (r + gl)],
                color=face, lw=max(gw * 0.055 * scale ** 0, 1.4), alpha=0.55,
                solid_capstyle="butt", zorder=2)
    ax.add_patch(Circle((cx, cy), r, facecolor=face, alpha=alpha, ec=INK,
                        lw=lw, zorder=3))
    ax.add_patch(Circle((cx, cy), r * 0.16, facecolor=SURFACE, ec=INK,
                        lw=0.8, zorder=4))
    if label:
        ax.text(cx, cy - r - gl - 26 * scale, label, ha="center", fontsize=8.5,
                color=INK2)


def state_chip(ax, x, y, colour, text, note=""):
    ax.add_patch(Circle((x, y), 3.4, color=colour, zorder=5))
    ax.text(x + 11, y + 1.5, text, fontsize=9.5, color=INK, va="center",
            fontweight="bold")
    if note:
        ax.text(x + 11, y - 8.0, note, fontsize=8.2, color=INK2, va="center")


fig = plt.figure(figsize=(16.4, 12.2), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 0.62], hspace=0.30,
                      wspace=0.24, left=0.045, right=0.975, top=0.885,
                      bottom=0.055)

# ---------------------------------------------------------------- main plan --
ax = fig.add_subplot(gs[0, :], facecolor=SURFACE)
ax.set_aspect("equal")

N_DISKS = 4
PITCH_REAL = UNIT                 # disks cannot sit closer than their footprint
PROBE_PITCH = 250.0               # what film1's 12-channel router actually gives
SPAN = (N_DISKS - 1) * PITCH_REAL
FILM_H = SPAN + UNIT
FILM_W = 520.0
GAP = 480.0

# --- film1: the router -----------------------------------------------------
f1x = 0.0
ax.add_patch(Rectangle((f1x, -FILM_H / 2), FILM_W, FILM_H,
                       facecolor=VERIFIED, alpha=0.13, ec=VERIFIED, lw=1.6,
                       zorder=1))
ax.text(f1x + FILM_W / 2, FILM_H / 2 + 62, "film1 — routing", ha="center",
        fontsize=12, fontweight="bold", color=INK)
ax.text(f1x + FILM_W / 2, FILM_H / 2 + 24,
        "1 in, 12 channels", ha="center",
        fontsize=9, color=INK2)
n_probe = int(FILM_H // PROBE_PITCH)
for j in range(n_probe):
    yy = -FILM_H / 2 + (j + 0.5) * PROBE_PITCH
    used = abs((yy + SPAN / 2) % PITCH_REAL) < 1.0 or abs(
        (yy + SPAN / 2) % PITCH_REAL - PITCH_REAL) < 1.0
    ax.plot([f1x + FILM_W * 0.86], [yy], "o", ms=5.2 if used else 3.4,
            color=VERIFIED if used else MUTED, zorder=4,
            alpha=1.0 if used else 0.45)
ax.annotate("", xy=(f1x + 40, 0), xytext=(f1x - 150, 0),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
ax.text(f1x - 155, 34, "u(t)", fontsize=11, color=INK, fontweight="bold",
        ha="right")

# --- the array, at the pitch physics permits -------------------------------
a0 = f1x + FILM_W + GAP
for k in range(N_DISKS):
    cy = -SPAN / 2 + k * PITCH_REAL
    ported_disk(ax, a0 + UNIT / 2, cy, scale=1.0, face=STABLE, alpha=0.85)
    ax.annotate("", xy=(a0 + 30, cy), xytext=(f1x + FILM_W * 0.90, cy),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
ax.text(a0 + UNIT / 2, SPAN / 2 + UNIT / 2 + 152, "ported vortex disk array",
        ha="center", fontsize=12, fontweight="bold", color=INK)
ax.text(a0 + UNIT / 2, SPAN / 2 + UNIT / 2 + 116,
        f"{N_DISKS} disks · 500 nm footprint · 6 ports each at 46°",
        ha="center", fontsize=9, color=INK2)

# the mismatch, drawn to scale instead of asserted
mx = a0 - 120
ax.annotate("", xy=(mx, -SPAN / 2), xytext=(mx, -SPAN / 2 + PITCH_REAL),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.8))
ax.text(mx - 16, -SPAN / 2 + PITCH_REAL / 2, "500 nm\ndisk pitch\n(footprint)",
        ha="right", va="center", fontsize=8.6, color=INK2)
ax.annotate("", xy=(mx + 56, -SPAN / 2), xytext=(mx + 56, -SPAN / 2 + PROBE_PITCH),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2.2))
ax.text(mx + 70, -SPAN / 2 + PROBE_PITCH / 2, "250 nm\nprobe pitch",
        ha="left", va="center", fontsize=8.6, color=ORANGE, fontweight="bold")
ax.text(a0 + UNIT / 2, -SPAN / 2 - UNIT / 2 - 78,
        "2× mismatch — only every other probe can address a disk;\n"
        "the router or the footprint has to change",
        ha="center", fontsize=9.2, color=ORANGE, fontweight="bold")

# --- film2 -----------------------------------------------------------------
f2x = a0 + UNIT + GAP
ax.add_patch(Rectangle((f2x, -FILM_H / 2), FILM_W, FILM_H,
                       facecolor=PROPOSED, alpha=0.12, ec=PROPOSED, lw=1.6,
                       ls=(0, (6, 3)), zorder=1))
ax.text(f2x + FILM_W / 2, FILM_H / 2 + 62, "film2 — denoising readout",
        ha="center", fontsize=12, fontweight="bold", color=INK)
ax.text(f2x + FILM_W / 2, FILM_H / 2 + 24,
        "24 in → 6 out, Noise2Noise", ha="center",
        fontsize=9, color=INK2)
for j in range(6):
    yy = -FILM_H / 2 + (j + 0.5) * FILM_H / 6
    ax.plot([f2x + FILM_W * 0.86], [yy], "o", ms=5.4, color=PROPOSED, zorder=4)
for k in range(N_DISKS):
    cy = -SPAN / 2 + k * PITCH_REAL
    ax.annotate("", xy=(f2x - 20, cy * 0.55), xytext=(a0 + UNIT - 30, cy),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
ax.annotate("", xy=(f2x + FILM_W + 150, 0), xytext=(f2x + FILM_W - 20, 0),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2))
ax.text(f2x + FILM_W + 158, 34, "ŷ(t)", fontsize=11, color=INK,
        fontweight="bold")

ax.set_xlim(f1x - 320, f2x + FILM_W + 320)
ax.set_ylim(-FILM_H / 2 - 190, FILM_H / 2 + 200)
ax.set_xlabel("nanometres — every element at true relative scale",
              fontsize=9.5, color=INK2)
ax.set_yticks([])
ax.tick_params(colors=MUTED, labelsize=8.5)
for s_ in ax.spines.values():
    s_.set_color(GRID)

# ---------------------------------------------------------- state of proof --
axL = fig.add_subplot(gs[1, 0], facecolor=SURFACE)
axL.axis("off"); axL.set_xlim(0, 100); axL.set_ylim(0, 152)
axL.set_title("what is actually established", fontsize=11.5, fontweight="bold",
              color=INK, loc="left", pad=6)
items = [
    (VERIFIED, "guides propagate", "80×20 nm, cutoff 11 GHz, Q 8.0→22.8"),
    (VERIFIED, "ports decompose modes", "3/3, |n| share 0.70 / 0.83 / 0.86"),
    (VERIFIED, "disk discriminates order", "AB/BA 23.4×, nonlinear"),
    (VERIFIED, "it reaches the ports", "24.9×, better than the interior"),
    (STABLE, "two-disk array stable", "λ −0.164/ns, converged at 5000 steps"),
    (PROPOSED, "film1 routing at this pitch", "250 nm pitch vs 500 nm footprint"),
    (PROPOSED, "film2 denoiser in the loop", "trained standalone, never end-to-end"),
]
y = 144
for col, head, note in items:
    state_chip(axL, 4, y, col, head, note)
    y -= 20.5

# ------------------------------------------------------------- the verdict --
axR = fig.add_subplot(gs[1, 1:], facecolor=SURFACE)
axR.axis("off"); axR.set_xlim(0, 100); axR.set_ylim(0, 100)
axR.set_title("and what it does NOT yet do", fontsize=11.5, fontweight="bold",
              color=INK, loc="left", pad=6)
axR.text(0, 84,
         "NARMA-10, single ported disk, read out only at the ports:",
         fontsize=10, color=INK2)
rows = [("input only", "1.041", MUTED),
        ("linear 10-lag  (a shift register)", "0.589", ORANGE),
        ("this device", "0.607", BLUE)]
yy = 70
for name, val, col in rows:
    axR.add_patch(Circle((2.2, yy + 1.4), 1.9, color=col))
    axR.text(7, yy, name, fontsize=10, color=INK)
    axR.text(46, yy, val, fontsize=11, color=INK, fontweight="bold")
    yy -= 11
axR.text(52, 70, "test NMSE  (1.0 = predicting the mean)", fontsize=8.6,
         color=MUTED)
axR.text(0, 44,
         "It loses to a ten-tap linear filter. The cause is measured: memory\n"
         "capacity 2.6 against the 10 lags the task needs. A three-tap memory\n"
         "cannot compute a ten-tap function, so the nonlinearity — real and\n"
         "verified above — has nothing to work with. Published reservoirs\n"
         "reach NMSE 0.1–0.2 here.",
         fontsize=9.6, color=INK2, va="top")
axR.text(0, 2, "The physics is verified. The computation is not.",
         fontsize=11, color=INK, fontweight="bold", va="bottom")

fig.suptitle("Magnonic reservoir — the full proposed machine, and how much of "
             "it is real", fontsize=18, fontweight="bold", color=INK, x=0.045,
             ha="left", y=0.955)
fig.text(0.045, 0.906,
         "Green is measured on this geometry. Blue is simulated and stable but "
         "not yet shown to be useful. Orange is proposed and never simulated at "
         "this scale.", fontsize=11, color=INK2, ha="left")

out = Path("runs/full_architecture.png")
fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}")
