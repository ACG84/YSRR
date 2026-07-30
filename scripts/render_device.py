#!/usr/bin/env python3
"""The device as it would be fabricated, to scale, with the numbers checked.

Distinguishes three things that are easy to conflate in a schematic: geometry
that is BUILT and simulated (the ported disk, the two-disk unit), geometry
that is MEASURED but from the earlier film work (film1's router), and layout
that is PROPOSED and has never been simulated (the full array). Drawing them
together at one scale is the point -- it is how you find out whether the
stages fit.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import magnonic_nn as mnn
from magnonic_nn.dispersion import kittel_fmr, usable_band
from magnonic_nn.vortex import (CoupledPortedConfig, CoupledPortedArray,
                                PortedVortexConfig, ported_mask)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET = "#4a3aa7"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

mnn.set_precision("float32"); mnn.set_device("cpu")

pc = PortedVortexConfig()
cc = CoupledPortedConfig(chirality=+1, chirality_b=-1)
arr = CoupledPortedArray(cc, timesteps=10, dtype=torch.float32)
arr.relax(steps=1800, require_tol=1e-4)

unit_nm = 2 * (pc.radius + pc.guide_length) * 1e9        # ported disk footprint
film_cfg = mnn.get_preset("vowels"); film_cfg.mesh.nx = film_cfg.mesh.ny = 80
film_nm = 80 * film_cfg.mesh.dx * 1e9
probes = mnn.linear_probe_array(film_cfg.mesh, 12, x=65, r=1.0,
                                margin=film_cfg.material.abc_width)
pitch_nm = abs(probes[1].y - probes[0].y) * film_cfg.mesh.dx * 1e9
array_span_nm = 11 * cc.separation * 1e9 + unit_nm
fmr = kittel_fmr(film_cfg.fields, film_cfg.material) / 1e9
band = [b / 1e9 for b in usable_band(film_cfg)]

fig = plt.figure(figsize=(15.0, 13.0), facecolor=SURFACE)
gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.15, 0.66], hspace=0.42,
                      wspace=0.20, left=0.115, right=0.975, top=0.885, bottom=0.05)

# ---- A: the built two-disk unit, dimensioned ------------------------------
axA = fig.add_subplot(gs[0, :], facecolor=SURFACE)
nx, ny = cc.grid
ext = [-nx * cc.dx * 5e8, nx * cc.dx * 5e8, -ny * cc.dx * 5e8, ny * cc.dx * 5e8]
mz = np.where(arr.mask[:, :, 0, 0].numpy().T > 0,
              arr.m0[:, :, 0, 2].numpy().T, np.nan)
axA.imshow(mz, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.9)
st = 5
xs = np.linspace(ext[0], ext[1], nx)[::st]; ys = np.linspace(ext[2], ext[3], ny)[::st]
U = arr.m0[::st, ::st, 0, 0].numpy().T; V = arr.m0[::st, ::st, 0, 1].numpy().T
msk = arr.mask[::st, ::st, 0, 0].numpy().T > 0
axA.quiver(*np.meshgrid(xs, ys), np.where(msk, U, np.nan), np.where(msk, V, np.nan),
           color=INK, scale=30, width=0.0019, alpha=0.7)
sep = cc.separation * 1e9
axA.annotate("", xy=(-sep / 2, -230), xytext=(sep / 2, -230),
             arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
axA.text(0, -262, f"{sep:.0f} nm centre-to-centre", ha="center", fontsize=9.4, color=ORANGE)
axA.annotate("", xy=(-sep / 2 - 100, 195), xytext=(-sep / 2 + 100, 195),
             arrowprops=dict(arrowstyle="<->", color=AQUA, lw=1.6))
axA.text(-sep / 2, 213, "2R = 200 nm", ha="center", fontsize=9.4, color=AQUA)
axA.text(0, 60, f"link, {cc.link_width*1e9:.0f} nm wide", ha="center",
         fontsize=9.2, color=VIOLET)
axA.text(sep / 2 + 175, 150, "6 guides,\n40 x 150 nm,\nabsorbing ends",
         fontsize=9.0, color=INK2, ha="center")
axA.set_title("BUILT & SIMULATED — the two-disk unit, relaxed. Permalloy, 20 nm thick, "
              "5 nm cells.\nOpposite chirality: the link magnetises uniformly, no domain "
              "wall, drift 3e-5, lambda -0.292/ns at 30 mT.",
              fontsize=11.0, color=INK, loc="left", pad=9)
axA.set_xlabel("nm", fontsize=9.5, color=INK2); axA.set_ylabel("nm", fontsize=9.5, color=INK2)
axA.tick_params(labelsize=8.5, colors=INK2)

# ---- B: the proposed array against film1's real pitch --------------------
axB = fig.add_subplot(gs[1, :], facecolor=SURFACE)
# film1 on top, the disk row beneath it, both to scale -- laid out horizontally
# so the span comparison is the thing you see first.
axB.add_patch(Rectangle((-film_nm / 2, 900), film_nm, film_nm * 0.42,
                        fc=BLUE, alpha=0.08, ec=BLUE, lw=1.8))
axB.text(-film_nm / 2, 900 + film_nm * 0.42 + 90,
         f"film₁ router — {film_nm:.0f} nm wide (MEASURED)",
         fontsize=9.8, color=BLUE, fontweight="bold")
for i, pr in enumerate(probes):
    px = (pr.y * film_cfg.mesh.dx * 1e9) - film_nm / 2
    axB.add_patch(Circle((px, 900), 34, fc=BLUE, ec="none"))
axB.annotate("", xy=(-11 * pitch_nm / 2, 760), xytext=(11 * pitch_nm / 2, 760),
             arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.7))
axB.text(0, 640, f"12 probes span {11 * pitch_nm:.0f} nm  ({pitch_nm:.0f} nm pitch)",
         ha="center", fontsize=9.6, color=BLUE)

for j in range(12):
    cx = (j - 5.5) * cc.separation * 1e9
    axB.add_patch(Rectangle((cx - unit_nm / 2, -unit_nm / 2), unit_nm, unit_nm,
                            fc="none", ec=AQUA, lw=0.7, ls=":", alpha=0.55))
    axB.add_patch(Circle((cx, 0), pc.radius * 1e9, fc=AQUA, alpha=0.45,
                         ec=AQUA, lw=1.1))
axB.annotate("", xy=(-array_span_nm / 2, -unit_nm / 2 - 190),
             xytext=(array_span_nm / 2, -unit_nm / 2 - 190),
             arrowprops=dict(arrowstyle="<->", color=AQUA, lw=1.7))
axB.text(0, -unit_nm / 2 - 320, f"12 ported disks span {array_span_nm:.0f} nm  "
         f"({sep:.0f} nm pitch, {unit_nm:.0f} nm footprint each)",
         ha="center", fontsize=9.6, color=AQUA)
axB.text(0, -unit_nm / 2 - 700,
         f"THE MISMATCH — {array_span_nm / (11 * pitch_nm):.1f}x. A ported disk is "
         f"{unit_nm:.0f} nm across but the probe pitch is only {pitch_nm:.0f} nm, so the "
         f"disks cannot sit one-per-probe.\nEither the router grows to "
         f"{array_span_nm:.0f} nm (4x the demag cost), or fan-out guides carry each probe "
         f"outward, or the guides shorten at the cost of the aperture budget.",
         ha="center", fontsize=9.6, color=ORANGE)
axB.set_xlim(-array_span_nm / 2 - 300, array_span_nm / 2 + 300)
axB.set_ylim(-unit_nm / 2 - 950, 900 + film_nm * 0.42 + 320)
axB.set_aspect("equal"); axB.axis("off")
axB.set_title("PROPOSED — never simulated. Both stages at one scale, which is how the "
              "conflict shows up.",
              fontsize=11.0, color=INK, loc="left", pad=6)

# ---- C: the frequency stack ----------------------------------------------
axC = fig.add_subplot(gs[2, :], facecolor=SURFACE)
rows = [
    ("film₁ FMR floor", fmr, fmr, MUTED, "waves below this do not propagate"),
    ("film₁ usable band", band[0], band[1], BLUE, "measured, 60 mT bias"),
    ("disk gyrotropic mode", 0.9, 1.1, ORANGE, "below the floor — cannot enter the film"),
    ("disk azimuthal modes", 5.0, 15.0, AQUA, "ABOVE the floor — direct coupling works"),
]
for i, (name, lo, hi, col, note) in enumerate(rows):
    y = len(rows) - i
    if hi > lo:
        axC.barh(y, hi - lo, left=lo, height=0.5, color=col, alpha=0.8)
    else:
        axC.plot([lo, lo], [y - 0.3, y + 0.3], color=col, lw=2.4)
    axC.text(16.4, y, note, fontsize=9.0, color=INK2, va="center")
axC.set_yticks(range(1, len(rows) + 1))
axC.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.6)
axC.set_xlim(0, 30); axC.set_xlabel("frequency (GHz)", fontsize=9.6, color=INK2)
axC.tick_params(labelsize=8.6, colors=INK2)
for s in ("top", "right"): axC.spines[s].set_visible(False)
for s in ("left", "bottom"): axC.spines[s].set_color(MUTED)
axC.set_title("The modal pivot fixed the frequency problem without anyone aiming at it: "
              "azimuthal modes sit above film₁'s FMR,\nso the AM-carrier workaround the "
              "gyrotropic mode needed is no longer required.",
              fontsize=11.0, color=INK, loc="left", pad=8)

fig.suptitle("Magnonic modal reservoir — the device", fontsize=15.5, color=INK,
             x=0.06, ha="left", y=0.977)
fig.text(0.06, 0.925,
         "Built geometry is relaxed live from the solver. The router is the measured film "
         "from the earlier work. The array layout is proposed and\nuntested — and drawing "
         "it to scale shows the two stages do not currently fit together.",
         fontsize=9.9, color=INK2, ha="left")
fig.savefig("runs/device.png", dpi=145, facecolor=SURFACE)
print(f"wrote runs/device.png")
print(f"  ported disk footprint {unit_nm:.0f} nm; probe pitch {pitch_nm:.0f} nm; "
      f"array span {array_span_nm:.0f} nm vs probe span {11*pitch_nm:.0f} nm")
print(f"  film FMR {fmr:.2f} GHz, usable band {band[0]:.2f}-{band[1]:.2f} GHz")
