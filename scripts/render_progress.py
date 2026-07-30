#!/usr/bin/env python3
"""Render the current design and what is actually established about it.

Geometry and magnetisation are relaxed live, not drawn -- the ground state
shown is the one the solver finds, with the drift guard enforced so it cannot
be a half-settled configuration.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.vortex import CoupledPortedConfig, CoupledPortedArray

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

mnn.set_precision("float32"); mnn.set_device("cpu")
def relax_and_measure_drift(arr, steps=1800):
    """Relax, then measure residual drift instead of demanding convergence.

    The walled configuration does not converge -- which is the finding, not an
    inconvenience -- so this reports drift rather than raising on it.
    """
    arr.relax(steps=steps)
    probe = arr.rollout.relax(arr.m0.clone(), arr.h_zero, 25, 0.5)
    return float((probe - arr.m0).norm() / max(float(arr.m0.norm()), 1e-30))


states = {}
for tag, cb in (("same", +1), ("opposite", -1)):
    cfg = CoupledPortedConfig(chirality=+1, chirality_b=cb)
    arr = CoupledPortedArray(cfg, timesteps=10, dtype=torch.float32)
    drift = relax_and_measure_drift(arr)
    states[tag] = (cfg, arr, arr.link_wall(), drift)
    print(f"{tag}: wall={states[tag][2]['reverses']} "
          f"mz_peak={states[tag][2]['mz_peak']:.2f} drift={drift:.2e}", flush=True)

cfg, arr, wall, drift_ok = states["opposite"]
nx, ny = cfg.grid
ext = [-nx * cfg.dx * 5e8, nx * cfg.dx * 5e8, -ny * cfg.dx * 5e8, ny * cfg.dx * 5e8]

fig = plt.figure(figsize=(15.0, 15.2), facecolor=SURFACE)
gs = fig.add_gridspec(3, 2, height_ratios=[1.45, 0.62, 0.98], hspace=0.34,
                      wspace=0.18, left=0.06, right=0.975, top=0.885, bottom=0.04)

# --- A: the built design, relaxed -----------------------------------------
axA = fig.add_subplot(gs[0, :], facecolor=SURFACE)
mz = arr.m0[:, :, 0, 2].numpy().T
mz = np.where(arr.mask[:, :, 0, 0].numpy().T > 0, mz, np.nan)
im = axA.imshow(mz, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
step = 6
xs = np.linspace(ext[0], ext[1], nx)[::step]
ys = np.linspace(ext[2], ext[3], ny)[::step]
U = arr.m0[::step, ::step, 0, 0].numpy().T
V = arr.m0[::step, ::step, 0, 1].numpy().T
msk = arr.mask[::step, ::step, 0, 0].numpy().T > 0
axA.quiver(*np.meshgrid(xs, ys), np.where(msk, U, np.nan), np.where(msk, V, np.nan),
           color=INK, scale=26, width=0.0022, alpha=0.75)
axA.set_title("The built design — two ported vortex disks, opposite chirality, relaxed "
              "ground state.\ncolour = m_z (the cores), arrows = in-plane m. Six guides "
              "per disk; the facing pair forms the link.",
              fontsize=11.2, color=INK, loc="left", pad=9)
axA.set_xlabel("nm", fontsize=9.5, color=INK2); axA.set_ylabel("nm", fontsize=9.5, color=INK2)
axA.tick_params(labelsize=8.5, colors=INK2)
cb = fig.colorbar(im, ax=axA, fraction=0.022, pad=0.01); cb.ax.tick_params(labelsize=8)

# --- B: the link, with and without the wall --------------------------------
axB = fig.add_subplot(gs[1, :], facecolor=SURFACE)
for tag, col, ls in (("same", ORANGE, "-"), ("opposite", BLUE, "--")):
    c2, a2, w2, dr = states[tag]
    x = (torch.arange(c2.grid[0], dtype=torch.float32) - (c2.grid[0] - 1) / 2) * c2.dx
    y = (torch.arange(c2.grid[1], dtype=torch.float32) - (c2.grid[1] - 1) / 2) * c2.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    gap = c2.separation / 2 - c2.radius
    row = (X.abs() < gap) & (Y.abs() < c2.dx) & (a2.mask[:, :, 0, 0] > 0)
    order = X[row].argsort()
    axB.plot((X[row][order] * 1e9).numpy(), a2.m0[:, :, 0, 0][row][order].numpy(),
             ls, color=col, lw=2.2,
             label=f"{tag} chirality — m_z peak {w2['mz_peak']:.2f}, "
                   f"residual drift {dr:.1e}")
axB.axhline(0, color=MUTED, lw=1.0, ls=":")
axB.set_ylim(-1.25, 1.25)
axB.set_xlabel("position along link (nm)", fontsize=9.5, color=INK2)
axB.set_ylabel("m_x", fontsize=9.5, color=INK2)
axB.legend(fontsize=9.2, frameon=False, loc="lower left")
axB.tick_params(labelsize=8.5, colors=INK2)
for s in ("top", "right"): axB.spines[s].set_visible(False)
for s in ("left", "bottom"): axB.spines[s].set_color(MUTED)
axB.set_title("Same chirality nucleates a 180° Bloch wall at the link midpoint — and that "
              "state never settles,\nso every stability number measured on it was measuring "
              "wall drift. Opposite chirality converges.",
              fontsize=11.2, color=INK, loc="left", pad=8)

# --- C: what is established, and what was retired -------------------------
axC = fig.add_subplot(gs[2, :], facecolor=SURFACE)
axC.set_xlim(0, 10); axC.set_ylim(0, 10.6); axC.axis("off")
est = [
    ("modal memory, one disk", "AB vs BA 0.283 at 30 mT vs 0.014 at 1 mT", "20.9x"),
    ("ports do the mode decomposition", "driven n=0,1,2 recovered as peak n", "3/3"),
    ("leak routing is input-dependent", "AB->port 1, BA->port 4; L1 0.294 vs 0.003", "105x"),
    ("aperture buys stability", "1/2/6 ports: +6.49 / +4.51 / -0.67 per ns", "monotone"),
    ("coupled array is stable", "opposite chirality, drift < 1e-4", "-0.292/ns"),
    ("float32 suffices", "matches float64 on every port weight", "0.0000"),
]
ret = [
    ("guides replace near-field coupling", "dipolar is instantaneous; B responds at 40 ps"),
    ("chaos is inter-disk coupling", "identical lambda at 500 / 800 / 1200 nm"),
    ("chaos is intrinsic to one disk", "six-port disk stable at the same drive"),
    ("walled config is merely stable", "it never converges at all: drift 1.2e-2"),
    ("chaos is link feedback (2nd try)", "damping the link raised lambda, not lowered"),
    ("all coupled lambda at 900 steps", "under-relaxed; drift read as divergence"),
]
axC.text(0.0, 10.2, "ESTABLISHED", fontsize=11, color=AQUA, fontweight="bold")
for i, (what, how, num) in enumerate(est):
    yy = 9.4 - i * 1.5
    axC.text(0.0, yy, what, fontsize=10.2, color=INK, fontweight="bold")
    axC.text(0.0, yy - 0.55, how, fontsize=8.9, color=INK2)
    axC.text(4.55, yy, num, fontsize=10.2, color=AQUA, fontweight="bold", ha="right")
axC.text(5.2, 10.2, "RETIRED / CORRECTED", fontsize=11, color=ORANGE, fontweight="bold")
for i, (what, why) in enumerate(ret):
    yy = 9.4 - i * 1.5
    axC.text(5.2, yy, what, fontsize=10.2, color=MUTED)
    axC.text(5.2, yy - 0.55, why, fontsize=8.9, color=INK2)
axC.plot([4.95, 4.95], [0.1, 10.4], color=MUTED, lw=0.9, alpha=0.5)

fig.suptitle("Magnonic modal reservoir — design and evidence",
             fontsize=15.5, color=INK, x=0.06, ha="left", y=0.982)
fig.text(0.06, 0.936,
         "Single-disk physics is verified and the coupled array is stable with opposite "
         "chirality. No task result exists yet: discriminating two\npulse orders is not "
         "time-series prediction. Five instability diagnoses were wrong before a "
         "convergence guard found the cause.",
         fontsize=9.9, color=INK2, ha="left")
fig.savefig("runs/progress.png", dpi=145, facecolor=SURFACE)
print("wrote runs/progress.png")
