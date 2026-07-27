#!/usr/bin/env python3
"""Render the physical assembly: film1, the disk array, film2 -- to scale.

Three stages that span two orders of magnitude in size, which is the point of
drawing them together: the films are micron-scale wave optics, the disks are
sub-micron magnetic textures, and the coupling between them is what makes the
architecture unusual.

Everything geometric here is read from the real configs and the measured film
response rather than drawn by hand -- probe positions come from
``linear_probe_array``, the scatterer is the actual design field, the disk
firing radius is solved from the Thiele parameters, and the channel responses
are the measured sweep.

    python scripts/render_assembly.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import magnonic_nn as mnn
from magnonic_nn.denoise import DenoiserConfig
from magnonic_nn.reservoir import FilmResponse
from magnonic_nn.thiele import ThieleConfig, ThieleDisks

# validated categorical slots (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"


def firing_radius(cfg: ThieleConfig) -> float:
    """Solve r/R where core speed reaches v_crit, from the Thiele parameters."""
    d = cfg.derived()
    ceiling = d["omega0"] * cfg.R
    for x in np.linspace(0.01, 1.5, 3000):
        if x * ceiling * (1 + cfg.kappa_nl * x * x) >= cfg.v_crit:
            return float(x)
    return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--response", default="runs/film_response/response.pt")
    p.add_argument("--out", default="runs/assembly.png")
    args = p.parse_args()

    mnn.set_precision("float32")
    mnn.set_device("cpu")

    # ---- real geometry from the real configs -------------------------------
    cfg1 = mnn.get_preset("vowels")
    cfg1.mesh.nx = cfg1.mesh.ny = 80
    nx1, ny1, _ = cfg1.mesh.n
    dx1 = cfg1.mesh.dx
    margin1 = cfg1.material.abc_width
    torch.manual_seed(7)
    src1_x = margin1 + 1
    probes1 = mnn.linear_probe_array(cfg1.mesh, 12, x=nx1 - 15, r=1.0, margin=margin1)
    probe_xy = [(pr.x, pr.y) for pr in probes1]

    model1 = mnn.SpinWaveNetwork(
        cfg1, [mnn.LineSource(cfg1.mesh, cfg1.fields, src1_x, 0, src1_x, ny1 - 1)],
        probes1)
    with torch.no_grad():
        model1.geometry.rho.copy_(torch.randn_like(model1.geometry.rho) * 0.3)
    rho = model1.geometry.rho.detach()[:, :, 0].numpy() if model1.geometry.rho.dim() == 3 \
        else model1.geometry.rho.detach().numpy()

    tcfg = ThieleConfig(coupling=0.08, phase_capture=0.5, drive_scale=0.12,
                        temperature=300.0, alpha_spread=0.5)
    td = tcfg.derived()
    r_fire = firing_radius(tcfg)

    dcfg = DenoiserConfig()
    cfg2 = mnn.get_preset("vowels")
    cfg2.mesh.nx = cfg2.mesh.ny = dcfg.nx
    cfg2.material.abc_width = 3
    nx2, ny2, _ = cfg2.mesh.n
    dx2 = cfg2.mesh.dx

    film = FilmResponse.load(args.response)
    u_grid, P = film.u_grid, film.P

    # probe pitch sets the disk pitch: the disks sit where the power lands
    pitch_nm = abs(probe_xy[1][1] - probe_xy[0][1]) * dx1 * 1e9

    fig = plt.figure(figsize=(15.5, 13.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.30, 0.52, 0.95],
                          hspace=0.46, wspace=0.28,
                          left=0.055, right=0.975, top=0.885, bottom=0.055)

    # ================= panel A: film1, to scale ==============================
    axA = fig.add_subplot(gs[0, :2], facecolor=SURFACE)
    ext = [0, nx1 * dx1 * 1e6, 0, ny1 * dx1 * 1e6]
    axA.imshow(rho.T, origin="lower", extent=ext, cmap="RdBu_r",
               vmin=-2.5, vmax=2.5, alpha=0.62, interpolation="bilinear")
    for side in ("left", "right"):
        axA.add_patch(Rectangle((0 if side == "left" else (nx1 - margin1) * dx1 * 1e6, 0),
                                margin1 * dx1 * 1e6, ny1 * dx1 * 1e6,
                                fc=MUTED, alpha=0.28, ec="none"))
    axA.add_patch(Rectangle((0, 0), nx1 * dx1 * 1e6, margin1 * dx1 * 1e6,
                            fc=MUTED, alpha=0.28, ec="none"))
    axA.add_patch(Rectangle((0, (ny1 - margin1) * dx1 * 1e6), nx1 * dx1 * 1e6,
                            margin1 * dx1 * 1e6, fc=MUTED, alpha=0.28, ec="none"))
    axA.plot([src1_x * dx1 * 1e6] * 2, [margin1 * dx1 * 1e6, (ny1 - margin1) * dx1 * 1e6],
             color=ORANGE, lw=3.4, solid_capstyle="butt", zorder=5)
    axA.text(src1_x * dx1 * 1e6 - 0.06, ny1 * dx1 * 1e6 * 0.5, "antenna",
             rotation=90, ha="right", va="center", color=ORANGE, fontsize=10,
             fontweight="bold")
    for i, (px, py) in enumerate(probe_xy):
        axA.add_patch(Circle((px * dx1 * 1e6, py * dx1 * 1e6), 0.075,
                             fc=BLUE, ec=SURFACE, lw=1.4, zorder=6))
        if i in (0, 11):
            axA.annotate(f"probe {i + 1}", (px * dx1 * 1e6, py * dx1 * 1e6),
                         xytext=(11, 0), textcoords="offset points", fontsize=9,
                         color=INK2, va="center")
    lam = 0.46
    axA.annotate("", xy=(1.05, 0.42), xytext=(1.05 + lam, 0.42),
                 arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.3))
    axA.text(1.05 + lam / 2, 0.30, "λ ≈ 460 nm", ha="center", fontsize=9, color=INK2)
    axA.set_title("film₁ — routing:  80×80 cells, 4.0 × 4.0 µm YIG, 60 mT bias.  One antenna in,\n"
                  "twelve probes out; background is the trained scatterer (static field offset)",
                  fontsize=10.8, color=INK, loc="left", pad=8)
    axA.set_xlabel("µm", fontsize=9.5, color=INK2)
    axA.set_ylabel("µm", fontsize=9.5, color=INK2)
    axA.tick_params(labelsize=8.5, colors=INK2)

    # ================= panel B: one disk, blown up ===========================
    axB = fig.add_subplot(gs[0, 2], facecolor=SURFACE)
    R_nm = tcfg.R * 1e9
    axB.add_patch(Circle((0, 0), R_nm, fc=AQUA, alpha=0.14, ec=AQUA, lw=2.2))
    th = np.linspace(0, 2 * np.pi, 400)
    for frac, style, lab, col, ty in (
            (0.30, ":", "sub-threshold", INK2, -0.30),
            (r_fire, "-", f"firing  r/R = {r_fire:.2f}", ORANGE, r_fire)):
        axB.plot(frac * R_nm * np.cos(th), frac * R_nm * np.sin(th),
                 style, color=col, lw=2.0)
        axB.annotate(lab, (0, ty * R_nm), xytext=(150, ty * R_nm * 1.15),
                     textcoords="data", fontsize=8.8, color=col, va="center",
                     ha="right",
                     arrowprops=dict(arrowstyle="-", color=col, lw=0.9,
                                     shrinkA=2, shrinkB=2))
    axB.add_patch(Circle((r_fire * R_nm * 0.72, r_fire * R_nm * 0.69), 6.5,
                         fc=ORANGE, ec=SURFACE, lw=1.2, zorder=6))
    axB.annotate("", xy=(-30, 118), xytext=(30, 118),
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                 connectionstyle="arc3,rad=0.42"))
    axB.text(0, 130, "gyration", ha="center", fontsize=9, color=ORANGE)
    axB.set_xlim(-190, 190)
    axB.set_ylim(-150, 160)
    axB.set_aspect("equal")
    axB.set_title(f"one disk, ×{4000 / (2 * R_nm):.0f} vs film₁\n"
                  f"permalloy, 2R = {2 * R_nm:.0f} nm, L = {tcfg.L * 1e9:.0f} nm",
                  fontsize=10.5, color=INK, loc="left", pad=8)
    axB.set_xlabel("nm", fontsize=9.5, color=INK2)
    axB.tick_params(labelsize=8.5, colors=INK2)

    # ================= panel B2: the array, at true pitch ====================
    axB2 = fig.add_subplot(gs[1, :], facecolor=SURFACE)
    for j in range(12):
        cx = j * pitch_nm
        axB2.add_patch(Circle((cx, 0), R_nm, fc=AQUA, alpha=0.16, ec=AQUA, lw=1.9))
        axB2.add_patch(Circle((cx, 0), r_fire * R_nm, fc="none", ec=ORANGE,
                              lw=1.0, ls="--", alpha=0.75))
        axB2.plot([cx], [0], marker="o", ms=3.5, color=INK2)
        axB2.text(cx, -R_nm - 34, f"{j + 1}", ha="center", fontsize=8.4, color=INK2)
    gap = pitch_nm - 2 * R_nm
    axB2.annotate("", xy=(R_nm, R_nm + 26), xytext=(pitch_nm - R_nm, R_nm + 26),
                  arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
    axB2.text(pitch_nm / 2, R_nm + 36, f"edge gap {gap:.0f} nm",
              ha="center", fontsize=9, color=ORANGE)
    axB2.annotate("", xy=(0, -R_nm - 12), xytext=(pitch_nm, -R_nm - 12),
                  arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.3))
    axB2.text(pitch_nm / 2, -R_nm - 30, f"pitch {pitch_nm:.0f} nm",
              ha="center", fontsize=9, color=INK2, va="top")
    axB2.set_xlim(-R_nm - 60, 11 * pitch_nm + R_nm + 60)
    axB2.set_ylim(-R_nm - 78, R_nm + 82)
    axB2.set_aspect("equal")
    axB2.axis("off")
    axB2.set_title("the array — twelve disks on film₁'s probe pitch, to scale.  "
                   f"Dashed = firing radius.  A {gap:.0f} nm edge gap is why "
                   "dipolar coupling is a design parameter, not a nuisance.",
                   fontsize=10.8, color=INK, loc="left", pad=6)

    # ================= panel C: the cascade ==================================
    axC = fig.add_subplot(gs[2, :2], facecolor=SURFACE)
    axC.set_xlim(0, 10)
    axC.set_ylim(0, 3.1)
    axC.axis("off")
    stages = [
        (0.75, "input\nu(t)", MUTED, "scalar\namplitude"),
        (2.75, "film₁\n80×80", BLUE, "12 channels\n3.3 + 3.9 GHz"),
        (5.05, "12 disks\nspiking", AQUA, "1.0 GHz gyration\n300 K, stochastic"),
        (7.35, "film₂\n20×20", YELLOW, "6 probes\n3.6 GHz carrier"),
        (9.3, "ridge\nreadout", MUTED, "trained\n(linear)"),
    ]
    for x, label, color, sub in stages:
        axC.add_patch(Rectangle((x - 0.62, 1.42), 1.24, 0.86, fc=color, alpha=0.20,
                                ec=color, lw=2.0, zorder=3))
        axC.text(x, 1.85, label, ha="center", va="center", fontsize=10.5,
                 color=INK, fontweight="bold", zorder=4)
        axC.text(x, 1.10, sub, ha="center", va="top", fontsize=8.8, color=INK2)
    for x0, x1, lab in ((0.75, 2.75, "drive"), (2.75, 5.05, "leaked power"),
                        (5.05, 7.35, "stray field\n(AM carrier)"),
                        (7.35, 9.3, "features")):
        axC.add_patch(FancyArrowPatch((x0 + 0.66, 1.85), (x1 - 0.66, 1.85),
                                      arrowstyle="-|>", mutation_scale=15,
                                      color=INK2, lw=1.5))
        axC.text((x0 + x1) / 2, 2.02, lab, ha="center", fontsize=8.6, color=INK2)
    axC.text(5.05, 2.72, "trained by backprop  ·  film₂ trained Noise2Noise, "
                         "never on the task", ha="center", fontsize=9.2,
             color=INK2, style="italic")
    axC.text(0.05, 0.42, "Frequencies are the constraint: the disks gyrate at "
                         "1.0 GHz, below the films' 3.33 GHz FMR, so a disk cannot "
                         "emit a propagating spin wave.\nIts stray field "
                         "amplitude-modulates film₂'s local drive instead — which "
                         "is why the coupling is AM rather than direct emission.",
             fontsize=9.2, color=INK, va="bottom")

    # ================= panel D: the measured channels ========================
    axD = fig.add_subplot(gs[2, 2], facecolor=SURFACE)
    for i in range(12):
        axD.plot(u_grid, P[:, i], color=BLUE, alpha=0.30, lw=1.4)
    axD.plot(u_grid, P.mean(axis=1), color=BLUE, lw=2.4, label="mean of 12")
    axD.axvline(0.5, color=ORANGE, lw=1.6, ls="--")
    axD.text(0.52, 0.06, "decorrelation\nonset", color=ORANGE, fontsize=8.8,
             va="bottom")
    axD.set_title("measured film₁ channels\n(normalised probe power)",
                  fontsize=11, color=INK, loc="left", pad=9)
    axD.set_xlabel("input amplitude  u", fontsize=9.5, color=INK2)
    axD.tick_params(labelsize=8.5, colors=INK2)
    axD.legend(fontsize=8.6, frameon=False, loc="upper left")
    for s in ("top", "right"):
        axD.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axD.spines[s].set_color(MUTED)

    fig.suptitle("Magnonic spiking reservoir — physical assembly",
                 fontsize=15.5, color=INK, x=0.055, ha="left", y=0.972)
    fig.text(0.055, 0.936,
             f"film₁ and film₂ drawn to their own scales; a disk is "
             f"{4000 / (2 * R_nm):.0f}× smaller than film₁'s width.  "
             f"Disk gyration f₀ = {td['f0'] / 1e9:.2f} GHz, "
             f"firing ceiling ω₀R = {td['omega0'] * tcfg.R:.0f} m/s vs "
             f"v_crit = {tcfg.v_crit:.0f} m/s.",
             fontsize=9.8, color=INK2, ha="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=155, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"  disk f0 = {td['f0'] / 1e9:.3f} GHz, omega0*R = "
          f"{td['omega0'] * tcfg.R:.0f} m/s, firing at r/R = {r_fire:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
