#!/usr/bin/env python3
"""Render the AB/BA mode spectra: the evidence for the modal pivot.

Identical average INPUT spectra by construction, so a linear disk must give
identical average OUTPUT spectra. The low-power column is that control; the
high-power column is the claim.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch

BLUE, ORANGE, INK, INK2, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
dt_ps, record_every = 1.0, 2
d = {k: torch.load(f"runs/modal_disk/spectra_{k}.pt", weights_only=False)
     for k in ("low", "high")}
nf = d["high"]["AB"].shape[1]
freq = np.fft.rfftfreq(2 * (nf - 1), d=dt_ps * 1e-12 * record_every) / 1e9

modes = [1, 3, 5]
fig, axes = plt.subplots(len(modes), 2, figsize=(12.4, 8.2), facecolor=SURFACE,
                         sharex=True)
for r, n in enumerate(modes):
    for c, (key, title) in enumerate((("low", "1 mT — linear control"),
                                      ("high", "30 mT — scattering on"))):
        ax = axes[r, c]; ax.set_facecolor(SURFACE)
        a = d[key]["AB"][n].numpy(); b = d[key]["BA"][n].numpy()
        sel = freq <= 20
        ax.plot(freq[sel], a[sel], color=BLUE, lw=1.7, label='"AB"')
        ax.plot(freq[sel], b[sel], color=ORANGE, lw=1.7, ls="--", label='"BA"')
        cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        ax.set_yscale("log")
        ax.text(0.975, 0.88, f"cos = {cos:.4f}", transform=ax.transAxes,
                ha="right", fontsize=10, color=INK,
                fontweight="bold" if cos < 0.99 else "normal")
        if r == 0:
            ax.set_title(title, fontsize=12, color=INK, loc="left", pad=8)
            ax.legend(fontsize=9.5, frameon=False, loc="upper left")
        if c == 0:
            ax.set_ylabel(f"mode n = {n}", fontsize=10.5, color=INK2)
        if r == len(modes) - 1:
            ax.set_xlabel("frequency (GHz)", fontsize=10, color=INK2)
        ax.tick_params(labelsize=8.5, colors=INK2)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)

fig.suptitle("One vortex disk distinguishes \"AB\" from \"BA\" — only when scattering is on",
             fontsize=14.5, color=INK, x=0.055, ha="left", y=0.985)
fig.text(0.055, 0.905,
         "Both orders contain the same two pulses, so the average input spectra are identical and a "
         "linear disk must return identical outputs.\nAt 1 mT it does (cos > 0.999). At 30 mT the "
         "dipole-active odd modes separate — memory and nonlinearity, no delay line. Ratio 20.9x.",
         fontsize=9.8, color=INK2, ha="left")
fig.tight_layout(rect=[0.045, 0.02, 0.985, 0.865])
fig.savefig("runs/modal_spectra.png", dpi=150, facecolor=SURFACE)
print("wrote runs/modal_spectra.png")
