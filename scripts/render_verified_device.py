#!/usr/bin/env python3
"""The device as verified: geometry to scale, with every claim's evidence beside it.

Five panels, each one a measurement rather than a schematic assertion:

  A  the ported disk as patterned, to scale, with the vortex ground state and
     the port angular budget drawn -- six 80 nm guides subtend 275 of 360 deg
  B  guide quality factor Q = decay/wavelength against frequency, which is what
     separates a waveguide from a hole in the film
  C  the disk's azimuthal mode ladder, measured from a ring-down, against the
     band the guides actually pass
  D  the AB/BA discrimination scored by PASSBAND -- the panel that reversed the
     design, since a guide with a cutoff passes everything above it rather than
     a convenient slice
  E  port mode decomposition at both widths, folded onto |n|

Panel D is the one worth reading closely. The same disk, the same guides, two
drive frequencies: at 6/9 GHz the discrimination the guides transport is 1.2x,
indistinguishable from a linear transient. At 11.7/13.7 GHz it is 15.0x. The
guides were never the problem.

    python scripts/render_verified_device.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
import magnonic_nn as mnn
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
GRID = "#e4e2dd"

mnn.set_precision("float32"); mnn.set_device("cpu")


def load_journal(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def qfac(r):
    lam, d = r.get("wavelength_nm"), r["decay_nm"]
    if lam is None or (isinstance(lam, float) and math.isnan(lam)) or lam <= 0:
        return None
    return None if math.isinf(d) else d / lam


def band_ratio(indir, cutoffs, fmax=40.0, dt_rec=2e-12):
    """AB/BA high/low ratio for a guide that passes everything above cutoff."""
    spec = {lab: torch.load(Path(indir) / f"spectra_{lab}.pt", weights_only=False)
            for lab in ("low", "high")}
    n_freq = spec["low"]["AB"].shape[1]
    f = np.fft.rfftfreq(2 * (n_freq - 1), d=dt_rec) / 1e9
    out = []
    for c in cutoffs:
        m = (f >= c) & (f <= fmax)
        vals = {}
        for lab in ("low", "high"):
            a, b = spec[lab]["AB"].numpy()[:, m], spec[lab]["BA"].numpy()[:, m]
            den = np.linalg.norm(a) + np.linalg.norm(b)
            vals[lab] = np.linalg.norm(a - b) / max(den, 1e-30)
        out.append(vals["high"] / max(vals["low"], 1e-12))
    return out


# ---------------------------------------------------------------- geometry --
pc = PortedVortexConfig()
disk = PortedVortexDisk(pc, timesteps=10, dtype=torch.float32)
disk.relax(steps=900)
arc_deg = pc.guide_width / pc.radius * 180 / math.pi

fig = plt.figure(figsize=(15.5, 12.6), facecolor=SURFACE)
gs = fig.add_gridspec(3, 3, height_ratios=[1.30, 0.92, 0.92], hspace=0.46,
                      wspace=0.30, left=0.065, right=0.975, top=0.885,
                      bottom=0.055)

# ---- A: the ported disk, to scale -----------------------------------------
axA = fig.add_subplot(gs[0, :2], facecolor=SURFACE)
n = pc.n_cells
half = n * pc.dx * 0.5e9
ext = [-half, half, -half, half]
mask = disk.mask[:, :, 0, 0].numpy().T
mz = np.where(mask > 0, disk.m0[:, :, 0, 2].numpy().T, np.nan)
axA.imshow(mz, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1,
           alpha=0.92, interpolation="nearest")
st = 4
xs = np.linspace(ext[0], ext[1], n)[::st]
ys = np.linspace(ext[2], ext[3], n)[::st]
U = disk.m0[::st, ::st, 0, 0].numpy().T
V = disk.m0[::st, ::st, 0, 1].numpy().T
mk = disk.mask[::st, ::st, 0, 0].numpy().T > 0
axA.quiver(*np.meshgrid(xs, ys), np.where(mk, U, np.nan), np.where(mk, V, np.nan),
           color=INK2, alpha=0.5, scale=26, width=0.0028, headwidth=3.4)
axA.add_patch(Circle((0, 0), pc.radius * 1e9, fill=False, ec=INK, lw=1.6,
                     ls=(0, (5, 3))))
for k, th in enumerate(pc.port_angles()):
    d = math.degrees(th)
    axA.add_patch(Wedge((0, 0), pc.radius * 1e9, d - arc_deg / 2,
                        d + arc_deg / 2, width=pc.radius * 1e9 * 0.13,
                        facecolor=ORANGE, alpha=0.55, ec="none"))
    rr = (pc.radius + pc.guide_length) * 1e9 - 26
    axA.text(rr * math.cos(th), rr * math.sin(th), f"P{k}", color=ORANGE,
             fontsize=10, fontweight="bold", ha="center", va="center")
pad = half * 1.06
axA.set_xlim(-pad, pad); axA.set_ylim(-pad, pad); axA.set_aspect("equal")
axA.set_xlabel("nm", fontsize=10, color=INK2)
axA.set_title("A · ported vortex disk as patterned, to scale",
              fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
axA.text(0.015, 0.022,
         f"disk r = {pc.radius*1e9:.0f} nm, t = {pc.thickness*1e9:.0f} nm\n"
         f"{pc.n_ports} guides x {pc.guide_width*1e9:.0f} nm wide, "
         f"{pc.guide_length*1e9:.0f} nm long\n"
         f"each subtends {arc_deg:.0f}°  ·  "
         f"{pc.n_ports*arc_deg:.0f}° of 360° used\n"
         f"far {int(pc.absorb_frac*100)}% of each guide is an absorbing taper",
         transform=axA.transAxes, fontsize=9.5, color=INK2, va="bottom",
         bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec=GRID, alpha=0.94))
axA.tick_params(colors=MUTED, labelsize=9)
for s in axA.spines.values():
    s.set_color(GRID)

# ---- A2: what the colours mean + the operating point ----------------------
axK = fig.add_subplot(gs[0, 2], facecolor=SURFACE)
axK.axis("off")
axK.set_title("verified operating point", fontsize=13, fontweight="bold",
              color=INK, loc="left", pad=10)
lines = [
    ("drive", "11.7 & 13.7 GHz", "on real modes n = ±5, ±7"),
    ("guide cutoff", "11 GHz", "Q = 8.0, rising to 22.8 by 18 GHz"),
    ("AB/BA, full", "23.4×", "high vs low drive; >3 is nonlinear"),
    ("AB/BA, in passband", "15.0×", "≥11 GHz, carrying 73% of power"),
    ("port decomposition", "3 / 3", "|n| concentration 0.70 / 0.83 / 0.86"),
    ("float32 vs float64", "0.0000", "max weight deviation"),
]
y = 0.90
for label, value, note in lines:
    axK.text(0.0, y, label, fontsize=10, color=MUTED, transform=axK.transAxes)
    axK.text(0.0, y - 0.052, value, fontsize=15, color=INK, fontweight="bold",
             transform=axK.transAxes)
    axK.text(0.0, y - 0.098, note, fontsize=9, color=INK2, transform=axK.transAxes)
    y -= 0.163

# ---- B: guide Q against frequency -----------------------------------------
axB = fig.add_subplot(gs[1, 0], facecolor=SURFACE)
rows = load_journal("runs/guide_thick20/journal.jsonl")
rows += load_journal("runs/guide_knee20/journal.jsonl")
for w, col in ((40.0, YELLOW), (80.0, BLUE), (100.0, AQUA), (120.0, ORANGE)):
    pts = sorted([(r["ghz"], qfac(r)) for r in rows
                  if r["w"] == w and qfac(r) is not None])
    if not pts:
        continue
    fx, fy = zip(*pts)
    lw = 2.6 if w == 80 else 2.0
    axB.plot(fx, fy, "-o", color=col, lw=lw, ms=6, zorder=3 if w == 80 else 2,
             mec=SURFACE, mew=1.5)
    dy = {40.0: -13, 80.0: -1, 100.0: 0, 120.0: 11}[w]
    xoff = 7 if w != 100 else -46
    axB.annotate(f"{w:.0f} nm", (fx[-1], fy[-1]), textcoords="offset points",
                 xytext=(xoff, dy), fontsize=10, color=col, fontweight="bold")
axB.axhline(5, color=MUTED, lw=1.4, ls=(0, (4, 3)))
axB.text(10.2, 5.6, "Q = 5: propagating", fontsize=9, color=MUTED)
axB.axvspan(11, 14, color=BLUE, alpha=0.07, zorder=0)
axB.text(12.5, 27.5, "operating\nband", fontsize=9, color=BLUE, ha="center")
axB.set_xlabel("frequency (GHz)", fontsize=10, color=INK2)
axB.set_ylabel("Q = decay / wavelength", fontsize=10, color=INK2)
axB.set_title("B · do the guides propagate?  (t = 20 nm)", fontsize=12,
              fontweight="bold", color=INK, loc="left", pad=8)
axB.set_xlim(9.2, 28.6)
axB.set_ylim(0, 32)
axB.grid(color=GRID, lw=0.8); axB.set_axisbelow(True)
axB.tick_params(colors=MUTED, labelsize=9)
for s in axB.spines.values():
    s.set_color(GRID)

# ---- C: the disk's azimuthal mode ladder ----------------------------------
axC = fig.add_subplot(gs[1, 1], facecolor=SURFACE)
modes = json.loads(Path("runs/disk_modes/modes.json").read_text())["modes"]
pos = sorted([(m["n"], m["peak_ghz"]) for m in modes if m["n"] >= 1])
neg = sorted([(-m["n"], m["peak_ghz"]) for m in modes if m["n"] < 0])
n0 = [m["peak_ghz"] for m in modes if m["n"] == 0][0]
axC.axhspan(11, 26, color=BLUE, alpha=0.08, zorder=0)
axC.text(7.75, 14.95, "80 nm guides pass ≥11 GHz", fontsize=9, color=BLUE,
         ha="right", fontweight="bold")
for pts, col, lab in ((pos, BLUE, "+n"), (neg, ORANGE, "−n  (degenerate\n     with +n for |n|≥2)")):
    nx_, ny_ = zip(*pts)
    axC.plot(nx_, ny_, "-o", color=col, lw=2.2, ms=7, mec=SURFACE, mew=1.5)
    yo = -2 if col == BLUE else -30
    axC.annotate(lab, (nx_[-1], ny_[-1]), textcoords="offset points",
                 xytext=(-92 if col != BLUE else 8, yo), fontsize=10,
                 color=col, fontweight="bold")
axC.plot([0], [n0], "s", color=MUTED, ms=8, mec=SURFACE, mew=1.5)
axC.annotate("n = 0\n(radial, not\na ladder rung)", (0, n0),
             textcoords="offset points", xytext=(10, -30), fontsize=8.5,
             color=MUTED)
axC.annotate("", xy=(1, 9.4), xytext=(1, 11.3),
             arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.4))
axC.text(1.3, 10.35, "n = ±1 split\n1.9 GHz", fontsize=9, color=INK2,
         va="center", fontweight="bold")
axC.set_xlabel("azimuthal order |n|", fontsize=10, color=INK2)
axC.set_ylabel("mode frequency (GHz)", fontsize=10, color=INK2)
axC.set_title("C · where the disk's modes actually are", fontsize=12,
              fontweight="bold", color=INK, loc="left", pad=8)
axC.set_xlim(-0.4, 7.9); axC.set_ylim(8.6, 15.4)
axC.grid(color=GRID, lw=0.8); axC.set_axisbelow(True)
axC.tick_params(colors=MUTED, labelsize=9)
for s in axC.spines.values():
    s.set_color(GRID)

# ---- D: discrimination by passband, the panel that reversed the design -----
axD = fig.add_subplot(gs[1, 2], facecolor=SURFACE)
cut = [8, 9, 10, 11, 12, 13, 14]
old = band_ratio("runs/modal_disk", cut)
new = band_ratio("runs/modal_inband", cut)
axD.plot(cut, new, "-o", color=BLUE, lw=2.6, ms=7, mec=SURFACE, mew=1.5, zorder=3)
axD.plot(cut, old, "-o", color=ORANGE, lw=2.2, ms=7, mec=SURFACE, mew=1.5)
axD.annotate("drive 11.7 / 13.7 GHz", (cut[1], new[1]), textcoords="offset points",
             xytext=(6, 8), fontsize=10, color=BLUE, fontweight="bold")
axD.annotate("drive 6 / 9 GHz", (cut[4], old[4]), textcoords="offset points",
             xytext=(-4, 22), fontsize=10, color=ORANGE, fontweight="bold",
             ha="center")
axD.axhline(3, color=MUTED, lw=1.4, ls=(0, (4, 3)))
axD.text(8.05, 3.6, "nonlinear threshold", fontsize=9, color=MUTED)
axD.axvline(11, color=INK2, lw=1.2, ls=(0, (2, 3)))
axD.text(11.18, 5.6, "80 nm guide\ncutoff", fontsize=9, color=INK2)
axD.set_xlabel("guide cutoff — passband is everything above", fontsize=10, color=INK2)
axD.set_ylabel("AB/BA discrimination (high ÷ low)", fontsize=10, color=INK2)
axD.set_title("D · does the discrimination survive the guides?", fontsize=12,
              fontweight="bold", color=INK, loc="left", pad=8)
axD.set_ylim(0, 24)
axD.grid(color=GRID, lw=0.8); axD.set_axisbelow(True)
axD.tick_params(colors=MUTED, labelsize=9)
for s in axD.spines.values():
    s.set_color(GRID)

# ---- E: port mode decomposition -------------------------------------------
axE = fig.add_subplot(gs[2, :2], facecolor=SURFACE)


def abs_weights(path):
    r = json.loads(Path(path).read_text())["float64"]["orders"]
    out = {}
    for k, d in r.items():
        w = d["weights"]
        mags = []
        for j in range(len(w) // 2 + 1):
            v = w[j] + (w[len(w) - j] if 0 < j < len(w) - j else 0.0)
            mags.append(v)
        out[int(k)] = mags
    return out


w80 = abs_weights("runs/ports80/results.json")
w40 = abs_weights("runs/ports/results.json")
labels, bar40, bar80 = [], [], []
for drive in (0, 1, 2):
    labels.append(f"drive n={drive}")
    bar40.append(w40[drive][drive])
    bar80.append(w80[drive][drive])
x = np.arange(3)
axE.bar(x - 0.19, bar40, 0.34, color=YELLOW, label="40 nm ports @ 8 GHz",
        edgecolor=SURFACE, lw=2)
axE.bar(x + 0.19, bar80, 0.34, color=BLUE, label="80 nm ports @ 12 GHz",
        edgecolor=SURFACE, lw=2)
for xi, v in zip(x - 0.19, bar40):
    axE.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, color=INK2)
for xi, v in zip(x + 0.19, bar80):
    axE.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, color=INK,
             fontweight="bold")
axE.set_xticks(x); axE.set_xticklabels(labels, fontsize=10, color=INK2)
axE.set_ylabel("share of port DFT at the driven |n|", fontsize=10, color=INK2)
axE.set_title("E · the ports perform the decomposition — and widening them "
              "did not cost selectivity", fontsize=12, fontweight="bold",
              color=INK, loc="left", pad=8)
axE.set_ylim(0, 1.12)
axE.legend(frameon=False, fontsize=10, loc="upper center", ncol=2,
           labelcolor=INK2, bbox_to_anchor=(0.5, 1.02))
axE.grid(color=GRID, lw=0.8, axis="y"); axE.set_axisbelow(True)
axE.tick_params(colors=MUTED, labelsize=9)
for s in axE.spines.values():
    s.set_color(GRID)

# ---- F: what is settled and what is not -----------------------------------
axF = fig.add_subplot(gs[2, 2], facecolor=SURFACE)
axF.axis("off")
axF.set_title("status", fontsize=12, fontweight="bold", color=INK,
              loc="left", pad=8)
items = [
    (AQUA, "guides propagate", "80×20 nm, cutoff 11 GHz"),
    (AQUA, "six ports fit", "46° each, 275° of 360°"),
    (AQUA, "modes resolved", "3/3, both widths"),
    (AQUA, "disk discriminates", "23.4×, in-band 15.0×"),
    (ORANGE, "inter-disk coupling", "retired — retake at 11–14 GHz"),
    (ORANGE, "NARMA-10 on array", "not yet attempted"),
]
y = 0.88
for col, head, note in items:
    axF.add_patch(plt.Circle((0.035, y + 0.012), 0.026, color=col,
                             transform=axF.transAxes, clip_on=False))
    axF.text(0.10, y, head, fontsize=10.5, color=INK, fontweight="bold",
             transform=axF.transAxes)
    axF.text(0.10, y - 0.058, note, fontsize=9, color=INK2,
             transform=axF.transAxes)
    y -= 0.152

fig.suptitle("Magnonic modal reservoir — the verified device",
             fontsize=19, fontweight="bold", color=INK, x=0.065, ha="left",
             y=0.962)
fig.text(0.065, 0.917,
         "Every panel is a measurement. Panel D is the one that reversed the "
         "design: the guides were never the problem — the drive sat below them.",
         fontsize=11.5, color=INK2, ha="left")

out = Path("runs/verified_device.png")
fig.savefig(out, dpi=155, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}")
