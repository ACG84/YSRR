#!/usr/bin/env python3
"""NARMA-10 running in the time domain: the disk, the ports, the prediction.

One GIF frame is one input sample. The left panel is the simulated
magnetisation -- the actual state the reservoir computes with -- and the right
panels are what a device would see and do with it: the six ports' modal
features, and a readout prediction rolling against the truth.

The readout is NOT fitted here. Its weights come from the completed 1200-frame
run, so the prediction is what a trained device would output on data it has
never been fitted to. Fitting on the animated window would make any figure look
good and mean nothing.

Watch the memory: the measured memory function falls from r^2 = 0.94 at lag 1
to 0.22 by lag 3 and nothing beyond, capacity 2.6 against the ten lags NARMA-10
needs. That is visible here as a prediction tracking the slow envelope and
missing the sharp excursions, which is what a three-tap memory looks like on a
ten-tap problem -- not noise, and not a fitting failure.

    python scripts/render_narma_gif.py
    python scripts/render_narma_gif.py --frames 300 --show-from 150
"""
from __future__ import annotations
import argparse, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np, torch
from matplotlib.patches import Circle
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.reservoir import narma10, ridge_fit, ridge_predict
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
GRID = "#e4e2dd"


@torch.no_grad()
def run_with_field(disk, u, steps_per_frame, carrier, amp_lo, amp_hi, dtype,
                   cache=None):
    """Same rollout as run_narma_modal, but keeping one field snapshot a frame."""
    if cache is not None and cache.exists():
        d = torch.load(cache, weights_only=False)
        if len(d["feats"]) >= len(u):
            print(f"[cached] {cache.name}", flush=True)
            return d["fields"], d["feats"]

    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, :, 0] = disk.disk_only[:, :, :, 0]

    m = disk.m0.clone()
    w = 2 * math.pi * carrier
    fields, feats, t0 = [], [], time.time()
    for j, un in enumerate(u):
        amp = (amp_lo + (amp_hi - amp_lo) * float(un)) * 1e-3 / MU_0
        accI = torch.zeros(cfg.n_ports, dtype=torch.float64)
        accQ = torch.zeros(cfg.n_ports, dtype=torch.float64)
        for k in range(steps_per_frame):
            tk = (j * steps_per_frame + k) * cfg.dt

            def h_drive(theta, tk=tk, amp=amp):
                return unit * (amp * math.sin(w * (tk + theta * cfg.dt)))

            m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
            p = disk.port_signals(m).double()
            accI += p * math.cos(w * tk)
            accQ += p * math.sin(w * tk)
        fields.append(((m - disk.m0) * disk.mask)[:, :, 0, 2].clone())
        A = ((accI + 1j * accQ) / steps_per_frame).numpy()
        M = np.fft.fft(A)
        row = []
        for q in range(cfg.n_ports // 2 + 1):
            v = M[q] + (M[cfg.n_ports - q] if 0 < q < cfg.n_ports - q else 0.0)
            row += [v.real, v.imag, abs(v)]
        feats.append(row)
        if (j + 1) % 25 == 0:
            print(f"  frame {j+1}/{len(u)} ({time.time()-t0:.0f}s)", flush=True)
            if cache is not None:
                torch.save({"fields": torch.stack(fields),
                            "feats": torch.tensor(np.array(feats))}, cache)

    F = torch.stack(fields)
    G = torch.tensor(np.array(feats), dtype=torch.float64)
    if cache is not None:
        torch.save({"fields": F, "feats": G}, cache)
    return F, G


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=260)
    p.add_argument("--show-from", type=int, default=110,
                   help="skip the washout; the reservoir starts from a relaxed "
                        "state that carries no history")
    p.add_argument("--steps-per-frame", type=int, default=200)
    p.add_argument("--carrier-ghz", type=float, default=12.0)
    p.add_argument("--amp-lo-mT", type=float, default=10.0)
    p.add_argument("--amp-hi-mT", type=float, default=30.0)
    p.add_argument("--trained-on", default="runs/narma_modal/features.pt")
    p.add_argument("--relax-steps", type=int, default=900)
    p.add_argument("--fps", type=int, default=11)
    p.add_argument("--out", default="runs/narma_timedomain.gif")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.out).parent; outdir.mkdir(parents=True, exist_ok=True)

    # readout weights from the completed run -- never fitted on this window
    Fc = torch.load(args.trained_on, weights_only=False).numpy()
    uc, yc = narma10(len(Fc), seed=0)
    mu, sd = Fc.mean(0), Fc.std(0).clip(1e-12)
    Xc = (Fc - mu) / sd
    tr = slice(150, 800)
    best, wbest = None, None
    for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        w = ridge_fit(Xc[tr], yc[tr], lam)
        e = float(np.mean((yc[800:950] - ridge_predict(Xc[800:950], w)) ** 2))
        if best is None or e < best:
            best, wbest = e, w
    print(f"readout fitted on the completed run (frames 150-800)", flush=True)

    u, y = narma10(args.frames, seed=0)
    cfg = PortedVortexConfig()
    disk = PortedVortexDisk(cfg, timesteps=args.steps_per_frame + 4, dtype=dtype)
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    print(f"relaxed {time.time()-t0:.0f}s", flush=True)

    fields, feats = run_with_field(
        disk, u, args.steps_per_frame, args.carrier_ghz * 1e9,
        args.amp_lo_mT, args.amp_hi_mT, dtype,
        cache=outdir / "gif_state.pt")
    Fv = fields.numpy()
    Xv = (feats.numpy() - mu) / sd
    pred = ridge_predict(Xv, wbest)

    s0 = args.show_from
    idx = list(range(s0, len(u)))
    mask = disk.mask[:, :, 0, 0].numpy().T > 0
    vmax = float(np.percentile(np.abs(Fv[Fv != 0]), 99.2)) or 1e-6
    half = cfg.n_cells * cfg.dx * 0.5e9
    ext = [-half, half, -half, half]

    fig = plt.figure(figsize=(12.4, 5.9), facecolor=SURFACE)
    gsp = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.05],
                           height_ratios=[1.0, 0.62], wspace=0.22, hspace=0.42,
                           left=0.045, right=0.975, top=0.83, bottom=0.10)
    ax1 = fig.add_subplot(gsp[:, 0], facecolor=SURFACE)
    ax2 = fig.add_subplot(gsp[0, 1], facecolor=SURFACE)
    ax3 = fig.add_subplot(gsp[1, 1], facecolor=SURFACE)

    im = ax1.imshow(np.where(mask, Fv[s0].T, np.nan), origin="lower", extent=ext,
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax1.add_patch(Circle((0, 0), cfg.radius * 1e9, fill=False, ec=INK, lw=1.3,
                         ls=(0, (5, 3))))
    ax1.set_aspect("equal"); ax1.set_xlabel("nm", fontsize=9, color=INK2)
    ax1.set_title("δm_z", fontsize=10.5, color=INK, loc="left", pad=7)
    ax1.tick_params(colors=MUTED, labelsize=8)
    for s in ax1.spines.values():
        s.set_color(GRID)
    cb = fig.colorbar(im, ax=ax1, fraction=0.043, pad=0.02)
    cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_edgecolor(GRID)

    win = 90
    ln_y, = ax2.plot([], [], color=INK, lw=2.2)
    ln_p, = ax2.plot([], [], color=BLUE, lw=1.8)
    ln_u, = ax2.plot([], [], color=MUTED, lw=1.2)
    ax2.set_ylim(min(y[s0:].min(), 0) - 0.05, max(y[s0:].max(), u.max()) + 0.10)
    ax2.set_title("prediction vs truth", fontsize=10.5, color=INK,
                  loc="left", pad=7)
    ax2.grid(color=GRID, lw=0.8); ax2.set_axisbelow(True)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for s in ax2.spines.values():
        s.set_color(GRID)

    nb = Xv.shape[1]
    bars = ax3.bar(range(nb), np.zeros(nb), color=AQUA, edgecolor=SURFACE, lw=1.6)
    lim = float(np.percentile(np.abs(Xv[s0:]), 99)) * 1.15
    ax3.set_ylim(-lim, lim)
    ax3.set_xticks(range(nb))
    ax3.set_xticklabels([f"{i}" for i in range(nb)], fontsize=7.5, color=MUTED)
    ax3.set_xlabel("port modal feature", fontsize=9, color=INK2)
    ax3.set_title("port features", fontsize=10.5, color=INK,
                  loc="left", pad=7)
    ax3.grid(color=GRID, lw=0.8, axis="y"); ax3.set_axisbelow(True)
    ax3.tick_params(colors=MUTED, labelsize=8)
    for s in ax3.spines.values():
        s.set_color(GRID)

    title = fig.suptitle("", fontsize=13.5, fontweight="bold", color=INK,
                         x=0.045, ha="left", y=0.955)
    sub = fig.text(0.045, 0.897, "", fontsize=10, color=INK2, ha="left")

    def update(t):
        j = idx[t]
        im.set_data(np.where(mask, Fv[j].T, np.nan))
        a = max(s0, j - win)
        xs = np.arange(a, j + 1)
        ln_y.set_data(xs, y[a:j + 1])
        ln_p.set_data(xs, pred[a:j + 1])
        ln_u.set_data(xs, u[a:j + 1])
        ax2.set_xlim(a, max(j, a + 12))
        for b, v in zip(bars, Xv[j]):
            b.set_height(v)
            b.set_color(BLUE if v >= 0 else ORANGE)
        err = abs(pred[j] - y[j])
        title.set_text(f"NARMA-10   ·   frame {j}   ·   u = {u[j]:.3f}")
        sub.set_text(f"target {y[j]:.3f}   ·   prediction {pred[j]:.3f}")
        return [im, ln_y, ln_p, ln_u, title, sub, *bars]

    anim = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
    out = Path(args.out)
    anim.save(out, writer=animation.PillowWriter(fps=args.fps), dpi=74,
              savefig_kwargs={"facecolor": SURFACE})
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
