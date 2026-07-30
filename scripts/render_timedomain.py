#!/usr/bin/env python3
"""Time-domain animation of the ported disk at the verified operating point.

Not an illustration -- every frame is the simulated state. The disk is relaxed
into its vortex ground state, driven with the AB pulse sequence (11.7 GHz then
13.7 GHz, both on real azimuthal modes), and then left to ring down while the
ports are watched.

Two panels, because the interesting claim is about the relationship between
them:

  left    delta m_z over the patterned geometry -- the spin waves themselves,
          leaving the disk through the guides
  right   power at each of the six ports, running

The routing criterion the array architecture rests on is that the port ranking
is INPUT-DEPENDENT and MIGRATES. A static splitter would show the same bar
heights throughout; recruitment shows the ranking changing as the second pulse
lands on a disk whose modes the first pulse already populated. The animation is
where that is visible rather than inferred from a summary statistic.

The colour scale is set from a percentile of the actual response, not from
+-1: the deviation from the ground state is small, and a full-range scale would
render the whole thing as an unchanging flat field.

    python scripts/render_timedomain.py
    python scripts/render_timedomain.py --order BA --fps 22
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
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
GRID = "#e4e2dd"


def pulse_sequence(n_steps, dt, order, f_a, f_b, pulse_steps, amp, ramp=0.15):
    """Two RF pulses in the given order, then silence for the readout window."""
    sig = torch.zeros(n_steps, dtype=torch.float64)
    freqs = (f_a, f_b) if order == "AB" else (f_b, f_a)
    edge = max(int(pulse_steps * ramp), 1)
    env = torch.ones(pulse_steps, dtype=torch.float64)
    taper = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, edge, dtype=torch.float64)))
    env[:edge] = taper
    env[-edge:] = taper.flip(0)
    for i, f in enumerate(freqs):
        s = i * pulse_steps
        t = torch.arange(pulse_steps, dtype=torch.float64) * dt
        sig[s:s + pulse_steps] = amp * env * torch.sin(2 * np.pi * f * t)
    return sig


@torch.no_grad()
def simulate(disk, signal, record_every, dtype):
    """Step the disk; return (frames of delta m_z, port signals, frame steps)."""
    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, :, 0] = disk.disk_only[:, :, :, 0]   # drive the disk, not the guides

    m = disk.m0.clone()
    frames, ports, ks, t0 = [], [], [], time.time()
    for k in range(len(signal) - 1):
        s0, s1 = float(signal[k]), float(signal[k + 1])

        def h_drive(theta, s0=s0, s1=s1):
            return unit * (s0 + theta * (s1 - s0))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        ports.append(disk.port_signals(m).double().clone())
        if k % record_every == 0:
            frames.append(((m - disk.m0) * disk.mask)[:, :, 0, 2].clone())
            ks.append(k)
            if len(frames) % 25 == 0:
                print(f"  frame {len(frames)} (step {k}, {time.time()-t0:.0f}s)",
                      flush=True)
    return torch.stack(frames), torch.stack(ports), ks


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f-a", type=float, default=11.7, help="GHz")
    p.add_argument("--f-b", type=float, default=13.7, help="GHz")
    p.add_argument("--order", default="AB", choices=["AB", "BA"])
    p.add_argument("--pulse-ns", type=float, default=1.5)
    p.add_argument("--read-ns", type=float, default=3.0)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--relax-steps", type=int, default=900)
    p.add_argument("--record-every", type=int, default=30)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default="runs/timedomain.gif")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    cfg = PortedVortexConfig()
    pulse_steps = int(args.pulse_ns * 1e-9 / cfg.dt)
    n_steps = 2 * pulse_steps + int(args.read_ns * 1e-9 / cfg.dt)
    print(f"mesh {cfg.n_cells}x{cfg.n_cells}, {n_steps} steps = "
          f"{n_steps*cfg.dt*1e9:.1f} ns", flush=True)

    disk = PortedVortexDisk(cfg, timesteps=n_steps, dtype=dtype)
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    print(f"relaxed ({time.time()-t0:.0f}s), core mz "
          f"{float(disk.m0[:, :, 0, 2].max()):+.3f}", flush=True)

    sig = pulse_sequence(n_steps, cfg.dt, args.order, args.f_a * 1e9,
                         args.f_b * 1e9, pulse_steps,
                         args.amp_mT * 1e-3 / MU_0)
    frames, ports, ks = simulate(disk, sig, args.record_every, dtype)
    print(f"{len(frames)} frames", flush=True)

    # Port power in a running window, so the bars show the envelope rather
    # than the carrier -- at 12 GHz the raw signal oscillates far faster than
    # the frame rate and would alias into noise.
    win = max(args.record_every * 4, 40)
    P = ports.numpy() ** 2
    csum = np.cumsum(np.vstack([np.zeros((1, P.shape[1])), P]), axis=0)
    env = np.array([(csum[min(k + 1, len(P))] - csum[max(k + 1 - win, 0)])
                    / max(min(k + 1, win), 1) for k in ks])
    env = np.sqrt(env)
    pmax = float(env.max()) or 1.0

    F = frames.numpy()
    mask = disk.mask[:, :, 0, 0].numpy().T > 0
    vmax = float(np.percentile(np.abs(F[F != 0]), 99.3)) or 1e-6
    half = cfg.n_cells * cfg.dx * 0.5e9
    ext = [-half, half, -half, half]

    fig = plt.figure(figsize=(11.6, 5.6), facecolor=SURFACE)
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.14, 0.86], wspace=0.26,
                           left=0.045, right=0.975, top=0.845, bottom=0.115)
    ax1 = fig.add_subplot(gsp[0, 0], facecolor=SURFACE)
    ax2 = fig.add_subplot(gsp[0, 1], facecolor=SURFACE)

    im = ax1.imshow(np.where(mask, F[0].T, np.nan), origin="lower", extent=ext,
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax1.add_patch(Circle((0, 0), cfg.radius * 1e9, fill=False, ec=INK, lw=1.4,
                         ls=(0, (5, 3))))
    for k, th in enumerate(cfg.port_angles()):
        rr = (cfg.radius + cfg.guide_length) * 1e9 - 26
        ax1.text(rr * math.cos(th), rr * math.sin(th), f"P{k}", color=INK2,
                 fontsize=9, fontweight="bold", ha="center", va="center")
    ax1.set_xlim(-half * 1.04, half * 1.04); ax1.set_ylim(-half * 1.04, half * 1.04)
    ax1.set_aspect("equal"); ax1.set_xlabel("nm", fontsize=9, color=INK2)
    ax1.tick_params(colors=MUTED, labelsize=8)
    for s in ax1.spines.values():
        s.set_color(GRID)
    cb = fig.colorbar(im, ax=ax1, fraction=0.043, pad=0.02)
    cb.set_label("δm_z  (deviation from the vortex ground state)", fontsize=8.5,
                 color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(GRID)

    cols = [BLUE, ORANGE, AQUA, YELLOW, "#4a3aa7", "#b0568a"]
    bars = ax2.bar(range(cfg.n_ports), env[0], color=cols, edgecolor=SURFACE, lw=2)
    ax2.set_ylim(0, pmax * 1.12)
    ax2.set_xticks(range(cfg.n_ports))
    ax2.set_xticklabels([f"P{k}" for k in range(cfg.n_ports)], fontsize=10,
                        color=INK2)
    ax2.set_ylabel("port amplitude (running RMS)", fontsize=9.5, color=INK2)
    ax2.set_title("which port leaks hardest — and does the ranking migrate?",
                  fontsize=10.5, color=INK, loc="left", pad=8)
    ax2.grid(color=GRID, lw=0.8, axis="y"); ax2.set_axisbelow(True)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for s in ax2.spines.values():
        s.set_color(GRID)

    seq = (args.f_a, args.f_b) if args.order == "AB" else (args.f_b, args.f_a)
    title = fig.suptitle("", fontsize=13, fontweight="bold", color=INK,
                         x=0.045, ha="left", y=0.955)
    sub = fig.text(0.045, 0.898, "", fontsize=10, color=INK2, ha="left")

    def update(i):
        im.set_data(np.where(mask, F[i].T, np.nan))
        for b, v in zip(bars, env[i]):
            b.set_height(v)
        t_ns = ks[i] * cfg.dt * 1e9
        if ks[i] < pulse_steps:
            phase, col = f"pulse 1 — {seq[0]:g} GHz", ORANGE
        elif ks[i] < 2 * pulse_steps:
            phase, col = f"pulse 2 — {seq[1]:g} GHz", ORANGE
        else:
            phase, col = "ring-down — drive off", INK2
        title.set_text(f"Ported vortex disk, order {args.order}   ·   "
                       f"t = {t_ns:5.2f} ns")
        sub.set_text(phase)
        sub.set_color(col)
        return [im, title, sub, *bars]

    anim = animation.FuncAnimation(fig, update, frames=len(F), blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=animation.PillowWriter(fps=args.fps), dpi=76,
              savefig_kwargs={"facecolor": SURFACE})
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
