#!/usr/bin/env python3
"""Where are the disk's azimuthal modes, actually?

Every frequency choice so far has rested on "5-15 GHz", which came from the
literature rather than from this disk. That was tolerable while the drive sat
at 6 and 9 GHz in the middle of the quoted range. It is not tolerable now: the
guides at the device's own 20 nm thickness are evanescent at 10 GHz (decay
225 nm) and propagate from 14 GHz up (decay 781 nm, Q = 10), so the operating
point has to move into 14-26 GHz -- and whether the disk has anything to say up
there is a measurement, not an assumption.

Method: relax, hit the disk with one short pulse, and watch it ring down. A
pulse short in time is broad in frequency; a pulse localised in space is broad
in azimuthal order. So a single impulse excites every (n, f) the disk supports,
and the ring-down spectrum decomposed by azimuthal index n gives the mode
structure directly.

The excitation is placed off-centre deliberately. A centred or axially
symmetric drive couples only to n = 0 -- the geometry forbids the rest -- and
would return a spectrum that looks empty for reasons that have nothing to do
with the disk's modes.

Reported per n: the peak frequency, and how much of that mode's spectral power
falls inside the guide band. The second number is the one that decides whether
a ported device can work at all, since a mode the guides cannot carry is a mode
the readout cannot see.

    python scripts/check_disk_modes.py
    python scripts/check_disk_modes.py --band 14 26 --ns 12
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import VortexConfig, VortexDisk


def impulse(n_steps, dt, t0_ps=40.0, sigma_ps=4.0):
    """Gaussian pulse; sigma = 4 ps is flat in amplitude out to ~40 GHz."""
    t = torch.arange(n_steps, dtype=torch.float64) * dt * 1e12
    return torch.exp(-0.5 * ((t - t0_ps) / sigma_ps) ** 2)


@torch.no_grad()
def ring_down(disk, signal, spot_frac=0.65, spot_cells=3, record_every=2):
    """Kick the disk once off-centre, then record the free ring-down."""
    n = disk.cfg.n_cells
    c = (n - 1) / 2
    off = spot_frac * disk.cfg.radius / disk.cfg.dx
    ix, iy = int(round(c + off)), int(round(c))

    h = torch.zeros(n, n, 1, 3, dtype=disk.m0.dtype)
    lo_x, hi_x = max(ix - spot_cells, 0), min(ix + spot_cells + 1, n)
    lo_y, hi_y = max(iy - spot_cells, 0), min(iy + spot_cells + 1, n)
    h[lo_x:hi_x, lo_y:hi_y, 0, 2] = disk.mask[lo_x:hi_x, lo_y:hi_y, 0, 0]
    if float(h.abs().sum()) == 0.0:
        raise RuntimeError("excitation spot fell outside the disk")

    m, snaps = disk.m0.clone(), []
    for k in range(len(signal) - 1):
        s0, s1 = float(signal[k]), float(signal[k + 1])

        def h_drive(theta, s0=s0, s1=s1):
            return h * (s0 + theta * (s1 - s0))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        if k % record_every == 0:
            snaps.append(((m - disk.m0) * disk.mask)[:, :, 0, 2].clone())
    return torch.stack(snaps)


def azimuthal_spectra(mz, mask, dt_rec, n_modes=8):
    """(T, nx, ny) m_z -> (n_modes, n_freq) power, plus the frequency axis."""
    T, nx, ny = mz.shape
    idx = torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2
    X, Y = torch.meshgrid(idx, torch.arange(ny, dtype=torch.float64)
                          - (ny - 1) / 2, indexing="ij")
    phi = torch.atan2(Y, X)
    inside = mask[:, :, 0, 0] > 0
    n_in = int(inside.sum())

    # Hann window: the ring-down is truncated, and a hard cut smears every
    # peak into its neighbours -- which for a mode-identification measurement
    # is the difference between resolving n and inventing it.
    win = torch.hann_window(T, periodic=False, dtype=torch.float64)

    out = []
    for n in range(n_modes):
        basis = torch.exp(-1j * n * phi) * inside
        amp = (mz.to(torch.complex128) * basis).sum(dim=(1, 2)) / n_in
        out.append(torch.fft.rfft(amp * win).abs() ** 2)
    freqs = torch.fft.rfftfreq(T, d=dt_rec)
    return torch.stack(out), freqs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=float, default=10.0, help="ring-down length")
    p.add_argument("--amp-mT", type=float, default=5.0,
                   help="small, so the ring-down is the linear mode structure")
    p.add_argument("--relax-steps", type=int, default=600)
    p.add_argument("--n-modes", type=int, default=8)
    p.add_argument("--band", type=float, nargs=2, default=[14.0, 26.0],
                   help="guide passband in GHz")
    p.add_argument("--fmax", type=float, default=40.0, help="GHz, for reporting")
    p.add_argument("--precision", default="float32", choices=["float32", "float64"])
    p.add_argument("--outdir", default="runs/disk_modes")
    args = p.parse_args()

    mnn.set_precision(args.precision)
    mnn.set_device("cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = VortexConfig()
    n_steps = int(args.ns * 1e-9 / cfg.dt)
    print(f"mesh {cfg.n_cells}x{cfg.n_cells}, {n_steps} steps = {args.ns:g} ns, "
          f"df = {1e-9 / (args.ns * 1e-9) / 1e0:.3f} GHz resolution", flush=True)

    disk = VortexDisk(cfg, timesteps=n_steps)
    t0 = time.time()
    disk.relax(steps=args.relax_steps)
    print(f"relaxed ({time.time()-t0:.0f}s), core mz "
          f"{float(disk.m0[:, :, 0, 2].max()):+.3f}", flush=True)

    sig = impulse(n_steps, cfg.dt) * (args.amp_mT * 1e-3 / MU_0)
    t0 = time.time()
    record_every = 2
    mz = ring_down(disk, sig)
    print(f"ring-down ({time.time()-t0:.0f}s)", flush=True)

    P, freqs = azimuthal_spectra(mz.double(), disk.mask.double(),
                                 cfg.dt * record_every, args.n_modes)
    f_ghz = (freqs / 1e9).numpy()
    lo, hi = args.band
    # Ignore the DC shoulder: a pulse deposits a static offset, and its skirt
    # would otherwise be read as a mode at 0 GHz in every channel.
    keep = (f_ghz > 1.0) & (f_ghz <= args.fmax)
    in_band = keep & (f_ghz >= lo) & (f_ghz <= hi)

    torch.save({"power": P, "freqs": freqs}, outdir / "spectra.pt")
    rows = []
    print(f"\n{'n':>3} {'peak GHz':>9} {'power in':>9} {'band':>7}   top peaks (GHz)")
    print(f"{'':>3} {'':>9} {f'{lo:g}-{hi:g}G':>9} {'frac':>7}")
    for n in range(args.n_modes):
        pk = P[n].numpy()
        tot = pk[keep].sum()
        frac = float(pk[in_band].sum() / tot) if tot > 0 else 0.0
        peak = float(f_ghz[keep][pk[keep].argmax()]) if tot > 0 else float("nan")
        # local maxima, strongest first, for the mode's fine structure
        w = np.where(keep)[0]
        loc = [i for i in w[1:-1] if pk[i] > pk[i - 1] and pk[i] > pk[i + 1]]
        loc.sort(key=lambda i: -pk[i])
        tops = " ".join(f"{f_ghz[i]:.1f}" for i in loc[:4])
        rows.append({"n": n, "peak_ghz": peak, "in_band_fraction": frac,
                     "top_peaks_ghz": [float(f_ghz[i]) for i in loc[:4]]})
        print(f"{n:>3} {peak:>9.2f} {'':>9} {frac:>7.3f}   {tops}")

    good = [r for r in rows if r["in_band_fraction"] >= 0.20]
    (outdir / "modes.json").write_text(json.dumps(
        {"band_ghz": [lo, hi], "modes": rows}, indent=2))

    print()
    if good:
        ns = ", ".join(f"n={r['n']}" for r in good)
        print(f"Modes with >=20% of their power inside the {lo:g}-{hi:g} GHz guide")
        print(f"band: {ns}.")
        print("Those are the modes a ported readout could actually see, so the")
        print("AB/BA re-measurement should drive two frequencies inside the band")
        print("that both land on populated modes.")
    else:
        print(f"No azimuthal mode puts 20% of its power in {lo:g}-{hi:g} GHz.")
        print("The disk and the guides do not overlap: a ported device would")
        print("transport nothing the disk computes with. Changing the operating")
        print("frequency cannot fix that -- the geometry has to change, either")
        print("thinner guides on a thicker disk (a multilayer mesh, since one")
        print("dz cannot do both) or a disk whose modes sit higher.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
