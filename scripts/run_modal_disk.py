#!/usr/bin/env python3
"""The pivot test: can one vortex disk tell "AB" from "BA"?

Körber et al. (Nat Commun 14, 3954, 2023) put two RF pulses at frequencies
f_A and f_B through a vortex disk in both orders. The *average input spectra*
are identical by construction -- same two tones, same durations, only the
order differs -- so a linear system must return identical average output
spectra. Any difference is memory plus nonlinearity, with no delay line
anywhere.

That is the property the Thiele model could not have: with two degrees of
freedom per disk and a reset that erases both, there was nothing to carry the
first pulse's effect into the second. Here the first pulse populates magnon
modes, three-magnon scattering above threshold redistributes that energy, and
the second pulse lands on a disk whose mode populations already encode which
pulse came first.

Scored honestly: the discriminability of AB from BA is compared against the
same measurement at LOW power, where scattering is off and the response should
be linear. A difference that survives at low power would mean the test is
measuring a transient artefact rather than nonlinear memory.

    python scripts/run_modal_disk.py
    python scripts/run_modal_disk.py --amp-mT 40 --pulse-ns 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import VortexConfig, VortexDisk


def pulse_sequence(n_steps, dt, order, f_a, f_b, pulse_steps, amp, ramp=0.15):
    """Two RF pulses in the given order, then silence for the readout window.

    Both orders contain exactly the same two pulses, so the average input
    spectrum is identical and any output difference is the reservoir's.
    """
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
def run(disk, signal, drive_axis=0, record_every=2):
    """Drive the relaxed disk with a uniform in-plane RF field; return m(t).

    Stepped explicitly rather than through ``LLGRollout.run``: that path is
    built for probe intensities from point sources, and what is needed here is
    a *uniform* field over the whole disk and the full magnetisation, since the
    readout is a spatial mode decomposition rather than a point measurement.
    ``rk4_step`` takes a callable giving the drive at each sub-step, so the
    interpolation inside the step stays correct.
    """
    n = disk.cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=torch.float64)
    unit[:, :, :, drive_axis] = disk.mask[:, :, :, 0]

    m, snaps = disk.m0.clone(), []
    for k in range(len(signal) - 1):
        s0, s1 = float(signal[k]), float(signal[k + 1])

        def h_drive(theta, s0=s0, s1=s1):
            return unit * (s0 + theta * (s1 - s0))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        if k % record_every == 0:
            snaps.append(((m - disk.m0) * disk.mask).clone())
    return torch.stack(snaps)


def mode_spectrum(traces, mask, n_modes=6):
    """Decompose the response into azimuthal modes, then take each one's FFT.

    Azimuthal index n is the natural basis for a vortex: three-magnon
    scattering conserves it in a way it does not conserve position, which is
    why the paper calls this computing in reciprocal space.
    """
    T, nx, ny, _, _ = traces.shape
    idx = torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    phi = torch.atan2(Y, X)
    inside = mask[:, :, 0, 0] > 0

    mz = traces[:, :, :, 0, 2]
    out = []
    for n in range(n_modes):
        basis = torch.exp(-1j * n * phi) * inside
        amp = (mz.to(torch.complex128) * basis).sum(dim=(1, 2)) / inside.sum()
        out.append(torch.fft.rfft(amp.real).abs())
    return torch.stack(out)          # (n_modes, freq)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f-a", type=float, default=6.0, help="GHz")
    p.add_argument("--f-b", type=float, default=9.0, help="GHz")
    p.add_argument("--pulse-ns", type=float, default=2.0)
    p.add_argument("--read-ns", type=float, default=4.0)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--low-amp-mT", type=float, default=1.0)
    p.add_argument("--relax-steps", type=int, default=600)
    p.add_argument("--outdir", default="runs/modal_disk")
    args = p.parse_args()

    mnn.set_precision("float64")
    mnn.set_device("cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = VortexConfig()
    pulse_steps = int(args.pulse_ns * 1e-9 / cfg.dt)
    n_steps = 2 * pulse_steps + int(args.read_ns * 1e-9 / cfg.dt)
    print(f"mesh {cfg.n_cells}x{cfg.n_cells}, dt {cfg.dt*1e12:g} ps, "
          f"{n_steps} steps = {n_steps*cfg.dt*1e9:.1f} ns", flush=True)

    disk = VortexDisk(cfg, timesteps=n_steps)
    t0 = time.time()
    disk.relax(steps=args.relax_steps)
    print(f"relaxed ({time.time()-t0:.0f}s), core mz {float(disk.m0[:,:,0,2].max()):+.3f}",
          flush=True)

    results = {}
    for label, amp_mT in (("high", args.amp_mT), ("low", args.low_amp_mT)):
        spectra = {}
        for order in ("AB", "BA"):
            sig = pulse_sequence(n_steps, cfg.dt, order, args.f_a * 1e9,
                                 args.f_b * 1e9, pulse_steps, amp_mT * 1e-3 / MU_0)
            t0 = time.time()
            traces = run(disk, sig)
            del sig
            spectra[order] = mode_spectrum(traces, disk.mask)
            print(f"  {label} {order}: {time.time()-t0:.0f}s", flush=True)

        a, b = spectra["AB"], spectra["BA"]
        # cosine distance per azimuthal mode: 0 means indistinguishable
        num = (a * b).sum(dim=1)
        den = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-30)
        cos = (num / den).tolist()
        rel = float((a - b).norm() / (a.norm() + b.norm()).clamp_min(1e-30))
        results[label] = {"amp_mT": amp_mT, "cos_per_mode": cos,
                          "relative_difference": rel}
        torch.save({"AB": a, "BA": b}, outdir / f"spectra_{label}.pt")
        print(f"{label} power ({amp_mT} mT): relative AB/BA difference {rel:.4f}",
              flush=True)
        print("  cos per azimuthal mode n=0..5: "
              + " ".join(f"{c:.4f}" for c in cos), flush=True)

    ratio = results["high"]["relative_difference"] / max(
        results["low"]["relative_difference"], 1e-12)
    results["nonlinearity_ratio"] = ratio
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nhigh/low difference ratio: {ratio:.1f}x")
    if ratio > 3:
        print("AB and BA separate far more at high power than low: the")
        print("discrimination is nonlinear, which is the claim under test.")
    else:
        print("No strong power dependence -- the difference is not coming from")
        print("nonlinear scattering, and this configuration does not reproduce")
        print("the effect. Raise the drive or lengthen the pulses before")
        print("reading anything into the spectra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
