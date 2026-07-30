#!/usr/bin/env python3
"""Does the discrimination reach the PORTS, or stay in the disk interior?

The question matters more than any other open one: the interior mode spectrum
separates AB from BA by 23x, but a real device cannot measure the interior. It
measures the ports. If the discrimination does not reach them, the modal
architecture does not work however good the interior physics is.

Three previous attempts at this all failed the same way, by comparing readouts
taken from DIFFERENT runs or with different parameters, so the readout was
confounded with the drive:

    total port power        80.8x -- but total power over a fixed window is not
                            order-invariant even for a LINEAR damped system,
                            since pulse 1 decays longer than pulse 2
    raw port spectra        1.0x -- swamped by a linear ordering background of
                            0.41, against 0.012 for the interior
    modal port readout      1.1x -- same background, same problem

That background is the tell. It came from inheriting 1.5 ns pulses from
check_leak_routing: shorter pulses are broader in frequency, the two tones
overlap more, and a linear system then separates AB from BA all by itself.

So this changes one thing at a time. ONE simulation, TWO readouts computed from
it:

    interior   azimuthal decomposition over the disk, what run_modal_disk scores
    ports      the cross-port DFT the six guides physically implement

Same drive, same timing as run_modal_disk (2 ns pulses, 4 ns read), same
high-against-low honesty check. The readout is then the only variable, and the
comparison is finally an answer rather than an artefact.

The first version of this script scored BOTH readouts at ~1.0x, including the
interior, where run_modal_disk gets 23.4x on the same geometry. A bare-disk
control cleared the device and a sixteen-way bisection found the cause: a Hann
window I had carried over from the ring-down measurement. Windowing an order
comparison weights the two pulse slots unequally and fabricates an AB/BA
difference in any system whatsoever. It is gone, and the ratio is measured on
the raw transform.

    python scripts/check_readout_comparison.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import (PortedVortexConfig, PortedVortexDisk,
                                VortexConfig, VortexDisk)


def pulse_sequence(n_steps, dt, order, f_a, f_b, pulse_steps, amp, ramp=0.15):
    """Identical to run_modal_disk's, so the two are comparable."""
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
def run(disk, signal, dtype, record_every=2):
    """One simulation; return interior m_z snapshots AND port signals."""
    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, :, 0] = disk.disk_only[:, :, :, 0]

    m, snaps, ports = disk.m0.clone(), [], []
    for k in range(len(signal) - 1):
        s0, s1 = float(signal[k]), float(signal[k + 1])

        def h_drive(theta, s0=s0, s1=s1):
            return unit * (s0 + theta * (s1 - s0))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        if disk.port_signals is not None:
            ports.append(disk.port_signals(m).double().clone())
        if k % record_every == 0:
            snaps.append(((m - disk.m0) * disk.disk_only)[:, :, 0, 2].double().clone())
    return torch.stack(snaps), (torch.stack(ports) if ports else None)


def interior_spectra(mz, disk_only, dt_rec, n_modes=4, fmax=40.0):
    """Azimuthal decomposition over the disk, folded onto |n|."""
    T, nx, ny = mz.shape
    idx = torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2
    X, Y = torch.meshgrid(idx, torch.arange(ny, dtype=torch.float64)
                          - (ny - 1) / 2, indexing="ij")
    phi = torch.atan2(Y, X)
    inside = disk_only[:, :, 0, 0] > 0
    n_in = int(inside.sum())
    # NO window. A Hann window over the whole record weights the two pulse
    # slots unequally -- w ~ 0.15 at 1 ns against ~0.6 at 3 ns over 8 ns -- so
    # AB has f_A attenuated and f_B emphasised while BA has the reverse. That
    # manufactures an AB/BA difference in ANY system, linear or not: it drove
    # the low-power baseline from 0.0119 to 0.3977 and collapsed the measured
    # ratio from 23.4x to 1.4x. Leakage control is right for a single-impulse
    # ring-down (check_disk_modes) and fatal for an order comparison.
    half = T // 2
    f = np.fft.fftfreq(T, d=dt_rec)[1:half] / 1e9
    keep = (f > 1.0) & (f <= fmax)

    out = []
    for n in range(n_modes):
        basis = torch.exp(-1j * n * phi) * inside
        amp = (mz.to(torch.complex128) * basis).sum(dim=(1, 2)) / n_in
        S = np.abs(np.fft.fft(amp.numpy()))
        pos, neg = S[1:half], S[T - 1:T - half:-1]
        v = pos + (neg if n > 0 else 0.0)
        out.append(v[keep])
    return np.array(out)


def port_spectra(sig, dt, n_modes=4, fmax=40.0):
    """The cross-port DFT the guides implement, then a time FFT, folded on |n|."""
    x = sig.numpy().astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    A = np.fft.fft(x, axis=1)                       # azimuthal modes from ports
    T, n_ports = A.shape
    S = np.abs(np.fft.fft(A, axis=0))       # unwindowed: see interior_spectra
    half = T // 2
    f = np.fft.fftfreq(T, d=dt)[1:half] / 1e9
    keep = (f > 1.0) & (f <= fmax)
    pos, neg = S[1:half], S[T - 1:T - half:-1]
    out = []
    for k in range(n_modes):
        v = pos[:, k] + (neg[:, k] if 0 < k < n_ports - k else 0.0)
        out.append(v[keep])
    return np.array(out)


def rel_diff(a, b):
    den = np.linalg.norm(a) + np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / max(den, 1e-30))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f-a", type=float, default=11.7)
    p.add_argument("--f-b", type=float, default=13.7)
    p.add_argument("--pulse-ns", type=float, default=2.0)
    p.add_argument("--read-ns", type=float, default=4.0)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--low-amp-mT", type=float, default=1.0)
    p.add_argument("--relax-steps", type=int, default=900)
    p.add_argument("--n-modes", type=int, default=4)
    p.add_argument("--geometry", default="ported",
                   choices=["ported", "bare", "ported-noabsorb"],
                   help="bare is the control: run_modal_disk's own geometry, so\n"
                        "a disagreement with its 23.4x is a methods difference\n"
                        "in THIS script rather than a fact about the guides")
    p.add_argument("--outdir", default="runs/readout_cmp")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.geometry == "bare":
        cfg = VortexConfig()
    elif args.geometry == "ported-noabsorb":
        cfg = PortedVortexConfig(absorb_frac=0.0)
    else:
        cfg = PortedVortexConfig()
    pulse_steps = int(args.pulse_ns * 1e-9 / cfg.dt)
    n_steps = 2 * pulse_steps + int(args.read_ns * 1e-9 / cfg.dt)
    if args.geometry == "bare":
        disk = VortexDisk(cfg, timesteps=n_steps, dtype=dtype)
        disk.disk_only = disk.mask               # no guides: the disk is all of it
        disk.port_signals = None
    else:
        disk = PortedVortexDisk(cfg, timesteps=n_steps, dtype=dtype)
    has_ports = disk.port_signals is not None
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    print(f"relaxed {time.time()-t0:.0f}s, core mz "
          f"{float(disk.m0[:, :, 0, 2].max()):+.3f}\n"
          f"{n_steps} steps = {n_steps*cfg.dt*1e9:.1f} ns", flush=True)

    rec = 2
    results = {}
    for label, amp_mT in (("high", args.amp_mT), ("low", args.low_amp_mT)):
        ins, prt = {}, {}
        for order in ("AB", "BA"):
            sig = pulse_sequence(n_steps, cfg.dt, order, args.f_a * 1e9,
                                 args.f_b * 1e9, pulse_steps,
                                 amp_mT * 1e-3 / MU_0)
            t0 = time.time()
            mz, ps = run(disk, sig, dtype, rec)
            ins[order] = interior_spectra(mz, disk.disk_only, cfg.dt * rec,
                                          args.n_modes)
            prt[order] = (port_spectra(ps, cfg.dt, args.n_modes)
                          if ps is not None else None)
            del mz, ps
            print(f"  {label} {order}: {time.time()-t0:.0f}s", flush=True)

        results[label] = {
            "amp_mT": amp_mT,
            "interior": rel_diff(ins["AB"], ins["BA"]),
            "ports": (rel_diff(prt["AB"], prt["BA"]) if has_ports else float("nan")),
            "interior_per_n": [rel_diff(ins["AB"][k], ins["BA"][k])
                               for k in range(args.n_modes)],
            "ports_per_n": ([rel_diff(prt["AB"][k], prt["BA"][k])
                             for k in range(args.n_modes)] if has_ports else []),
        }
        print(f"{label} ({amp_mT} mT): interior {results[label]['interior']:.5f}   "
              f"ports {results[label]['ports']:.5f}", flush=True)

    ri = results["high"]["interior"] / max(results["low"]["interior"], 1e-12)
    rp = results["high"]["ports"] / max(results["low"]["ports"], 1e-12)
    results["interior_ratio"], results["ports_ratio"] = ri, rp
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'readout':<12} {'high':>9} {'low':>9} {'ratio':>8}")
    print(f"{'interior':<12} {results['high']['interior']:>9.5f} "
          f"{results['low']['interior']:>9.5f} {ri:>8.1f}")
    print(f"{'ports':<12} {results['high']['ports']:>9.5f} "
          f"{results['low']['ports']:>9.5f} {rp:>8.1f}")
    print("\nper |n| at high drive")
    print("  interior: " + " ".join(f"{v:.3f}" for v in results["high"]["interior_per_n"]))
    print("  ports:    " + " ".join(f"{v:.3f}" for v in results["high"]["ports_per_n"]))
    print()
    if rp >= 3.0:
        print("The discrimination REACHES the ports. The modal readout is")
        print("physically realisable, and the array work can proceed on it.")
    elif ri >= 3.0:
        print("The interior discriminates and the ports do not: the information")
        print("stays inside the disk. A device that can only measure the ports")
        print("cannot use it, so the readout -- not the drive -- is what has to")
        print("change before anything downstream is worth building.")
    else:
        print("Neither readout shows nonlinear separation at this operating")
        print("point. Check the drive amplitude before drawing any conclusion")
        print("about the geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
