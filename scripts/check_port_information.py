#!/usr/bin/env python3
"""Do the PORTS carry the discrimination, or only the disk's interior?

scripts/check_leak_routing.py returned a clean negative: the port power pattern
for AB and BA agrees to cos 1.00000, equally at high and low drive. Read
carelessly that kills the ported readout. Read carefully it kills one
observable, because that script collapses each port's time series to a single
number -- total power -- and the AB/BA discrimination is not in the total. It
is in the frequency structure, which is exactly what summing squares throws
away. The same mistake, in a different costume, made the port DFT read 1/3
before it was switched from an RMS to a lock-in.

So this measures the port signals the way the readout would actually use them:
per-port SPECTRA, keeping the frequency axis. Both are computed side by side
from the identical runs --

    power-only   the six numbers check_leak_routing compared
    spectral     the same six ports, frequency resolved

-- so the comparison isolates the readout choice rather than the geometry. And
both are scored high-drive against low-drive, since a difference that survives
at 1 mT is a linear transient rather than the nonlinear memory under test.

If the spectral score is high while the power score is ~1, the ports are a
working readout and only the winner-take-all routing idea is dead.

    python scripts/check_port_information.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk


@torch.no_grad()
def drive_sequence(disk, order, f_a, f_b, pulse_steps, read_steps, amp, dtype,
                   ramp=0.15):
    """Two pulses in the given order, then silence; return (T, n_ports)."""
    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, :, 0] = disk.disk_only[:, :, :, 0]

    freqs = (f_a, f_b) if order == "AB" else (f_b, f_a)
    total = 2 * pulse_steps + read_steps
    sig = torch.zeros(total, dtype=torch.float64)
    edge = max(int(pulse_steps * ramp), 1)
    env = torch.ones(pulse_steps, dtype=torch.float64)
    taper = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, edge,
                                                dtype=torch.float64)))
    env[:edge] = taper
    env[-edge:] = taper.flip(0)
    for i, f in enumerate(freqs):
        s = i * pulse_steps
        t = torch.arange(pulse_steps, dtype=torch.float64) * cfg.dt
        sig[s:s + pulse_steps] = amp * env * torch.sin(2 * np.pi * f * t)

    m, out = disk.m0.clone(), []
    for k in range(total - 1):
        s0, s1 = float(sig[k]), float(sig[k + 1])

        def h_drive(theta, s0=s0, s1=s1):
            return unit * (s0 + theta * (s1 - s0))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        out.append(disk.port_signals(m).double().clone())
    return torch.stack(out)


def rel_diff(a, b):
    den = np.linalg.norm(a) + np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / max(den, 1e-30))


def port_spectra(sig, dt, fmax_ghz=40.0):
    """(T, n_ports) -> (n_ports, n_freq) magnitude, windowed, DC removed."""
    T = sig.shape[0]
    win = np.hanning(T)[:, None]
    x = sig.numpy()
    x = x - x.mean(axis=0, keepdims=True)
    S = np.abs(np.fft.rfft(x * win, axis=0))
    f = np.fft.rfftfreq(T, d=dt) / 1e9
    keep = (f > 1.0) & (f <= fmax_ghz)
    return S[keep].T, f[keep]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f-a", type=float, default=11.7)
    p.add_argument("--f-b", type=float, default=13.7)
    p.add_argument("--pulse-ns", type=float, default=1.5)
    p.add_argument("--read-ns", type=float, default=3.0)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--low-amp-mT", type=float, default=1.0)
    p.add_argument("--relax-steps", type=int, default=900)
    p.add_argument("--precision", default="float32", choices=["float32", "float64"])
    p.add_argument("--outdir", default="runs/port_information")
    args = p.parse_args()

    mnn.set_precision(args.precision); mnn.set_device("cpu")
    dtype = torch.float32 if args.precision == "float32" else torch.float64
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = PortedVortexConfig()
    pulse_steps = int(args.pulse_ns * 1e-9 / cfg.dt)
    read_steps = int(args.read_ns * 1e-9 / cfg.dt)
    disk = PortedVortexDisk(cfg, timesteps=2 * pulse_steps + read_steps,
                            dtype=dtype)
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    print(f"relaxed {time.time()-t0:.0f}s, core mz "
          f"{float(disk.m0[:, :, 0, 2].max()):+.3f}", flush=True)

    results = {}
    for label, amp_mT in (("high", args.amp_mT), ("low", args.low_amp_mT)):
        sigs = {}
        for order in ("AB", "BA"):
            t0 = time.time()
            sigs[order] = drive_sequence(disk, order, args.f_a * 1e9,
                                         args.f_b * 1e9, pulse_steps,
                                         read_steps, amp_mT * 1e-3 / MU_0, dtype)
            print(f"  {label} {order}: {time.time()-t0:.0f}s", flush=True)

        # the readout check_leak_routing used: total power per port
        pw = {o: (sigs[o].numpy() ** 2).sum(axis=0) for o in ("AB", "BA")}
        pw = {o: v / max(v.sum(), 1e-30) for o, v in pw.items()}
        power_score = rel_diff(pw["AB"], pw["BA"])

        # the readout a modal device would use: per-port spectra
        sa, freqs = port_spectra(sigs["AB"], cfg.dt)
        sb, _ = port_spectra(sigs["BA"], cfg.dt)
        spec_score = rel_diff(sa, sb)
        per_port = [rel_diff(sa[k], sb[k]) for k in range(cfg.n_ports)]

        results[label] = {"amp_mT": amp_mT, "power_score": power_score,
                          "spectral_score": spec_score,
                          "per_port_spectral": per_port}
        torch.save({"AB": sigs["AB"], "BA": sigs["BA"]},
                   outdir / f"ports_{label}.pt")
        print(f"{label} ({amp_mT} mT): power {power_score:.5f}   "
              f"spectral {spec_score:.5f}", flush=True)

    pr = results["high"]["power_score"] / max(results["low"]["power_score"], 1e-12)
    sr = (results["high"]["spectral_score"]
          / max(results["low"]["spectral_score"], 1e-12))
    results["power_ratio"] = pr
    results["spectral_ratio"] = sr
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'readout':<28} {'high':>9} {'low':>9} {'ratio':>8}")
    print(f"{'total power per port':<28} {results['high']['power_score']:>9.5f} "
          f"{results['low']['power_score']:>9.5f} {pr:>8.1f}")
    print(f"{'per-port spectra':<28} {results['high']['spectral_score']:>9.5f} "
          f"{results['low']['spectral_score']:>9.5f} {sr:>8.1f}")
    print("\nper-port spectral AB/BA difference at high drive: "
          + " ".join(f"P{k}={v:.3f}"
                     for k, v in enumerate(results["high"]["per_port_spectral"])))
    print()
    if sr >= 3.0 > pr:
        print("The ports DO carry the discrimination -- in their spectra, not in")
        print("how much power each one takes. Winner-take-all routing is dead;")
        print("the modal readout is not, and it reads the ports the way the")
        print("mode decomposition already does.")
    elif sr >= 3.0:
        print("Both readouts carry it; power is usable too.")
    else:
        print("Neither readout separates AB from BA nonlinearly. The guides")
        print("transport energy the readout cannot use, and the discrimination")
        print("stays trapped in the disk interior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
