#!/usr/bin/env python3
"""Where does the disk stop responding linearly to drive amplitude?

The capacity decomposition found the chain spending 98% of its measured capacity
on degree-1 targets, with NARMA-10's `u[n-9]*u[n]` product at exactly 0.000. The
readout was tested and cleared: a square-law detector does not supply the
products either, because squaring collapses the state from rank 21 to rank 5 by
discarding the phase that carries the lag information.

That leaves the drive. The runs so far modulate a 12 GHz carrier between 10 and
30 mT, and if the magnetisation's envelope responds linearly over that span then
the whole device is a linear filter and no readout or geometry can rescue it.

Screening this with full NARMA runs would cost hours per amplitude. The response
curve costs seconds per amplitude and answers the prior question -- IS there a
nonlinearity, and where does it start -- so the expensive runs can be aimed.

Three independent signatures, because they fail differently:

    |A|/a       amplitude compression. Flat = linear response. Bending = the
                cone angle is large enough for the LLG's own nonlinearity.
    arg(A)      the NONLINEAR FREQUENCY SHIFT, and the one that matters most
                here. A magnetic oscillator's frequency depends on amplitude, so
                drive amplitude leaks into response PHASE -- and phase is where
                this device's memory lives. Amplitude-into-phase is exactly the
                mixing that manufactures cross-lag products.
    |A_2f|/|A|  genuine second-harmonic generation, which must grow with drive
                if the dynamics are nonlinear. Note the 24 GHz READOUT tone is a
                0.99-correlated copy of the carrier (lock-in leakage), so this
                is measured against the leakage baseline at low amplitude rather
                than against zero.

And the constraint that bounds all of it: the vortex has to survive. Core
annihilation was already measured at 60 mT with lambda = +5.5, far past useful,
so this reports a stability check per amplitude and the useful window is where
nonlinearity has appeared and stability has not yet gone.

    python scripts/check_drive_nonlinearity.py
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
def drive(disk, amp, freq, dtype, n_settle, n_meas):
    """Drive at constant amplitude; lock in at f and 2f over the last n_meas.

    Returns (M_f, M_2f, m_final) with M the complex CROSS-PORT MODE amplitudes.

    The decomposition is not decoration. A uniform in-plane drive couples almost
    entirely to n = +-1 by symmetry -- 77.9% of the amplitude, measured on these
    features -- so the mean across the six ports lands on n = 0 and cancels the
    response almost exactly. Averaging first would have reported the disk's
    strongest mode as silence, and the phase of that silence as the nonlinear
    frequency shift. Locking in per port and transforming afterwards is also
    exactly what the reservoir readout does, so this measures the same quantity
    the capacity decomposition was run on.
    """
    cfg = disk.cfg
    # Uniform in-plane drive over the disk BODY -- what the chain runs used, and
    # therefore the drive whose linearity is in question.
    unit = torch.zeros_like(disk.m0)
    unit[:, :, :, 0] = disk.disk_only[:, :, :, 0].to(dtype)
    m = disk.m0.clone()
    n_ports = disk.port_signals(m).shape[0]
    accI = torch.zeros(2, n_ports, dtype=torch.float64)
    accQ = torch.zeros(2, n_ports, dtype=torch.float64)
    for k in range(n_settle + n_meas):
        tk = k * cfg.dt

        def h(theta, tk=tk):
            return unit * (amp * math.sin(2 * math.pi * freq * (tk + theta * cfg.dt)))

        m = disk.rollout.rk4_step(m, disk.h_zero, h)
        if k >= n_settle:
            p = disk.port_signals(m).double()
            for i, w in enumerate((2 * math.pi * freq, 4 * math.pi * freq)):
                accI[i] += p * math.cos(w * tk)
                accQ[i] += p * math.sin(w * tk)
    A = ((accI + 1j * accQ) / n_meas).numpy()
    return np.fft.fft(A[0]), np.fft.fft(A[1]), m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amps-mT", type=float, nargs="+",
                   default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--settle", type=int, default=600)
    p.add_argument("--meas", type=int, default=600)
    p.add_argument("--relax-steps", type=int, default=6000)
    p.add_argument("--outdir", default="runs/drive_nonlinearity")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = PortedVortexConfig()
    disk = PortedVortexDisk(cfg, timesteps=a.settle + a.meas + 8, dtype=dtype)
    m0c = outdir / "m0.pt"
    if m0c.exists():
        disk.m0 = torch.load(m0c, weights_only=False).to(dtype)
        print(f"[m0] restored from {m0c.name}")
    else:
        t0 = time.time()
        disk.relax(steps=a.relax_steps)
        torch.save(disk.m0.cpu(), m0c)
        print(f"relaxed in {time.time()-t0:.0f}s")

    mask = disk.mask[:, :, 0].to(dtype)
    n_cells = float(mask.sum())
    # The vortex signature: mean out-of-plane component over the disk. The core
    # is a few cells of m_z = 1, so this is small and POSITIVE for an intact
    # vortex and collapses when the core is expelled.
    mz0 = float((disk.m0[:, :, 0, 2] * mask).sum() / n_cells)
    print(f"ground state mean m_z = {mz0:.5f}\n")

    print(f"{'amp_mT':>7} {'|A|':>11} {'|A|/a':>11} {'norm':>7} "
          f"{'phase_deg':>10} {'d_phase':>8} {'2f/f':>9} {'mean_mz':>9} {'ok':>4}")
    rows, ref, mode = [], None, None
    for amp_mT in a.amps_mT:
        amp = amp_mT * 1e-3 / MU_0
        t0 = time.time()
        Mf, M2f, m = drive(disk, amp, a.freq * 1e9, dtype, a.settle, a.meas)
        # Fix the mode on the FIRST (smallest, most linear) amplitude and follow
        # that same one up the sweep. Re-picking the argmax per amplitude would
        # let the tracked phase jump between modes and read as a frequency shift.
        if mode is None:
            mode = int(np.argmax(np.abs(Mf)))
            print(f"tracking cross-port mode n = {mode} "
                  f"(strongest at {amp_mT:.0f} mT)\n")
        mag = float(abs(Mf[mode]))
        ph = float(np.angle(Mf[mode], deg=True))
        h2 = float(np.abs(M2f).max() / max(np.abs(Mf).max(), 1e-30))
        mz = float((m[:, :, 0, 2] * mask).sum() / n_cells)
        if ref is None:
            ref = (mag / amp_mT, ph)
        norm = (mag / amp_mT) / ref[0]
        dph = ((ph - ref[1] + 180) % 360) - 180
        # Stability: the core has to still be there. A vortex that has been
        # expelled shows mean m_z collapsing toward zero or flipping sign.
        ok = abs(mz) > 0.3 * abs(mz0) and np.sign(mz) == np.sign(mz0)
        rows.append({"amp_mT": amp_mT, "abs_A": mag, "per_mT": mag / amp_mT,
                     "normalised": norm, "phase_deg": ph, "d_phase_deg": dph,
                     "second_harmonic_ratio": h2, "mean_mz": mz,
                     "vortex_ok": bool(ok), "seconds": round(time.time() - t0, 1)})
        print(f"{amp_mT:>7.0f} {mag:>11.4e} {mag/amp_mT:>11.4e} {norm:>7.3f} "
              f"{ph:>10.2f} {dph:>8.2f} {h2:>9.4f} {mz:>9.5f} "
              f"{'yes' if ok else 'NO':>4}", flush=True)
        (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    live = [r for r in rows if r["vortex_ok"]]
    print("\ncolumns: 'norm' is |A|/a relative to the lowest amplitude, so 1.000")
    print("is perfectly linear response; 'd_phase' is the nonlinear frequency")
    print("shift, in degrees relative to the lowest amplitude.\n")
    if not live:
        print("No amplitude left the vortex intact -- nothing to conclude.")
    else:
        hi = live[-1]
        comp = abs(hi["normalised"] - 1.0)
        dph = abs(hi["d_phase_deg"])
        print(f"highest STABLE amplitude {hi['amp_mT']:.0f} mT: "
              f"compression {comp*100:.1f}%, phase shift {dph:.1f} deg, "
              f"2f/f {hi['second_harmonic_ratio']:.4f}")
        died = [r for r in rows if not r["vortex_ok"]]
        if died:
            print(f"vortex lost at and above {died[0]['amp_mT']:.0f} mT")
        if comp > 0.05 or dph > 10.0:
            print("\nThere IS usable nonlinearity below the stability limit. A\n"
                  "NARMA run driven in this window is worth its hours: the\n"
                  "amplitude-to-phase conversion is the mechanism that would\n"
                  "manufacture cross-lag products.")
        else:
            print("\nThe response is LINEAR everywhere the vortex survives.\n"
                  "Compression and phase shift both stay negligible right up to\n"
                  "the amplitude that destroys the core, so there is no window\n"
                  "where this disk is both nonlinear and stable. That is a limit\n"
                  "of the device on this task, not a tuning failure.")
    print(f"\nwrote {outdir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
