#!/usr/bin/env python3
"""Why does the reimplemented modal analysis score 1.2x where the original gets 23.4x?

A control run settled that the disagreement is in the ANALYSIS, not the device:
check_readout_comparison scored 1.2x on a bare disk, which is run_modal_disk's
own geometry and where run_modal_disk scores 23.4x. Same physics, same drive,
different number. So one of the steps I changed while reimplementing is
responsible, and guessing which would be a fourth guess in a row.

This bisects instead. One simulation produces the azimuthal mode amplitudes
a_n(t); every analysis variant is then computed from those same arrays, so
nothing but the analysis step differs. The variants toggle, one at a time, the
four things I changed:

    real part vs complex   the original takes rfft(a.real), which merges +n
                           with -n; the reimplementation keeps a complex and
                           folds, which separates them
    DC removed             the original keeps the static offset a pulse
                           deposits; the reimplementation drops everything
                           below 1 GHz
    band restricted        full spectrum against 1-40 GHz
    Hann window            none against windowed

Whichever toggle moves the ratio from ~1 to ~20 is the answer, and it matters
which: if the original's 23.4x depends on keeping DC, that number is measuring
a static energy offset rather than mode structure, and the headline result is
weaker than advertised. If it depends on the real part, the original is right
and my +-n separation is what broke it.

    python scripts/bisect_modal_analysis.py
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import VortexConfig, VortexDisk


def pulse_sequence(n_steps, dt, order, f_a, f_b, pulse_steps, amp, ramp=0.15):
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
def mode_amplitudes(disk, signal, dtype, n_modes=6, record_every=2):
    """Return complex a_n(t), (n_modes, T) -- the common input to every variant."""
    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, :, 0] = disk.mask[:, :, :, 0]

    idx = torch.arange(n, dtype=torch.float64) - (n - 1) / 2
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    phi = torch.atan2(Y, X)
    inside = disk.mask[:, :, 0, 0].double() > 0
    n_in = int(inside.sum())
    basis = torch.stack([torch.exp(-1j * k * phi) * inside for k in range(n_modes)])

    m, out = disk.m0.clone(), []
    for k in range(len(signal) - 1):
        s0, s1 = float(signal[k]), float(signal[k + 1])

        def h_drive(theta, s0=s0, s1=s1):
            return unit * (s0 + theta * (s1 - s0))

        m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
        if k % record_every == 0:
            mz = ((m - disk.m0) * disk.mask)[:, :, 0, 2].double()
            out.append((basis * mz).sum(dim=(1, 2)) / n_in)
    return torch.stack(out, dim=1)                     # (n_modes, T) complex


def spectrum(a, dt_rec, use_real, drop_dc, band, window):
    """One analysis variant applied to a_n(t)."""
    A = a.numpy()
    T = A.shape[1]
    if window:
        A = A * np.hanning(T)[None, :]
    if use_real:
        S = np.abs(np.fft.rfft(A.real, axis=1))
        f = np.fft.rfftfreq(T, d=dt_rec) / 1e9
    else:
        full = np.abs(np.fft.fft(A, axis=1))
        half = T // 2
        f = np.fft.fftfreq(T, d=dt_rec)[1:half] / 1e9
        pos, neg = full[:, 1:half], full[:, T - 1:T - half:-1]
        S = pos + neg                                  # fold -n onto +n
    keep = np.ones_like(f, dtype=bool)
    if drop_dc:
        keep &= f > 1.0
    if band:
        keep &= f <= 40.0
    return S[:, keep]


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
    p.add_argument("--relax-steps", type=int, default=600)
    p.add_argument("--outdir", default="runs/bisect_modal")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = VortexConfig()
    pulse_steps = int(args.pulse_ns * 1e-9 / cfg.dt)
    n_steps = 2 * pulse_steps + int(args.read_ns * 1e-9 / cfg.dt)
    disk = VortexDisk(cfg, timesteps=n_steps, dtype=dtype)
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    print(f"relaxed {time.time()-t0:.0f}s, core mz "
          f"{float(disk.m0[:, :, 0, 2].max()):+.3f}", flush=True)

    rec = 2
    amps = {}
    cache = outdir / "amps.pt"
    if cache.exists():
        amps = torch.load(cache, weights_only=False)
        print("[cached] amps.pt", flush=True)
    else:
        for label, amp_mT in (("high", args.amp_mT), ("low", args.low_amp_mT)):
            for order in ("AB", "BA"):
                sig = pulse_sequence(n_steps, cfg.dt, order, args.f_a * 1e9,
                                     args.f_b * 1e9, pulse_steps,
                                     amp_mT * 1e-3 / MU_0)
                t0 = time.time()
                amps[(label, order)] = mode_amplitudes(disk, sig, dtype,
                                                       record_every=rec)
                print(f"  {label} {order}: {time.time()-t0:.0f}s", flush=True)
        torch.save(amps, cache)

    dt_rec = cfg.dt * rec
    print(f"\n{'real':>5} {'dropDC':>7} {'band':>5} {'hann':>5}   "
          f"{'high':>8} {'low':>8} {'ratio':>8}")
    rows = []
    for use_real, drop_dc, band, window in itertools.product(
            (True, False), (False, True), (False, True), (False, True)):
        sc = {}
        for label in ("high", "low"):
            a = spectrum(amps[(label, "AB")], dt_rec, use_real, drop_dc, band, window)
            b = spectrum(amps[(label, "BA")], dt_rec, use_real, drop_dc, band, window)
            sc[label] = rel_diff(a, b)
        ratio = sc["high"] / max(sc["low"], 1e-12)
        rows.append({"real": use_real, "drop_dc": drop_dc, "band": band,
                     "hann": window, "high": sc["high"], "low": sc["low"],
                     "ratio": ratio})
        print(f"{str(use_real):>5} {str(drop_dc):>7} {str(band):>5} "
              f"{str(window):>5}   {sc['high']:>8.4f} {sc['low']:>8.4f} "
              f"{ratio:>8.1f}", flush=True)

    (outdir / "results.json").write_text(json.dumps(rows, indent=2))
    orig = next(r for r in rows if r["real"] and not r["drop_dc"]
                and not r["band"] and not r["hann"])
    mine = next(r for r in rows if not r["real"] and r["drop_dc"]
                and r["band"] and r["hann"])
    print(f"\noriginal recipe (real, keep DC, full band, no window): "
          f"{orig['ratio']:.1f}x")
    print(f"reimplementation (complex fold, drop DC, 1-40 GHz, hann): "
          f"{mine['ratio']:.1f}x")
    best = max(rows, key=lambda r: r["ratio"])
    print(f"largest ratio: {best['ratio']:.1f}x at real={best['real']} "
          f"dropDC={best['drop_dc']} band={best['band']} hann={best['hann']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
