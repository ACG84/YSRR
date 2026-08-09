#!/usr/bin/env python3
"""Measure tap delay by cross-correlation, not by threshold or peak.

Three builds have now been judged on estimators that cannot support the
verdict. The 10% threshold fires on whatever arrives first, so a small
instantaneous precursor reads as "no delay" even when 85-98% of the energy
arrives late -- which is what the envelope shapes showed. The envelope PEAK
fails the other way: each tap is a resonator with ~8 frames of ring-down driven
by a 2-frame burst, so the peak marks the disk ringing up, not the wave
arriving, and the two estimators disagree on the same data.

Cross-correlating each tap's ENVELOPE against the injected burst's envelope is
insensitive to both. Envelope rather than carrier because the carrier period is
83 ps -- 0.42 frames -- so a carrier-level correlation is ambiguous modulo the
period, while the burst envelope is unique.

Ring-up still biases the absolute number late, by roughly the disk's rise time,
and that bias is COMMON to every tap. So the quantity to trust is the
DIFFERENCE between taps: taps designed 3 frames apart should measure 3 frames
apart whatever the common bias. That comparison is what this reports, and it is
the one the architecture actually rests on.

    python scripts/check_polytap_delay.py --coupler-len 400
    python scripts/check_polytap_delay.py --gap 0          # the stub control
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import (PolyTapConfig, PolyTapArray,
                                DirCouplerConfig, DirCouplerArray)

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--n-taps", type=int, default=4)
p.add_argument("--lags", type=float, nargs="+", default=[5, 8, 11, 14])
p.add_argument("--coupler-len", type=float, default=0.0)
p.add_argument("--gap", type=float, default=0.0)
p.add_argument("--amp-mT", type=float, default=30.0)
p.add_argument("--freq", type=float, default=12.0)
p.add_argument("--burst", type=int, default=200)
p.add_argument("--quiet", type=int, default=2600)
p.add_argument("--outdir", default="runs/polytap_impulse")
a = p.parse_args()

mnn.set_precision("float32"); mnn.set_device("cpu")
dtype = torch.float32
if a.coupler_len > 0:
    cfg = DirCouplerConfig(n_taps=a.n_taps, tap_lags=tuple(a.lags),
                           coupler_len=a.coupler_len * 1e-9,
                           coupler_gap=(a.gap or 20.0) * 1e-9)
    arr = DirCouplerArray(cfg, timesteps=a.burst + a.quiet + 8, dtype=dtype)
    tag = f"n{cfg.n_taps}_cpl{int(a.coupler_len)}"
else:
    cfg = PolyTapConfig(n_taps=a.n_taps, tap_lags=tuple(a.lags),
                        coupling_gap=a.gap * 1e-9)
    arr = PolyTapArray(cfg, timesteps=a.burst + a.quiet + 8, dtype=dtype)
    tag = f"n{cfg.n_taps}_gap{int(a.gap)}"
arr.m0 = torch.load(Path(a.outdir) / f"m0_{tag}.pt", weights_only=False).to(dtype)
print(f"[m0] m0_{tag}.pt   burst {a.burst} steps")

unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
unit[:, :, 0, 2] = arr.inject_mask
amp = a.amp_mT * 1e-3 / MU_0
w = 2 * math.pi * a.freq * 1e9
m = arr.m0.clone()
out, drive = [], []
t0 = time.time()
with torch.no_grad():
    for k in range(a.burst + a.quiet):
        tk = k * cfg.dt
        on = k < a.burst
        drive.append(1.0 if on else 0.0)
        def h(theta, tk=tk, on=on):
            if not on:
                return torch.zeros_like(unit)
            return unit * (amp * math.sin(w * (tk + theta * cfg.dt)))
        m = arr.rollout.rk4_step(m, arr.h_zero, h)
        out.append(arr.port_signals(m).double().cpu().numpy().copy())
        if (k + 1) % 800 == 0:
            print(f"  step {k+1}/{a.burst+a.quiet} ({time.time()-t0:.0f}s)", flush=True)
sig = np.asarray(out); drv = np.asarray(drive)

win = max(4, int(round(1e3 / a.freq)))
frame = 200
npr = arr.n_readout

def env(x):
    return np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode="same"))

ref = drv - drv.mean()
print(f"\n{'tap':>4} {'designed':>9} {'xcorr_lag':>10} {'corr':>7} {'peak amp':>11}")
rows = []
for d in range(cfg.n_taps):
    e = env(np.sqrt((sig[:, d*npr:(d+1)*npr] ** 2).mean(axis=1)))
    ec = e - e.mean()
    c = np.correlate(ec, ref, mode="full")
    lags_steps = np.arange(-len(ref) + 1, len(ec))
    ok = lags_steps >= 0
    i = int(np.argmax(c[ok]))
    lag_frames = lags_steps[ok][i] / frame
    denom = math.sqrt(float((ec ** 2).sum()) * float((ref ** 2).sum()))
    rows.append({"tap": d + 1, "designed_lag": cfg.tap_lags[d],
                 "xcorr_lag": lag_frames,
                 "corr": float(c[ok][i] / max(denom, 1e-30)),
                 "peak": float(e.max())})
    print(f"{d+1:>4} {cfg.tap_lags[d]:>9.1f} {lag_frames:>10.2f} "
          f"{rows[-1]['corr']:>7.3f} {rows[-1]['peak']:>11.4e}")

print("\nDIFFERENCES, which is where the common ring-up bias cancels:")
print(f"{'pair':>10} {'designed':>9} {'measured':>9} {'error':>7}")
good = 0
for k in range(cfg.n_taps - 1):
    dd = cfg.tap_lags[k + 1] - cfg.tap_lags[k]
    dm = rows[k + 1]["xcorr_lag"] - rows[k]["xcorr_lag"]
    err = dm - dd
    good += abs(err) <= max(1.0, 0.4 * dd)
    print(f"{f'{k+1}->{k+2}':>10} {dd:>9.1f} {dm:>9.2f} {err:>+7.2f}")
Path(a.outdir).mkdir(parents=True, exist_ok=True)
(Path(a.outdir) / f"delay_{tag}.json").write_text(json.dumps(rows, indent=2))
print(f"\ntap-spacing deltas within tolerance: {good}/{cfg.n_taps-1}")
print(f"wrote {Path(a.outdir)}/delay_{tag}.json")
