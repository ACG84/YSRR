#!/usr/bin/env python3
"""Is there a DELAYED component at each tap, or only stray field?

The arrival metric used so far is the first crossing of 10% of a tap's own
envelope peak. That is fine when one component dominates and misleading when
two are present: a small instantaneous stray precursor trips the threshold
early even if the guided arrival, several frames later, carries most of the
energy. Every coupler variant so far has been judged by that number, so before
calling a third architecture dead it is worth asking the question the threshold
cannot answer.

This reports the whole shape instead: the time of the envelope PEAK, and the
envelope sampled at lag 0 against the tap's designed lag. A tap receiving a
genuine delayed copy peaks near its designed lag and has a lag-0/peak ratio
well below 1. A tap reading stray field peaks immediately and stays flat.

Bus injection only, no fresh drive. Reuses the cached m0.

    python scripts/check_polytap_envelope.py --coupler-len 400
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
p.add_argument("--coupler-len", type=float, default=400.0)
p.add_argument("--gap", type=float, default=0.0)
p.add_argument("--amp-mT", type=float, default=30.0)
p.add_argument("--freq", type=float, default=12.0)
p.add_argument("--burst", type=int, default=400)
p.add_argument("--quiet", type=int, default=2000)
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
print(f"[m0] m0_{tag}.pt")

unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
unit[:, :, 0, 2] = arr.inject_mask
amp = a.amp_mT * 1e-3 / MU_0
w = 2 * math.pi * a.freq * 1e9
m = arr.m0.clone()
out = []
t0 = time.time()
with torch.no_grad():
    for k in range(a.burst + a.quiet):
        tk = k * cfg.dt
        on = k < a.burst
        def h(theta, tk=tk, on=on):
            if not on:
                return torch.zeros_like(unit)
            return unit * (amp * math.sin(w * (tk + theta * cfg.dt)))
        m = arr.rollout.rk4_step(m, arr.h_zero, h)
        out.append(arr.port_signals(m).double().cpu().numpy().copy())
        if (k + 1) % 800 == 0:
            print(f"  step {k+1}/{a.burst+a.quiet} ({time.time()-t0:.0f}s)", flush=True)
sig = np.asarray(out)

win = max(4, int(round(1e3 / a.freq)))
frame = 200
npr = arr.n_readout
print(f"\n{'tap':>4} {'designed':>9} {'peak_lag':>9} {'lag0/peak':>10} "
      f"{'at_designed/peak':>17} {'peak amp':>11}")
rows = []
for d in range(cfg.n_taps):
    s = np.sqrt((sig[:, d*npr:(d+1)*npr] ** 2).mean(axis=1))
    e = np.sqrt(np.convolve(s ** 2, np.ones(win) / win, mode="same"))
    pk_i = int(np.argmax(e)); pk = float(e[pk_i])
    des_i = min(len(e) - 1, int(round(cfg.tap_lags[d] * frame)))
    r0 = float(e[:win].max()) / max(pk, 1e-30)
    rd = float(e[des_i]) / max(pk, 1e-30)
    rows.append({"tap": d + 1, "designed_lag": cfg.tap_lags[d],
                 "peak_lag": pk_i / frame, "lag0_over_peak": r0,
                 "at_designed_over_peak": rd, "peak": pk})
    print(f"{d+1:>4} {cfg.tap_lags[d]:>9.1f} {pk_i/frame:>9.2f} {r0:>10.3f} "
          f"{rd:>17.3f} {pk:>11.4e}")
Path(a.outdir).mkdir(parents=True, exist_ok=True)
(Path(a.outdir) / f"envelope_{tag}.json").write_text(json.dumps(rows, indent=2))

print("\nreading: a tap fed a real delayed copy PEAKS near its designed lag and")
print("has lag0/peak well below 1. A stray-dominated tap peaks at ~0 and has")
print("lag0/peak ~1.")
delayed = [r for r in rows if r["peak_lag"] > 0.5 * r["designed_lag"]]
print(f"\ntaps peaking near their designed lag: {len(delayed)}/{len(rows)}")
print(f"wrote {Path(a.outdir)}/envelope_{tag}.json")
