#!/usr/bin/env python3
"""Is the bus being drained by its own taps? Probe the bus between them.

The impulse check found tap amplitudes falling ~10x per tap (1.50e-3, 1.37e-4,
1.82e-5, 4.77e-6) while the BARE bus attenuates only 1.8x over the same 567 nm.
Two explanations fit and they need different fixes:

  loading   tap 1 absorbs most of the wave and passes little on. Fix: couple
            each tap weakly (a gap), so each takes a few percent.
  lossy     the loaded bus itself attenuates far more than the bare strip --
            the disks' stray fields perturbing the bus ground state. Fix: none
            cheap; the architecture would be in trouble.

Probing the bus between taps separates them: under loading the bus steps DOWN
at each tap and is flat between; under loss it decays smoothly everywhere.

Reuses the cached m0, so this costs one burst rather than another relax.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import PolyTapConfig, PolyTapArray

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--n-taps", type=int, default=4)
p.add_argument("--lags", type=float, nargs="+", default=[5, 8, 11, 14])
p.add_argument("--amp-mT", type=float, default=30.0)
p.add_argument("--freq", type=float, default=12.0)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--gap", type=float, default=0.0)
p.add_argument("--outdir", default="runs/polytap_impulse")
a = p.parse_args()

mnn.set_precision("float32"); mnn.set_device("cpu")
dtype = torch.float32
cfg = PolyTapConfig(n_taps=a.n_taps, tap_lags=tuple(a.lags),
                    coupling_gap=a.gap * 1e-9)
arr = PolyTapArray(cfg, timesteps=a.steps + 8, dtype=dtype)
m0c = Path(a.outdir) / f"m0_n{cfg.n_taps}_gap{int(a.gap)}.pt"
arr.m0 = torch.load(m0c, weights_only=False).to(dtype)
print(f"[m0] {m0c.name}")

nx, ny = cfg.grid
dx = cfg.dx
x0 = -(nx - 1) / 2 * dx
j_bus = int(round((cfg._bus_y() + (ny - 1) / 2 * dx) / dx))
strip = arr.bus_mask.bool()

# probe every 100 nm along the bus, from the injection to the far absorber
probe_nm = np.arange(cfg.inject_at * 1e9 + 200, (cfg.bus_length() - cfg.bus_absorb) * 1e9, 100.0)
probe_i = [int(round(v * 1e-9 / dx)) for v in probe_nm]
tap_nm = [x * 1e9 for x in cfg.tap_x()]

unit = torch.zeros(nx, ny, 1, 3, dtype=dtype)
unit[:, :, 0, 2] = arr.inject_mask
amp = a.amp_mT * 1e-3 / MU_0
w = 2 * math.pi * a.freq * 1e9
m = arr.m0.clone()
sig = np.zeros((a.steps, len(probe_i)))
t0 = time.time()
with torch.no_grad():
    for k in range(a.steps):
        tk = k * cfg.dt
        def h(theta, tk=tk):
            return unit * (amp * math.sin(w * (tk + theta * cfg.dt)))
        m = arr.rollout.rk4_step(m, arr.h_zero, h)
        dmz = (m - arr.m0)[:, :, 0, 2]
        for j, xi in enumerate(probe_i):
            col = dmz[xi][strip[xi]]
            sig[k, j] = float(col.mean()) if col.numel() else 0.0
        if (k + 1) % 1000 == 0:
            print(f"  step {k+1}/{a.steps} ({time.time()-t0:.0f}s)", flush=True)

print(f"\ntaps sit at x = {[round(v) for v in tap_nm]} nm")
print(f"{'x_nm':>7} {'bus rms':>11} {'vs first':>9}  {'':<3}")
ref = None
rows = []
for j, v in enumerate(probe_nm):
    rms = float(np.sqrt((sig[-a.steps // 3:, j] ** 2).mean()))
    if ref is None:
        ref = rms
    near = min(range(len(tap_nm)), key=lambda t: abs(tap_nm[t] - v))
    marker = f"<- tap {near+1}" if abs(tap_nm[near] - v) < 60 else ""
    rows.append({"x_nm": float(v), "rms": rms, "rel": rms / max(ref, 1e-30)})
    print(f"{v:>7.0f} {rms:>11.4e} {rms/max(ref,1e-30):>9.4f}  {marker}")
Path(a.outdir).mkdir(parents=True, exist_ok=True)
(Path(a.outdir) / f"bus_profile_gap{int(a.gap)}.json").write_text(json.dumps(rows, indent=2))

# step at each tap vs decay between taps
print("\nratio across each tap vs the bare-bus expectation over the same span:")
for t, xv in enumerate(tap_nm):
    before = [r for r in rows if r["x_nm"] < xv - 60]
    after = [r for r in rows if r["x_nm"] > xv + 60]
    if not before or not after:
        continue
    b, af = before[-1]["rms"], after[0]["rms"]
    span = (after[0]["x_nm"] - before[-1]["x_nm"]) * 1e-9
    bare = math.exp(-span / 951e-9)
    print(f"  tap {t+1}: {b:.3e} -> {af:.3e} = {af/max(b,1e-30):.3f}   "
          f"bare bus over {span*1e9:.0f} nm would give {bare:.3f}")
print(f"\nwrote {Path(a.outdir)}/bus_profile_gap{int(a.gap)}.json")
