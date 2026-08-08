#!/usr/bin/env python3
"""Verify the directional-coupler geometry before spending physics on it.

The stub build's checker caught two bugs that would each have wasted an hour of
relax, so the coupler gets the same treatment plus the checks specific to it:

  arm parallel    the coupler arm must run alongside the bus at a CONSTANT gap
                  over its whole length. A coupler that touches the bus is a
                  galvanic stub with extra steps -- the thing being replaced.
  arm isolated    arm and bus must not be galvanically connected anywhere.
  arms separate   adjacent arms must not merge. At 567 nm tap spacing an arm
                  longer than ~547 nm fuses two taps into one waveguide.
  arm terminated  the upstream end of each arm must be damped, or the arm is a
                  resonator and its tap reads a standing wave whose phase has
                  nothing to do with the delay it was placed for.
  link connected  arm -> vertical link -> disk must be one connected path, or
                  the coupled power never reaches the disk.
  phase matched   arm width must equal bus width. Different widths mean
                  different dispersion, no phase matching, and power that beats
                  back out of the arm as fast as it couples in.

    python scripts/check_dircoupler_geometry.py --coupler-len 400
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.vortex import (DirCouplerConfig, DirCouplerArray,
                                dircoupler_mask, dircoupler_alpha)

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--n-taps", type=int, default=4)
p.add_argument("--lags", type=float, nargs="+", default=[5, 8, 11, 14])
p.add_argument("--coupler-len", type=float, default=400.0)
p.add_argument("--coupler-gap", type=float, default=20.0)
a = p.parse_args()

mnn.set_precision("float32"); mnn.set_device("cpu")
dtype = torch.float32
cfg = DirCouplerConfig(n_taps=a.n_taps, tap_lags=tuple(a.lags),
                       coupler_len=a.coupler_len * 1e-9,
                       coupler_gap=a.coupler_gap * 1e-9)
nx, ny = cfg.grid
dx = cfg.dx
print(f"grid {nx} x {ny} = {nx*ny:,} cells   bus {cfg.bus_length()*1e9:.0f} nm")
print(f"coupler: len {a.coupler_len:.0f} nm (max {cfg.max_coupler_len()*1e9:.0f}), "
      f"gap {a.coupler_gap:.0f} nm, width {cfg.coupler_width*1e9:.0f} nm")
print(f"taps at {[round((x-cfg.inject_at)*1e9) for x in cfg.tap_x()]} nm\n")

mask = dircoupler_mask(cfg, dtype=dtype)[:, :, 0, 0].numpy() > 0.5
alpha = dircoupler_alpha(cfg, dtype=dtype)[:, :, 0, 0].numpy()
arr = DirCouplerArray(cfg, timesteps=8, dtype=dtype)
yb = cfg._bus_y()
j_bus = int(round((yb + (ny - 1) / 2 * dx) / dx))
lo, hi = cfg.arm_y()
j_lo = int(round((yb + lo + (ny - 1) / 2 * dx) / dx))
j_hi = int(round((yb + hi + (ny - 1) / 2 * dx) / dx))
ok = True

print(f"phase matching: bus {cfg.bus_width*1e9:.0f} nm, arm "
      f"{cfg.coupler_width*1e9:.0f} nm "
      f"{'OK' if abs(cfg.bus_width-cfg.coupler_width) < dx/2 else '<-- MISMATCHED'}")
ok &= abs(cfg.bus_width - cfg.coupler_width) < dx / 2

print(f"\ncoupler gap (must be clear film everywhere along each arm):")
for k, (cx, _) in enumerate(cfg.centres()):
    i0 = int(round((cx - cfg.coupler_len + (nx - 1) / 2 * dx) / dx))
    i1 = int(round((cx + (nx - 1) / 2 * dx) / dx))
    band = mask[i0:i1, j_bus + int(round(cfg.bus_width / 2 / dx)) + 1:j_lo]
    filled = int(band.sum())
    print(f"  arm {k+1}: {filled} filled cells in the gap band "
          f"{'OK' if filled == 0 else '<-- ARM TOUCHES BUS'}")
    ok &= filled == 0

# flood fill from the bus: arms must NOT be reachable (isolated), disks likewise
xi = int(round((cfg.inject_x() + (nx - 1) / 2 * dx) / dx))
seen = np.zeros_like(mask, dtype=bool)
stack = [(xi, j_bus)]; seen[xi, j_bus] = True
while stack:
    i, j = stack.pop()
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        u, v = i + di, j + dj
        if 0 <= u < nx and 0 <= v < ny and mask[u, v] and not seen[u, v]:
            seen[u, v] = True; stack.append((u, v))
print(f"\nisolation from the bus (evanescent coupling, so nothing should connect):")
for k, dm in enumerate(arr.disk_masks.numpy() > 0.5):
    n = int((seen & dm).sum())
    print(f"  disk {k+1}: {n} cells reachable from bus "
          f"{'OK' if n == 0 else '<-- GALVANIC'}")
    ok &= n == 0

# arm -> link -> disk must be one connected component
print(f"\narm -> link -> disk connectivity:")
for k, (cx, cy) in enumerate(cfg.centres()):
    i_arm = int(round((cx - cfg.coupler_len / 2 + (nx - 1) / 2 * dx) / dx))
    j_arm = (j_lo + j_hi) // 2
    seen2 = np.zeros_like(mask, dtype=bool)
    st = [(i_arm, j_arm)]; seen2[i_arm, j_arm] = True
    while st:
        i, j = st.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = i + di, j + dj
            if 0 <= u < nx and 0 <= v < ny and mask[u, v] and not seen2[u, v]:
                seen2[u, v] = True; st.append((u, v))
    dm = arr.disk_masks.numpy()[k] > 0.5
    n = int((seen2 & dm).sum())
    tot = int(dm.sum())
    print(f"  tap {k+1}: {n}/{tot} disk cells reachable from its arm "
          f"{'OK' if n == tot else '<-- LINK BROKEN'}")
    ok &= n == tot

print(f"\nadjacent arms separate:")
xs = cfg.tap_x()
for k in range(len(xs) - 1):
    gap = (xs[k + 1] - cfg.coupler_len) - xs[k]
    print(f"  arms {k+1}-{k+2}: {gap*1e9:.0f} nm "
          f"{'OK' if gap > 2 * dx else '<-- ARMS MERGE'}")
    ok &= gap > 2 * dx

print(f"\narm upstream termination:")
for k, (cx, _) in enumerate(cfg.centres()):
    i0 = int(round((cx - cfg.coupler_len + (nx - 1) / 2 * dx) / dx))
    i1 = i0 + int(round(cfg.arm_absorb / dx))
    seg = alpha[i0:i1, j_lo:j_hi]
    print(f"  arm {k+1}: upstream alpha max {seg.max():.3f} "
          f"{'OK' if seg.max() > cfg.alpha * 5 else '<-- NOT DAMPED'}")
    ok &= seg.max() > cfg.alpha * 5

w = arr._tap_masks.numpy()
empty = [i for i in range(len(w)) if w[i].sum() == 0]
print(f"\nreadout taps: {len(w)} ({arr.n_readout} per disk x {cfg.n_taps}) "
      f"{'OK' if not empty else f'EMPTY: {empty}'}")
ok &= not empty

print("\nmap ('o' disk, '-' arm, '=' bus, '#' other):")
dms = arr.disk_masks.numpy() > 0.5
sx, sy = max(1, nx // 100), max(1, ny // 28)
for j in range(ny - 1, -1, -sy):
    row = ""
    for i in range(0, nx, sx):
        blk = mask[i:i + sx, max(0, j - sy):j + 1]
        if not blk.any():
            row += " "
        elif any(d[i:i + sx, max(0, j - sy):j + 1].any() for d in dms):
            row += "o"
        elif j_lo - sy <= j <= j_hi + sy:
            row += "-"
        elif abs(j - j_bus) <= sy:
            row += "="
        else:
            row += "#"
    print("  " + row)

print(f"\n{'GEOMETRY OK' if ok else 'GEOMETRY BROKEN'}")
raise SystemExit(0 if ok else 1)
