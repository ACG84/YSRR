#!/usr/bin/env python3
"""Verify the poly-tap geometry before spending any physics on it.

Everything downstream assumes four things about the patterned film, and all
four are cheap to check and expensive to discover later:

  connected     each disk's 270-degree guide must actually REACH the bus. If it
                falls a cell short the tap is dipolar-coupled at best, and the
                run would look like a mysteriously weak delay line rather than
                a broken one.
  separated     adjacent disks must not touch. At 567 nm spacing and a 433 nm
                x-footprint the gap is ~130 nm, and a single overlapping cell
                would silently fuse two taps into one object.
  absorbers     the readout guide ends must be damped and the coupling
                corridor must NOT be. An absorber spilling into the corridor
                attenuates the signal the tap exists to receive -- the mistake
                the chain avoided by cutting its links out of the ramp.
  bus intact    the bus must be one strip from injection to the far absorber,
                with the taps hanging off it.

Prints a coarse ASCII map as well, because a geometry bug is usually obvious
at a glance and invisible in a summary statistic.

    python scripts/check_polytap_geometry.py
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.vortex import (PolyTapConfig, PolyTapArray, polytap_mask,
                                polytap_alpha)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-taps", type=int, default=4)
    p.add_argument("--lags", type=float, nargs="+", default=[5, 8, 11, 14])
    p.add_argument("--gap", type=float, default=0.0,
                   help="nm between bus edge and coupling-guide tip. With a gap\n"
                        "the disks are DELIBERATELY not touching the bus, so the\n"
                        "connectivity test inverts: galvanic contact would be the\n"
                        "bug, and the gap must be intact at every tap.")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    cfg = PolyTapConfig(n_taps=a.n_taps, tap_lags=tuple(a.lags),
                        coupling_gap=a.gap * 1e-9)
    nx, ny = cfg.grid
    print(f"grid {nx} x {ny} = {nx*ny:,} cells   bus {cfg.bus_length()*1e9:.0f} nm")
    print(f"taps at {[round((x-cfg.inject_at)*1e9) for x in cfg.tap_x()]} nm "
          f"from injection")
    print(f"predicted lags {list(cfg.tap_lags[:cfg.n_taps])} frames")
    print(f"fresh-drive scale {[round(s,3) for s in cfg.fresh_scale()]}\n")

    mask = polytap_mask(cfg, dtype=dtype)[:, :, 0, 0].numpy() > 0.5
    alpha = polytap_alpha(cfg, dtype=dtype)[:, :, 0, 0].numpy()
    arr = PolyTapArray(cfg, timesteps=8, dtype=dtype)
    dx = cfg.dx
    yb = cfg._bus_y()
    j_bus = int(round((yb + (ny - 1) / 2 * dx) / dx))

    ok = True

    # 1. connectivity: flood fill from the injection cell and require every
    #    disk to be reached. A disk that is only dipolar-coupled will not be.
    xi = int(round((cfg.inject_x() + (nx - 1) / 2 * dx) / dx))
    seen = np.zeros_like(mask, dtype=bool)
    stack = [(xi, j_bus)]
    seen[xi, j_bus] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = i + di, j + dj
            if 0 <= u < nx and 0 <= v < ny and mask[u, v] and not seen[u, v]:
                seen[u, v] = True
                stack.append((u, v))
    want_contact = a.gap <= 0
    print(f"connectivity from the injection point "
          f"({'galvanic taps' if want_contact else f'{a.gap:.0f} nm gap taps'}):")
    for k, dm in enumerate(arr.disk_masks.numpy() > 0.5):
        n_reached = int((seen & dm).sum())
        touching = n_reached > 0
        good = touching == want_contact
        ok &= good
        if want_contact:
            good = n_reached == int(dm.sum())
            ok &= good
            print(f"  disk {k+1}: {n_reached}/{int(dm.sum())} cells reached "
                  f"{'OK' if good else 'NOT CONNECTED'}")
        else:
            print(f"  disk {k+1}: {n_reached} cells reached "
                  f"{'OK (isolated, as designed)' if good else 'TOUCHING THE BUS'}")
    if want_contact:
        unreached = int(mask.sum() - seen.sum())
        print(f"  material not reachable from the bus: {unreached} cells "
              f"{'OK' if unreached == 0 else '<-- floating islands'}")
        ok &= unreached == 0

    # 2. separation: no two disks' material may touch.
    print("\nadjacent-tap separation:")
    dms = arr.disk_masks.numpy() > 0.5
    for k in range(len(dms) - 1):
        xa = np.where(dms[k].any(axis=1))[0]
        xb = np.where(dms[k + 1].any(axis=1))[0]
        gap_cells = int(xb.min() - xa.max()) - 1
        # material gap along the row of disk centres, not just disk bodies
        row = int(round((cfg.centres()[k][1] + (ny - 1) / 2 * dx) / dx))
        seg = mask[xa.max():xb.min(), row]
        good = gap_cells > 0 and not seg.all()
        ok &= good
        print(f"  disks {k+1}-{k+2}: body gap {gap_cells*dx*1e9:.0f} nm "
              f"{'OK' if good else 'TOUCHING'}")

    # 3. absorbers in the right places
    print("\ndamping:")
    corridor = np.zeros_like(mask)
    for cx, cy in cfg.centres():
        i0 = int(round((cx - cfg.guide_width / 2 + (nx - 1) / 2 * dx) / dx))
        i1 = int(round((cx + cfg.guide_width / 2 + (nx - 1) / 2 * dx) / dx))
        j1 = int(round((cy + (ny - 1) / 2 * dx) / dx))
        corridor[i0:i1 + 1, j_bus:j1] = True
    cor = alpha[corridor & mask]
    print(f"  coupling corridors: alpha max {cor.max():.4f} "
          f"(bulk is {cfg.alpha}) "
          f"{'OK' if cor.max() < cfg.alpha * 1.5 else '<-- ABSORBING'}")
    ok &= cor.max() < cfg.alpha * 1.5
    ends = alpha[mask & (alpha > cfg.alpha * 5)]
    print(f"  absorbing cells: {ends.size} at alpha up to {alpha[mask].max():.2f} "
          f"{'OK' if ends.size > 0 else '<-- NO ABSORBERS'}")
    ok &= ends.size > 0

    # 4. readout taps land on material
    w = arr._tap_masks.numpy()
    empty = [i for i in range(len(w)) if w[i].sum() == 0]
    print(f"\nreadout taps: {len(w)} "
          f"({arr.n_readout} per disk x {cfg.n_taps}) "
          f"{'OK' if not empty else f'EMPTY: {empty}'}")
    ok &= not empty

    # coarse map
    print("\nmap (x -> right, y -> up; '#' material, '=' bus, 'o' disk):")
    sx, sy = max(1, nx // 100), max(1, ny // 26)
    for j in range(ny - 1, -1, -sy):
        row = ""
        for i in range(0, nx, sx):
            blk = mask[i:i + sx, max(0, j - sy):j + 1]
            if not blk.any():
                row += " "
            elif any(dm[i:i + sx, max(0, j - sy):j + 1].any() for dm in dms):
                row += "o"
            elif abs(j - j_bus) <= sy:
                row += "="
            else:
                row += "#"
        print("  " + row)

    print(f"\n{'GEOMETRY OK' if ok else 'GEOMETRY BROKEN'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
