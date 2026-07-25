#!/usr/bin/env python3
"""Measure whether the film is reciprocal: does H_ab equal H_ba?

Reciprocity is the property that makes *in-situ* (physical) gradient schemes
conceivable at all. If driving at A and listening at B gives the same transfer
as driving at B and listening at A, then a field launched backwards from the
detectors is the transpose of the forward operator -- which is what an adjoint
gradient needs. It is measured here rather than assumed, because the standard
expectation for a biased magnetic film is that it is *not* reciprocal.

Two traps this script is built to avoid:

* **Mirror symmetry.** Placing A and B symmetrically about the mesh centre makes
  ``H_ab = H_ba`` hold by symmetry whatever the reciprocity, so the default
  points are deliberately off-axis in both coordinates.
* **An unsaturated film.** A perpendicular bias must exceed the film's own
  demagnetising field, ``mu_0 Ms`` = 176 mT for this YIG, or the magnetisation
  never leaves the plane and the "forward volume" geometry does not exist.
  Measuring it at 60 mT reports a meaningless number with no warning.

    python scripts/check_reciprocity.py
    python scripts/check_reciprocity.py --freq 4.5 --nx 48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import magnonic_nn as mnn
from magnonic_nn.config import MU_0

GEOMETRIES = [
    # name, B0, bias axis, driven component
    ("in-plane (Damon-Eshbach)", 60e-3, (0.0, 1.0, 0.0), 2),
    ("out-of-plane (forward volume)", 250e-3, (0.0, 0.0, 1.0), 0),
]


def transfer(cfg, a, b, drive_axis, freq):
    """Drive a point source at ``a``, return the field trace at ``b``."""
    src = mnn.PointSource(cfg.mesh, cfg.fields, *a)
    model = mnn.SpinWaveNetwork(cfg, [src], [mnn.PointProbe(cfg.mesh, *b)])

    with torch.no_grad():
        result = model.run(mnn.tone(cfg, freq), snapshot_every=2)
        m0 = model.equilibrium()

    axis = torch.tensor(cfg.fields.bias_axis, dtype=m0.dtype, device=m0.device)
    saturation = float((m0 * axis).sum(dim=-1).mean())
    trace = (result.snapshots - m0.unsqueeze(0))[:, b[0], b[1], 0, drive_axis]
    return trace, saturation


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--freq", type=float, default=4.0, help="GHz")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--tolerance", type=float, default=0.99,
                   help="minimum cosine to call the medium reciprocal")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    mnn.set_precision("float64")   # the whole point is a precise symmetry test
    mnn.set_device("cpu")

    demag = MU_0 * mnn.get_preset("vowels").material.Ms
    print(f"film demagnetising field mu_0*Ms = {demag * 1e3:.0f} mT "
          f"(a perpendicular bias below this cannot saturate the film)\n")

    # deliberately not mirror-symmetric in either coordinate
    A, B = (args.nx // 4, args.nx // 3), (3 * args.nx // 4, 2 * args.nx // 3)
    print(f"source A = {A}, detector B = {B}, {args.freq:.2f} GHz\n")

    worst = 1.0
    for name, b0, axis, drive in GEOMETRIES:
        cfg = mnn.get_preset("vowels")
        cfg.mesh.nx = cfg.mesh.ny = args.nx
        cfg.material.abc_width = max(args.nx // 8, 2)
        cfg.solver.timesteps = args.timesteps
        cfg.solver.relax_steps = 120
        cfg.fields.B0, cfg.fields.bias_axis, cfg.fields.drive_axis = b0, axis, drive
        cfg.fields.Bt = 1e-3

        ab, sat = transfer(cfg, A, B, drive, args.freq * 1e9)
        ba, _ = transfer(cfg, B, A, drive, args.freq * 1e9)

        cos = float((ab * ba).sum() / (ab.norm() * ba.norm()).clamp_min(1e-30))
        ratio = float(ab.norm() / ba.norm().clamp_min(1e-30))
        worst = min(worst, cos)

        flag = "" if sat > 0.95 else "   <-- FILM NOT SATURATED, result meaningless"
        print(f"{name:<34} B0={b0 * 1e3:5.0f} mT  saturation {sat:.3f}{flag}")
        print(f"{'':34} cos(H_ab, H_ba) = {cos:+.4f}   |H_ab|/|H_ba| = {ratio:.4f}\n")

    if worst >= args.tolerance:
        print("Reciprocal. An adjoint field launched from the detectors is the")
        print("transpose of the forward operator, so a physical gradient scheme")
        print("is not ruled out on symmetry grounds.")
        print()
        print("Necessary, not sufficient: the linearised LLG operator is")
        print("gyroscopic (m x .), so the adjoint dynamics are not simply the")
        print("forward dynamics run backwards, and reciprocity of the transfer")
        print("function alone does not deliver the gradient.")
        return 0

    print(f"Non-reciprocal (worst cosine {worst:+.4f}). A physical adjoint would be")
    print("biased by the asymmetry.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
