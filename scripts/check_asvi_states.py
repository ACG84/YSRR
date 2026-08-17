#!/usr/bin/env python3
"""Does a 3D multilayer ASVI island hold its four states, where the disk did not?

THE REMATCH. This project's ported vortex disks are not vortices: initialised
with a proper core (m_z = 1 at the centre), they hold it to about relax step
400 and lose it by 800, while every array run relaxes 8000. Measured on an
ISOLATED ported disk, so neither the bus nor the neighbours can be blamed --
six 80 nm guides give the flux a closure path a bare disk does not have. Every
interpretation resting on "the disk's mode" fell with it, including the choice
of 9 GHz.

The element here is named for holding vortices. Four states per magnetic layer,
two macrospin and two vortex, 16 per 3D island, which is where the 16^N
microstate space comes from. So the first question is not what it computes but
whether the core survives, and the honest way to ask it is the way that caught
the disk: sample the core through the relax rather than only at the end.

    Dion et al., Nat. Commun. 15 (2024), doi:10.1038/s41467-024-48080-z

WHAT IS REPORTED, per magnetic layer, at every checkpoint:

    |mz|max     peak out-of-plane. A vortex core is ~1. A layer that has gone
                planar reads ~0 -- which is exactly what the disks return, and
                what the old whole-mask check could not detect because it read
                0.133 and had no threshold to fail.
    mean mz     order (core area / island area). Separates a core from a
                uniformly canted layer, which |mz|max alone cannot.
    |m| in-plane net moment. A macrospin layer reads ~1, a flux-closed one ~0.
    circ        mean (r_hat x m)_z: +1 anticlockwise, -1 clockwise. Chirality
                needs its own number -- |mz| says a core is there and the
                moment says the layer is closed, and neither tells ACW from CW.
                The 50 nm inter-layer offset exists to select between them, so
                a run that cannot read chirality cannot see the paper's knob.

The two macrospin combinations are CONTROLS, not results. They should survive
trivially; if they do not, the mesh or the relax is wrong and no vortex number
from the same run means anything.

    python scripts/check_asvi_states.py --device cuda
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import magnonic_nn as mnn
from magnonic_nn.spinice import (ASVIConfig, ASVIIsland, STATES,
                                 MACROSPIN_POS, MACROSPIN_NEG,
                                 VORTEX_ACW, VORTEX_CW)

COMBOS = [
    (MACROSPIN_POS, MACROSPIN_POS),   # control: parallel macrospin
    (MACROSPIN_POS, MACROSPIN_NEG),   # control: antiparallel macrospin
    (VORTEX_ACW,    MACROSPIN_POS),   # mixed: vortex in the 30 nm hard layer
    (MACROSPIN_POS, VORTEX_ACW),      # mixed: vortex in the 20 nm soft layer
    (VORTEX_ACW,    VORTEX_ACW),      # both vortex, same chirality
    (VORTEX_ACW,    VORTEX_CW),       # both vortex, opposite -- the offset
                                      # breaks the degeneracy between these two
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--alpha-relax", type=float, default=0.5)
    # Sampled where the disk died. The disk held to 400 and was gone by 800, so
    # a checkpoint list that jumps 0 -> 8000 would report the endpoint and miss
    # the fact that there was ever a core at all.
    p.add_argument("--checkpoints", type=int, nargs="+",
                   default=[0, 100, 200, 400, 800, 1600, 3200, 8000])
    p.add_argument("--length-nm", type=float, default=550.0)
    p.add_argument("--width-nm", type=float, default=140.0)
    p.add_argument("--offset-nm", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--dx-nm", type=float, default=5.0)
    p.add_argument("--outdir", default="runs/asvi")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = ASVIConfig(length=a.length_nm * 1e-9, width=a.width_nm * 1e-9,
                     layer_offset=a.offset_nm * 1e-9, alpha=a.alpha,
                     dx=a.dx_nm * 1e-9, dz=a.dx_nm * 1e-9)
    nx, ny, nz = cfg.grid
    isl = ASVIIsland(cfg, timesteps=32, dtype=dtype)
    print(f"ASVI island {a.length_nm:g} x {a.width_nm:g} x "
          f"{cfg.thickness()*1e9:.0f} nm, aspect {cfg.aspect():.2f}")
    print(f"mesh {nx} x {ny} x {nz} = {nx*ny*nz:,} cells, "
          f"{int(isl.mask.sum()):,} magnetic")
    print(f"layers: " + ", ".join(
        f"{(z1-z0)*cfg.dz*1e9:.0f} nm at y {cfg.layer_cy(k)*1e9:+.0f} nm"
        for k, (z0, z1, _) in enumerate(cfg.magnetic_layers())))
    print(f"alpha {cfg.alpha:g}, Ms {cfg.Ms/1e3:.0f} kA/m, A {cfg.A*1e12:g} pJ/m, "
          f"dt {cfg.dt*1e12:g} ps\n", flush=True)

    cps = sorted(set(c for c in a.checkpoints if c <= a.relax_steps))
    rows = []
    for combo in COMBOS:
        tag = " / ".join(combo)
        print(f"===== {tag} =====", flush=True)
        m = isl.initial_state(list(combo), dtype=dtype)
        hdr = (f"{'step':>6} " +
               " ".join(f"{'L'+str(k)+' |mz|max':>11}{'meanmz':>9}"
                        f"{'|m|ip':>8}{'circ':>8}" for k in range(isl.n_layers)))
        print(hdr, flush=True)
        done, hist = 0, []
        t0 = time.time()
        for cp in cps:
            if cp > done:
                m = isl.rollout.relax(m, isl.h_zero, cp - done, a.alpha_relax)
                done = cp
            rec = {"step": done, "layers": []}
            line = f"{done:>6} "
            for k in range(isl.n_layers):
                pk, mn = isl.layer_core(m, k)
                mx, my = isl.layer_moment(m, k)
                ip = (mx**2 + my**2) ** 0.5
                cc = isl.layer_circulation(m, k)
                rec["layers"].append({"peak_mz": pk, "mean_mz": mn,
                                      "m_inplane": ip, "circulation": cc,
                                      "mx": mx, "my": my})
                line += f"{pk:>11.3f}{mn:>9.4f}{ip:>8.3f}{cc:>8.3f}"
            hist.append(rec)
            print(line, flush=True)
        rows.append({"states": list(combo), "seconds": round(time.time()-t0, 1),
                     "history": hist})
        (outdir / "asvi_states.json").write_text(json.dumps(rows, indent=2))
        print(flush=True)

    # ------------------------------------------------------------- verdict
    print("=" * 72)
    print(f"{'initial state':<26} " + " ".join(
        f"{'L'+str(k)+' final':>22}" for k in range(isl.n_layers)))
    # Peak |m_z| ALONE does not identify a vortex, and the smoke test proved it:
    # a saturated 90 nm-thick island reads |m_z|max 0.40-0.48 from edge canting
    # at the stadium caps, which would clear any sensible core threshold and be
    # reported as a vortex. Its mean m_z is 0.0000 and its in-plane moment is
    # 0.98, so flux closure is the discriminator and the core is the
    # confirmation. Classifying on the core alone is exactly the error that let
    # a whole-mask reading of 0.133 stand in for a vortex on the disks.
    CORE = 0.5          # a core is present if peak |m_z| clears this
    CLOSED = 0.5        # a layer is flux-closed if its in-plane moment is below
    for r in rows:
        line = f"{' / '.join(r['states']):<26} "
        for k in range(isl.n_layers):
            f = r["history"][-1]["layers"][k]
            closed = f["m_inplane"] < CLOSED
            if closed and f["peak_mz"] >= CORE:
                chir = "ACW" if f["circulation"] > 0 else "CW"
                what = f"vortex {chir} ({f['circulation']:+.2f})"
            elif not closed:
                what = f"macro {'+' if f['mx'] > 0 else '-'} ({f['m_inplane']:.2f})"
            else:
                what = f"closed,no core ({f['peak_mz']:.2f})"
            line += f"{what:>22}"
        print(line)

    print()
    vortex_runs = [r for r in rows if any(s.startswith("vortex") for s in r["states"])]
    survived = 0
    for r in vortex_runs:
        for k, s in enumerate(r["states"]):
            if s.startswith("vortex"):
                f = r["history"][-1]["layers"][k]
                if f["peak_mz"] >= CORE and f["m_inplane"] < CLOSED:
                    survived += 1
    total = sum(sum(1 for s in r["states"] if s.startswith("vortex"))
                for r in vortex_runs)
    print(f"vortex layers surviving {a.relax_steps} relax steps: {survived}/{total}")
    if survived == total:
        print("Every vortex held. The element has the state multiplicity the\n"
              "architecture needs, which the ported disk did not -- that lost\n"
              "its core between step 400 and 800.")
    elif survived:
        print("Partial. Compare the two layers: the hard layer is 30 nm and the\n"
              "soft 20 nm, and vortex stability rises with thickness, so a\n"
              "thickness-ordered failure is physics and an unordered one is a\n"
              "bug. Check the step at which each died against the disk's 400-800.")
    else:
        print("No vortex survived. Before concluding the element cannot hold\n"
              "one, check the controls: if the macrospin combinations also look\n"
              "wrong the mesh or the relax is at fault, not the state. If they\n"
              "are clean, the aspect ratio is the next suspect -- 3.93 here,\n"
              "and an elongated stadium favours macrospin over flux closure.")
    print(f"\nwrote {outdir / 'asvi_states.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
