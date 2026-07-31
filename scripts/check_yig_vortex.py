#!/usr/bin/env python3
"""Does a YIG disk hold a vortex at all, and above what radius?

This gates the whole YIG idea, so it runs before anything else.

YIG is worth the trouble for one reason: memory. Expressed in reservoir frames,
the ring-down is 1/(2*pi*alpha*N_cycles) where N_cycles is the carrier periods
per frame -- INDEPENDENT of frequency. Permalloy at alpha = 0.008 with 2.4
cycles gives 8.3 frames, and measurement put the real figure at 2.6 once the
port drains are counted. YIG at alpha = 1e-4 gives 663. Nothing geometric comes
close to 80x.

But YIG is not a parameter swap. Its exchange length

    l_ex = sqrt(2A / (mu0 Ms^2)) = 17.1 nm

against permalloy's 5.7 nm, because A is 3.6e-12 rather than 1.3e-11 and Ms is
140 kA/m rather than 800. Flux closure is what stabilises a vortex, and it is
the magnetostatic energy -- scaling as Ms^2 -- that pays for the exchange cost
of the core. With Ms down 5.7x that bargain is much worse, so the disk has to
be bigger in absolute terms AND in units of l_ex. The 100 nm radius used
throughout is only 5.9 l_ex in YIG; permalloy's 100 nm is 17.5 l_ex. There is
no reason to expect a vortex to survive at that size, and every downstream YIG
measurement is void if it does not.

So: initialise a vortex, relax, and ask whether it is still a vortex. Two
independent checks, since either alone can be fooled --

    circulation   the in-plane texture's mean curl. A collapsed single-domain
                  state has curl ~ 0 while keeping a perfectly good core-like
                  m_z spot at the centre if one is imposed.
    core          peak |m_z| inside a core-sized disc, which a genuine vortex
                  holds near 1 and an in-plane state does not.

    python scripts/check_yig_vortex.py
    python scripts/check_yig_vortex.py --radii 150 200 250 300 --material yig
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.vortex import VortexConfig, VortexDisk

MU0 = 4e-7 * math.pi

MATERIALS = {
    # Ms (A/m), A (J/m), alpha, label
    "permalloy": dict(Ms=800e3, A=1.3e-11, alpha=0.008),
    "yig": dict(Ms=140e3, A=3.6e-12, alpha=1e-4),
    # thin-film YIG is routinely an order worse than bulk; carried as a
    # pessimistic bound so the memory claim is not resting on the best case
    "yig-film": dict(Ms=140e3, A=3.6e-12, alpha=1e-3),
}


def exchange_length(Ms, A):
    return math.sqrt(2 * A / (MU0 * Ms ** 2))


@torch.no_grad()
def vortex_survives(cfg, relax_steps, dtype=torch.float32):
    """Relax a vortex initial state; return circulation and core polarisation."""
    disk = VortexDisk(cfg, timesteps=10, dtype=dtype)
    t0 = time.time()
    disk.relax(steps=relax_steps)
    m0 = disk.m0
    n = cfg.n_cells
    idx = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    r = torch.sqrt(X ** 2 + Y ** 2)
    inside = (disk.mask[:, :, 0, 0] > 0)

    # circulation: projection of the in-plane texture onto the azimuthal
    # direction. +-1 is a clean vortex, ~0 a single domain.
    phi_x, phi_y = -Y / r.clamp_min(1e-30), X / r.clamp_min(1e-30)
    mx, my = m0[:, :, 0, 0].double(), m0[:, :, 0, 1].double()
    ring = inside & (r > 0.35 * cfg.radius) & (r < 0.9 * cfg.radius)
    circ = float(((mx * phi_x + my * phi_y)[ring]).mean()) if ring.any() else 0.0

    core_r = max(cfg.core_width, 3 * cfg.dx)
    core = inside & (r <= core_r)
    core_mz = float(m0[:, :, 0, 2].double()[core].abs().max()) if core.any() else 0.0
    return circ, core_mz, time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", default="yig", choices=sorted(MATERIALS))
    p.add_argument("--radii", type=float, nargs="+",
                   default=[100, 150, 200, 250, 300])
    p.add_argument("--thickness", type=float, default=20.0)
    p.add_argument("--dx", type=float, default=None,
                   help="nm; defaults to l_ex/2, which is the resolution the "
                        "material demands rather than one inherited from "
                        "permalloy")
    p.add_argument("--relax-steps", type=int, default=3000)
    p.add_argument("--outdir", default="runs/yig_vortex")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mat = MATERIALS[args.material]
    lex = exchange_length(mat["Ms"], mat["A"]) * 1e9
    dx = args.dx if args.dx is not None else max(round(lex / 2), 4)

    print(f"{args.material}: Ms {mat['Ms']/1e3:.0f} kA/m, A {mat['A']:.2e} J/m, "
          f"alpha {mat['alpha']:.0e}")
    print(f"exchange length {lex:.1f} nm  (permalloy 5.7 nm), mesh dx {dx:.0f} nm")
    tau_frames = 1.0 / (2 * math.pi * mat["alpha"] * 2.4)
    print(f"ring-down at 2.4 cycles/frame: {tau_frames:.0f} frames "
          f"(permalloy 8.3, measured 2.6)\n")

    print(f"{'R (nm)':>7} {'R/l_ex':>7} {'circulation':>12} {'core |mz|':>10} "
          f"{'verdict':>10}")
    rows = []
    for R in args.radii:
        cfg = VortexConfig(radius=R * 1e-9, thickness=args.thickness * 1e-9,
                           dx=dx * 1e-9, Ms=mat["Ms"], A=mat["A"],
                           alpha=mat["alpha"],
                           core_width=max(2 * lex, 10.0) * 1e-9)
        circ, core, secs = vortex_survives(cfg, args.relax_steps)
        ok = abs(circ) > 0.6 and core > 0.5
        rows.append({"radius_nm": R, "r_over_lex": R / lex, "circulation": circ,
                     "core_mz": core, "vortex": bool(ok), "seconds": secs})
        print(f"{R:>7.0f} {R/lex:>7.1f} {circ:>12.3f} {core:>10.3f} "
              f"{'VORTEX' if ok else 'collapsed':>10}", flush=True)

    (outdir / f"results_{args.material}.json").write_text(
        json.dumps({"material": args.material, "l_ex_nm": lex, "dx_nm": dx,
                    "rows": rows}, indent=2))

    good = [r for r in rows if r["vortex"]]
    print()
    if good:
        rmin = min(r["radius_nm"] for r in good)
        print(f"Vortex holds from R = {rmin:.0f} nm ({rmin/lex:.1f} l_ex).")
        print(f"Footprint per ported disk would be "
              f"{2*(rmin+150):.0f} nm against 500 nm in permalloy.")
        print("Next: the mode ladder and the guide band both scale with Ms, so")
        print("expect them ~5.7x lower -- around 1.6-2.5 GHz rather than")
        print("9-14 GHz. Every frequency in the design has to be re-measured.")
    else:
        print("No radius tested holds a vortex. Either go larger, or YIG's low")
        print("Ms makes flux closure too weak at this thickness and the vortex")
        print("reservoir does not transfer to it -- in which case the memory")
        print("gain is unreachable by this route.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
