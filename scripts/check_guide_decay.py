#!/usr/bin/env python3
"""Does the guide propagate? Measured as a decay length, not a transmission.

The transmission ratio used earlier compared power in a slice near the source
against a slice at the far end. That was fragile in two ways and both bit: at
40 nm width the near slice sat partly inside the source's near field (so a
sub-cutoff guide read 0.0038 rather than a clean zero), and at 80 nm the mesh
grew while the slice indices did not, producing ratios above 1.0 -- which is
impossible for a passive guide and made the whole row uninterpretable.

A decay length has neither problem. Drive one end, let it reach steady state,
and fit log|m_z(x)| along the guide. A propagating mode decays only through
damping, giving lengths of order v_g/(alpha*omega) -- microns here. An
evanescent mode decays as exp(-kappa x) with kappa set by how far below cutoff
it is, giving tens of nanometres. The two regimes differ by orders of
magnitude, so the measurement does not need to be precise to be decisive, and
it is independent of source coupling, guide area and mode profile -- the three
things that broke the ratio.

    python scripts/check_guide_decay.py
    python scripts/check_guide_decay.py --widths 40 80 --thicknesses 5
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0, MeshConfig, SolverConfig
from magnonic_nn.solver import LLGRollout


@torch.no_grad()
def decay_length_nm(freq, width_nm, thickness_nm, length_nm=900.0, steps=1600,
                    amp_mT=3.0, dx=5e-9, margin=6, dtype=torch.float32):
    """Fit log|m_z(x)| along a straight guide; return the 1/e length in nm."""
    nx = int(round(length_nm * 1e-9 / dx)) + 2 * margin
    ny = int(round(width_nm * 1e-9 / dx)) + 2 * margin
    x = (torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2) * dx
    y = (torch.arange(ny, dtype=torch.float64) - (ny - 1) / 2) * dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    L = length_nm * 1e-9
    mask = ((X.abs() <= L / 2) & (Y.abs() <= width_nm * 1e-9 / 2)).to(dtype)
    mask = mask.reshape(nx, ny, 1, 1)

    mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=dx, dy=dx, dz=thickness_nm * 1e-9)
    solver = SolverConfig(dt=1e-12, timesteps=steps, checkpoint=False,
                          renormalize=True, demag=True)
    alpha = torch.full((nx, ny, 1, 1), 0.008, dtype=dtype)
    roll = LLGRollout(mesh, solver, A=1.3e-11, alpha=alpha, Ms_ref=800e3)
    roll.set_Ms(800e3 * mask)
    h0 = torch.zeros(nx, ny, 1, 3, dtype=dtype)

    m = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    m[:, :, 0, 0] = 1.0
    m0 = roll.relax(m * mask, h0, 900, 0.5)

    src = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    src[margin:margin + 3, :, 0, 2] = mask[margin:margin + 3, :, 0, 0]
    amp = amp_mT * 1e-3 / MU_0

    mm = m0.clone()
    acc = torch.zeros(nx, dtype=dtype)
    n_acc = 0
    for k in range(steps):
        t0 = k * 1e-12
        def h(theta, t0=t0):
            return src * (amp * math.sin(2 * math.pi * freq * (t0 + theta * 1e-12)))
        mm = roll.rk4_step(mm, h0, h)
        if k > steps // 2:                      # steady state only
            acc += ((mm - m0)[:, :, 0, 2] ** 2).sum(dim=1)
            n_acc += 1
    rms = (acc / max(n_acc, 1)).sqrt()

    # fit over the guide interior, clear of the source and the far end
    lo = margin + int(80e-9 / dx)
    hi = nx - margin - int(60e-9 / dx)
    xs = (torch.arange(lo, hi, dtype=torch.float64) * dx * 1e9).numpy()
    ys = rms[lo:hi].double().clamp_min(1e-30).log().numpy()
    good = np.isfinite(ys) & (ys > ys.max() - 25)     # ignore the numerical floor
    if good.sum() < 8:
        return float("nan"), float(rms.max())
    slope = np.polyfit(xs[good], ys[good], 1)[0]
    return (float("inf") if slope >= 0 else -1.0 / slope), float(rms.max())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--widths", type=float, nargs="+", default=[40, 80, 120])
    p.add_argument("--thicknesses", type=float, nargs="+", default=[5])
    p.add_argument("--freqs", type=float, nargs="+", default=[6, 8, 10, 14, 18])
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--outdir", default="runs/guide_decay")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / "journal.jsonl"
    done = {}
    if jpath.exists():
        for line in jpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["w"], r["t"], r["ghz"])] = r["decay_nm"]

    print("1/e decay length in nm along a 900 nm guide "
          "(> ~450 means it crosses; < ~50 is evanescent)\n")
    print(f"{'w x t':>10}   " + "  ".join(f"{f:>7.0f}" for f in args.freqs))
    for t in args.thicknesses:
        for w in args.widths:
            cells = []
            for f in args.freqs:
                key = (w, t, f)
                if key in done:
                    cells.append(done[key]); continue
                d, _ = decay_length_nm(f * 1e9, w, t, steps=args.steps)
                with jpath.open("a") as fh:
                    fh.write(json.dumps({"w": w, "t": t, "ghz": f,
                                         "decay_nm": d}) + "\n")
                cells.append(d)
            txt = "  ".join("    inf" if math.isinf(c) else
                            ("    nan" if math.isnan(c) else f"{c:>7.0f}")
                            for c in cells)
            print(f"{w:>4.0f} x {t:<3.0f}   {txt}", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    prop = [r for r in rows if (math.isinf(r["decay_nm"]) or r["decay_nm"] > 450)]
    print()
    if prop:
        low = min(r["ghz"] for r in prop)
        best = min((r for r in prop if r["ghz"] == low), key=lambda r: r["w"])
        print(f"Lowest propagating frequency {low:.0f} GHz, at "
              f"{best['w']:.0f} x {best['t']:.0f} nm.")
        if low <= 10:
            print("That reaches the disk's low-order azimuthal modes, so the guides")
            print("carry the modes that actually hold the computation. Mode")
            print("selectivity at this width still has to be re-measured.")
        else:
            print("Still above the disk's low-order modes -- the guides would carry")
            print("only the high-n tail of the spectrum.")
    else:
        print("Nothing propagated at any geometry tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
