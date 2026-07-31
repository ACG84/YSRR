#!/usr/bin/env python3
"""Does the guide propagate? Decay length and wavelength, on a terminated guide.

Three measurement designs preceded this one and each failed on a different
property of sub-cutoff behaviour:

  transmission ratio   near slice sat in the source's near field, and the slice
                       indices did not scale with the mesh, giving ratios > 1
  decay length         below cutoff the guide responds weakly but *uniformly*,
                       and a flat profile fits as infinite decay
  phase slope          reported finite wavelengths everywhere, including where
                       transmission said nothing was getting through

The phase-slope run's own numbers showed why: 80 x 5 nm returned a decay length
of 293 um at 8 GHz -- three hundred times what alpha = 0.008 permits -- and
wavelengths of 1271, 663, 130 nm at 6, 8, 10 GHz, which is not a dispersion
relation. Both are the signature of a REFLECTION. The guide simply ended, so a
propagating mode bounced off the terminus and set up a standing wave: a standing
wave transports no net power (no decay) and its phase advances in steps rather
than linearly (nonsense wavelengths). The measurement was being corrupted
precisely in the propagating case it existed to detect.

So the guide is terminated in a graded damping ramp -- the same absorbing
boundary used everywhere else in this codebase -- and the fit window is required
to hold at least ``--min-cycles`` wavelengths, since a wavelength longer than
the window is fitted, not measured.

The quantity reported is decay/wavelength: how many oscillations the mode
completes before dying. Below cutoff that is order unity, evanescent by any
name. A real propagating mode gives tens.

    python scripts/check_guide_decay.py
    python scripts/check_guide_decay.py --widths 40 80 --freqs 6 10
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0, MeshConfig, SolverConfig
from magnonic_nn.solver import LLGRollout

ALPHA_ABS = 0.5

# Material sets the band as much as geometry does: the strip cutoff goes as Ms,
# so YIG at 140 kA/m puts it ~5.7x below permalloy's. Hardcoding permalloy is
# how the guides came to be characterised at a thickness the device never had.
MATERIALS = {
    "permalloy": dict(Ms=800e3, A=1.3e-11, alpha=0.008),
    "yig": dict(Ms=140e3, A=3.6e-12, alpha=1e-4),
    "yig-film": dict(Ms=140e3, A=3.6e-12, alpha=1e-3),
}
ALPHA_BULK = MATERIALS["permalloy"]["alpha"]


@torch.no_grad()
def guide_response(freq, width_nm, thickness_nm, length_nm=1400.0,
                   absorber_nm=300.0, steps=2000, amp_mT=3.0, dx=5e-9,
                   margin=6, min_cycles=1.5, dtype=torch.float32,
                   material="permalloy"):
    """Drive one end of a terminated guide; fit decay length and wavelength.

    Returns ``(decay_nm, wavelength_nm, amp, cycles_in_window)``. ``amp`` is
    per-cell so it is comparable across widths -- summing over y made a wide
    guide look like a strong one.
    """
    nx = int(round(length_nm * 1e-9 / dx)) + 2 * margin
    ny = int(round(width_nm * 1e-9 / dx)) + 2 * margin
    x = (torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2) * dx
    y = (torch.arange(ny, dtype=torch.float64) - (ny - 1) / 2) * dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    L = length_nm * 1e-9
    mask = ((X.abs() <= L / 2) & (Y.abs() <= width_nm * 1e-9 / 2)).to(dtype)
    n_wide = int(mask[nx // 2].sum().item())          # cells across the guide
    mask = mask.reshape(nx, ny, 1, 1)

    mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=dx, dy=dx, dz=thickness_nm * 1e-9)
    solver = SolverConfig(dt=1e-12, timesteps=steps, checkpoint=False,
                          renormalize=True, demag=True)

    # Terminate the far end. Quadratic grading rather than a step, so the
    # absorber itself does not reflect what it is there to absorb.
    mat = MATERIALS[material]
    a_bulk = mat["alpha"]
    n_abs = int(round(absorber_nm * 1e-9 / dx))
    alpha = torch.full((nx, ny, 1, 1), a_bulk, dtype=dtype)
    if n_abs > 0:
        ramp = torch.linspace(0.0, 1.0, n_abs, dtype=dtype) ** 2
        a0 = nx - margin - n_abs
        alpha[a0:nx - margin, :, 0, 0] = (
            a_bulk + ramp[:, None] * (ALPHA_ABS - a_bulk))

    roll = LLGRollout(mesh, solver, A=mat["A"], alpha=alpha, Ms_ref=mat["Ms"])
    roll.set_Ms(mat["Ms"] * mask)
    h0 = torch.zeros(nx, ny, 1, 3, dtype=dtype)

    m = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    m[:, :, 0, 0] = 1.0
    m0 = roll.relax(m * mask, h0, 900, 0.5)

    src = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    src[margin:margin + 3, :, 0, 2] = mask[margin:margin + 3, :, 0, 0]
    amp = amp_mT * 1e-3 / MU_0

    # Lock-in at the drive frequency for the COMPLEX amplitude per x. Phase is
    # what settles propagation; decay length alone cannot, since a weak uniform
    # response fits as infinite decay. Phase advance is amplitude-independent.
    #
    # Accumulated in two separate quarters, because "long enough to settle" is
    # not something to guess at. The smoke test guessed wrong: at 800 steps the
    # wavefront covers ~920 nm of a 1400 nm guide, so the far end was dark
    # because the wave had not ARRIVED, and that fits as a 157 nm decay length
    # against the ~1280 nm damping alone allows at this group velocity. A guide
    # that has reached steady state gives the same profile in both quarters; one
    # still filling does not.
    mm = m0.clone()
    acc = [torch.zeros(nx, dtype=torch.complex128) for _ in range(2)]
    n_acc = [0, 0]
    q3 = steps // 2
    for k in range(steps):
        t0 = k * 1e-12

        def h(theta, t0=t0):
            return src * (amp * math.sin(2 * math.pi * freq * (t0 + theta * 1e-12)))

        mm = roll.rk4_step(mm, h0, h)
        if k >= q3:
            prof = (mm - m0)[:, :, 0, 2].double().sum(dim=1) / max(n_wide, 1)
            ph = 2 * math.pi * freq * (k * 1e-12)
            j = 0 if k < q3 + (steps - q3) // 2 else 1
            acc[j] += prof * complex(math.cos(ph), math.sin(ph))
            n_acc[j] += 1
    q_a = (acc[0] / max(n_acc[0], 1)).numpy()
    q_b = (acc[1] / max(n_acc[1], 1)).numpy()
    A = 0.5 * (q_a + q_b)
    den = np.abs(q_a).max() + np.abs(q_b).max()
    settle = float(np.abs(np.abs(q_a) - np.abs(q_b)).max() / max(den, 1e-30))

    # Fit clear of the source's near field and clear of the absorber.
    lo = margin + int(round(150e-9 / dx))
    hi = nx - margin - n_abs - int(round(50e-9 / dx))
    if hi - lo < 20:
        raise ValueError("fit window too short; lengthen the guide")
    win = A[lo:hi]
    xs = (np.arange(lo, hi) * dx * 1e9)
    span_nm = float(xs[-1] - xs[0])
    mag = np.abs(win)

    ys = np.log(np.clip(mag, 1e-30, None))
    good = np.isfinite(ys) & (mag > mag.max() * 1e-4)   # above the numeric floor
    if good.sum() < 20:
        return dict(decay_nm=float("nan"), wavelength_nm=float("nan"),
                    amp=float(mag.max()), cycles=0.0, settle=settle)
    slope = np.polyfit(xs[good], ys[good], 1)[0]
    decay = float("inf") if slope >= 0 else -1.0 / slope

    phase = np.unwrap(np.angle(win))
    kfit = np.polyfit(xs[good], phase[good], 1)[0]      # radians per nm
    lam = abs(2 * math.pi / kfit) if abs(kfit) > 1e-9 else float("inf")
    cycles = span_nm / lam if math.isfinite(lam) and lam > 0 else 0.0
    # A wavelength longer than the window was fitted, not measured.
    if cycles < min_cycles:
        lam = float("nan")
    return dict(decay_nm=decay, wavelength_nm=lam, amp=float(mag.max()),
                cycles=cycles, settle=settle)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--widths", type=float, nargs="+", default=[40, 80, 120])
    p.add_argument("--thicknesses", type=float, nargs="+", default=[5])
    p.add_argument("--freqs", type=float, nargs="+", default=[6, 8, 10, 14, 18])
    p.add_argument("--length", type=float, default=1400.0, help="guide length nm")
    p.add_argument("--absorber", type=float, default=300.0, help="terminus nm")
    p.add_argument("--min-cycles", type=float, default=1.5)
    p.add_argument("--settle-tol", type=float, default=0.10,
                   help="max profile drift between the last two quarters")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--material", default="permalloy", choices=sorted(MATERIALS))
    p.add_argument("--outdir", default="runs/guide_decay_term")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / "journal.jsonl"
    done = {}
    if jpath.exists():
        for line in jpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["w"], r["t"], r["ghz"])] = r

    print("Guide terminated in a graded absorber, so a propagating mode is not\n"
          "reflected back on itself. Reported: decay/wavelength -- oscillations\n"
          "completed before the mode dies. Order 1 is evanescent; tens is a\n"
          "real propagating mode. '--' means the wavelength exceeded the fit\n"
          f"window ({args.min_cycles:g} cycles required) and was not measured;\n"
          "'~' means the guide had not reached steady state, so the far end was\n"
          "dark because the wave had not arrived rather than from damping.\n")
    print(f"{'w x t':>10}   " + "  ".join(f"{f:>7.0f}" for f in args.freqs))

    def qual(r):
        """decay/wavelength, or None if this entry cannot be trusted."""
        lam, d = r.get("wavelength_nm"), r["decay_nm"]
        if r.get("settle", 0.0) > args.settle_tol:
            return None
        if lam is None or math.isnan(lam) or lam <= 0:
            return None
        return float("inf") if math.isinf(d) else d / lam

    for t in args.thicknesses:
        for w in args.widths:
            cells = []
            for f in args.freqs:
                key = (w, t, f)
                if key in done:
                    cells.append(done[key]); continue
                rec = guide_response(f * 1e9, w, t, length_nm=args.length,
                                     absorber_nm=args.absorber, steps=args.steps,
                                     min_cycles=args.min_cycles)
                rec.update({"w": w, "t": t, "ghz": f})
                with jpath.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                cells.append(rec)
            txt = []
            for r in cells:
                if r.get("settle", 0.0) > args.settle_tol:
                    txt.append("       ~")
                    continue
                q = qual(r)
                if q is None:
                    txt.append("      --")
                else:
                    txt.append("    inf" if math.isinf(q) else f"{q:>7.1f}")
            print(f"{w:>4.0f} x {t:<3.0f}   " + "  ".join(txt), flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    unsettled = sum(1 for r in rows if r.get("settle", 0.0) > args.settle_tol)
    if unsettled:
        print(f"\n{unsettled}/{len(rows)} points had not settled at "
              f"{args.steps} steps -- rerun those with more.")
    prop = [r for r in rows if (q := qual(r)) is not None and q >= 5.0]
    print()
    if prop:
        low = min(r["ghz"] for r in prop)
        best = min((r for r in prop if r["ghz"] == low), key=lambda r: r["w"])
        print(f"Lowest propagating frequency {low:.0f} GHz, at "
              f"{best['w']:.0f} x {best['t']:.0f} nm "
              f"(lambda {best['wavelength_nm']:.0f} nm).")
        if low <= 10:
            print("That reaches the disk's low-order azimuthal modes, so the guides")
            print("carry the modes that hold the computation. Mode selectivity at")
            print("this width still has to be re-measured -- a wider guide subtends")
            print("more of the disk perimeter, and six 120 nm ports need 720 nm")
            print("against the disk's 628 nm circumference.")
        else:
            print("Still above the disk's low-order modes -- the guides would carry")
            print("only the high-n tail of the spectrum.")
    else:
        print("Nothing propagated at any geometry tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
