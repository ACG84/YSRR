#!/usr/bin/env python3
"""Is 12 GHz inside the strip's propagating band at all?

The near-field test showed the taps read a field that decays over a few hundred
nm and arrives everywhere at once: flat to 0.04 frames over 1701 nm where the
fitted 945 m/s demands 9.0. That is what an evanescent field does, and it leaves
one prior question the whole delay-line architecture rested on without ever
asking it -- whether the bus supports a travelling wave at the drive frequency.

Geometry says it might not. The bus is a bare 80 x 20 nm permalloy strip with NO
applied field, so shape anisotropy puts the ground state along the strip and the
drive wavevector runs along the magnetisation. That is the BACKWARD-VOLUME
geometry, whose magnetostatic band lies BELOW the k = 0 resonance, not above it.
A drive above that resonance has no real k to couple to and can only produce a
localised, non-propagating response -- which is what was measured.

This measures the dispersion directly rather than arguing from a formula that
was derived for the other geometry (magnonic_nn.dispersion is Damon-Eshbach:
k PERPENDICULAR to M, which is not this).

Method: excite a two-cell line source with a temporal sinc -- flat spectrum to
`--fc` GHz, broad in k because the source is narrow in x -- record the
width-averaged out-of-plane deviation at every x, and take the 2D FFT. A
travelling mode appears as a sharp ridge in (k, f); a frequency with no mode
appears as power smeared over all k, because a response localised at the source
is broad in k by construction.

Two independent read-outs per frequency, because ridge-finding alone invites
the same estimator failure that has already produced one false delay here:

  k_peak, contrast   from the (k, f) map. Contrast is peak-over-median across
                     k, so a real ridge scores high and a smeared response
                     scores near 1 whatever the absolute power.
  decay length       from |M(x, f)| along the strip, fitted away from the
                     source. This needs no peak-finding at all, and it is the
                     quantity the architecture actually cares about: a delay
                     line has to carry microns.

The two must agree. Where they do, v_g follows from the slope of the ridge and
gives the tap spacing directly -- the number the whole poly-tap design was
built on and, on present evidence, got wrong.

    python scripts/check_bus_dispersion.py                    # as built
    python scripts/check_bus_dispersion.py --width 240        # wider strip
    python scripts/check_bus_dispersion.py --bias-mT 50       # biased along x
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0, MeshConfig, SolverConfig
from magnonic_nn.solver import LLGRollout
from magnonic_nn.vortex import VortexConfig
from magnonic_nn._compat import get_device

FRAME_STEPS = 200


def build(cfg, length, width, absorb, dtype):
    """A straight strip with absorbing ends -- same construction as the
    transport measurement, so the two are comparable."""
    dx = cfg.dx
    nx = int(round(length / dx))
    ny = int(round(width / dx)) + 16
    mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=dx, dy=dx, dz=cfg.thickness)
    y = (torch.arange(ny, dtype=dtype) - (ny - 1) / 2) * dx
    x = torch.arange(nx, dtype=dtype) * dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    mask = (Y.abs() <= width / 2).to(dtype).reshape(nx, ny, 1, 1)
    ramp_l = ((absorb - X) / absorb).clamp(0.0, 1.0) ** 2
    ramp_r = ((X - (length - absorb)) / absorb).clamp(0.0, 1.0) ** 2
    ramp = torch.maximum(ramp_l, ramp_r).reshape(nx, ny, 1, 1)
    alpha = (cfg.alpha + (0.5 - cfg.alpha) * ramp) * mask
    return mesh, mask, alpha, nx, ny, X


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=float, default=5000.0, help="nm")
    p.add_argument("--width", type=float, default=80.0, help="nm")
    p.add_argument("--absorb", type=float, default=400.0, help="nm, each end")
    p.add_argument("--src-len", type=float, default=10.0,
                   help="nm; narrow on purpose -- the source's own k-spectrum\n"
                        "is flat only out to about 2*pi/src_len, and anything\n"
                        "beyond that the map cannot see because nothing drove it")
    p.add_argument("--amp-mT", type=float, default=5.0,
                   help="small: this is a linear-response measurement, and the\n"
                        "disk was already shown to compress by 24% at 30 mT")
    p.add_argument("--fc", type=float, default=50.0,
                   help="GHz; sinc cutoff, so the drive is flat to here")
    p.add_argument("--steps", type=int, default=8192,
                   help="8192 ps gives 122 MHz of frequency resolution")
    p.add_argument("--t0-ps", type=float, default=1000.0,
                   help="sinc centre; the source is symmetric about it")
    p.add_argument("--bias-mT", type=float, default=0.0,
                   help="static field along the strip (x). The device as built\n"
                        "has none; this exists to test whether a bias moves the\n"
                        "band somewhere useful")
    p.add_argument("--relax-steps", type=int, default=3000)
    p.add_argument("--fmax", type=float, default=40.0, help="GHz, for the report")
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="runs/bus_dispersion")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = VortexConfig()
    L, W, AB = a.length * 1e-9, a.width * 1e-9, a.absorb * 1e-9

    mesh, mask, alpha, nx, ny, X = build(cfg, L, W, AB, dtype)
    solver = SolverConfig(dt=cfg.dt, timesteps=a.steps + 8, checkpoint=False,
                          renormalize=True, demag=True)
    roll = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha, Ms_ref=cfg.Ms)
    roll.set_Ms(cfg.Ms * mask)

    h_static = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    if a.bias_mT:
        h_static[:, :, :, 0] = (a.bias_mT * 1e-3 / MU_0) * mask[:, :, :, 0]
    tag = f"{int(a.length)}_{int(a.width)}_b{a.bias_mT:g}"
    print(f"strip {a.length:.0f} x {a.width:.0f} nm, mesh {nx}x{ny}, "
          f"bias {a.bias_mT:g} mT, dt {cfg.dt*1e12:.2f} ps, device {a.device}")

    m0c = outdir / f"m0_{tag}.pt"
    if m0c.exists():
        m0 = torch.load(m0c, weights_only=False).to(device=get_device(), dtype=dtype)
        print(f"[m0] restored from {m0c.name}")
    else:
        m = torch.zeros(nx, ny, 1, 3, dtype=dtype)
        m[:, :, 0, 0] = 1.0                    # shape anisotropy puts it here
        m = m * mask
        t0 = time.time()
        m0 = roll.relax(m, h_static, a.relax_steps, 0.5)
        torch.save(m0.cpu(), m0c)
        print(f"relaxed in {time.time()-t0:.0f}s")
    mx0 = float((m0[:, :, 0, 0] * mask[:, :, 0, 0]).sum() / mask.sum())
    print(f"ground state mean m_x = {mx0:.4f} "
          f"({'along the strip' if mx0 > 0.9 else 'NOT uniform -- check'})")

    # A narrow line source in the middle. Narrow in x so the excitation is broad
    # in k; out of plane because that is what the real injector does.
    xc = L / 2
    seg = ((X >= xc - a.src_len * 1e-9 / 2) &
           (X <= xc + a.src_len * 1e-9 / 2)).reshape(nx, ny, 1)
    unit = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    unit[:, :, :, 2] = (seg * mask[:, :, :, 0].bool()).to(dtype)
    n_src = int(seg[:, 0, 0].sum())
    print(f"source {n_src} cells wide at x = {xc*1e9:.0f} nm, "
          f"sinc flat to {a.fc:.0f} GHz")

    amp = a.amp_mT * 1e-3 / MU_0
    wc = 2 * math.pi * a.fc * 1e9
    t0s = a.t0_ps * 1e-12

    def sinc_h(t):
        z = wc * (t - t0s)
        return amp * (1.0 if abs(z) < 1e-12 else math.sin(z) / z)

    per_x = mask[:, :, 0, 0].sum(dim=1).clamp(min=1)
    rec = np.zeros((a.steps, nx), dtype=np.float32)
    m = m0.clone()
    t_start = time.time()
    with torch.no_grad():
        for k in range(a.steps):
            tk = k * cfg.dt

            def h(theta, tk=tk):
                return unit * sinc_h(tk + theta * cfg.dt)

            m = roll.rk4_step(m, h_static, h)
            dmz = (m - m0)[:, :, 0, 2] * mask[:, :, 0, 0]
            rec[k] = (dmz.sum(dim=1) / per_x).float().cpu().numpy()
            if (k + 1) % 1000 == 0:
                print(f"  step {k+1}/{a.steps} ({time.time()-t_start:.0f}s)",
                      flush=True)

    # --- 2D FFT over the interior, away from the absorbers ---------------
    dx = cfg.dx
    i0 = int(round((AB + 200e-9) / dx))
    i1 = int(round((L - AB - 200e-9) / dx))
    seg_x = rec[:, i0:i1]
    nt, nxu = seg_x.shape
    wt = np.hanning(nt)[:, None]
    wx = np.hanning(nxu)[None, :]
    F = np.fft.fftshift(np.fft.fft2(seg_x * wt * wx), axes=(0, 1))
    freqs = np.fft.fftshift(np.fft.fftfreq(nt, cfg.dt)) / 1e9        # GHz
    kk = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nxu, dx))        # rad/m
    P = np.abs(F)
    pos = freqs > 0
    freqs, P = freqs[pos], P[pos]

    # --- decay length per frequency, independent of any peak finding -----
    # One-sided: from 200 nm past the source out to the absorber. A single
    # temporal DFT per x, so this shares no machinery with the map above.
    js = int(round((xc + 200e-9) / dx))
    je = int(round((L - AB - 100e-9) / dx))
    xs = (np.arange(js, je) * dx - xc)                                # metres
    tw = np.hanning(a.steps)[:, None]
    Fx = np.fft.rfft(rec[:, js:je] * tw, axis=0)
    fx = np.fft.rfftfreq(a.steps, cfg.dt) / 1e9

    def decay_len_at(f_ghz):
        j = int(np.argmin(np.abs(fx - f_ghz)))
        A = np.abs(Fx[j])
        good = A > 0
        if good.sum() < 10:
            return float("nan"), 0.0
        lg = np.log(A[good])
        s, c = np.polyfit(xs[good], lg, 1)
        pred = np.polyval((s, c), xs[good])
        ss = 1.0 - ((lg - pred) ** 2).sum() / max(((lg - lg.mean()) ** 2).sum(), 1e-30)
        return (float("inf") if s >= 0 else -1.0 / s), float(ss)

    print(f"\n{'f_GHz':>7} {'k_peak':>11} {'lambda_nm':>10} {'contrast':>9} "
          f"{'decay_nm':>10} {'fit_R2':>7}")
    rows = []
    for f_t in np.arange(2.0, a.fmax + 0.01, 1.0):
        j = int(np.argmin(np.abs(freqs - f_t)))
        row = P[j]
        ip = int(np.argmax(row))
        kp = float(kk[ip])
        contrast = float(row[ip] / max(np.median(row), 1e-30))
        dl, r2 = decay_len_at(f_t)
        lam = float("inf") if abs(kp) < 1e-9 else 2 * np.pi / abs(kp) * 1e9
        rows.append({"f_GHz": float(f_t), "k_peak": kp, "lambda_nm": lam,
                     "contrast": contrast, "decay_nm": dl * 1e9, "fit_R2": r2,
                     "power": float(row[ip])})
        print(f"{f_t:>7.1f} {kp:>11.3e} {lam:>10.1f} {contrast:>9.1f} "
              f"{dl*1e9:>10.1f} {r2:>7.3f}")

    (outdir / f"results_{tag}.json").write_text(json.dumps(
        {"args": vars(a), "rows": rows}, indent=2))

    # --- what the numbers mean -------------------------------------------
    # A propagating mode has to carry MICRONS to be a delay line; a decay
    # length under the tap spacing is not a weak wave, it is not a wave.
    prop = [r for r in rows if r["decay_nm"] > 1000.0 and r["contrast"] > 3.0]
    print()
    if prop:
        lo, hi = prop[0]["f_GHz"], prop[-1]["f_GHz"]
        print(f"Propagating band (decay > 1 um and a resolved ridge): "
              f"{lo:.0f} - {hi:.0f} GHz")
        # group velocity along the ridge, where it exists
        fs = np.array([r["f_GHz"] for r in prop]) * 1e9
        ks = np.array([abs(r["k_peak"]) for r in prop])
        if len(fs) > 2:
            dk = np.gradient(ks)
            vg = 2 * np.pi * np.gradient(fs) / np.where(np.abs(dk) < 1e-9,
                                                        np.nan, dk)
            for r, v in zip(prop, vg):
                if np.isfinite(v):
                    r["v_g_m_s"] = float(v)
            fin = [r for r in prop if np.isfinite(r.get("v_g_m_s", np.nan))]
            if fin:
                print(f"\n{'f_GHz':>7} {'v_g_m_s':>11} {'nm_per_frame':>13} "
                      f"{'decay_nm':>10}")
                for r in fin:
                    print(f"{r['f_GHz']:>7.1f} {r['v_g_m_s']:>11.1f} "
                          f"{abs(r['v_g_m_s'])*FRAME_STEPS*cfg.dt*1e9:>13.1f} "
                          f"{r['decay_nm']:>10.1f}")
    else:
        print("NO frequency in the scanned range both resolves a ridge and\n"
              "carries a micron. On this strip, at this bias, there is no\n"
              "delay line to be had at any drive frequency.")

    at12 = min(rows, key=lambda r: abs(r["f_GHz"] - 12.0))
    print(f"\nAt the drive frequency actually used, 12 GHz: contrast "
          f"{at12['contrast']:.1f}, decay {at12['decay_nm']:.0f} nm.")
    if at12["decay_nm"] < 1000.0:
        print("That is the measurement the whole poly-tap campaign needed\n"
              "first. A tap array spanning 1.7 um cannot be fed by this.")

    # The launcher is a separate failure mode from the band, and the map can
    # only report what the source excited: a 100 nm segment has its first
    # spectral zero at 2*pi/100nm, so modes needing shorter wavelengths are
    # invisible to it however well the strip carries them.
    k_launch = 2 * np.pi / 100e-9
    print(f"\nLauncher check: the 100 nm injector is flat in k only to "
          f"{k_launch:.2e} rad/m\n(wavelengths down to 100 nm). Modes needing "
          f"shorter wavelengths than that\nare not driven, however well the "
          f"strip would carry them.")
    print(f"\nwrote {outdir / f'results_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
