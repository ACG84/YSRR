#!/usr/bin/env python3
"""Can a plain waveguide carry the input far enough to tap it at many delays?

The capacity accounting says what the device is missing: ~40 independent
LONG-SEPARATION product dimensions, against the ~5 it has spare. Products
between samples one or two frames apart are worthless on NARMA-10 (measured:
every short-separation family sits at the 0.124 baseline while far pairs score
0.0401), and the serial chain makes only the worthless kind.

The shape that would make the useful kind is a delay-line BUS tapped in
parallel: one guide carrying the input, tapped at many distances, each tap
feeding its own nonlinear disk that also receives the fresh sample. Each disk
then mixes a different delay against the present, so the products land at many
separations instead of one.

Two measurements gate that design, and both are about the bus rather than the
disks:

  attenuation   the delayed copy has to arrive comparable in amplitude to the
                fresh injection at the same tap. The chain failed exactly here
                -- transfer is 0.092 per hop THROUGH a disk, so by stage 3 the
                delayed copy is ~100x down, and co-driving that stage with a
                full-amplitude fresh sample drowned it. Measured: co-drive cut
                degree-1 capacity from 8.91 to 4.95 at matched sample size and
                pulled the memory horizon in from lag 12 to lag 7. A bus avoids
                the disks in the path, so its loss is the guide's own -- but
                that number has never been measured over micron distances.

  group speed   the tap positions follow from it. At the 1173 m/s fitted from
                the chain, one frame of delay (0.2 ns) is 235 nm, so lags 5, 10
                and 15 sit at 1.17, 2.35 and 3.52 um. If the speed differs on a
                straight guide, every tap position moves.

A bare strip, driven near one end, tapped along its length. No disks, no vortex
-- this isolates the transport question from everything else.

    python scripts/check_bus_transport.py
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


def build(cfg, length, width, absorb, dtype):
    """A straight strip with absorbing ends. Returns (rollout, mask, geometry)."""
    dx = cfg.dx
    nx = int(round(length / dx))
    ny = int(round(width / dx)) + 16          # margin either side of the strip
    mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=dx, dy=dx, dz=cfg.thickness)

    y = (torch.arange(ny, dtype=dtype) - (ny - 1) / 2) * dx
    x = torch.arange(nx, dtype=dtype) * dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    mask = (Y.abs() <= width / 2).to(dtype).reshape(nx, ny, 1, 1)

    # Absorbers at BOTH ends. Without them the strip is a resonator and every
    # tap reads a standing wave set by the strip length rather than a travelling
    # wave -- the same failure the disk ports were given tapers to avoid.
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
    p.add_argument("--drive-at", type=float, default=600.0, help="nm")
    p.add_argument("--drive-len", type=float, default=100.0, help="nm")
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--steps", type=int, default=12000,
                   help="must be long enough for the FARTHEST tap to reach\n"
                        "steady state, not merely to be reached. The first run\n"
                        "of this used 4000 and averaged over the last third --\n"
                        "a window that began at 2.67 ns while the 2.7 um tap was\n"
                        "still arriving at 2.6 ns, so the far taps were measured\n"
                        "mid-transient and the fitted attenuation length came out\n"
                        "~2x too short.")
    p.add_argument("--fit-from", type=float, default=900.0,
                   help="nm from the drive; excludes the near field, where the\n"
                        "amplitude RISES with distance and has no business in an\n"
                        "exponential fit")
    p.add_argument("--relax-steps", type=int, default=3000)
    p.add_argument("--outdir", default="runs/bus_transport")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cfg = VortexConfig()
    L, W, AB = a.length * 1e-9, a.width * 1e-9, a.absorb * 1e-9

    mesh, mask, alpha, nx, ny, X = build(cfg, L, W, AB, dtype)
    solver = SolverConfig(dt=cfg.dt, timesteps=a.steps + 8, checkpoint=False,
                          renormalize=True, demag=True)
    roll = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha, Ms_ref=cfg.Ms)
    roll.set_Ms(cfg.Ms * mask)
    h_zero = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    print(f"strip {a.length:.0f} x {a.width:.0f} nm, mesh {nx}x{ny}, "
          f"dt {cfg.dt*1e12:.2f} ps")

    m0c = outdir / f"m0_{int(a.length)}_{int(a.width)}.pt"
    if m0c.exists():
        m0 = torch.load(m0c, weights_only=False).to(device=get_device(), dtype=dtype)
        print(f"[m0] restored from {m0c.name}")
    else:
        # Shape anisotropy puts the ground state along the strip.
        m = torch.zeros(nx, ny, 1, 3, dtype=dtype)
        m[:, :, 0, 0] = 1.0
        m = m * mask
        t0 = time.time()
        m0 = roll.relax(m, h_zero, a.relax_steps, 0.5)
        torch.save(m0.cpu(), m0c)
        print(f"relaxed in {time.time()-t0:.0f}s")

    # Drive a short transverse-field segment; read m_z deviation along the strip.
    unit = torch.zeros(nx, ny, 1, 3, dtype=dtype)
    seg = ((X >= a.drive_at * 1e-9) &
           (X <= (a.drive_at + a.drive_len) * 1e-9)).reshape(nx, ny, 1)
    unit[:, :, :, 2] = (seg * mask[:, :, :, 0].bool()).to(dtype)

    taps_nm = np.arange(a.drive_at + 150, a.length - a.absorb - 100, 150.0)
    tap_idx = [int(round(t * 1e-9 / cfg.dx)) for t in taps_nm]
    strip = mask[:, :, 0, 0].bool()

    amp = a.amp_mT * 1e-3 / MU_0
    w = 2 * math.pi * a.freq * 1e9
    m = m0.clone()
    sig = np.zeros((a.steps, len(tap_idx)))
    t0 = time.time()
    with torch.no_grad():
        for k in range(a.steps):
            tk = k * cfg.dt

            def h(theta, tk=tk):
                return unit * (amp * math.sin(w * (tk + theta * cfg.dt)))

            m = roll.rk4_step(m, h_zero, h)
            dmz = (m - m0)[:, :, 0, 2]
            for j, xi in enumerate(tap_idx):
                col = dmz[xi][strip[xi]]
                sig[k, j] = float(col.mean())
            if (k + 1) % 1000 == 0:
                print(f"  step {k+1}/{a.steps} ({time.time()-t0:.0f}s)", flush=True)

    dt_ns = cfg.dt * 1e9
    frame_ns = 200 * cfg.dt * 1e9
    # Steady-state amplitude from the last third; arrival from a 10% threshold
    # on the running envelope.
    win = max(4, int(round(1e3 / a.freq)))
    rows = []
    print(f"\n{'x_nm':>7} {'dist_nm':>8} {'amp':>11} {'rel':>8} "
          f"{'arrive_ns':>10} {'lag_frames':>11}")
    a0 = None
    for j, xnm in enumerate(taps_nm):
        s = sig[:, j]
        env = np.sqrt(np.convolve(s ** 2, np.ones(win) / win, mode="same"))
        ss = float(np.sqrt((s[-a.steps // 3:] ** 2).mean()))
        pk = float(env.max())
        idx = int(np.argmax(env > 0.1 * pk)) if (env > 0.1 * pk).any() else -1
        t_arr = idx * dt_ns if idx >= 0 else float("nan")
        if a0 is None:
            a0 = ss
        d = xnm - a.drive_at
        rows.append({"x_nm": float(xnm), "dist_nm": float(d), "amp": ss,
                     "rel": ss / max(a0, 1e-30), "arrive_ns": t_arr,
                     "lag_frames": t_arr / frame_ns})
        print(f"{xnm:>7.0f} {d:>8.0f} {ss:>11.4e} {ss/max(a0,1e-30):>8.4f} "
              f"{t_arr:>10.3f} {t_arr/frame_ns:>11.2f}")
    (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    # Fit only where the measurement is trustworthy: past the near field, and
    # above a noise floor taken from the quietest tap. Arrival times below that
    # floor are threshold crossings on numerical noise -- visible in the first
    # run as arrival times that FELL with distance.
    # Trustworthy = past the near field AND arriving later than the tap before
    # it. A tap whose arrival time DROPS with distance is triggering on
    # numerical noise, which is what the far taps did before the run was long
    # enough for them to reach steady state. (An amplitude-threshold floor was
    # tried first and was worse: scaled off the quietest tap it excluded almost
    # every tap once the run converged.)
    cand = [r for r in rows if r["dist_nm"] >= a.fit_from
            and np.isfinite(r["arrive_ns"])]
    ok, last = [], -1.0
    for r in cand:
        if r["arrive_ns"] > last:
            ok.append(r); last = r["arrive_ns"]
    print(f"\nfitting {len(ok)} of {len(rows)} taps "
          f"(past {a.fit_from:.0f} nm, arrival monotonic)")
    if len(ok) >= 3:
        d = np.array([r["dist_nm"] for r in ok]) * 1e-9
        lg = np.log(np.array([r["amp"] for r in ok]))
        s, _ = np.linalg.lstsq(np.vstack([d, np.ones_like(d)]).T, lg, rcond=None)[0]
        Ld = -1.0 / s if s < 0 else float("inf")
        t = np.array([r["arrive_ns"] for r in ok]) * 1e-9
        sv, _ = np.linalg.lstsq(np.vstack([d, np.ones_like(d)]).T, t, rcond=None)[0]
        v = 1.0 / sv if sv > 0 else float("nan")
        print(f"\nattenuation length {Ld*1e9:.0f} nm   "
              f"group velocity {v:.0f} m/s (chain fit gave 1173)")
        print(f"one frame of delay = {v*200*cfg.dt*1e9:.0f} nm")
        # Read the amplitudes off the MEASURED taps nearest each target delay
        # rather than extrapolating the fit, so the dynamic range quoted below
        # is data.
        print(f"\n{'lag (frames)':>13} {'tap at (nm)':>12} {'measured amp':>13} "
              f"{'vs lag-5':>9}")
        ref5 = None
        for lag in (5, 10, 15):
            dist = v * lag * 200 * cfg.dt * 1e9
            r = min(ok, key=lambda q: abs(q["dist_nm"] - dist))
            if ref5 is None:
                ref5 = r["amp"]
            print(f"{lag:>13} {r['dist_nm']:>12.0f} {r['amp']:>13.3e} "
                  f"{r['amp']/ref5:>9.3f}")
        r15 = min(ok, key=lambda q: abs(q["dist_nm"] - v * 15 * 200 * cfg.dt * 1e9))
        worst = r15["amp"] / ref5
        print()
        if worst > 0.05:
            print(f"The bus carries. At the lag-15 tap the delayed copy is still\n"
                  f"{worst:.3f} of the lag-5 tap, so a fresh injection scaled to\n"
                  f"match it is a sane thing to build -- which is exactly what the\n"
                  f"chain could not offer at 0.092 per hop.")
        else:
            print(f"The bus does NOT carry far enough: {worst:.4f} of the near tap\n"
                  f"survives to lag 15. Taps at long delay would need their fresh\n"
                  f"injection cut by the same factor, putting the product term\n"
                  f"below the noise -- the same failure as the chain, in a\n"
                  f"straighter line.")
    print(f"\nwrote {outdir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
