#!/usr/bin/env python3
"""Is core reversal a usable threshold element, or just a noise source?

The tapped bus is limited by one inequality: balancing the two operands caps the
fresh drive a tap can receive at ~1.6 mT, and the permalloy vortex disk does not
compress below ~10 mT. Nine configurations produced s[n]*s[n-k] = 0.000 at every
lag because of it.

A low-moment disk at matched R/l_ex (Ms 450 kA/m, radius 178 nm) relaxes to a
clean vortex -- circulation +0.968, |dm| = 0.0000 -- and REVERSES ITS CORE at
about 2 mT, which is within a factor of 1.25 of what the coupling delivers. That
is the first element here whose nonlinearity is reachable at its own operating
point, and its smooth response is unmeasurable precisely BECAUSE it switches.

Switching is worth more than smooth compression, not less. A memoryless
nonlinearity ahead of a linear reservoir is a Wiener system: it fills the
P2(s[n-k]) column and leaves the cross-lag families at zero, which is the
failure already measured. A hysteretic node is different. Core polarity
persists, the gyrotropic sense flips with it, so the response to a fresh sample
is p[n]*u[n] with p[n] set by input history -- a current-times-past product,
which is the thing the architecture cannot otherwise make. The nonlinearity has
to sit AFTER the memory, and a bistable node puts it there.

But only if the switching is USABLE, and three things have to hold:

  deterministic   the map from drive amplitude to final polarity must be a step,
                  not a scatter. A core that flips at 2, 6 and 15 mT but not 4,
                  8 or 10 is a random number generator with extra steps.
  persistent      polarity must survive after the drive stops. If it relaxes
                  back within a frame there is no memory to multiply against.
  reversible      it must be settable BOTH ways, or the element saturates after
                  one event and stops responding to input.

This measures all three. Each amplitude starts from the same relaxed state,
drives for `--frames-drive` frames, then holds in silence for `--frames-hold`,
and the polarity is read after the drive and again after the hold.

    python scripts/check_core_switching.py --radius 178 --Ms 450e3 --device cuda
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk
from magnonic_nn._compat import get_device

FRAME_STEPS = 200


def core_mz(m, body):
    """m_z at the core: the extremal out-of-plane cell inside the disk body."""
    mz = m[:, :, 0, 2] * body
    return float(mz.flatten()[int(torch.argmax(mz.abs()))])


@torch.no_grad()
def drive_and_hold(disk, m0, body, amp, freq, n_drive, n_hold, dtype):
    """Drive from m0 for n_drive frames, then hold silent for n_hold.

    Returns (core polarity after the drive, after the hold, |core m_z| after).
    Every amplitude starts from the SAME state, so the map from amplitude to
    outcome is a function of amplitude alone.
    """
    unit = torch.zeros_like(disk.m0)
    unit[:, :, :, 0] = disk.disk_only[:, :, :, 0].to(dtype)
    w = 2 * math.pi * freq
    m = m0.clone()
    hz = disk.h_zero
    n_tot = (n_drive + n_hold) * FRAME_STEPS
    cz_drive = None
    for k in range(n_tot):
        tk = k * disk.cfg.dt
        a0 = amp if k < n_drive * FRAME_STEPS else 0.0

        def h(theta, tk=tk, a0=a0):
            return unit * (a0 * math.sin(w * (tk + theta * disk.cfg.dt)))

        m = disk.rollout.rk4_step(m, hz, h)
        if k + 1 == n_drive * FRAME_STEPS:
            cz_drive = core_mz(m, body)
    return cz_drive, core_mz(m, body), m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--amps-mT", type=float, nargs="+",
                   default=[0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15])
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--frames-drive", type=int, default=1)
    p.add_argument("--frames-hold", type=int, default=10)
    p.add_argument("--radius", type=float, default=None, help="nm")
    p.add_argument("--Ms", type=float, default=None, help="A/m")
    p.add_argument("--relax-steps", type=int, default=6000)
    p.add_argument("--relax-alpha", type=float, default=0.5)
    p.add_argument("--relax-tol", type=float, default=2e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="runs/core_switching")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    kw = {}
    if a.radius is not None:
        kw["radius"] = a.radius * 1e-9
    if a.Ms is not None:
        kw["Ms"] = a.Ms
    cfg = PortedVortexConfig(**kw)
    steps = (a.frames_drive + a.frames_hold) * FRAME_STEPS + 8
    disk = PortedVortexDisk(cfg, timesteps=steps, dtype=dtype)
    tag = f"r{cfg.radius*1e9:.0f}_Ms{cfg.Ms/1e3:.0f}_f{a.freq:g}"
    m0c = outdir / f"m0_{tag}.pt"
    if m0c.exists():
        disk.m0 = torch.load(m0c, weights_only=False).to(device=get_device(),
                                                         dtype=dtype)
        print(f"[m0] restored from {m0c.name}")
    else:
        t0 = time.time()
        disk.relax(steps=a.relax_steps, alpha_relax=a.relax_alpha)
        torch.save(disk.m0.cpu(), m0c)
        print(f"relaxed in {time.time()-t0:.0f}s")

    nx_, ny_ = disk.mask.shape[0], disk.mask.shape[1]
    xs = (torch.arange(nx_, dtype=dtype) - (nx_ - 1) / 2)
    ys = (torch.arange(ny_, dtype=dtype) - (ny_ - 1) / 2)
    Xc, Yc = torch.meshgrid(xs, ys, indexing="ij")
    Rc = torch.sqrt(Xc ** 2 + Yc ** 2).clamp(min=1e-6)
    m2d = disk.mask[:, :, 0, 0].to(dtype)
    body = ((Rc <= (cfg.radius / cfg.dx)) & (m2d > 0)).to(dtype)
    n_body = float(body.sum())
    circ = float(((Xc * disk.m0[:, :, 0, 1] - Yc * disk.m0[:, :, 0, 0]) / Rc
                  * body).sum() / max(n_body, 1.0))
    cz0 = core_mz(disk.m0, body)
    _m1 = disk.rollout.relax(disk.m0.clone(), disk.h_zero, 200, a.relax_alpha)
    dm = float((_m1 - disk.m0).abs().max())
    print(f"radius {cfg.radius*1e9:.0f} nm, Ms {cfg.Ms/1e3:.0f} kA/m, "
          f"{a.freq:g} GHz")
    print(f"circulation {circ:+.3f}, core m_z {cz0:+.3f}, "
          f"relax |dm| {dm:.4f} ({'converged' if dm <= a.relax_tol else 'NOT CONVERGED'})")
    if abs(circ) < 0.5 or dm > a.relax_tol:
        raise SystemExit("ground state is not a converged vortex; a switching "
                         "map measured here would not be one.")

    print(f"\ndrive {a.frames_drive} frame(s), then {a.frames_hold} frames of "
          f"silence, from the same state each time\n")
    print(f"{'amp_mT':>7} {'core after drive':>17} {'after hold':>12} "
          f"{'|cz| held':>10} {'switched':>9} {'persisted':>10}")
    rows = []
    for amp_mT in a.amps_mT:
        amp = amp_mT * 1e-3 / MU_0
        t0 = time.time()
        czd, czh, _ = drive_and_hold(disk, disk.m0, body, amp, a.freq * 1e9,
                                     a.frames_drive, a.frames_hold, dtype)
        sw = np.sign(czd) != np.sign(cz0) and abs(czd) > 0.3 * abs(cz0)
        pers = np.sign(czh) == np.sign(czd) and abs(czh) > 0.3 * abs(cz0)
        rows.append({"amp_mT": amp_mT, "core_after_drive": czd,
                     "core_after_hold": czh, "switched": bool(sw),
                     "persisted": bool(pers),
                     "seconds": round(time.time() - t0, 1)})
        print(f"{amp_mT:>7.2f} {czd:>17.3f} {czh:>12.3f} {abs(czh):>10.3f} "
              f"{'yes' if sw else '-':>9} {'yes' if pers else '-':>10}",
              flush=True)
        (outdir / f"results_{tag}.json").write_text(json.dumps(rows, indent=2))

    sws = [r for r in rows if r["switched"]]
    print()
    if not sws:
        print("No amplitude reversed the core. This element does not threshold "
              "in this range.")
    else:
        amps = [r["amp_mT"] for r in rows]
        flags = [r["switched"] for r in rows]
        first = sws[0]["amp_mT"]
        # A THRESHOLD element switches for every amplitude above some value. A
        # noise source switches at scattered amplitudes. The difference is
        # whether the flags, read in amplitude order, are monotone.
        above = [f for A, f in zip(amps, flags) if A >= first]
        monotone = all(above)
        held = sum(1 for r in sws if r["persisted"])
        print(f"first reversal at {first:.2f} mT; "
              f"{sum(flags)}/{len(flags)} amplitudes reversed, "
              f"{held}/{len(sws)} held polarity through the silence")
        if monotone and held == len(sws):
            print("STEP-LIKE and PERSISTENT: every amplitude above the first\n"
                  "reverses, and each holds its new polarity. That is a "
                  "threshold element\nwith memory -- the p[n] a hysteretic "
                  "layer needs.")
        elif not monotone:
            print("SCATTERED: some amplitudes above the first do NOT reverse.\n"
                  "The map from input to polarity is not a threshold, so this "
                  "element is\na noise source rather than a comparator.")
        else:
            print(f"Reverses but does not hold: only {held} of {len(sws)} "
                  f"kept polarity through\nthe silence. Without persistence "
                  f"there is no memory to multiply against.")
    print(f"\nwrote {outdir / f'results_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
