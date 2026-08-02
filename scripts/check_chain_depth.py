#!/usr/bin/env python3
"""How deep can the cascade practically go? Measure every stage at once.

The two-stage run established that the link carries a guided wave: arrival grows
linearly with separation at ~1173 m/s, 90x stronger than the dipolar background
at 700 nm, and the delay it introduces is what shifted stage B's memory window
from lags 0-4 to 3-12.

Depth then has a budget set by two measured numbers. Guided transfer is 0.383
per hop, so stage n receives 0.383^(n-1); dipolar crosstalk reaching any disk
straight from the driven one is a fixed 0.0043. The guided path falls to that
floor at stage 6, which predicts a ceiling of five stages at this separation --
and depth 4 puts the deepest stage's response near lag 10, where NARMA-10's
product term lives.

    stage   predicted amplitude   vs floor   predicted peak lag
      1           1.000             234x            4.0
      2           0.383              90x            6.2   <- measured 6
      3           0.147              34x            8.3
      4           0.056              13x           10.4
      5           0.022               5x           12.6
      6           0.008               1.9x         14.7   <- at the floor

This measures all of it in ONE simulation. A chain of N disks pulsed at stage 1
gives every stage's arrival and amplitude simultaneously, so the decay constant
and the delay per hop are fitted across depth rather than assumed from a single
hop. An impulse, not a task run: thousands of steps, not 600 frames.

Why it matters for the architecture: if the deep stages hold the memory, the
input stage is free to be driven into annihilation -- its own state destroyed
each frame -- because its job is transduction, not storage. That only works if
the signal still arrives intact several hops down, which is what this measures.

    python scripts/check_chain_depth.py --n-disks 5
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import ChainPortedConfig, ChainPortedArray


@torch.no_grad()
def pulse(arr, amp, freq, dtype, n_burst, n_quiet, stage=0):
    """Drive one stage's disk body; record every tap each step."""
    cfg = arr.cfg
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    unit[:, :, 0, 0] = arr.disk_masks[stage].to(dtype)
    m = arr.m0.clone()
    out = []
    for k in range(n_burst + n_quiet):
        tk = k * cfg.dt
        a = amp if k < n_burst else 0.0

        def h(theta, tk=tk, a=a):
            return unit * (a * math.sin(2 * math.pi * freq * (tk + theta * cfg.dt)))

        m = arr.rollout.rk4_step(m, arr.h_zero, h)
        out.append(arr.port_signals(m).double().cpu().numpy().copy())
    return np.asarray(out)


def envelope(x, win):
    return np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode="same"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-disks", type=int, default=5)
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--burst", type=int, default=400)
    p.add_argument("--quiet", type=int, default=2400)
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--require-tol", type=float, default=2e-3)
    p.add_argument("--no-link", action="store_true",
                   help="delete the link corridors: the crosstalk floor")
    p.add_argument("--outdir", default="runs/chain_depth")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    amp = a.amp_mT * 1e-3 / MU_0
    win = max(4, int(round(1e3 / a.freq)))          # ~one carrier period, in steps

    cfg = ChainPortedConfig(n_disks=a.n_disks, separation=a.separation * 1e-9,
                            link_width=(0.0 if a.no_link else 80e-9))
    tag = f"n{a.n_disks}_{'nolink' if a.no_link else 'linked'}"
    arr = ChainPortedArray(cfg, timesteps=a.burst + a.quiet + 8, dtype=dtype)
    print(f"chain of {a.n_disks} at {a.separation:.0f} nm, grid {cfg.grid}, "
          f"chirality {cfg.chiralities()}, link={not a.no_link}")

    m0c = outdir / f"m0_{tag}.pt"
    t0 = time.time()
    if m0c.exists():
        arr.m0 = torch.load(m0c, weights_only=False).to(dtype)
        print(f"[m0] restored from {m0c.name}")
    else:
        arr.relax(steps=a.relax_steps, require_tol=a.require_tol)
        torch.save(arr.m0.cpu(), m0c)
        print(f"relaxed in {time.time()-t0:.0f}s")

    sig = pulse(arr, amp, a.freq * 1e9, dtype, a.burst, a.quiet, stage=0)
    npb = cfg.n_ports
    dt_ns = cfg.dt * 1e9
    frame_ns = 200 * cfg.dt * 1e9

    print(f"\n{'stage':>6} {'arrival_ns':>11} {'peak_ns':>9} {'peak_lag':>9} "
          f"{'amp':>11} {'rel_to_1':>9} {'pred_rel':>9}")
    rows = []
    env0 = None
    for d in range(a.n_disks):
        e = envelope(np.sqrt((sig[:, d * npb:(d + 1) * npb] ** 2).mean(axis=1)), win)
        pk = float(e.max())
        if d == 0:
            env0 = pk
        thr = 0.1 * pk
        idx = int(np.argmax(e > thr)) if (e > thr).any() else -1
        t_arr = idx * dt_ns if idx >= 0 else float("nan")
        t_pk = float(np.argmax(e) * dt_ns)
        rel = pk / max(env0, 1e-30)
        rows.append({"stage": d + 1, "arrival_ns": t_arr, "peak_ns": t_pk,
                     "peak_lag_frames": t_pk / frame_ns, "amp": pk,
                     "rel_to_stage1": rel, "pred_rel": 0.383 ** d})
        print(f"{d+1:>6} {t_arr:>11.3f} {t_pk:>9.3f} {t_pk/frame_ns:>9.2f} "
              f"{pk:>11.4e} {rel:>9.4f} {0.383**d:>9.4f}", flush=True)
    (outdir / f"results_{tag}.json").write_text(json.dumps(rows, indent=2))

    # Fit the per-hop decay and delay across depth, rather than trusting one hop.
    ok = [r for r in rows if r["rel_to_stage1"] > 0 and np.isfinite(r["arrival_ns"])]
    if len(ok) >= 3:
        n = np.array([r["stage"] - 1 for r in ok], dtype=float)
        lg = np.log(np.array([r["rel_to_stage1"] for r in ok]))
        A = np.vstack([n, np.ones_like(n)]).T
        s, _ = np.linalg.lstsq(A, lg, rcond=None)[0]
        print(f"\nfitted transfer per hop: {math.exp(s):.3f}  "
              f"(single-hop measurement gave 0.383)")
        t = np.array([r["peak_ns"] for r in ok])
        sd, _ = np.linalg.lstsq(A, t, rcond=None)[0]
        print(f"fitted delay per hop:    {sd:.3f} ns = {sd/frame_ns:.2f} frames "
              f"(single-hop gave 0.430 ns = 2.15 frames)")
        floor = 0.00427
        depth = int(np.floor(math.log(3 * floor) / s)) + 1 if s < 0 else -1
        print(f"\ndepth where guided signal falls to 3x the dipolar floor: "
              f"stage {depth}")
    print(f"\nwrote {outdir / f'results_{tag}.json'}")


if __name__ == "__main__":
    main()
