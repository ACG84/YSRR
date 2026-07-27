#!/usr/bin/env python3
"""End-to-end: film1 -> spiking disks -> film2 denoiser -> readout, on NARMA-10.

Stages, each cached to disk so a container reclaim costs one stage rather than
the run:

  1  film1 response (measured) maps the input series to twelve drive channels
  2  the spiking disk array runs at 300 K, TWICE, with different noise seeds
  3  film2 is trained Noise2Noise -- encode realisation A, reconstruct
     realisation B -- which needs no clean target and no task labels
  4  ridge readouts are compared on a common protocol

The comparison is the point, so it is nested and includes the control that
matters. A physical denoiser is only interesting if it beats *temporal
averaging*, since that is the cheap way to exploit the same statistics
(independent spike jitter, correlated signal). So the boxcar control is
matched to film2's own effective window rather than being a token baseline.

    input only        u_n alone
    raw noisy         the disk features as they come off the reservoir
    boxcar            those features, temporally averaged -- the cheap denoiser
    film2 bottleneck  the trained physical autoencoder

film2 never sees the NARMA targets: it is trained purely on reconstruction, so
any win is denoising rather than task fitting.

    python scripts/run_full_system.py
    python scripts/run_full_system.py --frames 900 --epochs 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import magnonic_nn as mnn
from magnonic_nn.denoise import DenoiserConfig, PhysicalDenoiser
from magnonic_nn.reservoir import FilmResponse, fit_eval, narma10
from magnonic_nn.thiele import ThieleConfig, ThieleDisks

COLS = [0, 1, 2, 3, 6]           # analog features; polarity/spikes excluded


def disk_features(P, cfg, steps, noise_seed, tag, outdir):
    cache = outdir / f"feats_{tag}.pt"
    if cache.exists():
        print(f"  [cached] {cache.name}", flush=True)
        return torch.load(cache, weights_only=False)
    d = ThieleDisks(cfg, noise_seed=noise_seed)
    drive = torch.zeros(cfg.n_disks, dtype=torch.float64)
    out, t0 = [], time.time()
    for n in range(len(P)):
        drive.copy_(torch.from_numpy(P[n]))
        out.append(d.run_frame(drive, steps))
        if (n + 1) % 200 == 0:
            print(f"  {tag}: frame {n + 1}/{len(P)} ({time.time() - t0:.0f}s)", flush=True)
    F = torch.stack(out)
    torch.save(F, cache)
    return F


def boxcar(F, k):
    if k <= 1:
        return F
    pad = torch.cat([F[:1].expand(k - 1, *F.shape[1:]), F], dim=0)
    return torch.stack([pad[i:i + k].mean(0) for i in range(len(F))])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--response", default="runs/film_response/response.pt")
    p.add_argument("--frames", type=int, default=1100)
    p.add_argument("--splits", type=int, nargs=3, default=(100, 600, 150))
    p.add_argument("--steps-per-frame", type=int, default=300)
    p.add_argument("--train-frames", type=int, default=250)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--outdir", default="runs/full_system")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mnn.set_precision("float32")
    mnn.set_device("cpu")

    # -- stage 1: input -> film1 -> drive channels ---------------------------
    u, y = narma10(args.frames, seed=0)
    film1 = FilmResponse.load(args.response)
    P = film1(u / 0.5)[:, :12]

    # -- stage 2: the spiking reservoir, twice -------------------------------
    # Operating point from the sweep: contractive (lambda -0.35), firing, and
    # the least noise-dominated corner available at coupling 0.08.
    # Waveguide-encased: near-field dipolar coupling drops (the guides
    # separate the disks), and the coupling returns with a delay set by guide
    # length. tau is chosen as ~3 frames so the delay line spans the memory
    # NARMA-10 needs -- the disks themselves cannot hold it, since firing
    # resets them inside their own damping horizon (measured lag-1
    # autocorrelation -0.006 without guides).
    frame_s = args.steps_per_frame * (ThieleConfig().derived()["dt"])
    res_cfg = ThieleConfig(coupling=0.01, spike_kick=0.0, phase_capture=0.5,
                           drive_scale=0.12, temperature=300.0, alpha_spread=0.5,
                           wg_coupling=0.10, wg_delay=3 * frame_s,
                           wg_feedback=0.06, wg_feedback_delay=7 * frame_s,
                           port_neighbour=0.6, port_readout=0.3, port_return=0.1,
                           wg_return_delay=4 * frame_s)
    print("stage 2: reservoir (two noise realisations)", flush=True)
    A = disk_features(P, res_cfg, args.steps_per_frame, 0, "Awg", outdir)
    B = disk_features(P, res_cfg, args.steps_per_frame, 7, "Bwg", outdir)

    # -- stage 3: train film2, Noise2Noise -----------------------------------
    dc = DenoiserConfig(nx=20, steps_per_frame=40, n_out=6)
    sim = mnn.get_preset("vowels")
    sim.mesh.nx = sim.mesh.ny = dc.nx
    sim.material.abc_width = 3
    sim.solver.timesteps = dc.steps_per_frame * args.train_frames
    sim.fields.Bt = 5e-3
    den = PhysicalDenoiser(sim, dc)

    ckpt = outdir / "denoiser.pt"
    if ckpt.exists():
        den.load_state_dict(torch.load(ckpt, weights_only=False))
        print("stage 3: [cached] denoiser.pt", flush=True)
    else:
        print(f"stage 3: training film2 on {args.train_frames} frames", flush=True)
        opt = torch.optim.Adam(den.parameters(), lr=args.lr)
        a_tr = A[:args.train_frames].float()
        b_tr = B[:args.train_frames][:, :, COLS].reshape(args.train_frames, -1).float()
        b_tr = (b_tr - b_tr.mean(0)) / b_tr.std(0).clamp_min(1e-6)
        for ep in range(args.epochs):
            t0 = time.time()
            opt.zero_grad()
            z, rec = den(a_tr)
            loss = torch.nn.functional.mse_loss(rec, b_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            opt.step()
            print(f"  epoch {ep}  loss {float(loss):.4f}  ({time.time() - t0:.0f}s)",
                  flush=True)
            torch.save(den.state_dict(), ckpt)

    # -- stage 4: encode everything, then compare readouts -------------------
    zc = outdir / "z.pt"
    if zc.exists():
        Z = torch.load(zc, weights_only=False)
        print("stage 4: [cached] z.pt", flush=True)
    else:
        print("stage 4: encoding full sequence through film2", flush=True)
        sim.solver.timesteps = dc.steps_per_frame * args.frames
        den.film.rollout.timesteps = sim.solver.timesteps
        with torch.no_grad():
            Z = den.encode(A.float())
        torch.save(Z, zc)

    raw = A[:, :, COLS].reshape(len(A), -1).numpy()
    zc_ = (raw - raw.mean(0)) / raw.std(0).clip(1e-12)
    ac1 = float(np.mean([np.corrcoef(zc_[:-1, i], zc_[1:, i])[0, 1]
                         for i in range(zc_.shape[1])
                         if np.isfinite(np.corrcoef(zc_[:-1, i], zc_[1:, i])[0, 1])]))
    print(f"\nreservoir lag-1 autocorrelation: {ac1:+.3f}  "
          f"(was -0.006 without waveguides; NARMA-10 needs memory)", flush=True)
    splits = tuple(args.splits)
    # match the boxcar window to film2's own memory: propagation across the
    # mesh at the group velocity, expressed in frames
    k_eff = max(int(round(dc.nx * sim.mesh.dx / 500.0
                          / (dc.steps_per_frame * sim.solver.dt))), 2)

    results = {
        "input_only": fit_eval(u[:, None], y, splits),
        "raw_noisy": fit_eval(raw, y, splits),
        f"boxcar_k{k_eff}": fit_eval(boxcar(torch.from_numpy(raw), k_eff).numpy(),
                                     y, splits),
        "film2_bottleneck": fit_eval(Z.double().numpy(), y, splits),
    }
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'readout':<20} {'dim':>5} {'NARMA-10 test NMSE':>20}")
    dims = {"input_only": 1, "raw_noisy": raw.shape[1],
            f"boxcar_k{k_eff}": raw.shape[1], "film2_bottleneck": Z.shape[1]}
    for name, r in results.items():
        print(f"{name:<20} {dims[name]:>5} {r['nmse_test']:>20.4f}")
    print(f"\nfilm2 effective window {k_eff} frames; it never saw the targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
