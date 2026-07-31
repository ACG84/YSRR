#!/usr/bin/env python3
"""The whole machine, end to end: film1 -> coupled ported array -> film2 -> readout.

Every earlier full-system attempt ran on the Thiele model, which has since been
retired: two degrees of freedom per disk and a reset that erased both, so there
was nothing to carry one input into the next. This runs the chain on the
geometry that has actually been measured -- guides that propagate at 11 GHz,
six ports that decompose modes 3/3, a disk whose AB/BA discrimination reaches
those ports at 24.9x, and a two-disk array that is stable at lambda -0.164/ns
on a converged ground state.

    stage 1   film1 (measured response) maps the input to drive channels
    stage 2   the coupled array runs TWICE at 300 K with different noise seeds
    stage 3   film2 is trained Noise2Noise -- encode realisation A, reconstruct
              realisation B -- which needs no clean target and no task labels
    stage 4   ridge readouts compared on one protocol

Stage 2 needs the thermal field added to the solver for this run, and that is
not decoration. Noise2Noise assumes two INDEPENDENT noisy views of the same
signal; a deterministic rollout returns byte-identical features twice, and
film2 would be trained to reproduce its own input exactly -- a perfect score
that means nothing. The Langevin term is fixed by fluctuation-dissipation from
the damping already in the model (verified against the analytic amplitude to
0.981 with exact sqrt(T) scaling), so it is not a tuning knob.

The baselines decide whether any of it matters:

    input only        u_n alone
    linear 10-lag     a shift register: all the memory, no nonlinearity
    raw ports         the features as they come off the array, at 300 K
    boxcar            those features temporally averaged -- the CHEAP denoiser,
                      matched to film2's own window. A physical autoencoder is
                      only interesting if it beats this, since averaging
                      exploits the same statistics for free.
    film2 bottleneck  the trained physical readout

film2 never sees the task targets, so any win is denoising rather than fitting.

    python scripts/run_full_system_modal.py
    python scripts/run_full_system_modal.py --frames 700 --temperature 300
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.denoise import DenoiserConfig, PhysicalDenoiser
from magnonic_nn.reservoir import FilmResponse, fit_eval, narma10
from magnonic_nn.vortex import CoupledPortedConfig, CoupledPortedArray


def lag_matrix(u, n_lags):
    out = np.zeros((len(u), n_lags))
    for k in range(n_lags):
        out[k:, k] = u[:len(u) - k]
    return out


def boxcar(F, k):
    if k <= 1:
        return F
    pad = np.concatenate([np.repeat(F[:1], k - 1, axis=0), F], axis=0)
    return np.stack([pad[i:i + k].mean(0) for i in range(len(F))])


@torch.no_grad()
def run_array(arr, drive, steps_per_frame, carrier, tones, amp_lo, amp_hi,
              temperature, seed, dtype, cache=None):
    """Drive each disk from its own film1 channel; return per-frame features."""
    if cache is not None and cache.exists():
        prev = torch.load(cache, weights_only=False)
        if len(prev) >= len(drive):
            print(f"  [cached] {cache.name}", flush=True)
            return prev

    cfg = arr.cfg
    arr.rollout.set_temperature(temperature, seed=seed)
    masks = arr.disk_masks.to(dtype)
    n_disks = masks.shape[0]

    m = arr.m0.clone()
    n_ports_total = arr.port_signals(m).shape[0]
    ws = [2 * math.pi * f for f in (carrier, *tones)]
    feats, t0 = [], time.time()
    for j in range(len(drive)):
        # each disk gets its own amplitude from its own routed channel -- that
        # spatial diversity is the only thing film1 contributes, so collapsing
        # it to one shared amplitude would make the router pointless
        amps = [(amp_lo + (amp_hi - amp_lo) * float(drive[j, d])) * 1e-3 / MU_0
                for d in range(n_disks)]
        unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
        for d in range(n_disks):
            unit[:, :, 0, 0] += masks[d] * amps[d]

        accI = torch.zeros(len(ws), n_ports_total, dtype=torch.float64)
        accQ = torch.zeros(len(ws), n_ports_total, dtype=torch.float64)
        for k in range(steps_per_frame):
            tk = (j * steps_per_frame + k) * cfg.dt

            def h_drive(theta, tk=tk):
                return unit * math.sin(2 * math.pi * carrier
                                       * (tk + theta * cfg.dt))

            m = arr.rollout.rk4_step(m, arr.h_zero, h_drive)
            p = arr.port_signals(m).double()
            for i, wi in enumerate(ws):
                accI[i] += p * math.cos(wi * tk)
                accQ[i] += p * math.sin(wi * tk)

        row = []
        for i in range(len(ws)):
            A = ((accI[i] + 1j * accQ[i]) / steps_per_frame).numpy()
            for d in range(0, n_ports_total, cfg.n_ports):
                M = np.fft.fft(A[d:d + cfg.n_ports])
                for q in range(cfg.n_ports):
                    row += [M[q].real, M[q].imag]
        feats.append(row)
        if (j + 1) % 50 == 0:
            print(f"  frame {j+1}/{len(drive)} ({time.time()-t0:.0f}s)", flush=True)
            if cache is not None:
                torch.save(torch.tensor(np.array(feats), dtype=torch.float64),
                           cache)

    F = torch.tensor(np.array(feats), dtype=torch.float64)
    if cache is not None:
        torch.save(F, cache)
    return F


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=550)
    p.add_argument("--splits", type=int, nargs=3, default=(80, 260, 90))
    p.add_argument("--steps-per-frame", type=int, default=200)
    p.add_argument("--carrier-ghz", type=float, default=12.0)
    p.add_argument("--tones-ghz", type=float, nargs="*", default=[9.9, 13.7])
    p.add_argument("--amp-lo-mT", type=float, default=10.0)
    p.add_argument("--amp-hi-mT", type=float, default=30.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--absorb-frac", type=float, default=0.12)
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--link-width", type=float, default=80.0)
    p.add_argument("--relax-steps", type=int, default=5000)
    p.add_argument("--require-tol", type=float, default=1e-3)
    p.add_argument("--response", default="runs/film_response/response.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--outdir", default="runs/full_modal")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -- stage 1: film1 -----------------------------------------------------
    u, y = narma10(args.frames, seed=0)
    film1 = FilmResponse.load(args.response)
    P = film1(u / 0.5)[:, :12]
    P = np.asarray(P, dtype=np.float64)
    P = (P - P.min(0)) / np.ptp(P, axis=0).clip(1e-12)
    n_disks = 2
    drive = np.stack([P[:, 0], P[:, 6]], axis=1)      # two well-separated channels
    print(f"stage 1: film1 -> {drive.shape[1]} routed channels "
          f"(corr {np.corrcoef(drive[:,0], drive[:,1])[0,1]:+.3f})", flush=True)

    # -- stage 2: the array, twice ------------------------------------------
    cfg = CoupledPortedConfig(chirality=+1, chirality_b=-1,
                              separation=args.separation * 1e-9,
                              link_width=args.link_width * 1e-9,
                              absorb_frac=args.absorb_frac)
    arr = CoupledPortedArray(cfg, timesteps=args.steps_per_frame + 4, dtype=dtype)
    t0 = time.time()
    arr.relax(steps=args.relax_steps, require_tol=args.require_tol)
    print(f"stage 2: relaxed {time.time()-t0:.0f}s, running at "
          f"{args.temperature:g} K twice", flush=True)
    tones = [t * 1e9 for t in args.tones_ghz]
    A = run_array(arr, drive, args.steps_per_frame, args.carrier_ghz * 1e9,
                  tones, args.amp_lo_mT, args.amp_hi_mT, args.temperature,
                  seed=0, dtype=dtype, cache=outdir / "featsA.pt")
    B = run_array(arr, drive, args.steps_per_frame, args.carrier_ghz * 1e9,
                  tones, args.amp_lo_mT, args.amp_hi_mT, args.temperature,
                  seed=7, dtype=dtype, cache=outdir / "featsB.pt")

    Xa, Xb = A.numpy(), B.numpy()
    noise = float(np.abs(Xa - Xb).mean() / (np.abs(Xa).mean() + 1e-30))
    print(f"  realisation disagreement {noise:.4f} "
          f"(0 would mean the thermal field did nothing)", flush=True)

    # -- stage 3: film2, Noise2Noise ----------------------------------------
    n_feat = Xa.shape[1]
    dc = DenoiserConfig(nx=20, steps_per_frame=40, n_out=8)
    sim = mnn.get_preset("vowels")
    sim.mesh.nx = sim.mesh.ny = dc.nx
    sim.material.abc_width = 3
    sim.solver.timesteps = dc.steps_per_frame * args.frames
    sim.fields.Bt = 5e-3
    den = PhysicalDenoiser(sim, dc, n_disks=min(n_feat, 12), n_features=1)

    ckpt = outdir / "denoiser.pt"
    a_t = torch.from_numpy(Xa[:, :den.n_disks]).float()
    b_t = torch.from_numpy(Xb).float()
    b_t = (b_t - b_t.mean(0)) / b_t.std(0).clamp_min(1e-6)
    if ckpt.exists():
        den.load_state_dict(torch.load(ckpt, weights_only=False))
        print("stage 3: [cached] denoiser.pt", flush=True)
    else:
        print(f"stage 3: training film2 Noise2Noise ({args.epochs} epochs)",
              flush=True)
        den.decoder = torch.nn.Linear(dc.n_out, b_t.shape[1])
        opt = torch.optim.Adam(den.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            t0 = time.time(); opt.zero_grad()
            z, rec = den(a_t.unsqueeze(-1))
            loss = torch.nn.functional.mse_loss(rec, b_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0)
            opt.step()
            print(f"  epoch {ep} loss {float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(den.state_dict(), ckpt)

    with torch.no_grad():
        Z = den.encode(a_t.unsqueeze(-1)).double().numpy()

    # -- stage 4: compare ----------------------------------------------------
    Xn = (Xa - Xa.mean(0)) / Xa.std(0).clip(1e-12)
    k_eff = max(int(round(dc.nx * sim.mesh.dx / 500.0
                          / (dc.steps_per_frame * sim.solver.dt))), 2)
    splits = tuple(args.splits)
    results = {
        "input_only": fit_eval(u[:, None], y, splits),
        "linear_10lag": fit_eval(lag_matrix(u, 10), y, splits),
        "raw_ports": fit_eval(Xn, y, splits),
        f"boxcar_k{k_eff}": fit_eval(boxcar(Xn, k_eff), y, splits),
        "film2_bottleneck": fit_eval(Z, y, splits),
    }
    results["realisation_disagreement"] = noise
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    dims = {"input_only": 1, "linear_10lag": 10, "raw_ports": Xn.shape[1],
            f"boxcar_k{k_eff}": Xn.shape[1], "film2_bottleneck": Z.shape[1]}
    print(f"\n{'readout':<20} {'dim':>5} {'NARMA-10 test NMSE':>20}")
    for name in results:
        if name in dims:
            print(f"{name:<20} {dims[name]:>5} "
                  f"{results[name]['nmse_test']:>20.4f}")

    lin = results["linear_10lag"]["nmse_test"]
    box = results[f"boxcar_k{k_eff}"]["nmse_test"]
    f2 = results["film2_bottleneck"]["nmse_test"]
    print()
    if f2 < min(lin, box) * 0.9:
        print("film2 beats both the linear tap-delay and temporal averaging.")
        print("The physical denoiser is doing something neither supplies.")
    elif f2 < box * 0.9:
        print("film2 beats the boxcar but not the linear baseline: it denoises,")
        print("and the array still lacks the memory the task needs.")
    else:
        print(f"film2 ({f2:.3f}) does not beat temporal averaging ({box:.3f}).")
        print("A boxcar is free; a trained micromagnetic film is not. On this")
        print("evidence the physical autoencoder is not earning its place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
