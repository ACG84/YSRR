#!/usr/bin/env python3
"""NARMA-10 through the ported vortex disk, read out at the ports.

The task the whole project has been building toward, run on the geometry that
has actually been verified rather than an assumed one:

    guides propagate          80 x 20 nm, cutoff 11 GHz, Q = 8.0 rising to 22.8
    six ports fit             46 deg each, 275 of 360
    modes resolved            3/3, |n| concentration 0.70 / 0.83 / 0.86
    disk discriminates        AB/BA 23.4x, nonlinear
    readout sees it           24.9x at the PORTS, better than the interior

Encoding. Each input sample amplitude-modulates a 12 GHz carrier -- inside the
passband, on the disk's own mode ladder -- for one frame, with no reset between
frames. The reservoir's memory is then physical: the ring-down time is
1/(alpha*omega) ~ 1.7 ns against a 0.2 ns frame, so roughly nine frames of the
past are still present in the magnetisation when the next sample arrives, which
is the order NARMA-10 needs.

Readout. Per-port lock-in at the carrier, then the cross-port DFT the six
guides physically implement, giving complex mode amplitudes for |n| <= 3. Only
what a real device could measure -- no interior state, no phase reference the
hardware would not have.

Scored against baselines that make the result falsifiable rather than
impressive:

    input only        u_n alone; anything above this is memory
    linear 10-lag     u_n ... u_n-9 through the same ridge readout. This is the
                      one that matters: it has ALL the memory the task needs and
                      no nonlinearity, so beating it is the only evidence that
                      the disk's nonlinear mixing is doing work. A reservoir
                      that ties with it is an expensive delay line.
    ports             the physical modal readout

    python scripts/run_narma_modal.py
    python scripts/run_narma_modal.py --frames 1400 --steps-per-frame 250
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.reservoir import fit_eval, narma10
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk


@torch.no_grad()
def run_reservoir(disk, u, steps_per_frame, carrier, amp_lo, amp_hi, dtype,
                  cache=None, tones=()):
    """Drive frame by frame with no reset; return (n_frames, n_features).

    The magnetisation is carried across frames deliberately -- that continuity
    IS the reservoir's memory, and resetting between samples would leave a
    stateless nonlinearity with nothing to compute over.
    """
    # Resume from a partial run. A 50-minute rollout was lost to a stray
    # signal with nothing on disk, because only the COMPLETED feature set was
    # cached. Checkpointing every ckpt_every frames caps that loss, and the
    # reservoir state is deterministic given the input prefix, so replaying the
    # first j frames reproduces the state exactly -- the cost of resuming is
    # bounded by the checkpoint interval, not the run length.
    done = []
    if cache is not None and cache.exists():
        prev = torch.load(cache, weights_only=False)
        if len(prev) >= len(u):
            print(f"[cached] {cache.name}", flush=True)
            return prev
        done = [r.tolist() for r in prev]
        print(f"[resume] {cache.name} has {len(done)}/{len(u)} frames; "
              f"replaying to rebuild reservoir state", flush=True)

    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    unit[:, :, :, 0] = disk.disk_only[:, :, :, 0]

    m = disk.m0.clone()
    feats, t0 = [], time.time()
    w = 2 * math.pi * carrier
    # Lock in at several frequencies, not just the drive. Three-magnon
    # scattering is the entire nonlinear mechanism here -- it is what produces
    # the 23.4x AB/BA discrimination -- and it moves energy AWAY from the
    # carrier, into harmonics and sum/difference tones and the disk's own mode
    # ladder at 9.4-13.7 GHz. A lock-in at the carrier alone measures the
    # linear response and discards the nonlinear products, which is exactly the
    # signature the first run showed: good memory, no advantage over a linear
    # tap-delay.
    ws = [2 * math.pi * f for f in (carrier, *tones)]
    for j, un in enumerate(u):
        amp = (amp_lo + (amp_hi - amp_lo) * float(un)) * 1e-3 / MU_0
        accI = torch.zeros(len(ws), cfg.n_ports, dtype=torch.float64)
        accQ = torch.zeros(len(ws), cfg.n_ports, dtype=torch.float64)
        for k in range(steps_per_frame):
            tk = (j * steps_per_frame + k) * cfg.dt

            def h_drive(theta, tk=tk, amp=amp):
                return unit * (amp * math.sin(w * (tk + theta * cfg.dt)))

            m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
            p = disk.port_signals(m).double()
            for i, wi in enumerate(ws):
                accI[i] += p * math.cos(wi * tk)
                accQ[i] += p * math.sin(wi * tk)
        # per tone: lock-in amplitude per port, then the cross-port DFT the
        # guides implement. Modes are kept SIGNED (+n and -n separately)
        # rather than folded onto |n|: the vortex splits them -- n = +-1 sit
        # 1.9 GHz apart -- so folding discards a real, measured channel.
        row = []
        for i in range(len(ws)):
            A = ((accI[i] + 1j * accQ[i]) / steps_per_frame).numpy()
            M = np.fft.fft(A)
            for q in range(cfg.n_ports):
                row += [M[q].real, M[q].imag]
        feats.append(row)
        if (j + 1) % 100 == 0:
            print(f"  frame {j+1}/{len(u)} ({time.time()-t0:.0f}s)", flush=True)
            if cache is not None and j + 1 > len(done):
                torch.save(torch.tensor(np.array(feats), dtype=torch.float64),
                           cache)

    F = torch.tensor(np.array(feats), dtype=torch.float64)
    if cache is not None:
        torch.save(F, cache)
    return F


def lag_matrix(u, n_lags):
    """u_n ... u_{n-n_lags+1}; the memory the task needs, with no nonlinearity."""
    out = np.zeros((len(u), n_lags))
    for k in range(n_lags):
        out[k:, k] = u[:len(u) - k]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=1200)
    p.add_argument("--splits", type=int, nargs=3, default=(150, 650, 150),
                   help="(washout, train, val). TEST is the REMAINDER, so\n"
                        "these must sum to less than --frames: summing to\n"
                        "exactly it leaves an empty test slice and every\n"
                        "score comes back nan, including input-only.")
    p.add_argument("--steps-per-frame", type=int, default=200)
    p.add_argument("--carrier-ghz", type=float, default=12.0)
    p.add_argument("--tones-ghz", type=float, nargs="*",
                   default=[9.9, 10.3, 13.7, 24.0],
                   help="extra lock-in frequencies. Defaults are where the\n"
                        "scattering products actually land: the strongest modes\n"
                        "n=+-2 and +-3 (9.9, 10.3), the n=+-7 rung (13.7), and\n"
                        "the second harmonic of the carrier (24.0).")
    p.add_argument("--amp-lo-mT", type=float, default=10.0)
    p.add_argument("--amp-hi-mT", type=float, default=30.0)
    p.add_argument("--relax-steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="runs/narma_modal")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    u, y = narma10(args.frames, seed=args.seed)
    cfg = PortedVortexConfig()
    disk = PortedVortexDisk(cfg, timesteps=args.steps_per_frame + 4, dtype=dtype)
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    tau_ns = 1.0 / (cfg.alpha * 2 * math.pi * args.carrier_ghz * 1e9) * 1e9
    frame_ns = args.steps_per_frame * cfg.dt * 1e9
    print(f"relaxed {time.time()-t0:.0f}s, core mz "
          f"{float(disk.m0[:, :, 0, 2].max()):+.3f}\n"
          f"frame {frame_ns:.2f} ns, ring-down {tau_ns:.2f} ns "
          f"-> ~{tau_ns/frame_ns:.1f} frames of physical memory", flush=True)

    F = run_reservoir(disk, u, args.steps_per_frame, args.carrier_ghz * 1e9,
                      args.amp_lo_mT, args.amp_hi_mT, dtype,
                      cache=outdir / "features_multitone.pt",
                      tones=[t * 1e9 for t in args.tones_ghz])
    X = F.numpy()
    X = (X - X.mean(0)) / X.std(0).clip(1e-12)

    # how much of the past actually survives, measured rather than assumed
    ac1 = float(np.nanmean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1]
                            for i in range(X.shape[1])]))
    print(f"\nreservoir lag-1 autocorrelation: {ac1:+.3f}", flush=True)

    splits = tuple(args.splits)
    results = {
        "input_only": fit_eval(u[:, None], y, splits),
        "linear_10lag": fit_eval(lag_matrix(u, 10), y, splits),
        "ports_modal": fit_eval(X, y, splits),
    }
    results["lag1_autocorrelation"] = ac1
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    dims = {"input_only": 1, "linear_10lag": 10, "ports_modal": X.shape[1]}
    print(f"\n{'readout':<16} {'dim':>5} {'NARMA-10 test NMSE':>20}")
    for name in ("input_only", "linear_10lag", "ports_modal"):
        print(f"{name:<16} {dims[name]:>5} {results[name]['nmse_test']:>20.4f}")

    lin = results["linear_10lag"]["nmse_test"]
    res = results["ports_modal"]["nmse_test"]
    print()
    if res < lin * 0.9:
        print(f"The ported disk beats a linear readout with the same memory "
              f"({res:.3f} vs {lin:.3f}).")
        print("That margin is the nonlinear mixing doing work -- the part a")
        print("delay line cannot supply.")
    elif res < 1.0:
        print(f"Predicts better than the input alone but does not beat the")
        print(f"linear 10-lag baseline ({res:.3f} vs {lin:.3f}). On this task")
        print("the disk is behaving as a delay line with extra steps; the")
        print("nonlinearity is not contributing.")
    else:
        print(f"No better than predicting the mean ({res:.3f}). Check the")
        print("lag-1 autocorrelation above: if it is near zero the frame is")
        print("long against the ring-down and no memory survives between")
        print("samples, which no readout can repair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
