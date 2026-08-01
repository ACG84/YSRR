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
    linear 10-lag     u_n ... u_n-9 through the same ridge readout. Reported for
                      continuity with this project's earlier numbers ONLY. It
                      looks like the matched baseline -- NARMA-10's memory
                      requirement is ten -- but it is not one: the 0.3*y[t] term
                      feeds every past y forward, so u's influence on y reaches
                      well past ten steps and a ten-lag window truncates
                      predictability that is genuinely linear.
    linear best       the same filter at the depth validation prefers. Measured
                      on the target alone, ten lags gives 0.71 and twenty gives
                      0.17 -- a factor of four. THIS is the baseline that
                      decides anything; beating 10-lag and losing to this means
                      the disk is an expensive delay line.
    ports             the physical modal readout

    python scripts/run_narma_modal.py
    python scripts/run_narma_modal.py --frames 1400 --steps-per-frame 250
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.reservoir import fit_eval, narma10
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk


# How often to checkpoint, in frames. This is not a tuning knob, it is a bet on
# how long the process gets to live. A checkpoint every 25 frames is ~137 s of
# compute, and a restart spends another ~90-120 s on imports, the demag kernel
# and the relax before it reaches frame one -- so the first save after a restart
# lands about four minutes in. When this container was rebooting hourly that was
# free. When it started killing runs every three to eight minutes, every cycle
# died before its first save and the sweep banked ZERO frames across 24 minutes
# while looking perfectly healthy: six live processes, all busy, all discarding
# their work. Five frames is ~27 s, which fits inside the short end of the
# observed kill window.
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "5"))


def save_ckpt(cache, feats, m):
    """Write the checkpoint atomically, or not at all.

    torch.save straight onto the live path is a torn-write waiting to happen:
    the kills land at arbitrary points, and one that arrives mid-write leaves a
    truncated file that torch.load cannot read -- which does not cost the last
    interval, it costs the ENTIRE run, because the next resume finds no usable
    checkpoint and starts from frame 0. Saving 5x more often multiplies that
    exposure by five, so the interval change above is only safe together with
    this. Write to a temp file, fsync, then os.replace, which is atomic on POSIX:
    a reader sees either the previous checkpoint or the new one, never a
    half-written one.
    """
    if cache is None:
        return
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save({"feats": feats, "m": m.detach().cpu()}, fh)  # host: a CUDA
        fh.flush()               # checkpoint would only reload on a CUDA machine
        os.fsync(fh.fileno())    # else the rename can beat the data to disk
    os.replace(tmp, cache)


@torch.no_grad()
def run_reservoir(disk, u, steps_per_frame, carrier, amp_lo, amp_hi, dtype,
                  cache=None, tones=(), drive="uniform"):
    """Drive frame by frame with no reset; return (n_frames, n_features).

    The magnetisation is carried across frames deliberately -- that continuity
    IS the reservoir's memory, and resetting between samples would leave a
    stateless nonlinearity with nothing to compute over.
    """
    # Resume cheaply. The checkpoint carries the MAGNETISATION as well as the
    # features, so a restart reloads the reservoir state instead of replaying
    # to rebuild it. That distinction is the whole point on this machine: the
    # container has rebooted three times in a day, roughly hourly, and the
    # previous scheme replayed from frame 0 -- so a crash at minute 39 of a
    # 40-minute run cost 39 minutes, not the checkpoint interval. It also kept
    # any run longer than the reboot period from EVER finishing, since each
    # attempt restarted from nothing.
    done, m_resume = [], None
    prev = None
    if cache is not None and cache.exists():
        try:
            prev = torch.load(cache, weights_only=False)
        except Exception as e:
            # A checkpoint written before saves became atomic can be torn. Losing
            # the run's history is bad; refusing to start is worse, because the
            # supervisor would relaunch into the same exception forever and the
            # seed would never advance again.
            print(f"[resume] {cache.name} unreadable ({type(e).__name__}: {e}); "
                  "starting this seed from frame 0", flush=True)
    if prev is not None:
        if isinstance(prev, dict):
            feats_prev, m_resume = prev["feats"], prev.get("m")
        else:
            feats_prev = prev                     # old format: features only
        if len(feats_prev) >= len(u):
            print(f"[cached] {cache.name}", flush=True)
            return feats_prev if not isinstance(prev, dict) else prev["feats"]
        done = [list(r) for r in feats_prev.tolist()]
        print(f"[resume] {cache.name} has {len(done)}/{len(u)} frames"
              + (" (state restored)" if m_resume is not None
                 else " (no state saved; replaying)"), flush=True)

    cfg = disk.cfg
    n = cfg.n_cells
    unit = torch.zeros(n, n, 1, 3, dtype=dtype)
    if drive == "uniform":
        unit[:, :, :, 0] = disk.disk_only[:, :, :, 0]
    else:
        # Single-port injection. A uniform in-plane field couples almost
        # entirely to n = +-1 by symmetry -- 77.9% of the amplitude, measured on
        # exactly these features -- so five of the six readout channels were
        # near-empty and the extra features bought nothing. One port at angle
        # theta_0 has azimuthal content exp(-i n theta_0), FLAT in n, so it
        # excites the whole basis the six ports decompose.
        t0m = disk._tap_masks.to(dtype)[0]
        unit[:, :, 0, 2] = t0m / t0m.sum().clamp_min(1e-30)

    # Graph-capturable stepper: the four RK4 substep fields are handed in as
    # tensors so the step captures once and replays, instead of breaking the
    # graph on a Python callable every substep. 0.63 ms/step against 16.26 on
    # CPU. Off CUDA this returns the eager function, so the loop below is
    # identical either way.
    stepper, graphed = disk.rollout.graph_stepper()
    hz = disk.h_zero
    m = disk.m0.clone() if m_resume is None else m_resume.clone().to(dtype)
    start = len(done) if m_resume is not None else 0
    feats, t0 = list(done), time.time()
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
        if j < start:
            continue                       # state was restored, not replayed
        amp = (amp_lo + (amp_hi - amp_lo) * float(un)) * 1e-3 / MU_0
        dev = m.device
        accI = torch.zeros(len(ws), cfg.n_ports, dtype=torch.float64, device=dev)
        accQ = torch.zeros(len(ws), cfg.n_ports, dtype=torch.float64, device=dev)
        for k in range(steps_per_frame):
            tk = (j * steps_per_frame + k) * cfg.dt
            # h at theta = 0, 1/2, 1/2, 1. The two half-step fields are the
            # same value, so this is three scalars, not four.
            s0 = amp * math.sin(w * tk)
            sh = amp * math.sin(w * (tk + 0.5 * cfg.dt))
            s1 = amp * math.sin(w * (tk + cfg.dt))
            h0, hh, h1 = hz + unit * s0, hz + unit * sh, hz + unit * s1
            if graphed:
                torch.compiler.cudagraph_mark_step_begin()
                m = stepper(m, h0, hh, hh, h1).clone()
            else:
                m = stepper(m, h0, hh, hh, h1)
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
            A = ((accI[i] + 1j * accQ[i]) / steps_per_frame).cpu().numpy()
            M = np.fft.fft(A)
            for q in range(cfg.n_ports):
                row += [M[q].real, M[q].imag]
        feats.append(row)
        if (j + 1) % CKPT_EVERY == 0:
            print(f"  frame {j+1}/{len(u)} ({time.time()-t0:.0f}s)", flush=True)
            save_ckpt(cache, torch.tensor(np.array(feats), dtype=torch.float64),
                      m)

    F = torch.tensor(np.array(feats), dtype=torch.float64)
    save_ckpt(cache, F.cpu(), m)
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
    p.add_argument("--absorb-frac", type=float, default=None,
                   help="fraction of each guide used as absorbing taper.\n"
                        "The default 0.35 is why memory capacity is 2.6: six\n"
                        "tapers are engineered DRAINS, and they cut the\n"
                        "effective decay from the 1.66 ns bulk damping time to\n"
                        "~0.6 ns. Lowering it keeps energy in the disk and lets\n"
                        "guide-end reflections return some -- both lengthen\n"
                        "memory, at the cost of a less clean readout.")
    p.add_argument("--drive", default="uniform", choices=["uniform", "port"],
                   help="'port' injects through one guide, which is flat in "
                        "azimuthal order; 'uniform' puts 77.9%% into n=+-1")
    p.add_argument("--relax-steps", type=int, default=900)
    p.add_argument("--device", default="cpu",
                   help="'cuda' takes the graph-captured step: 0.63 ms against\n"
                        "16.26 on CPU, so a 400-frame point is ~50 s instead of\n"
                        "~18 min -- which also puts a run inside the window\n"
                        "between this container's reboots.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="runs/narma_modal")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(args.device)
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    u, y = narma10(args.frames, seed=args.seed)
    cfg = (PortedVortexConfig() if args.absorb_frac is None
           else PortedVortexConfig(absorb_frac=args.absorb_frac))
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
                      tones=[t * 1e9 for t in args.tones_ghz],
                      drive=args.drive)
    X = F.cpu().numpy()
    X = (X - X.mean(0)) / X.std(0).clip(1e-12)

    # how much of the past actually survives, measured rather than assumed
    ac1 = float(np.nanmean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1]
                            for i in range(X.shape[1])]))
    print(f"\nreservoir lag-1 autocorrelation: {ac1:+.3f}", flush=True)

    splits = tuple(args.splits)
    # the linear filter's depth is chosen on VALIDATION, so it is the best one
    # the data supports rather than the one the task's name suggests
    cand = {k: fit_eval(lag_matrix(u, k), y, splits) for k in (10, 20, 40)}
    best_k = min(cand, key=lambda k: cand[k]["nmse_val"])
    results = {
        "input_only": fit_eval(u[:, None], y, splits),
        "linear_10lag": cand[10],
        "linear_best": cand[best_k],
        "best_lags": best_k,
        "ports_modal": fit_eval(X, y, splits),
    }
    results["lag1_autocorrelation"] = ac1
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    dims = {"input_only": 1, "linear_10lag": 10, "linear_best": best_k,
            "ports_modal": X.shape[1]}
    print(f"\n{'readout':<16} {'dim':>5} {'NARMA-10 test NMSE':>20}")
    for name in ("input_only", "linear_10lag", "linear_best", "ports_modal"):
        print(f"{name:<16} {dims[name]:>5} {results[name]['nmse_test']:>20.4f}")

    lin10 = results["linear_10lag"]["nmse_test"]
    lin = results["linear_best"]["nmse_test"]
    res = results["ports_modal"]["nmse_test"]
    print()
    if res < lin * 0.9:
        print(f"The ported disk beats the best linear filter on the same input "
              f"({res:.3f} vs {lin:.3f}\nat {best_k} lags). That margin is the "
              f"nonlinear mixing doing work -- the part a delay\nline cannot "
              f"supply at any length.")
    elif res < lin10 * 0.9:
        print(f"Beats the 10-lag baseline ({res:.3f} vs {lin10:.3f}) but loses "
              f"to the same filter at\n{best_k} lags ({lin:.3f}). That is not a "
              f"result: a longer linear window is free, and\nthe disk is an "
              f"expensive delay line against it.")
    elif res < 1.0:
        print(f"Predicts better than the mean but beats no linear baseline "
              f"({res:.3f} against\n{lin10:.3f} at 10 lags and {lin:.3f} at "
              f"{best_k}). The nonlinearity is not reaching the readout\nin "
              f"usable form.")
    else:
        print(f"No better than predicting the mean ({res:.3f}). Check the")
        print("lag-1 autocorrelation above: if it is near zero the frame is")
        print("long against the ring-down and no memory survives between")
        print("samples, which no readout can repair.")
    print("\nThe certification protocol -- six independent draws, paired "
          "intervals, and a\nleakage guard -- is scripts/certify_narma10.py; "
          "one run is not evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
