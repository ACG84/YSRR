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


def save_ckpt(cache, feats, m, waves=None, states=None):
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
    obj = {"feats": feats, "m": m.detach().cpu()}   # host: a CUDA checkpoint
    if waves is not None:                           # would only reload on CUDA
        # Same file, so waveform and features can never resume out of step with
        # each other. Optional key, so checkpoints written before this existed
        # still load.
        obj["waves"] = torch.tensor(np.array(waves), dtype=torch.float32)
    if states is not None:
        obj["state"] = torch.tensor(np.array(states), dtype=torch.float32)
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())    # else the rename can beat the data to disk
    os.replace(tmp, cache)


@torch.no_grad()
def run_reservoir(disk, u, steps_per_frame, carrier, amp_lo, amp_hi, dtype,
                  cache=None, tones=(), drive="uniform", keep_waves=False,
                  wave_stride=2, state_dims=0, state_lockin=0):
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
    done, m_resume, waves, states = [], None, None, None
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
        if (state_dims or state_lockin) and isinstance(prev, dict) \
                and prev.get("state") is not None:
            states = [r for r in prev["state"].numpy()[:len(done)]]
        if keep_waves and isinstance(prev, dict) and prev.get("waves") is not None:
            # Trim to the features' length: a checkpoint is written features-
            # first, so waves can only ever be equal or longer, never shorter.
            wprev = prev["waves"].numpy()
            waves = [row for row in wprev[:len(done)]]
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
    if keep_waves and waves is None:
        waves = []
    # A direct readout of the magnetisation, for asking whether the ports are
    # what limits the measured memory. A FIXED random projection, not more
    # lock-in tones: adding tones adds high-variance uninformative columns that
    # a single-lambda ridge cannot shrink selectively, which made a strict
    # superset of the shipped readout score WORSE than it (MC 2.07 vs 8.04). A
    # random projection is richer without being noisier, and at the shipped
    # readout's own width it is a like-for-like comparison.
    inside_m = disk.disk_only[:, :, 0, 0] > 0.5
    n_state = int(inside_m.sum()) * 3
    P = None
    # A frame-END SNAPSHOT of the state does not work, and the reason is the one
    # that killed the raw-waveform arm: the frame is 2.4 carrier cycles, so the
    # phase at the sampling instant advances 0.4 cycle per frame. Measured on the
    # snapshots, cosine similarity is +0.988 at frame separation 5 and -0.19 at
    # separation 1, and max |corr(projection, u[n])| is 0.084 against the
    # lock-in's 0.930. The state is there; the phase makes it unreadable by any
    # fixed linear map.
    #
    # So demodulate the state exactly as the ports are demodulated, and the
    # comparison then isolates the one thing left: SPATIAL SAMPLING. Six physical
    # port taps against `state_lockin` random spatial functionals of the whole
    # magnetisation, same tones, same I/Q, same width.
    PL = None
    if state_lockin:
        gl = torch.Generator().manual_seed(999)
        PL = torch.randn(n_state, state_lockin, generator=gl,
                         dtype=torch.float64) / math.sqrt(n_state)
        if states is None:
            states = []
    if state_dims:
        g = torch.Generator().manual_seed(12345)
        P = torch.randn(n_state, state_dims, generator=g,
                        dtype=torch.float64) / math.sqrt(n_state)
        if states is None:
            states = []
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
        if PL is not None:
            sI = torch.zeros(len(ws), state_lockin, dtype=torch.float64)
            sQ = torch.zeros(len(ws), state_lockin, dtype=torch.float64)
        wrow = [] if waves is not None else None
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
            if PL is not None:
                sv = ((m - disk.m0)[:, :, 0, :][inside_m]
                      .reshape(-1).double() @ PL)
                for i, wi in enumerate(ws):
                    sI[i] += sv * math.cos(wi * tk)
                    sQ[i] += sv * math.sin(wi * tk)
            # The raw port waveform, kept only when asked for. The lock-in below
            # collapses this to five tones; whether that projection is what
            # limits the measured memory is not answerable from the projection
            # itself, so the un-projected signal has to be stored to compare.
            if wrow is not None and k % wave_stride == 0:
                wrow.append(p.cpu().numpy().copy())
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
        if waves is not None:
            waves.append(np.array(wrow, dtype=np.float32).ravel())
        if states is not None:
            if PL is not None:
                row_s = []
                for i in range(len(ws)):
                    A = ((sI[i] + 1j * sQ[i]) / steps_per_frame).cpu().numpy()
                    for q in range(state_lockin):
                        row_s += [A[q].real, A[q].imag]
                states.append(np.array(row_s, dtype=np.float32))
            else:
                d_state = (m - disk.m0)[:, :, 0, :][inside_m].reshape(-1).double()
                states.append((d_state @ P).cpu().numpy().astype(np.float32))
        if (j + 1) % CKPT_EVERY == 0:
            print(f"  frame {j+1}/{len(u)} ({time.time()-t0:.0f}s)", flush=True)
            save_ckpt(cache, torch.tensor(np.array(feats), dtype=torch.float64),
                      m, waves, states)

    F = torch.tensor(np.array(feats), dtype=torch.float64)
    save_ckpt(cache, F.cpu(), m, waves, states)
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
    p.add_argument("--alpha", type=float, default=None,
                   help="Gilbert damping of the disk bulk (not the absorbing\n"
                        "tapers, which stay at absorb_alpha). This is the knob\n"
                        "the NARMA-10 certification points at: memory in frames\n"
                        "is 1/(2 pi alpha N_cycles), measured 8.12 against 8.29\n"
                        "predicted at the 0.008 default, so the model is good to\n"
                        "2% and lowering alpha should buy memory proportionally.\n"
                        "0.008 is permalloy-like, 0.004 CoFeB-like, 0.002 needs\n"
                        "an optimised low-damping alloy. Note this varies damping\n"
                        "ALONE: a real material swap would move Ms and A too, so\n"
                        "these are idealised points on one axis, not materials.")
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
    p.add_argument("--save-waveform", action="store_true",
                   help="also store the raw per-frame port waveform. The 60\n"
                        "lock-in features are a five-tone projection of it, and\n"
                        "whether that projection is what caps the measured\n"
                        "memory at 8 lags cannot be answered from the projection\n"
                        "itself -- MC was invariant at 8.3 across a fourfold\n"
                        "damping change and a sixfold port change, which is what\n"
                        "a saturated instrument looks like.")
    p.add_argument("--save-state", type=int, default=0, metavar="N",
                   help="also store N fixed random projections of the disk\n"
                        "magnetisation per frame. This is the readout-vs-disk\n"
                        "test: if the state carries memory past lag 8 that the\n"
                        "ports do not, the ceiling is in the readout.")
    p.add_argument("--save-state-lockin", type=int, default=0, metavar="N",
                   help="lock-in readout of the FULL magnetisation: N random\n"
                        "spatial functionals, demodulated at the same tones as\n"
                        "the ports. N=6 gives exactly the shipped readout's 60\n"
                        "features, so the only difference is WHERE the state is\n"
                        "sampled -- six physical taps versus the whole disk.")
    p.add_argument("--wave-stride", type=int, default=2,
                   help="keep every Nth step of the waveform")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="runs/narma_modal")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(args.device)
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    u, y = narma10(args.frames, seed=args.seed)
    kw = {}
    if args.absorb_frac is not None:
        kw["absorb_frac"] = args.absorb_frac
    if args.alpha is not None:
        kw["alpha"] = args.alpha
    cfg = PortedVortexConfig(**kw)
    disk = PortedVortexDisk(cfg, timesteps=args.steps_per_frame + 4, dtype=dtype)
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    tau_ns = 1.0 / (cfg.alpha * 2 * math.pi * args.carrier_ghz * 1e9) * 1e9
    frame_ns = args.steps_per_frame * cfg.dt * 1e9
    print(f"relaxed {time.time()-t0:.0f}s, core mz "
          f"{float(disk.m0[:, :, 0, 2].max()):+.3f}\n"
          f"alpha {cfg.alpha:.4f}, frame {frame_ns:.2f} ns, "
          f"ring-down {tau_ns:.2f} ns "
          f"-> ~{tau_ns/frame_ns:.1f} frames of physical memory", flush=True)

    F = run_reservoir(disk, u, args.steps_per_frame, args.carrier_ghz * 1e9,
                      args.amp_lo_mT, args.amp_hi_mT, dtype,
                      cache=outdir / "features_multitone.pt",
                      tones=[t * 1e9 for t in args.tones_ghz],
                      drive=args.drive, keep_waves=args.save_waveform,
                      wave_stride=args.wave_stride, state_dims=args.save_state,
                      state_lockin=args.save_state_lockin)
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
