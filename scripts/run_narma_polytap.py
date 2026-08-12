#!/usr/bin/env python3
"""NARMA-10 through the poly-tap delay bus.

This is the run the whole tapped-bus architecture exists for, and it is only
worth starting now because the delay line has finally been shown to work. With a
rollout long enough for the wave to arrive -- 12.2 ns, against the 2.81 ns every
previous poly-tap run used -- the as-built array resolves all four taps:

    arrival lags   13.13, 16.59, 17.80, 21.55 frames
    spacings       3.46, 1.20, 3.76   against 3.00 designed
    weakest tap    3.93e-05, four times over the placement bar

The capacity accounting says what the reservoir is missing: roughly 40
independent LONG-SEPARATION product dimensions against the ~5 it has spare.
Products between samples one or two frames apart are worth nothing on NARMA-10 --
every short-separation family sits at the 0.1243 linear baseline while far pairs,
|i-j| >= 5, score 0.0401. The serial chain makes only the worthless kind, because
its nonlinearity sits at stage 1 and stage 1 only remembers lags 0-6.

The tapped bus is supposed to fix that by giving each disk a DIFFERENT delay
against the same fresh sample. Which is why the co-drive arm is the point:

    --drive bus     bus only. Each disk sees one delayed copy and nothing to
                    multiply it against, so this measures the delay line's
                    memory and should NOT produce cross-lag products.
    --drive both    bus plus a fresh sample at every disk body, scaled per tap.
                    A nonlinearity can only multiply signals present in the same
                    state at the same time, so this is the arm that can make
                    u[n]*u[n-k] at four different k.

The fresh scaling matters and is measured rather than assumed. The chain's
co-drive failed exactly here: a full-amplitude fresh sample drowned a delayed
copy ~100x smaller, and degree-1 capacity fell from 8.91 to 4.95 while the
memory horizon pulled in from lag 12 to lag 7. Here the default per-tap scale is
the MEASURED delayed amplitude at each tap, normalised to tap 1, so the two
operands arrive comparable.

Baselines are the point, as always. Four disks give 200 feature columns and a
wider readout fits better whether or not the taps do anything, so the comparison
that decides this is against the linear filter at 0.1243 and against the
bus-only arm -- not against the input.

    python scripts/run_narma_polytap.py --device cuda --drive both
    python scripts/run_narma_polytap.py --device cuda --drive bus    # control
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.reservoir import fit_eval, narma10
from magnonic_nn._compat import get_device
from _polytap_probe import make_array, ensure_m0, check_record_length

CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "10"))

# Measured delayed amplitude at each tap on the as-built array with a valid
# record (bw80, gap 15, bus alpha x1), normalised to tap 1. The fresh sample is
# scaled by these so it arrives comparable to the delayed copy rather than
# swamping it -- which is precisely how the chain's co-drive failed.
MEASURED_TAP_AMPS = (1.1460e-3, 1.4156e-4, 7.0611e-5, 3.9348e-5)


def lag_matrix(u, n_lags):
    out = np.zeros((len(u), n_lags))
    for k in range(n_lags):
        out[k:, k] = u[:len(u) - k]
    return out


def save_ckpt(cache, feats, m):
    """Atomic checkpoint: temp file, fsync, rename.

    A 600-frame run is 120,000 steps on a 804x159 mesh. Both hosts this runs on
    reclaim machines -- the container reboots roughly hourly and Colab runtimes
    have been pruned twice in this project -- so an all-or-nothing run on those
    odds does not merely risk the work, it can fail to ever complete.
    """
    if cache is None:
        return
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save({"feats": feats, "m": m.detach().cpu()}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, cache)


@torch.no_grad()
def run_reservoir(arr, cfg, u, spf, carrier, amp_lo, amp_hi, dtype,
                  cache=None, tones=(), drive="both", fresh_scale=None,
                  quench_frac=0.0, quench_gain=1.0):
    """Drive frame by frame with no reset; return (n_frames, features)."""
    done, m_resume = [], None
    if cache is not None and cache.exists():
        try:
            prev = torch.load(cache, weights_only=False)
            feats_prev, m_resume = prev["feats"], prev.get("m")
            if len(feats_prev) >= len(u):
                print(f"[cached] {cache.name}", flush=True)
                return feats_prev
            done = [list(r) for r in feats_prev.tolist()]
            print(f"[resume] {cache.name} has {len(done)}/{len(u)} frames",
                  flush=True)
        except Exception as e:
            print(f"[resume] {cache.name} unreadable ({type(e).__name__}); "
                  f"starting from frame 0", flush=True)

    # Two drive regions with different geometry AND different orientation. The
    # bus injection is out of plane at the injection segment, which is what
    # launches a guided wave; the fresh sample is in-plane over each disk BODY,
    # never on a guide -- a drive on a readout guide injects straight into the
    # readout and the ports would be measuring the input rather than the state.
    # Split, because the two get different time envelopes. The bus is a
    # transmission line and its drive must stay a clean carrier or the delayed
    # copy is corrupted; the disk is a resonator and can be quenched.
    unit_bus = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    unit_disk = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    if drive in ("bus", "both"):
        unit_bus[:, :, 0, 2] += arr.inject_mask
    if drive in ("fresh", "both"):
        for k, s in enumerate(fresh_scale):
            unit_disk[:, :, 0, 0] += float(s) * arr.disk_masks[k]

    # Anti-phase quench: the last `quench_frac` of each frame drives the disk
    # bodies in antiphase, coherently cancelling the residual precession instead
    # of waiting for it to dissipate.
    #
    # This exists because raising Gilbert damping is a blunt way to buy a short
    # response time. Ring-down is 1/(alpha*omega) = 8.3 frames at alpha 0.008,
    # and the nonlinearity therefore acts on an ~8-sample average rather than on
    # single samples -- which is the leading explanation for why every product
    # family reads 0.00. tap_alpha_mult=10 fixes the timescale by dropping
    # Q = 1/(2*alpha) from 62.5 to 6.25, throwing away tenfold the stored energy
    # to do it, and the coupling sweep measured that cost as halving what each
    # tap receives.
    #
    # The quench sets the EFFECTIVE response time from the drive side, leaving
    # alpha and therefore the amplitude alone. It also acts on whatever is
    # ringing in the disk regardless of whether it arrived fresh or down the
    # bus, which is the point: a tap should respond to the instantaneous SUM of
    # its two inputs, not to a running average of them.
    n_drive = int(round(spf * (1.0 - quench_frac)))

    stepper, graphed = arr.rollout.graph_stepper()
    eager = arr.rollout.rk4_step_fields
    hz = arr.h_zero
    m = arr.m0.clone() if m_resume is None else m_resume.clone().to(dtype)
    start = len(done) if m_resume is not None else 0
    n_sig = arr.port_signals(m).shape[0]
    npr = arr.n_readout
    ws = [2 * math.pi * f for f in (carrier, *tones)]
    feats, t0 = list(done), time.time()
    for j, un in enumerate(u):
        if j < start:
            continue
        amp = (amp_lo + (amp_hi - amp_lo) * float(un)) * 1e-3 / MU_0
        accI = torch.zeros(len(ws), n_sig, dtype=torch.float64)
        accQ = torch.zeros(len(ws), n_sig, dtype=torch.float64)
        for k in range(spf):
            tk = (j * spf + k) * cfg.dt
            s0 = amp * math.sin(ws[0] * tk)
            sh = amp * math.sin(ws[0] * (tk + 0.5 * cfg.dt))
            s1 = amp * math.sin(ws[0] * (tk + cfg.dt))
            # One sign for the whole step: the boundary moves by at most one
            # step of 1 ps against a 200-step frame.
            q = 1.0 if k < n_drive else -quench_gain
            h0 = hz + unit_bus * s0 + unit_disk * (s0 * q)
            hh = hz + unit_bus * sh + unit_disk * (sh * q)
            h1 = hz + unit_bus * s1 + unit_disk * (s1 * q)
            if graphed:
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                    m = stepper(m, h0, hh, hh, h1).clone()
                except Exception as e:
                    print(f"  compiled step failed ({type(e).__name__}); "
                          f"falling back to eager", flush=True)
                    graphed, stepper = False, eager
                    m = stepper(m, h0, hh, hh, h1)
            else:
                m = stepper(m, h0, hh, hh, h1)
            p = arr.port_signals(m).double()
            for i, wi in enumerate(ws):
                accI[i] += p * math.cos(wi * tk)
                accQ[i] += p * math.sin(wi * tk)
        row = []
        for i in range(len(ws)):
            # .cpu() before .numpy(): mnn.set_device sets torch's GLOBAL default
            # device, so these accumulators were allocated on CUDA and the bare
            # .numpy() this was copied from raises there. The same unguarded
            # call is still in run_narma_chain.py, which has only ever been run
            # on CPU.
            A = ((accI[i] + 1j * accQ[i]) / spf).cpu().numpy()
            # Each disk's readout guides are their own ring, so they decompose
            # independently rather than as one twenty-port ring.
            for d in range(0, n_sig, npr):
                M = np.fft.fft(A[d:d + npr])
                for q in range(npr):
                    row += [M[q].real, M[q].imag]
        feats.append(row)
        if (j + 1) % CKPT_EVERY == 0:
            print(f"  frame {j+1}/{len(u)} ({time.time()-t0:.0f}s)", flush=True)
            save_ckpt(cache, torch.tensor(np.array(feats), dtype=torch.float64,
                                          device="cpu"), m)

    F = torch.tensor(np.array(feats), dtype=torch.float64, device="cpu")
    save_ckpt(cache, F.cpu(), m)
    return F


@torch.no_grad()
def probe_tau(arr, cfg, spf, carrier, amp, dtype, quench_frac, quench_gain,
              n_quiet=3000):
    """Ring-down time of the tap disks, in frames, under a given quench.

    Drive the disk bodies for exactly one frame, then go silent, and fit the
    decay of each disk's own readout. This is the number the whole argument
    turns on -- a disk that integrates ~8 input samples cannot make a product of
    two distinct ones -- so it is measured directly rather than taken from
    1/(alpha*omega).
    """
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    for k in range(arr.disk_masks.shape[0]):
        unit[:, :, 0, 0] += arr.disk_masks[k]
    n_drive = int(round(spf * (1.0 - quench_frac)))
    w = 2 * math.pi * carrier
    m = arr.m0.clone()
    hz = arr.h_zero
    npr, out = arr.n_readout, []
    for k in range(spf + n_quiet):
        tk = k * cfg.dt
        a0 = amp if k < spf else 0.0
        q = 1.0 if (k < n_drive or k >= spf) else -quench_gain
        s0 = a0 * q * math.sin(w * tk)
        sh = a0 * q * math.sin(w * (tk + 0.5 * cfg.dt))
        s1 = a0 * q * math.sin(w * (tk + cfg.dt))
        m = arr.rollout.rk4_step_fields(m, hz + unit * s0, hz + unit * sh,
                                        hz + unit * sh, hz + unit * s1)
        out.append(arr.port_signals(m).double().cpu().numpy().copy())
    sig = np.asarray(out)
    win = max(4, int(round(1e3 / (carrier / 1e9))))
    taus = []
    for d in range(arr.disk_masks.shape[0]):
        e = np.sqrt((sig[:, d * npr:(d + 1) * npr] ** 2).mean(axis=1))
        e = np.sqrt(np.convolve(e ** 2, np.ones(win) / win, mode="same"))
        tail = e[spf + win:]
        pk = float(tail.max()) if tail.size else 0.0
        good = tail > max(pk * 0.05, 1e-12)
        n = int(good.sum())
        if n < 50 or pk <= 0:
            taus.append(float("nan")); continue
        x = np.arange(n, dtype=float)
        sl = np.polyfit(x, np.log(tail[:n]), 1)[0]
        taus.append(float("inf") if sl >= 0 else -1.0 / sl / spf)
    return taus


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--splits", type=int, nargs=3, default=(100, 300, 100),
                   help="(washout, train, val); TEST is the remainder. Washout\n"
                        "must exceed the transit to the farthest tap -- 19\n"
                        "frames here -- or the line is still filling.")
    p.add_argument("--steps-per-frame", type=int, default=200)
    p.add_argument("--carrier-ghz", type=float, default=12.0)
    p.add_argument("--amp-lo-mT", type=float, default=10.0)
    p.add_argument("--amp-hi-mT", type=float, default=30.0)
    p.add_argument("--tones-ghz", type=float, nargs="*",
                   default=[9.9, 10.3, 13.7, 24.0],
                   help="extra lock-in tones; must match the chain runs or the\n"
                        "numbers this is compared against are a different\n"
                        "instrument")
    p.add_argument("--n-taps", type=int, default=4)
    p.add_argument("--lags", type=float, nargs="+", default=[5, 8, 11, 14])
    p.add_argument("--gap", type=float, default=15.0)
    p.add_argument("--tap-alpha", type=float, default=1.0)
    p.add_argument("--bus-alpha", type=float, default=1.0,
                   help="x1 is the as-built value and the only point in the\n"
                        "damping ladder that gave both 3/3 monotonic taps and a\n"
                        "mean spacing near the designed 3.0. Lower alpha buys\n"
                        "amplitude and scatters the timing.")
    p.add_argument("--bus-guide-width", type=float, default=None,
                   help="nm; None keeps the as-built 80. Narrowing it measured\n"
                        "WORSE at every tap once the record was long enough --\n"
                        "coupling goes as width, with no aperture null.")
    p.add_argument("--drive", choices=("bus", "fresh", "both"), default="both")
    p.add_argument("--quench-frac", type=float, default=0.0,
                   help="fraction of each frame spent driving the DISKS in\n"
                        "antiphase, cancelling their residual precession rather\n"
                        "than waiting for it to dissipate. Sets the effective\n"
                        "response time from the drive side, leaving alpha -- and\n"
                        "so Q = 1/(2*alpha) = 62.5, and the signal amplitude --\n"
                        "alone. 0 disables it.")
    p.add_argument("--quench-gain", type=float, default=1.0,
                   help="amplitude of the antiphase segment relative to the\n"
                        "drive segment")
    p.add_argument("--probe-tau", action="store_true",
                   help="measure the disk ring-down time and exit, instead of\n"
                        "running the task. Verifies the quench does what it\n"
                        "claims before a 600-frame run is spent on it.")
    p.add_argument("--fresh-scale", type=float, nargs="*", default=None,
                   help="per-tap fresh-drive amplitude. Default is the MEASURED\n"
                        "delayed amplitude at each tap, normalised to tap 1, so\n"
                        "the fresh and delayed operands arrive comparable.")
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="runs/narma_polytap")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pid").write_text(str(os.getpid()))

    u, y = narma10(a.frames, seed=a.seed)
    cfg, arr, geom, run = make_array(
        a.n_taps, a.lags, a.gap, 0.0, a.tap_alpha, a.steps_per_frame + 4, dtype,
        bus_alpha_mult=a.bus_alpha, bus_guide_width_nm=a.bus_guide_width,
        steps_per_frame=a.steps_per_frame)

    # The washout has to outlast the transit, for the same reason the delay
    # probe's record did: until the wave has crossed the array the far taps are
    # reporting near field, and a reservoir fitted through that window is fitted
    # on a state that does not yet exist.
    # 706 m/s, the LOCAL group velocity at the 12 GHz drive, not the 948 m/s
    # low-k phase velocity this used before. The faster figure underestimates
    # transit, which makes the guard below too lenient in exactly the direction
    # that has already cost four runs -- a record that ends before the wave
    # arrives reads the near field and reports it as a result.
    #
    # Note this is invariant under --steps-per-frame: frame_nm scales with the
    # frame, so lag*frame_nm/(spf*dt) does not move. That is the property the
    # coupled scaling exists to give.
    transit_frames = (max(a.lags[:a.n_taps]) * cfg.frame_nm / 706.0
                      / (a.steps_per_frame * cfg.dt))
    if a.splits[0] < transit_frames + 13:
        raise SystemExit(
            f"washout is {a.splits[0]} frames but the wave needs "
            f"{transit_frames:.0f}\nto reach the farthest tap, plus ~13 for the "
            f"disk to respond. Raise --splits.")
    print(f"poly-tap NARMA: {a.n_taps} taps, gap {a.gap:g} nm, "
          f"bus alpha x{a.bus_alpha:g}, drive={a.drive}\n"
          f"mesh {cfg.grid[0]}x{cfg.grid[1]}, {a.frames} frames x "
          f"{a.steps_per_frame} steps, device {a.device}\n"
          f"transit to the farthest tap {transit_frames:.1f} frames, "
          f"washout {a.splits[0]}", flush=True)

    ensure_m0(arr, outdir, run, relax_steps=a.relax_steps, dtype=dtype,
              log=lambda s: print(f"  {s}", flush=True))

    if a.probe_tau:
        amp = a.amp_hi_mT * 1e-3 / MU_0
        print(f"\nring-down of the tap disks, in frames "
              f"(1/(alpha*omega) predicts {1.0/(cfg.alpha*a.tap_alpha*2*math.pi*a.carrier_ghz*1e9)/(a.steps_per_frame*cfg.dt):.2f})")
        print(f"{'quench':>18} " + " ".join(f"{'disk '+str(k+1):>9}"
                                            for k in range(a.n_taps)))
        for qf, qg in ((0.0, 1.0), (a.quench_frac, a.quench_gain)):
            if qf == 0.0 and a.quench_frac == 0.0 and qg != 1.0:
                continue
            taus = probe_tau(arr, cfg, a.steps_per_frame, a.carrier_ghz * 1e9,
                             amp, dtype, qf, qg)
            lab = "off" if qf == 0.0 else f"frac {qf:g} gain {qg:g}"
            print(f"{lab:>18} " + " ".join(f"{t:>9.2f}" for t in taus),
                  flush=True)
            if qf == 0.0 and a.quench_frac == 0.0:
                break
        print("\nA disk that rings for ~8 frames averages its input over ~8\n"
              "samples, and cannot make a product of two distinct ones. Under\n"
              "1 frame is the target.")
        return 0

    fresh = a.fresh_scale
    if fresh is None:
        base = MEASURED_TAP_AMPS[:a.n_taps]
        fresh = [x / base[0] for x in base]
    print("fresh-drive scale per tap: "
          + ", ".join(f"{x:.4f}" for x in fresh), flush=True)

    tag = f"{run}_{a.drive}"
    F = run_reservoir(arr, cfg, u, a.steps_per_frame, a.carrier_ghz * 1e9,
                      a.amp_lo_mT, a.amp_hi_mT, dtype,
                      cache=outdir / f"features_{tag}.pt",
                      tones=[t * 1e9 for t in a.tones_ghz],
                      drive=a.drive, fresh_scale=fresh,
                      quench_frac=a.quench_frac, quench_gain=a.quench_gain)
    X = F.cpu().numpy()
    X = (X - X.mean(0)) / X.std(0).clip(1e-12)
    ac1 = float(np.nanmean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1]
                            for i in range(X.shape[1])]))
    print(f"\nreservoir lag-1 autocorrelation: {ac1:+.3f}", flush=True)

    splits = tuple(a.splits)
    results = {
        "input_only": fit_eval(u[:, None], y, splits),
        "linear_10lag": fit_eval(lag_matrix(u, 10), y, splits),
        "linear_20lag": fit_eval(lag_matrix(u, 20), y, splits),
        f"polytap_{tag}": fit_eval(X, y, splits),
        "lag1_autocorrelation": ac1,
        "drive": a.drive,
        "n_features": int(X.shape[1]),
    }
    (outdir / f"results_{tag}.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'readout':<28} {'dim':>5} {'NARMA-10 test NMSE':>20}")
    for name, dim in (("input_only", 1), ("linear_10lag", 10),
                      ("linear_20lag", 20), (f"polytap_{tag}", X.shape[1])):
        print(f"{name:<28} {dim:>5} {results[name]['nmse_test']:>20.4f}")

    dev = results[f"polytap_{tag}"]["nmse_test"]
    lin = min(results["linear_10lag"]["nmse_test"],
              results["linear_20lag"]["nmse_test"])
    print()
    if dev < lin:
        print(f"The device BEATS the linear filter, {dev:.4f} against {lin:.4f}.\n"
              f"That is the first time in this project, and it is the claim the\n"
              f"tapped-bus architecture was built to make.")
    else:
        print(f"The device does NOT beat the linear filter: {dev:.4f} against\n"
              f"{lin:.4f}. Resolved delays are a prerequisite for the products\n"
              f"this task needs, not a demonstration that the device makes them.\n"
              f"Compare the bus-only arm to see whether the co-drive did\n"
              f"anything at all.")
    print(f"\nwrote {outdir / f'results_{tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
