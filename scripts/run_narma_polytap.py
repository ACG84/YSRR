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
                  cache=None, tones=(), drive="both", fresh_scale=None):
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
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    if drive in ("bus", "both"):
        unit[:, :, 0, 2] += arr.inject_mask
    if drive in ("fresh", "both"):
        for k, s in enumerate(fresh_scale):
            unit[:, :, 0, 0] += float(s) * arr.disk_masks[k]

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
            h0, hh, h1 = hz + unit * s0, hz + unit * sh, hz + unit * s1
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
            A = ((accI[i] + 1j * accQ[i]) / spf).numpy()
            # Each disk's readout guides are their own ring, so they decompose
            # independently rather than as one twenty-port ring.
            for d in range(0, n_sig, npr):
                M = np.fft.fft(A[d:d + npr])
                for q in range(npr):
                    row += [M[q].real, M[q].imag]
        feats.append(row)
        if (j + 1) % CKPT_EVERY == 0:
            print(f"  frame {j+1}/{len(u)} ({time.time()-t0:.0f}s)", flush=True)
            save_ckpt(cache, torch.tensor(np.array(feats), dtype=torch.float64), m)

    F = torch.tensor(np.array(feats), dtype=torch.float64)
    save_ckpt(cache, F.cpu(), m)
    return F


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
        bus_alpha_mult=a.bus_alpha, bus_guide_width_nm=a.bus_guide_width)

    # The washout has to outlast the transit, for the same reason the delay
    # probe's record did: until the wave has crossed the array the far taps are
    # reporting near field, and a reservoir fitted through that window is fitted
    # on a state that does not yet exist.
    transit_frames = (max(a.lags[:a.n_taps]) * cfg.frame_nm / 948.0
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
                      drive=a.drive, fresh_scale=fresh)
    X = F.numpy()
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
