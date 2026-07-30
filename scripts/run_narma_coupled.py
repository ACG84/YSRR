#!/usr/bin/env python3
"""NARMA-10 through the COUPLED two-disk array, read out at both disks' ports.

The single-disk version asks whether one ported disk computes. This asks
whether coupling them buys anything, which is the claim the whole array
architecture rests on and the only reason to build more than one.

Measured inputs to this run, all on converged states:

    lambda -0.164/ns linked, -0.340/ns unlinked   stable, slightly negative,
                                                  which is the edge-of-chaos
                                                  point a reservoir wants
    B/A 0.00049 against 0.00000 dipolar-only      the link carries 121.7x

One limit carried forward, with its cause now identified. The first arrival
over the 700 nm link implied 17,500 m/s against a ~2,000 m/s magnon group
velocity, so the leading edge was magnetostatic rather than guided -- because
that link was 40 nm wide, and 40 x 20 nm is evanescent at 12 GHz (it reaches
Q = 10 only by 14 GHz). A link below its own cutoff is not a waveguide, so the
measured 121.7x transfer was stray field. The link here is 80 nm, matching the
ports, which is the width that propagates at 11 GHz and up.

The no-link control still deletes the link MATERIAL, so it cannot by itself
separate guided transport from the dipolar field of the extra material. What
distinguishes them is timing: a guided component must arrive no faster than the
group velocity allows, ~0.35 ns over 700 nm.

The baselines are the point, again. A second disk doubles the feature count,
and more features fit better whether or not the coupling does anything -- so
the comparison that matters is not against the input but against the SAME
readout with the link removed. If linked and unlinked score alike, the array is
two independent reservoirs sharing a substrate and the link is decoration.

    python scripts/run_narma_coupled.py
    python scripts/run_narma_coupled.py --no-link      # the control
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.reservoir import fit_eval, narma10
from magnonic_nn.vortex import CoupledPortedConfig, CoupledPortedArray


def lag_matrix(u, n_lags):
    out = np.zeros((len(u), n_lags))
    for k in range(n_lags):
        out[k:, k] = u[:len(u) - k]
    return out


@torch.no_grad()
def run_reservoir(arr, u, steps_per_frame, carrier, amp_lo, amp_hi, dtype,
                  cache=None):
    """Drive both disks, no reset between frames; return (n_frames, features)."""
    if cache is not None and cache.exists():
        print(f"[cached] {cache.name}", flush=True)
        return torch.load(cache, weights_only=False)

    cfg = arr.cfg
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    unit[:, :, :, 0] = arr.disk_only[:, :, :, 0]

    m = arr.m0.clone()
    n_ports_total = arr.port_signals(m).shape[0]
    w = 2 * math.pi * carrier
    feats, t0 = [], time.time()
    for j, un in enumerate(u):
        amp = (amp_lo + (amp_hi - amp_lo) * float(un)) * 1e-3 / MU_0
        accI = torch.zeros(n_ports_total, dtype=torch.float64)
        accQ = torch.zeros(n_ports_total, dtype=torch.float64)
        for k in range(steps_per_frame):
            tk = (j * steps_per_frame + k) * cfg.dt

            def h_drive(theta, tk=tk, amp=amp):
                return unit * (amp * math.sin(w * (tk + theta * cfg.dt)))

            m = arr.rollout.rk4_step(m, arr.h_zero, h_drive)
            p = arr.port_signals(m).double()
            accI += p * math.cos(w * tk)
            accQ += p * math.sin(w * tk)
        A = ((accI + 1j * accQ) / steps_per_frame).numpy()
        # each disk's ports decompose independently -- they are separate
        # six-port rings, not one twelve-port ring
        row = []
        for d in range(0, n_ports_total, cfg.n_ports):
            M = np.fft.fft(A[d:d + cfg.n_ports])
            for q in range(cfg.n_ports // 2 + 1):
                v = M[q] + (M[cfg.n_ports - q] if 0 < q < cfg.n_ports - q else 0.0)
                row += [v.real, v.imag, abs(v)]
        feats.append(row)
        if (j + 1) % 50 == 0:
            print(f"  frame {j+1}/{len(u)} ({time.time()-t0:.0f}s)", flush=True)

    F = torch.tensor(np.array(feats), dtype=torch.float64)
    if cache is not None:
        torch.save(F, cache)
    return F


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=900)
    p.add_argument("--splits", type=int, nargs=3, default=(120, 550, 230))
    p.add_argument("--steps-per-frame", type=int, default=200)
    p.add_argument("--carrier-ghz", type=float, default=12.0)
    p.add_argument("--amp-lo-mT", type=float, default=10.0)
    p.add_argument("--amp-hi-mT", type=float, default=30.0)
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--relax-steps", type=int, default=5000)
    p.add_argument("--require-tol", type=float, default=1e-3)
    p.add_argument("--link-width", type=float, default=80.0,
                   help="nm. 80 to match the ports, NOT the old 40 default: at\n"
                        "the disk's 20 nm thickness a 40 nm guide is evanescent\n"
                        "at 12 GHz, which is why the measured first arrival\n"
                        "implied 17500 m/s -- magnetostatic, not guided. A link\n"
                        "below its own cutoff is not a waveguide.")
    p.add_argument("--no-link", action="store_true",
                   help="the control: same two disks, link deleted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="runs/narma_coupled")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(args.device)
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    u, y = narma10(args.frames, seed=args.seed)
    cfg = CoupledPortedConfig(
        chirality=+1, chirality_b=-1,
        separation=args.separation * 1e-9,
        link_width=(0.0 if args.no_link else args.link_width * 1e-9))
    arr = CoupledPortedArray(cfg, timesteps=args.steps_per_frame + 4, dtype=dtype)
    t0 = time.time()
    # 5000 steps with a hard guard: every coupled lambda in this project
    # measured at 900 steps was reporting relaxation drift as dynamics, and a
    # reservoir run on a drifting ground state has the same problem silently.
    arr.relax(steps=args.relax_steps, require_tol=args.require_tol)
    tau_ns = 1.0 / (cfg.alpha * 2 * math.pi * args.carrier_ghz * 1e9) * 1e9
    frame_ns = args.steps_per_frame * cfg.dt * 1e9
    print(f"relaxed {time.time()-t0:.0f}s  link={not args.no_link}\n"
          f"frame {frame_ns:.2f} ns, ring-down {tau_ns:.2f} ns "
          f"-> ~{tau_ns/frame_ns:.1f} frames of memory", flush=True)

    tag = "nolink" if args.no_link else "linked"
    F = run_reservoir(arr, u, args.steps_per_frame, args.carrier_ghz * 1e9,
                      args.amp_lo_mT, args.amp_hi_mT, dtype,
                      cache=outdir / f"features_{tag}.pt")
    X = F.numpy()
    X = (X - X.mean(0)) / X.std(0).clip(1e-12)
    ac1 = float(np.nanmean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1]
                            for i in range(X.shape[1])]))
    print(f"\nreservoir lag-1 autocorrelation: {ac1:+.3f}", flush=True)

    splits = tuple(args.splits)
    results = {
        "input_only": fit_eval(u[:, None], y, splits),
        "linear_10lag": fit_eval(lag_matrix(u, 10), y, splits),
        f"coupled_{tag}": fit_eval(X, y, splits),
        "lag1_autocorrelation": ac1,
        "link": not args.no_link,
    }
    (outdir / f"results_{tag}.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'readout':<20} {'dim':>5} {'NARMA-10 test NMSE':>20}")
    for name, dim in (("input_only", 1), ("linear_10lag", 10),
                      (f"coupled_{tag}", X.shape[1])):
        print(f"{name:<20} {dim:>5} {results[name]['nmse_test']:>20.4f}")

    other = outdir / ("results_linked.json" if args.no_link
                      else "results_nolink.json")
    print()
    if other.exists():
        o = json.loads(other.read_text())
        okey = next(k for k in o if k.startswith("coupled_"))
        mine = results[f"coupled_{tag}"]["nmse_test"]
        theirs = o[okey]["nmse_test"]
        lo, hi = (mine, theirs) if not args.no_link else (theirs, mine)
        print(f"linked {lo:.4f}  vs  no-link {hi:.4f}")
        if lo < hi * 0.9:
            print("The link contributes: coupling the disks predicts better than")
            print("the same two disks measured independently.")
        else:
            print("The link does not contribute. Two disks sharing a substrate,")
            print("with the guide as decoration -- the extra features come from")
            print("having a second reservoir, not from coupling them.")
    else:
        print(f"Run the other arm to compare: "
              f"{'--no-link' if not args.no_link else '(without --no-link)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
