#!/usr/bin/env python3
"""NARMA-10 through an N-stage CHAIN, driven at stage 1 only.

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
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.reservoir import fit_eval, narma10
from magnonic_nn.vortex import ChainPortedConfig, ChainPortedArray
from magnonic_nn._compat import get_device


def lag_matrix(u, n_lags):
    out = np.zeros((len(u), n_lags))
    for k in range(n_lags):
        out[k:, k] = u[:len(u) - k]
    return out


CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "5"))


def save_ckpt(cache, feats, m):
    """Atomic checkpoint: temp file, fsync, rename.

    Without this a coupled run cannot finish on this machine at all. The mesh is
    248x108 against a single disk's 108x108, so a 600-frame run is ~55 minutes,
    and the container reboots every 10-30. The previous all-or-nothing cache
    ("if it exists, load it; otherwise compute everything and save at the end")
    guarantees zero progress under those conditions -- the exact failure that
    banked no frames for 24 minutes earlier in this project.
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
def run_reservoir(arr, u, steps_per_frame, carrier, amp_lo, amp_hi, dtype,
                  cache=None, tones=(), drive_stages=(0,)):
    """Drive, no reset between frames; return (n_frames, features).

    drive="one" is the CASCADE topology and the point of this script: only disk
    A is driven, so disk B sees the input solely through the link. Driving both
    (drive="both", the previous behaviour) gives two parallel reservoirs sharing
    a substrate, which tests whether coupling helps a wider readout -- a
    different question from whether two stages COMPOSE their memory.

    Readout matches run_narma_modal exactly: the same five tones, per-disk
    cross-port DFT, signed modes, real and imaginary parts only. It has to, or
    the single-disk memory numbers this is compared against are measuring a
    different instrument.
    """
    cfg = arr.cfg
    done, m_resume = [], None
    if cache is not None and cache.exists():
        try:
            prev = torch.load(cache, weights_only=False)
            if isinstance(prev, dict):
                feats_prev, m_resume = prev["feats"], prev.get("m")
            else:
                feats_prev = prev
            if len(feats_prev) >= len(u):
                print(f"[cached] {cache.name}", flush=True)
                return feats_prev
            done = [list(r) for r in feats_prev.tolist()]
            print(f"[resume] {cache.name} has {len(done)}/{len(u)} frames"
                  + (" (state restored)" if m_resume is not None else ""),
                  flush=True)
        except Exception as e:
            print(f"[resume] {cache.name} unreadable ({type(e).__name__}); "
                  "starting from frame 0", flush=True)

    # Drive the disk BODIES, never the guides: a drive on a guide injects
    # straight into the readout and the ports would be measuring the input.
    #
    # More than one stage may be driven, and that is the point of the co-drive
    # arm. A nonlinearity can only multiply signals that are present in the SAME
    # state at the SAME time. The capacity decomposition showed the disk is
    # strongly nonlinear over this very amplitude range, and that every product
    # it makes is between samples one or two frames apart -- because the
    # nonlinearity sits at stage 1 and stage 1 only remembers lags 0-6. The deep
    # stages remember out to lag 14 and have no fresh input to mix against.
    # Feeding u to a deep stage as well puts the delayed copy (~9.5 frames
    # through the chain) and the fresh sample in one nonlinear element.
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    for st in drive_stages:
        unit[:, :, 0, 0] += arr.disk_masks[st].to(dtype)

    # Graph-capturable stepper, for the same reason run_narma_modal uses one:
    # rk4_step takes a Python callable and evaluates it per substep, which forces
    # a graph break every time. Measured on the single disk that is 0.63 ms/step
    # captured against 9.65 eager. It also explains why the GPU was previously
    # written off for the array runs at "1.0-1.5x" -- those runs never used this
    # path, so they were paying ~4 us of launch overhead on ~4.8 us of work, 520
    # kernels per step. Off CUDA this returns the eager function and the loop is
    # identical, so nothing changes on CPU.
    stepper, graphed = arr.rollout.graph_stepper()
    hz = arr.h_zero
    m = arr.m0.clone() if m_resume is None else m_resume.clone().to(dtype)
    start = len(done) if m_resume is not None else 0
    n_ports_total = arr.port_signals(m).shape[0]
    ws = [2 * math.pi * f for f in (carrier, *tones)]
    feats, t0 = list(done), time.time()
    for j, un in enumerate(u):
        if j < start:
            continue
        amp = (amp_lo + (amp_hi - amp_lo) * float(un)) * 1e-3 / MU_0
        accI = torch.zeros(len(ws), n_ports_total, dtype=torch.float64)
        accQ = torch.zeros(len(ws), n_ports_total, dtype=torch.float64)
        for k in range(steps_per_frame):
            tk = (j * steps_per_frame + k) * cfg.dt
            # h at theta = 0, 1/2, 1/2, 1; the two half-step fields coincide.
            s0 = amp * math.sin(ws[0] * tk)
            sh = amp * math.sin(ws[0] * (tk + 0.5 * cfg.dt))
            s1 = amp * math.sin(ws[0] * (tk + cfg.dt))
            h0, hh, h1 = hz + unit * s0, hz + unit * sh, hz + unit * s1
            if graphed:
                torch.compiler.cudagraph_mark_step_begin()
                m = stepper(m, h0, hh, hh, h1).clone()
            else:
                m = stepper(m, h0, hh, hh, h1)
            p = arr.port_signals(m).double()
            for i, wi in enumerate(ws):
                accI[i] += p * math.cos(wi * tk)
                accQ[i] += p * math.sin(wi * tk)
        row = []
        for i in range(len(ws)):
            A = ((accI[i] + 1j * accQ[i]) / steps_per_frame).numpy()
            # each disk's ports are their own six-port ring, so they decompose
            # independently rather than as one twelve-port ring
            for d in range(0, n_ports_total, cfg.n_ports):
                M = np.fft.fft(A[d:d + cfg.n_ports])
                for q in range(cfg.n_ports):
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
    p.add_argument("--frames", type=int, default=900)
    p.add_argument("--splits", type=int, nargs=3, default=(120, 450, 130),
                   help="(washout, train, val); TEST is the remainder, so\n"
                        "these must sum to less than --frames.")
    p.add_argument("--steps-per-frame", type=int, default=200)
    p.add_argument("--carrier-ghz", type=float, default=12.0)
    p.add_argument("--amp-lo-mT", type=float, default=10.0)
    p.add_argument("--amp-hi-mT", type=float, default=30.0)
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--relax-steps", type=int, default=10000)
    p.add_argument("--require-tol", type=float, default=2e-3)
    p.add_argument("--link-width", type=float, default=80.0,
                   help="nm. 80 to match the ports, NOT the old 40 default: at\n"
                        "the disk's 20 nm thickness a 40 nm guide is evanescent\n"
                        "at 12 GHz, which is why the measured first arrival\n"
                        "implied 17500 m/s -- magnetostatic, not guided. A link\n"
                        "below its own cutoff is not a waveguide.")
    p.add_argument("--no-link", action="store_true",
                   help="the control: same two disks, link deleted")
    p.add_argument("--tones-ghz", type=float, nargs="*",
                   default=[9.9, 10.3, 13.7, 24.0],
                   help="extra lock-in tones; must match run_narma_modal or the\n"
                        "single-disk numbers this is compared against are a\n"
                        "different instrument")
    p.add_argument("--n-disks", type=int, default=3)
    p.add_argument("--Ms", type=float, default=None,
                   help="A/m. Permalloy is 800e3. A low-moment disk at matched\n"
                        "R/l_ex is the only element measured here whose\n"
                        "nonlinearity is reachable at the drive its coupling\n"
                        "delivers -- it REVERSES its core at ~2 mT rather than\n"
                        "compressing.")
    p.add_argument("--radius", type=float, default=None,
                   help="nm. Must be scaled with the exchange length when Ms is\n"
                        "lowered, or the vortex dissolves: l_ex = 5.7 nm at\n"
                        "800 kA/m and 10.1 at 450, so 100 nm pairs with 178.")
    p.add_argument("--relax-alpha", type=float, default=0.5,
                   help="a low-moment disk has long-wavelength modes that damp\n"
                        "slowly; 1.0 converges the 178 nm disk where 0.5 does not")
    p.add_argument("--drive-stages", type=int, nargs="+", default=[0],
                   help="which stages receive the drive; 0 is the input stage.\n"
                        "More than one co-drives: `--drive-stages 0 2` gives the\n"
                        "deep stage both the chain-delayed copy and the fresh\n"
                        "sample, which is what a cross-lag product needs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="runs/narma_coupled")
    args = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(args.device)
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    # Record the PID so a watchdog can tell THIS run from the other cascade arm.
    # It cannot be done from the process table: magnum.np renames the worker to
    # "magnumnp scripts/run_narma_coupled.py" a few minutes in, destroying argv
    # and with it the --outdir that distinguishes linked from nolink. A watchdog
    # matching on argv therefore sees zero running and launches a duplicate onto
    # a live checkpoint -- two writers, one file, frames going backwards.
    (outdir / "pid").write_text(str(os.getpid()))

    u, y = narma10(args.frames, seed=args.seed)
    kw = {}
    if args.Ms is not None:
        kw["Ms"] = args.Ms
    if args.radius is not None:
        kw["radius"] = args.radius * 1e-9
    cfg = ChainPortedConfig(
        n_disks=args.n_disks,
        separation=args.separation * 1e-9,
        link_width=(0.0 if args.no_link else args.link_width * 1e-9), **kw)
    arr = ChainPortedArray(cfg, timesteps=args.steps_per_frame + 4, dtype=dtype)
    t0 = time.time()
    # 5000 steps with a hard guard: every coupled lambda in this project
    # measured at 900 steps was reporting relaxation drift as dynamics, and a
    # reservoir run on a drifting ground state has the same problem silently.
    tag = f"n{args.n_disks}_" + ("nolink" if args.no_link else "linked")
    if args.Ms is not None or args.radius is not None:
        tag += f"_r{cfg.radius*1e9:.0f}_Ms{cfg.Ms/1e3:.0f}"
    # Cache the relaxed ground state. It is deterministic given the config, and
    # it costs minutes on this mesh -- which every restart would otherwise pay
    # again before even reading the checkpoint, on a container that reboots
    # every 10-30 minutes. The guard below still runs the first time.
    m0_cache = outdir / f"m0_{tag}.pt"
    if m0_cache.exists():
        try:
            arr.m0 = torch.load(m0_cache, weights_only=False).to(device=get_device(), dtype=dtype)
            print(f"[m0] restored from {m0_cache.name}", flush=True)
        except Exception as e:
            print(f"[m0] {m0_cache.name} unreadable ({type(e).__name__}); "
                  "relaxing", flush=True)
            arr.relax(steps=args.relax_steps, alpha_relax=args.relax_alpha,
                      require_tol=args.require_tol)
            torch.save(arr.m0.cpu(), m0_cache)
    else:
        arr.relax(steps=args.relax_steps, alpha_relax=args.relax_alpha,
                  require_tol=args.require_tol)
        torch.save(arr.m0.cpu(), m0_cache)
    tau_ns = 1.0 / (cfg.alpha * 2 * math.pi * args.carrier_ghz * 1e9) * 1e9
    frame_ns = args.steps_per_frame * cfg.dt * 1e9
    print(f"relaxed {time.time()-t0:.0f}s  link={not args.no_link}\n"
          f"frame {frame_ns:.2f} ns, ring-down {tau_ns:.2f} ns "
          f"-> ~{tau_ns/frame_ns:.1f} frames of memory", flush=True)

    tag = f"n{args.n_disks}_" + ("nolink" if args.no_link else "linked")
    if args.Ms is not None or args.radius is not None:
        tag += f"_r{cfg.radius*1e9:.0f}_Ms{cfg.Ms/1e3:.0f}"
    if list(args.drive_stages) != [0]:
        tag += "_drv" + "".join(str(s) for s in args.drive_stages)
    F = run_reservoir(arr, u, args.steps_per_frame, args.carrier_ghz * 1e9,
                      args.amp_lo_mT, args.amp_hi_mT, dtype,
                      cache=outdir / f"features_{tag}.pt",
                      tones=[t * 1e9 for t in args.tones_ghz],
                      drive_stages=tuple(args.drive_stages))
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
