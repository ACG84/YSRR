#!/usr/bin/env python3
"""Does hysteretic microstate memory satisfy the echo state property?

THE question for this architecture. A reservoir must FORGET: two different
initial states, driven by the same input sequence, have to converge, or the
readout is reading initial conditions rather than input. Non-volatile magnetic
memory is exactly what that forbids -- so the fact that the microstate persists,
which is what makes it attractive as memory, is also what puts ESP at risk.

The standard test, applied here: run the SAME pseudorandom field sequence
through two MAXIMALLY DIFFERENT initial states and watch the distance between
them. ESP holds if it decays to zero. It is a state-contraction test and it
needs no task, no readout and no training.

The measured switching fields bracket the answer in advance
(check_asvi_hysteresis, same element):

    30 nm hard layer   switches at +-60 mT
    20 nm soft layer   switches at +-40 mT, vortex window 40-50 mT
    separation         20 mT, so the layers are independently addressable

which predicts three regimes, and the sweep exists to find the boundaries:

    below ~40 mT   nothing switches. The two trajectories never meet, distance
                   stays at its initial value, ESP FAILS, and no input reaches
                   the state at all. Infinite memory is not good memory.
    above ~60 mT   both layers follow the field regardless of history. Distance
                   collapses in one step, ESP holds TRIVIALLY, and the state
                   carries no information about anything but the last input.
                   Capacity zero.
    40-60 mT       partial, history-dependent switching. The only regime that
                   can compute, and the question is whether ESP holds in it.

So convergence alone is not a pass. A pass needs convergence AND state
diversity: the trajectory must visit more than one state, or the device has
merely saturated. Both are reported, and the verdict requires both.

    python scripts/check_asvi_esp.py --device cuda --amps-mT 30 45 55 70
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.spinice import ASVIConfig, ASVIIsland


def label(isl, m):
    """Compact microstate label, per layer, using gate one's discriminator."""
    out = []
    for k in range(isl.n_layers):
        pk, _ = isl.layer_core(m, k)
        mx, my = isl.layer_moment(m, k)
        ip = (mx**2 + my**2) ** 0.5
        cc = isl.layer_circulation(m, k)
        if ip < 0.5 and pk >= 0.5:
            out.append("A" if cc > 0 else "C")     # vortex ACW / CW
        elif ip >= 0.5:
            out.append("+" if mx > 0 else "-")
        else:
            out.append("?")
    return "".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--amps-mT", type=float, nargs="+", default=[30, 45, 55, 70],
                   help="peak drive amplitude per sweep point. Defaults bracket\n"
                        "the measured 40 mT soft and 60 mT hard coercivities.")
    p.add_argument("--n-steps", type=int, default=25, help="input sequence length")
    p.add_argument("--settle", type=int, default=1500,
                   help="relax steps per input sample; must exceed the switching\n"
                        "time at alpha_relax or the input never lands")
    p.add_argument("--alpha-relax", type=float, default=0.5)
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--axis", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--starts", nargs="+",
                   default=["macro+/macro+", "macro-/macro-"],
                   help="initial states. Maximally different by default: if THESE\n"
                        "converge, less separated ones will.")
    p.add_argument("--length-nm", type=float, default=550.0)
    p.add_argument("--width-nm", type=float, default=140.0)
    p.add_argument("--offset-nm", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--dx-nm", type=float, default=5.0)
    p.add_argument("--outdir", default="runs/asvi_esp")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = ASVIConfig(length=a.length_nm * 1e-9, width=a.width_nm * 1e-9,
                     layer_offset=a.offset_nm * 1e-9, alpha=a.alpha,
                     dx=a.dx_nm * 1e-9, dz=a.dx_nm * 1e-9)
    nx, ny, nz = cfg.grid
    isl = ASVIIsland(cfg, timesteps=32, dtype=dtype)
    mask = isl.mask
    n_mag = float(mask.sum())
    print(f"ASVI island, mesh {nx}x{ny}x{nz} = {nx*ny*nz:,} cells, "
          f"{int(n_mag):,} magnetic")
    print(f"starts {a.starts}, {a.n_steps} inputs, {a.settle} steps each, "
          f"seed {a.seed}", flush=True)

    # ONE input sequence, shared by every amplitude and every start, so a
    # difference between rows is the amplitude and not a different draw.
    rng = np.random.default_rng(a.seed)
    u = rng.uniform(-1.0, 1.0, size=a.n_steps)
    print(f"input u[n] (shared): " + " ".join(f"{v:+.2f}" for v in u[:10])
          + (" ..." if a.n_steps > 10 else ""), flush=True)

    # Relax each start ONCE. The initial states do not depend on the drive, and
    # re-relaxing them per amplitude was a quarter of a run that has now been
    # killed mid-sweep by the host three times. Cloned per amplitude, in
    # process -- not reloaded from disk, which is the CUDA corruption path.
    starts0 = [isl.relax(s.split("/"), steps=a.relax_steps, dtype=dtype).clone()
               for s in a.starts]
    print("relaxed starts: " + "  ".join(
        f"{s} -> {label(isl, m)}" for s, m in zip(a.starts, starts0)), flush=True)

    rows = []
    for amp in a.amps_mT:
        print(f"\n===== peak {amp:g} mT =====", flush=True)
        ms = [m.clone() for m in starts0]
        d0 = float((ms[0] - ms[1]).norm() / (2 * n_mag) ** 0.5)
        print(f"{'n':>4} {'u':>7} " + " ".join(f"{'st'+str(i):>5}" for i in
                                               range(len(ms)))
              + f" {'distance':>10} {'rel':>7}")
        print(f"{0:>4} {'':>7} " + " ".join(f"{label(isl, m):>5}" for m in ms)
              + f" {d0:>10.4f} {1.0:>7.3f}", flush=True)
        hist, seen, t0 = [], set(), time.time()
        for n in range(a.n_steps):
            h = torch.zeros(nx, ny, nz, 3, dtype=dtype)
            h[:, :, :, a.axis] = (float(u[n]) * amp * 1e-3 / MU_0) * mask[:, :, :, 0]
            ms = [isl.rollout.relax(m, h, a.settle, a.alpha_relax) for m in ms]
            d = float((ms[0] - ms[1]).norm() / (2 * n_mag) ** 0.5)
            labs = [label(isl, m) for m in ms]
            seen.update(labs)
            hist.append({"n": n + 1, "u": float(u[n]), "labels": labs,
                         "distance": d, "rel": d / max(d0, 1e-30)})
            print(f"{n+1:>4} {u[n]:>+7.2f} " + " ".join(f"{l:>5}" for l in labs)
                  + f" {d:>10.4f} {d/max(d0,1e-30):>7.3f}", flush=True)
        rows.append({"amp_mT": amp, "d0": d0, "history": hist,
                     "states_seen": sorted(seen),
                     "seconds": round(time.time() - t0, 1)})
        (outdir / "asvi_esp.json").write_text(json.dumps(rows, indent=2))

    # ------------------------------------------------------------- verdict
    print(f"\n{'':=<78}")
    print(f"{'peak mT':>8} {'final rel':>10} {'converged at':>13} "
          f"{'states seen':>28} {'verdict':>10}")
    CONV = 0.05        # relative distance counting as converged
    for r in rows:
        rel = [h["rel"] for h in r["history"]]
        hit = next((h["n"] for h in r["history"] if h["rel"] <= CONV), None)
        # POST-CONVERGENCE only. States visited BEFORE convergence are the
        # transient from two deliberately different initial conditions, not
        # computation, and counting them scored a run USEFUL that converges at
        # step 4 and then sits on one state for the next ten inputs. What
        # matters is whether the state still moves once the initial condition
        # has been forgotten.
        post = [h for h in r["history"] if hit is not None and h["n"] >= hit]
        st = [h["labels"][0] for h in post]
        nstate, live = len(set(st)), len(set(st[-5:])) > 1
        if hit is None:
            v = "ESP FAILS"
        elif nstate <= 1:
            v = "trivial"
        elif not live:
            v = "FREEZES"
        else:
            v = "USEFUL"
        print(f"{r['amp_mT']:>8.0f} {rel[-1]:>10.3f} "
              f"{(str(hit) if hit else 'never'):>13} "
              f"{','.join(sorted(set(st))) if post else '-':>28} {v:>10}")
    print(f"\nconverged = relative distance <= {CONV:g}. A run that converges but\n"
          "visits one state has saturated, not computed: it forgets the input as\n"
          "completely as it forgets the initial condition. USEFUL needs both\n"
          "convergence and more than one state, over more than one step.")
    def useful(r):
        hit = next((h["n"] for h in r["history"] if h["rel"] <= CONV), None)
        if hit is None:
            return False
        st = [h["labels"][0] for h in r["history"] if h["n"] >= hit]
        return len(set(st)) > 1 and len(set(st[-5:])) > 1
    good = [r["amp_mT"] for r in rows if useful(r)]
    print(f"\namplitudes with fading memory AND state diversity: "
          f"{good if good else 'NONE'}")
    if not good:
        print("  No amplitude gives both. Either the window is narrower than the\n"
              "  sweep resolution, or a single island is too small a state space\n"
              "  to hold memory without also latching -- which an ARRAY, where\n"
              "  neighbours supply the partial fields a uniform drive cannot,\n"
              "  would be the way to test.")
    print(f"\nwrote {outdir / 'asvi_esp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
