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
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.spinice import (ASVIConfig, ASVIIsland,
                                 ASVIVertexConfig, ASVIVertex)


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
    p.add_argument("--angle-input", type=float, default=None,
                   help="encode the input as field ANGLE at FIXED magnitude,\n"
                        "sweeping u[n] over +-this many degrees about --field-deg.\n"
                        "Amplitude encoding drives along ONE axis of a state\n"
                        "space whose traps sit off that axis: the vertex falls\n"
                        "into the interlayer-antiparallel state below 90 mT and\n"
                        "saturates above it, with no window between. A rotating\n"
                        "field can walk between states without saturating, which\n"
                        "is why ASI reservoir work uses angle.")
    p.add_argument("--field-deg", type=float, default=None,
                   help="in-plane field direction in degrees from +x. Square ASI\n"
                        "has TWO sublattices at 90 degrees, so a field along a\n"
                        "lattice axis drives one and leaves the other transverse\n"
                        "to its own shape anisotropy, unable to switch at any\n"
                        "amplitude. 45 degrees addresses both. Overrides --axis.")
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
    p.add_argument("--lattice", type=int, nargs=2, default=None,
                   metavar=("NVX", "NVY"),
                   help="islands on the bonds of an NVX x NVY VERTEX lattice:\n"
                        "2*n*(n-1) islands, so 24 for 4x4. Overrides --vertex.")
    p.add_argument("--vertex", type=int, default=0,
                   help="islands meeting at a square-ASI vertex. 0 = the single\n"
                        "island; 2 = the minimal motif (one x-island, one\n"
                        "y-island); 4 = the full vertex.")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint file. Written after EVERY input step and\n"
                        "reloaded on restart, so a run longer than the host's\n"
                        "uptime survives being killed part-way.")
    p.add_argument("--chunk", type=int, default=0,
                   help="stop cleanly after this many input steps this\n"
                        "invocation (0 = run to the end). Sizes one leg of the\n"
                        "run to fit inside the window the host actually gives.")
    p.add_argument("--outdir", default="runs/asvi_esp")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    kw = dict(length=a.length_nm * 1e-9, width=a.width_nm * 1e-9,
              layer_offset=a.offset_nm * 1e-9, alpha=a.alpha,
              dx=a.dx_nm * 1e-9, dz=a.dx_nm * 1e-9)
    if a.lattice or a.vertex:
        P = (ASVIVertexConfig.square_lattice(*a.lattice, a.length_nm, 125.0)
             if a.lattice else
             ASVIVertexConfig.square_vertex(a.length_nm, 125.0, a.vertex))
        cfg = ASVIVertexConfig(placements=P, **kw)
        isl = ASVIVertex(cfg, timesteps=32, dtype=dtype)
        n_parts, lab = isl.n_parts, isl.label
    else:
        cfg = ASVIConfig(**kw)
        isl = ASVIIsland(cfg, timesteps=32, dtype=dtype)
        n_parts, lab = isl.n_layers, lambda m: label(isl, m)
    nx, ny, nz = cfg.grid
    mask = isl.mask
    n_mag = float(mask.sum())
    print(f"ASVI island, mesh {nx}x{ny}x{nz} = {nx*ny*nz:,} cells, "
          f"{int(n_mag):,} magnetic")
    fdir = (f"angle-encoded, {a.field_deg or 45:g} +- {a.angle_input:g} deg at "
            f"fixed magnitude" if a.angle_input is not None
            else f"{a.field_deg:g} deg from +x" if a.field_deg is not None
            else f"along {'xyz'[a.axis]}")
    print(f"starts {a.starts}, {a.n_steps} inputs, {a.settle} steps each, "
          f"field {fdir}, seed {a.seed}", flush=True)
    if a.vertex and a.vertex > 1 and a.field_deg is None and a.angle_input is None:
        print("  WARNING: a vertex has two sublattices at 90 degrees and the\n"
              "  field is along a lattice axis, so the transverse sublattice\n"
              "  cannot switch at any amplitude and will latch. Pass\n"
              "  --field-deg 45 to address both.", flush=True)

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
    def expand(spec):
        # A start is written per ISLAND ("macro+/macro+") and repeated across
        # islands, so the same string means the same thing for one island and
        # for a vertex and the two runs stay comparable.
        parts = spec.split("/")
        if len(parts) == n_parts:
            return parts
        if n_parts % len(parts) == 0:
            return parts * (n_parts // len(parts))
        raise SystemExit(f"start {spec!r} has {len(parts)} labels; need "
                         f"{n_parts} or a divisor of it")

    # ------------------------------------------------------- checkpointing
    # A 4x4 lattice ESP run is about two hours and the host gives roughly one,
    # so the run has to survive being killed. It is NOT enough to write state:
    # a stale checkpoint silently resumed under different geometry is the same
    # class of bug as the m0 reuse that corrupted two delay points sharing an
    # outdir. So every checkpoint carries a fingerprint of the things that
    # would make it meaningless, and a mismatch refuses rather than resumes.
    idx = mask[..., 0].bool()          # store magnetic cells only: 700k not 5M
    def fingerprint():
        return {"grid": [nx, ny, nz], "n_mag": int(n_mag), "n_parts": n_parts,
                "starts": list(a.starts), "seed": a.seed,
                "n_steps": a.n_steps, "settle": a.settle,
                "relax_steps": a.relax_steps, "amps": list(a.amps_mT),
                "angle_input": a.angle_input, "field_deg": a.field_deg,
                "alpha_relax": a.alpha_relax, "lattice": a.lattice,
                "vertex": a.vertex, "dx_nm": a.dx_nm}
    def pack(ts):
        return [t[idx].cpu().clone() for t in ts]
    def unpack(packed):
        out = []
        for q in packed:
            full = torch.zeros(nx, ny, nz, 3, dtype=dtype)
            full[idx] = q.to(full.dtype)
            out.append(full.to(mask.device) if hasattr(mask, "device") else full)
        return out
    def save_ckpt(**kw):
        if not a.ckpt:
            return
        tmp = str(a.ckpt) + ".tmp"
        torch.save({"fp": fingerprint(), **kw}, tmp)
        os.replace(tmp, a.ckpt)          # atomic: a kill mid-write cannot
                                         # leave a half-file that loads
    ck = None
    if a.ckpt and os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        if ck.get("fp") != fingerprint():
            diff = [k for k, v in fingerprint().items()
                    if ck.get("fp", {}).get(k) != v]
            raise SystemExit(
                f"checkpoint {a.ckpt} does not match this run; differs in "
                f"{diff}.\nRefusing to resume -- delete it to start over. "
                "Resuming a mismatched\nstate is how the m0 reuse corrupted a "
                "sweep without announcing it.")
        print(f"resumed from {a.ckpt}: {ck['done']}/{a.n_steps} inputs done",
              flush=True)

    if ck is not None:
        starts0 = unpack(ck["starts0"])
    else:
        starts0 = [isl.relax(expand(s), steps=a.relax_steps, dtype=dtype).clone()
                   for s in a.starts]
    print("relaxed starts: " + "  ".join(
        f"{s} -> {lab(m)}" for s, m in zip(a.starts, starts0)), flush=True)

    rows = []
    for amp in a.amps_mT:
        print(f"\n===== peak {amp:g} mT =====", flush=True)
        ms = [m.clone() for m in starts0]
        d0 = float((ms[0] - ms[1]).norm() / (2 * n_mag) ** 0.5)
        n_done, hist0, seen0 = 0, [], []
        if ck is not None and ck["amp"] == amp:
            ms, n_done = unpack(ck["ms"]), ck["done"]
            hist0, seen0, d0 = ck["hist"], ck["seen"], ck["d0"]
        print(f"{'n':>4} {'u':>7} " + " ".join(f"{'st'+str(i):>9}" for i in
                                               range(len(ms)))
              + f" {'distance':>10} {'rel':>7}")
        print(f"{0:>4} {'':>7} " + " ".join(f"{lab(m):>9}" for m in ms)
              + f" {d0:>10.4f} {1.0:>7.3f}", flush=True)
        hist, seen, t0 = list(hist0), set(seen0), time.time()
        for h_ in hist:
            print(f"{h_['n']:>4} {h_['u']:>+7.2f} "
                  + " ".join(f"{l:>9}" for l in h_["labels"])
                  + f" {h_['distance']:>10.4f} {h_['rel']:>7.3f}", flush=True)
        for n in range(n_done, a.n_steps):
            h = torch.zeros(nx, ny, nz, 3, dtype=dtype)
            if a.angle_input is not None:
                # Constant magnitude, direction carries the input. The state is
                # then walked around the easy-axis landscape rather than driven
                # along one axis of it.
                base = a.field_deg if a.field_deg is not None else 45.0
                th = math.radians(base + float(u[n]) * a.angle_input)
                g = amp * 1e-3 / MU_0
                h[:, :, :, 0] = g * math.cos(th) * mask[:, :, :, 0]
                h[:, :, :, 1] = g * math.sin(th) * mask[:, :, :, 0]
                ms = [isl.rollout.relax(m, h, a.settle, a.alpha_relax) for m in ms]
                d = float((ms[0] - ms[1]).norm() / (2 * n_mag) ** 0.5)
                labs = [lab(m) for m in ms]
                seen.update(labs)
                hist.append({"n": n + 1, "u": float(u[n]), "labels": labs,
                             "distance": d, "rel": d / max(d0, 1e-30)})
                print(f"{n+1:>4} {u[n]:>+7.2f} " + " ".join(f"{l:>9}" for l in labs)
                      + f" {d:>10.4f} {d/max(d0,1e-30):>7.3f}", flush=True)
                save_ckpt(amp=amp, done=n + 1, ms=pack(ms), hist=hist,
                          seen=sorted(seen), d0=d0,
                          starts0=pack(starts0))
                if a.chunk and (n + 1 - n_done) >= a.chunk and n + 1 < a.n_steps:
                    print(f"\nCHUNK DONE: {n+1}/{a.n_steps} inputs, state saved "
                          f"to {a.ckpt}.\nRe-run the same command to continue.",
                          flush=True)
                    return 0
                continue
            amp_am = float(u[n]) * amp * 1e-3 / MU_0
            if a.field_deg is None:
                h[:, :, :, a.axis] = amp_am * mask[:, :, :, 0]
            else:
                th = math.radians(a.field_deg)
                h[:, :, :, 0] = amp_am * math.cos(th) * mask[:, :, :, 0]
                h[:, :, :, 1] = amp_am * math.sin(th) * mask[:, :, :, 0]
            ms = [isl.rollout.relax(m, h, a.settle, a.alpha_relax) for m in ms]
            d = float((ms[0] - ms[1]).norm() / (2 * n_mag) ** 0.5)
            labs = [lab(m) for m in ms]
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
        print("  No amplitude gives both. Either the window is narrower than\n"
              "  the sweep resolution, or this geometry cannot hold memory\n"
              "  without also latching.")
        if not a.vertex:
            print("  A single island under a uniform field sees exactly two\n"
                  "  coercivities, so a layer either switches or latches. An\n"
                  "  ARRAY, where neighbours supply the state-dependent partial\n"
                  "  fields a uniform drive cannot, is the way to test that.")
        else:
            print(f"  This is a {a.vertex}-island vertex, so neighbour coupling\n"
                  "  was present and did not supply it at this amplitude. Re-sweep\n"
                  "  the amplitude before concluding anything about the lattice:\n"
                  "  inter-island coupling shifts the switching fields, and 70 mT\n"
                  "  was the SINGLE ISLAND's working point, not this one's.")
    print(f"\nwrote {outdir / 'asvi_esp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
