#!/usr/bin/env python3
"""Switching fields: the precondition for any echo-state-property test.

The reservoir question for this architecture is whether hysteretic microstate
memory can FORGET. A reservoir needs the echo state property -- two different
initial states, driven by the same input, must converge -- and non-volatile
memory is precisely what that forbids. If an early input latches the microstate
permanently, the device is a latch and not a reservoir.

That test cannot be designed without knowing where this element switches. The
failure modes bracket the useful window from both sides:

    field too weak     nothing switches. The state never moves, memory is
                       infinite, ESP fails, and no input reaches the readout.
    field too strong   everything saturates to the same state regardless of
                       history. ESP holds trivially and capacity is zero.
    in between         partial, history-dependent switching -- the only regime
                       in which this computes anything.

So this measures the map from applied field to microstate before anything is
built on it. It is also the paper's state-programming mechanism rather than a
side measurement: in artificial spin-VORTEX ice the vortex states are NUCLEATED
during field reversal, so the descending branch is where the four states per
layer actually come from.

Registered predictions:

  two coercivities   the layers are 30 nm (hard, lower) and 20 nm (soft,
                     upper) and the paper states they switch at different
                     fields, which is what makes each layer independently
                     addressable and the 16^N space reachable. Two distinct
                     steps should appear in the loop. One step, or two at the
                     same field, means the layers are not separately
                     addressable in this solver and the state space collapses
                     toward 4^N.
  vortex nucleation  vortex states should appear NEAR THE COERCIVE FIELD on
                     the descending branch, as flux closure beats a reversing
                     macrospin, and vanish again at saturation. Vortices that
                     never appear would mean the four states per layer are
                     reachable only by the artificial initialisation used in
                     gate one, not by any field protocol -- which would make
                     them useless as reservoir states however stable they are.

Reported per layer at every field: m_x (the loop), peak |m_z| and in-plane
moment (state classification), and circulation (chirality).

    python scripts/check_asvi_hysteresis.py --device cuda --b-max 200 --b-step 10
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.spinice import ASVIConfig, ASVIIsland


def classify(isl, m, k):
    pk, mn = isl.layer_core(m, k)
    mx, my = isl.layer_moment(m, k)
    ip = (mx**2 + my**2) ** 0.5
    cc = isl.layer_circulation(m, k)
    # Same discriminator as gate one, and for the same reason: a saturated
    # 90 nm island reads |m_z|max 0.40-0.50 from edge canting at the stadium
    # caps, so flux closure identifies the state and the core confirms it.
    if ip < 0.5 and pk >= 0.5:
        what = "vACW" if cc > 0 else "vCW"
    elif ip >= 0.5:
        what = "mac+" if mx > 0 else "mac-"
    else:
        what = "??"
    return {"mx": mx, "my": my, "ip": ip, "peak_mz": pk, "mean_mz": mn,
            "circ": cc, "state": what}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--b-max", type=float, default=200.0, help="mT")
    p.add_argument("--b-step", type=float, default=10.0, help="mT")
    p.add_argument("--settle", type=int, default=2000,
                   help="relax steps per field point; quasi-static, so this must\n"
                        "exceed the switching time (~1 ns) at alpha_relax")
    p.add_argument("--alpha-relax", type=float, default=0.5)
    p.add_argument("--relax-steps", type=int, default=8000,
                   help="initial relax, before the field is applied")
    p.add_argument("--axis", type=int, default=0, help="0=x (long axis), 1=y")
    p.add_argument("--start", default="macro+/macro+")
    p.add_argument("--length-nm", type=float, default=550.0)
    p.add_argument("--width-nm", type=float, default=140.0)
    p.add_argument("--offset-nm", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--dx-nm", type=float, default=5.0)
    p.add_argument("--outdir", default="runs/asvi_hyst")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = ASVIConfig(length=a.length_nm * 1e-9, width=a.width_nm * 1e-9,
                     layer_offset=a.offset_nm * 1e-9, alpha=a.alpha,
                     dx=a.dx_nm * 1e-9, dz=a.dx_nm * 1e-9)
    nx, ny, nz = cfg.grid
    isl = ASVIIsland(cfg, timesteps=32, dtype=dtype)
    print(f"ASVI island {a.length_nm:g} x {a.width_nm:g} x {cfg.thickness()*1e9:.0f} nm, "
          f"mesh {nx}x{ny}x{nz} = {nx*ny*nz:,} cells")
    print(f"field along {'xy'[a.axis]}, +-{a.b_max:g} mT in {a.b_step:g} mT steps, "
          f"{a.settle} relax steps each", flush=True)

    m = isl.relax(a.start.split("/"), steps=a.relax_steps, dtype=dtype)
    print(f"start {a.start}: " + "  ".join(
        f"L{k} {classify(isl, m, k)['state']}" for k in range(isl.n_layers)),
        flush=True)

    # Descending then ascending: a hysteresis loop needs both branches, and the
    # vortex states are expected on the descending one.
    down = np.arange(a.b_max, -a.b_max - 1e-9, -a.b_step)
    up = np.arange(-a.b_max, a.b_max + 1e-9, a.b_step)
    field = np.concatenate([down, up])

    mask = isl.mask
    rows, t0 = [], time.time()
    print(f"\n{'B(mT)':>8} " + " ".join(
        f"{'L'+str(k)+' mx':>8}{'state':>7}{'circ':>7}" for k in range(isl.n_layers)),
        flush=True)
    prev = [classify(isl, m, k)["state"] for k in range(isl.n_layers)]
    for i, B in enumerate(field):
        h = torch.zeros(nx, ny, nz, 3, dtype=dtype)
        h[:, :, :, a.axis] = (float(B) * 1e-3 / MU_0) * mask[:, :, :, 0]
        m = isl.rollout.relax(m, h, a.settle, a.alpha_relax)
        cs = [classify(isl, m, k) for k in range(isl.n_layers)]
        rows.append({"B_mT": float(B), "branch": "down" if i < len(down) else "up",
                     "layers": cs})
        flag = ""
        for k, c in enumerate(cs):
            if c["state"] != prev[k]:
                flag += f"   <-- L{k} {prev[k]} -> {c['state']}"
                prev[k] = c["state"]
        print(f"{B:>8.1f} " + " ".join(
            f"{c['mx']:>8.3f}{c['state']:>7}{c['circ']:>7.2f}" for c in cs) + flag,
            flush=True)
        (outdir / "asvi_hysteresis.json").write_text(json.dumps(rows, indent=2))

    # ------------------------------------------------------------- verdict
    print(f"\n{'':=<72}\n({time.time()-t0:.0f}s)")
    for k in range(isl.n_layers):
        z0, z1, _ = cfg.magnetic_layers()[k]
        th = (z1 - z0) * cfg.dz * 1e9
        trans = [(r["B_mT"], r["branch"]) for j, r in enumerate(rows[1:], 1)
                 if r["layers"][k]["state"] != rows[j-1]["layers"][k]["state"]]
        print(f"L{k} ({th:.0f} nm): {len(trans)} transitions at "
              + (", ".join(f"{b:+.0f} mT ({br})" for b, br in trans) or "none"))
    seen = {r["layers"][k]["state"] for r in rows for k in range(isl.n_layers)}
    print(f"\nstates visited by the field alone: {sorted(seen)}")
    vort = {s for s in seen if s.startswith("v")}
    print(f"vortex states nucleated by field: {sorted(vort) or 'NONE'}")
    if not vort:
        print("  No vortex appeared at any field. The four states per layer are\n"
              "  then reachable only by the artificial initialisation gate one\n"
              "  used, not by a field protocol -- which makes them unusable as\n"
              "  reservoir states however stable they are.")
    hc = []
    for k in range(isl.n_layers):
        t = [r["B_mT"] for j, r in enumerate(rows[1:], 1)
             if r["layers"][k]["state"] != rows[j-1]["layers"][k]["state"]
             and r["branch"] == "down"]
        hc.append(t[0] if t else None)
    if all(h is not None for h in hc):
        sep = abs(hc[0] - hc[1])
        print(f"\ndescending-branch switching: L0 {hc[0]:+.0f} mT, L1 {hc[1]:+.0f} mT, "
              f"separation {sep:.0f} mT")
        print("  " + ("layers are independently addressable -- the coercive offset\n"
                      "  the 16^N space needs is present"
                      if sep >= 2 * a.b_step else
                      "  layers switch together at this resolution; without a\n"
                      "  coercive offset they are not separately addressable and\n"
                      "  the state space collapses toward 4^N"))
    print(f"\nwrote {outdir / 'asvi_hysteresis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
