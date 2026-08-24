#!/usr/bin/env python3
"""NARMA-10 on the ASVI spin-ice reservoir, against the baselines that matter.

NARMA-10 is
    y[n+1] = 0.3 y[n] + 0.05 y[n] SUM_{i=0..9} y[n-i] + 1.5 u[n-9] u[n] + 0.1
with u drawn uniform on [0, 0.5]. Two properties make it the benchmark this
project has been aiming at, and they are the two this device has been measured
against separately:

    memory      the target depends on u[n-9], so a reservoir whose state has
                forgotten ten inputs back cannot express it. The measured ESP
                horizon is 6-8 at 70 mT and 17-34 at 62 mT, which is why the
                run below uses 62 mT.
    nonlinearity the term u[n-9]*u[n] is a degree-2 PRODUCT OF TWO LAGS. The
                poly-tap device failed exactly here -- it scored 0.07 and 0.00
                on cross-lag products even when both operands arrived in one
                frame with the DC bin demodulated. Whether this element does
                better is the open question.

THE BASELINES ARE THE POINT. A NARMA NRMSE quoted alone says almost nothing,
because a linear filter on recent inputs already does much of the job. Three
comparisons are reported:

    mean            predicting the training mean. NRMSE 1.0 by construction --
                    anything at or above this has learned nothing.
    linear on u     ridge on [u[n], u[n-1], ... u[n-K]]. This is a pure delay
                    line with no nonlinearity, and it is the bar the reservoir
                    must clear to have contributed anything. It cannot form
                    u[n-9]*u[n], so the gap between it and the reservoir is
                    the reservoir's nonlinear contribution.
    reservoir       ridge on the island-layer state vectors alone, with NO u
                    appended. Appending the input is standard practice and
                    makes numbers look better, but it lets the readout build
                    input terms the reservoir never computed, which is exactly
                    how a device can appear to solve a task its dynamics never
                    touched. Reported separately as `reservoir+u` so the two
                    can be told apart.

Ridge strength is chosen on a validation split and reported on a held-out test
split that is never used for selection.

    python scripts/run_narma_asvi.py --device cuda --vertex 2 --amps-mT 62
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.spinice import ASVIConfig, ASVIIsland, ASVIVertexConfig, ASVIVertex


def narma10(u):
    """Standard NARMA-10. The first 10 outputs are the warm-up transient."""
    y = np.zeros(len(u))
    for n in range(9, len(u) - 1):
        y[n + 1] = (0.3 * y[n] + 0.05 * y[n] * y[n - 9:n + 1].sum()
                    + 1.5 * u[n - 9] * u[n] + 0.1)
    return y


def ridge(X, y, alpha):
    """Closed-form ridge with an unpenalised bias column."""
    X = np.column_stack([X, np.ones(len(X))])
    A = X.T @ X + alpha * np.eye(X.shape[1])
    A[-1, -1] -= alpha
    return np.linalg.solve(A, X.T @ y)


def apply_w(X, w):
    return np.column_stack([X, np.ones(len(X))]) @ w


def nrmse(y, yh):
    v = y.var()
    return float(np.sqrt(((y - yh) ** 2).mean() / v)) if v > 0 else float("nan")


def fit_report(Xtr, ytr, Xva, yva, Xte, yte, alphas):
    """Select the ridge strength on validation, report on test."""
    best, bw = None, None
    for a in alphas:
        w = ridge(Xtr, ytr, a)
        e = nrmse(yva, apply_w(Xva, w))
        if best is None or e < best[1]:
            best, bw = (a, e), w
    return {"alpha": best[0], "val_nrmse": best[1],
            "test_nrmse": nrmse(yte, apply_w(Xte, bw)),
            "train_nrmse": nrmse(ytr, apply_w(Xtr, bw))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--washout", type=int, default=40,
                   help="discarded transient. Must be >= --baseline-taps\n"
                        "so the delay-line baseline has real history.")
    p.add_argument("--settle", type=int, default=500)
    p.add_argument("--alpha-relax", type=float, default=0.5)
    p.add_argument("--relax-steps", type=int, default=2000)
    p.add_argument("--amp-mT", type=float, default=62.0)
    p.add_argument("--angle-input", type=float, default=90.0)
    p.add_argument("--field-deg", type=float, default=45.0)
    p.add_argument("--start", default="macro+/macro+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline-taps", type=int, default=12,
                   help="delay-line length for the linear-on-u baseline. Must\n"
                        "reach lag 9 or the baseline is unfairly handicapped.")
    p.add_argument("--length-nm", type=float, default=550.0)
    p.add_argument("--width-nm", type=float, default=140.0)
    p.add_argument("--offset-nm", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--dx-nm", type=float, default=5.0)
    p.add_argument("--lattice", type=int, nargs=2, default=None)
    p.add_argument("--vertex", type=int, default=2)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--outdir", default="runs/narma_asvi")
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
        n_parts = isl.n_parts

        def feat(m):
            out = []
            for j in range(n_parts):
                c = isl.part_state(m, j)
                out += [c["m_long"], c["m_trans"], c["circ"], c["peak_mz"]]
            return out
    else:
        cfg = ASVIConfig(**kw)
        isl = ASVIIsland(cfg, timesteps=32, dtype=dtype)
        n_parts = isl.n_layers

        def feat(m):
            out = []
            for k in range(n_parts):
                mx, my = isl.layer_moment(m, k)
                pk, _ = isl.layer_core(m, k)
                out += [mx, my, isl.layer_circulation(m, k), pk]
            return out

    nx, ny, nz = cfg.grid
    mask = isl.mask
    print(f"ASVI mesh {nx}x{ny}x{nz} = {nx*ny*nz:,} cells, "
          f"{n_parts} island-layers -> {4*n_parts} features", flush=True)

    rng = np.random.default_rng(a.seed)
    u = rng.uniform(0.0, 0.5, size=a.n_steps)
    y = narma10(u)
    print(f"NARMA-10: {a.n_steps} steps, u ~ U(0,0.5), "
          f"target std {y[a.washout:].std():.4f}", flush=True)
    print(f"drive {a.amp_mT:g} mT, {a.field_deg:g} +- {a.angle_input:g} deg, "
          f"settle {a.settle}", flush=True)

    # ------------------------------------------------------------ checkpoint
    idx = mask[..., 0].bool()
    def fp():
        return {"grid": [nx, ny, nz], "n_parts": n_parts, "seed": a.seed,
                "n_steps": a.n_steps, "settle": a.settle, "amp": a.amp_mT,
                "angle": a.angle_input, "field_deg": a.field_deg,
                "start": a.start, "relax": a.relax_steps, "dx": a.dx_nm,
                "lattice": a.lattice, "vertex": a.vertex}
    ck, done, feats, m = None, 0, [], None
    if a.ckpt and os.path.exists(a.ckpt):
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        if ck.get("fp") != fp():
            bad = [k for k, v in fp().items() if ck.get("fp", {}).get(k) != v]
            raise SystemExit(f"checkpoint {a.ckpt} is for a different run; "
                             f"differs in {bad}. Delete it to start over.")
        done, feats = ck["done"], ck["feats"]
        full = torch.zeros(nx, ny, nz, 3, dtype=dtype, device=idx.device)
        full[idx] = ck["m"].to(device=idx.device, dtype=dtype)
        m = full
        print(f"resumed at input {done}/{a.n_steps}", flush=True)

    if m is None:
        parts = a.start.split("/")
        if len(parts) != n_parts:
            parts = parts * (n_parts // len(parts))
        m = isl.relax(parts, steps=a.relax_steps, dtype=dtype).clone()
        print("relaxed initial state", flush=True)

    t0 = time.time()
    for n in range(done, a.n_steps):
        # u is on [0, 0.5]; centre and scale to [-1, 1] before the angle map so
        # the drive sweeps the same range the ESP sweep characterised.
        us = (float(u[n]) - 0.25) / 0.25
        th = math.radians(a.field_deg + us * a.angle_input)
        g = a.amp_mT * 1e-3 / MU_0
        h = torch.zeros(nx, ny, nz, 3, dtype=dtype)
        h[:, :, :, 0] = g * math.cos(th) * mask[:, :, :, 0]
        h[:, :, :, 1] = g * math.sin(th) * mask[:, :, :, 0]
        m = isl.rollout.relax(m, h, a.settle, a.alpha_relax)
        feats.append(feat(m))
        if (n + 1) % 20 == 0 or n + 1 == a.n_steps:
            print(f"  {n+1}/{a.n_steps}  {(time.time()-t0)/60:.1f} min",
                  flush=True)
        if a.ckpt:
            tmp = a.ckpt + ".tmp"
            torch.save({"fp": fp(), "done": n + 1, "feats": feats,
                        "m": m[idx].cpu().clone()}, tmp)
            os.replace(tmp, a.ckpt)
        if a.chunk and (n + 1 - done) >= a.chunk and n + 1 < a.n_steps:
            print(f"\nCHUNK DONE: {n+1}/{a.n_steps}, state saved to {a.ckpt}",
                  flush=True)
            return 0

    X = np.asarray(feats, dtype=float)
    np.save(outdir / "features.npy", X)
    np.save(outdir / "input.npy", u)

    # ------------------------------------------------------------ readout
    W, K = a.washout, a.baseline_taps
    if W < K:
        raise SystemExit(f"--washout {W} must be at least --baseline-taps {K}, "
                         "or the delay line has no real history to draw on")
    # Index the lags on the FULL series rather than padding a washed-out one
    # with zeros. Zero padding fabricates K rows whose history is zeros while
    # their target is real, and those are severe outliers: on a synthetic check
    # they moved an ORACLE readout -- one handed NARMA's own terms, which fits
    # to 3e-15 when indexed properly -- all the way to 0.12. Worse, this
    # baseline is the bar the reservoir has to clear, so corrupting it would
    # have flattered the reservoir rather than penalising it.
    t = np.arange(W, a.n_steps)
    Xr, yr = X[t], y[t]
    U = np.column_stack([u[t - k] for k in range(K)])
    ntr, nva = int(0.5 * len(yr)), int(0.25 * len(yr))
    sl = (slice(0, ntr), slice(ntr, ntr + nva), slice(ntr + nva, None))
    alphas = [10.0 ** k for k in range(-8, 4)]

    res = {}
    res["linear on u"] = fit_report(U[sl[0]], yr[sl[0]], U[sl[1]], yr[sl[1]],
                                    U[sl[2]], yr[sl[2]], alphas)
    res["reservoir"] = fit_report(Xr[sl[0]], yr[sl[0]], Xr[sl[1]], yr[sl[1]],
                                  Xr[sl[2]], yr[sl[2]], alphas)
    XU = np.column_stack([Xr, U])
    res["reservoir+u"] = fit_report(XU[sl[0]], yr[sl[0]], XU[sl[1]], yr[sl[1]],
                                    XU[sl[2]], yr[sl[2]], alphas)
    res["mean"] = {"test_nrmse": nrmse(yr[sl[2]],
                                       np.full(len(yr[sl[2]]), yr[sl[0]].mean())),
                   "alpha": None, "val_nrmse": None, "train_nrmse": None}

    print(f"\n{'':=<64}")
    print(f"NARMA-10, {len(yr)} usable steps "
          f"(train {ntr} / val {nva} / test {len(yr)-ntr-nva}), "
          f"{X.shape[1]} reservoir features")
    print(f"{'readout':>16}{'alpha':>10}{'train':>9}{'val':>9}{'TEST':>9}")
    for k in ("mean", "linear on u", "reservoir", "reservoir+u"):
        r = res[k]
        al = f"{r['alpha']:.0e}" if r["alpha"] is not None else "-"
        tr = f"{r['train_nrmse']:.3f}" if r["train_nrmse"] is not None else "-"
        va = f"{r['val_nrmse']:.3f}" if r["val_nrmse"] is not None else "-"
        print(f"{k:>16}{al:>10}{tr:>9}{va:>9}{r['test_nrmse']:>9.3f}")

    lin, rsv = res["linear on u"]["test_nrmse"], res["reservoir"]["test_nrmse"]
    mn = res["mean"]["test_nrmse"]
    print("\nverdict")
    # Refuse to conclude anything from a degenerate split. A one-sample test
    # block has zero variance, so every NRMSE is nan -- and `nan >= mn` is
    # False, which walked the comparison chain straight into "the reservoir
    # beats the delay line" on a 16-step smoke run. A harness that announces a
    # win from nan will eventually put a false claim in the record.
    n_test = len(yr[sl[2]])
    if n_test < 20 or not all(math.isfinite(v) for v in (lin, rsv, mn)):
        print(f"  NO VERDICT: {n_test} test samples and scores "
              f"(linear {lin:.3f}, reservoir {rsv:.3f}, mean {mn:.3f}).\n"
              "  A test block needs at least 20 samples with finite variance\n"
              "  before any comparison between readouts means anything.")
        (outdir / "narma.json").write_text(json.dumps(
            {"results": res, "n_features": int(X.shape[1]),
             "n_test": n_test, "verdict": "insufficient data",
             "args": vars(a)}, indent=2, default=str))
        print(f"\nwrote {outdir/'narma.json'}")
        return 0
    # Compare against the MEASURED mean-predictor score. Predicting the training
    # mean on a test block whose mean differs scores above 1.0, so a nominal 1.0
    # is the wrong threshold and would call a useless readout useful.
    if rsv >= mn:
        print(f"  The reservoir ({rsv:.3f}) does no better than predicting the\n"
              f"  training mean ({mn:.3f}). Its state carries nothing about the\n"
              "  target.")
    elif rsv >= lin:
        print(f"  The reservoir ({rsv:.3f}) does NOT beat a linear filter on the\n"
              f"  last {K} inputs ({lin:.3f}). Whatever it computes, a delay line\n"
              "  already does, so the dynamics are not contributing.")
    else:
        print(f"  The reservoir ({rsv:.3f}) beats the linear delay line "
              f"({lin:.3f}).\n  The gap is the part of NARMA-10 that needs the "
              "u[n-9]*u[n] product,\n  which no linear function of the inputs can "
              "express.")
    (outdir / "narma.json").write_text(json.dumps(
        {"results": res, "n_features": int(X.shape[1]), "args": vars(a)},
        indent=2, default=str))
    print(f"\nwrote {outdir/'narma.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
