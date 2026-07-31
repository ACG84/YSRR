#!/usr/bin/env python3
"""Vowel classification on the ported disk -- the task this device actually fits.

Measured envelope: linear memory capacity 2.45, quadratic products recoverable
to lag 2. NARMA-10 needs ten lags and the disk ties a shift register there;
NARMA-2 needs two and it beats one by 2.9x. So the productive question is not
how to force a ten-lag task through a three-lag device, but which real task
lives inside the envelope.

Vowel classification is that task, and it is the benchmark the field uses for
physical reservoirs -- Torrejon et al. (Nature 2017) classified spoken digits
on a single spin-torque oscillator. It needs almost no memory and a great deal
of nonlinear separation, which is the exact shape of what has been verified
here: AB/BA discrimination 23.4x, reaching the ports at 24.9x.

Encoding maps each token's three formants into the disk's own mode ladder
(9.4-13.7 GHz measured, not assumed), so a vowel arrives as three tones landing
on real modes and the disk's three-magnon mixing does the separating.

The baseline is the one that matters. A linear classifier on the INPUT
representation -- the three mapped frequencies and their amplitudes -- already
knows everything about the token that was put in. Beating it means the disk
supplied separation the input did not already contain; failing to means the
reservoir is an expensive spectrum analyser. Jitter is set high enough that
classes genuinely overlap, since at zero jitter the task is three fixed spectra
and any method scores 100%.

    python scripts/run_vowels_modal.py
    python scripts/run_vowels_modal.py --classes iy ae uw ao er --jitter 0.12
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import PortedVortexConfig, PortedVortexDisk

FORMANTS = None            # filled from mnn at runtime


def token_frequencies(vowel, band, jitter, rng):
    """Map F1,F2,F3 affinely into the disk's mode band, with speaker jitter."""
    (f1, f2, f3), _ = FORMANTS[vowel]
    src = np.array([f1, f2, f3], dtype=float)
    src = src * (1.0 + jitter * rng.standard_normal(3))
    lo_s, hi_s = 300.0, 3000.0
    lo_b, hi_b = band
    frac = np.clip((src - lo_s) / (hi_s - lo_s), 0.0, 1.0)
    return lo_b + frac * (hi_b - lo_b)


@torch.no_grad()
def run_tokens(disk, tokens, steps_per_frame, frames_per_token, amp_mT,
               tones_read, dtype, cache=None):
    """Drive each token as three simultaneous tones; return per-token features."""
    if cache is not None and cache.exists():
        prev = torch.load(cache, weights_only=False)
        if len(prev) >= len(tokens):
            print(f"[cached] {cache.name}", flush=True)
            return prev

    cfg = disk.cfg
    n = cfg.n_cells
    amp = amp_mT * 1e-3 / MU_0
    # Drive THROUGH THE PORTS rather than uniformly. A uniform in-plane field
    # couples almost entirely to n = +-1 by symmetry -- measured at 77.9% of the
    # amplitude on the NARMA features -- so every token excited the same single
    # mode and the six-port decomposition had nothing to separate (Fisher ratio
    # 0.037). The guides are reciprocal: injecting a_k at port k makes the
    # excitation's azimuthal content the DFT of {a_k}, which addresses
    # n = 0..+-3 directly. This is what the routing film was always for.
    taps = disk._tap_masks.to(dtype)                     # (n_ports, nx, ny)
    port_drive = torch.zeros(cfg.n_ports, n, n, 1, 3, dtype=dtype)
    for k in range(cfg.n_ports):
        port_drive[k, :, :, 0, 2] = taps[k] / taps[k].sum().clamp_min(1e-30)
    ws_read = [2 * math.pi * f for f in tones_read]

    m = disk.m0.clone()
    feats, t0 = [], time.time()
    for ti, freqs in enumerate(tokens):
        w_tok = [2 * math.pi * f for f in freqs]
        # each formant addresses a different azimuthal order: formant j is
        # injected with the phase pattern exp(i (j+1) theta_k) across ports, so
        # the three formants land on n = 1, 2, 3 instead of all on n = 1
        angles = torch.tensor(cfg.port_angles(), dtype=torch.float64)
        pattern = torch.stack([torch.cos((j + 1) * angles)
                               for j in range(len(w_tok))])       # (3, n_ports)
        unit_t = torch.einsum("jp,pxyzc->jxyzc",
                              pattern.to(dtype), port_drive)      # (3, nx,ny,1,3)
        rows = []
        for fr in range(frames_per_token):
            accI = torch.zeros(len(ws_read), cfg.n_ports, dtype=torch.float64)
            accQ = torch.zeros(len(ws_read), cfg.n_ports, dtype=torch.float64)
            for k in range(steps_per_frame):
                tk = ((ti * frames_per_token + fr) * steps_per_frame + k) * cfg.dt

                def h_drive(theta, tk=tk):
                    t = tk + theta * cfg.dt
                    out = torch.zeros_like(unit_t[0])
                    for j, w in enumerate(w_tok):
                        out = out + unit_t[j] * math.sin(w * t)
                    return out * (amp / len(w_tok))

                m = disk.rollout.rk4_step(m, disk.h_zero, h_drive)
                p = disk.port_signals(m).double()
                for i, wi in enumerate(ws_read):
                    accI[i] += p * math.cos(wi * tk)
                    accQ[i] += p * math.sin(wi * tk)
            # keep only the settled frames: the first frame of a token still
            # carries the previous one, and with ~3 frames of memory that
            # carryover is the dominant contaminant between tokens
            if fr >= frames_per_token - 2:
                row = []
                for i in range(len(ws_read)):
                    A = ((accI[i] + 1j * accQ[i]) / steps_per_frame).numpy()
                    M = np.fft.fft(A)
                    for q in range(cfg.n_ports):
                        row += [M[q].real, M[q].imag]
                rows.append(row)
        feats.append(np.concatenate(rows))
        if (ti + 1) % 20 == 0:
            print(f"  token {ti+1}/{len(tokens)} ({time.time()-t0:.0f}s)",
                  flush=True)
            if cache is not None:
                torch.save(torch.tensor(np.array(feats), dtype=torch.float64),
                           cache)

    F = torch.tensor(np.array(feats), dtype=torch.float64)
    if cache is not None:
        torch.save(F, cache)
    return F


def ridge_classify(Xtr, ytr, Xte, yte, n_cls, lams=(1e-6, 1e-4, 1e-2, 1.0, 100.0),
                   val_frac=0.25, seed=0):
    """One-vs-rest ridge, lambda chosen on a VALIDATION split of the training set.

    The first version picked lambda by test accuracy, which is test-set leakage
    and inflates whatever it reports. It did not rescue the negative results it
    was used on, but it has to be gone before any positive one is believed.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    n_val = max(int(len(idx) * val_frac), n_cls)
    va, tr = idx[:n_val], idx[n_val:]

    def fit(X, y, lam):
        T = -np.ones((len(y), n_cls)); T[np.arange(len(y)), y] = 1.0
        A = np.hstack([X, np.ones((len(X), 1))])
        return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ T)

    def acc(W, X, y):
        B = np.hstack([X, np.ones((len(X), 1))])
        return float((np.argmax(B @ W, axis=1) == y).mean())

    best_lam, best_va = None, -1.0
    for lam in lams:
        a = acc(fit(Xtr[tr], ytr[tr], lam), Xtr[va], ytr[va])
        if a > best_va:
            best_va, best_lam = a, lam
    W = fit(Xtr, ytr, best_lam)          # refit on all training data
    return acc(W, Xte, yte), best_lam


def main():
    global FORMANTS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", nargs="+", default=["iy", "ae", "uw", "ao", "er"])
    p.add_argument("--n-per-class", type=int, default=24)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--jitter", type=float, default=0.10)
    p.add_argument("--band", type=float, nargs=2, default=[9.4, 13.7],
                   help="GHz; the disk's MEASURED mode ladder")
    p.add_argument("--tones-read", type=float, nargs="*",
                   default=[9.6, 10.4, 11.2, 12.0, 12.8, 13.6],
                   help="six bins across the band rather than three, so the "
                        "readout samples the spectrum instead of three points "
                        "of it")
    p.add_argument("--steps-per-frame", type=int, default=1000,
                   help="Sets the READOUT's frequency resolution, 1/T. At the\n"
                        "old 200 steps that was 5 GHz, and the whole token\n"
                        "encoding spans 4.3 GHz -- the entire task fitted inside\n"
                        "one resolution element and every lock-in measured the\n"
                        "same blur. 1000 steps gives 1 GHz; two settled frames\n"
                        "integrate to 0.5 GHz, enough to separate the classes.")
    p.add_argument("--frames-per-token", type=int, default=4)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--relax-steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="runs/vowels_modal")
    args = p.parse_args()

    FORMANTS = mnn.FORMANTS
    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    band = (args.band[0] * 1e9, args.band[1] * 1e9)
    tokens, labels = [], []
    for ci, v in enumerate(args.classes):
        for _ in range(args.n_per_class):
            tokens.append(token_frequencies(v, band, args.jitter, rng))
            labels.append(ci)
    tokens = np.array(tokens); labels = np.array(labels)
    order = rng.permutation(len(tokens))
    tokens, labels = tokens[order], labels[order]

    # how much class overlap the jitter actually produced -- without this the
    # headline accuracy is uninterpretable
    sep = []
    for ci in range(len(args.classes)):
        a = tokens[labels == ci].mean(0)
        for cj in range(ci + 1, len(args.classes)):
            b = tokens[labels == cj].mean(0)
            sd = tokens.std(0).mean()
            sep.append(float(np.linalg.norm(a - b) / (sd + 1e-30)))
    print(f"{len(args.classes)} classes x {args.n_per_class} tokens, "
          f"jitter {args.jitter:g}", flush=True)
    print(f"class separation in the INPUT: {min(sep):.2f} sigma (closest pair)",
          flush=True)

    cfg = PortedVortexConfig()
    disk = PortedVortexDisk(cfg, timesteps=args.steps_per_frame + 4, dtype=dtype)
    t0 = time.time(); disk.relax(steps=args.relax_steps)
    print(f"relaxed {time.time()-t0:.0f}s", flush=True)

    F = run_tokens(disk, tokens, args.steps_per_frame, args.frames_per_token,
                   args.amp_mT, [t * 1e9 for t in args.tones_read], dtype,
                   cache=outdir / "features.pt")
    X = F.numpy()
    X = (X - X.mean(0)) / X.std(0).clip(1e-12)

    n_tr = int(len(X) * args.train_frac)
    n_cls = len(args.classes)
    # the input representation: mapped frequencies, which already contain
    # everything about the token that was presented
    U = (tokens - tokens.mean(0)) / tokens.std(0).clip(1e-12)
    acc_in, lam_in = ridge_classify(U[:n_tr], labels[:n_tr], U[n_tr:],
                                    labels[n_tr:], n_cls)
    acc_res, lam_res = ridge_classify(X[:n_tr], labels[:n_tr], X[n_tr:],
                                      labels[n_tr:], n_cls)
    chance = 1.0 / n_cls

    results = {"classes": args.classes, "n_per_class": args.n_per_class,
               "jitter": args.jitter, "input_accuracy": acc_in,
               "reservoir_accuracy": acc_res, "chance": chance,
               "n_features": int(X.shape[1]),
               "min_class_separation_sigma": min(sep)}
    (outdir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'readout':<26} {'dim':>5} {'test accuracy':>14}")
    print(f"{'chance':<26} {'-':>5} {chance:>14.3f}")
    print(f"{'linear on input spectrum':<26} {U.shape[1]:>5} {acc_in:>14.3f}")
    print(f"{'ported disk (modal)':<26} {X.shape[1]:>5} {acc_res:>14.3f}")
    print()
    if acc_res > acc_in + 0.05:
        print("The disk separates vowels better than the spectrum it was given.")
        print("That gap is nonlinear expansion -- the mixing supplying structure")
        print("the input did not contain, on a task inside the memory envelope.")
    elif acc_res > chance + 0.1:
        print("The disk classifies well above chance but no better than a linear")
        print("readout on the input spectrum. It is preserving the input, not")
        print("enriching it -- an expensive spectrum analyser on this task.")
    else:
        print("At or near chance. Check that the mapped tones land on populated")
        print("modes before concluding anything about the device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
