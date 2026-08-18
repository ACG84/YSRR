#!/usr/bin/env python3
"""Do the microstates have distinguishable magnon spectra? The readout question.

Gate two. Gate one asks whether the element holds four states per layer; this
asks whether those states are READABLE, because a reservoir whose microstates
all look alike computes nothing however many of them there are.

This is "spin-wave fingerprinting" -- Gartside et al., Nat. Nanotech. 17, 460
(2022) -- the readout the artificial spin-vortex ice reservoir work is built on,
and it is why this architecture is worth the redirection. The feature vector is
an FMR POWER spectrum, and a power spectrum is an even-order detector. The
poly-tap readout demodulates coherently at the carrier and its tone bank, with
no zero-frequency bin, so for a narrowband drive through a compressive
nonlinearity it returns degree 1 and degree 3 in the envelope and NEVER the
degree-2 cross term s[n]*s[n-d], which lives at DC and 2w. It is blind by
parity to the family it was scored on. Nothing analogous applies here.

PROTOCOL, from the paper's Methods (doi:10.1038/s41467-024-48080-z):

    excitation   broadband sinc pulse along z, cutoff 15 GHz, amplitude 1 mT
    duration     26 ns, magnetisation saved every 33 ps
    processing   subtract the static relaxed state at t = 0, so only dynamic
                 components remain, then FFT along the time axis

33 ps sampling puts Nyquist at 15.15 GHz, which is why their cutoff is 15 --
the protocol is self-consistent and is reproduced rather than reinvented. 26 ns
gives 38 MHz resolution here; their quoted 18 MHz implies a longer record or
zero-padding, and the difference does not matter for GHz-scale mode shifts.

WHAT IS REPORTED

Per magnetic layer, the power spectrum of the spatially averaged dynamic
magnetisation. Per layer because the two layers hold independent states and the
whole 16^N microstate space is per-layer; a whole-island average would sum them
and throw away the distinction the architecture is made of.

The verdict is a DISTINGUISHABILITY matrix, not a list of peaks. What the
reservoir needs is not that spectra exist but that different microstates give
different ones, so every pair of states is compared by spectral correlation.
Two states whose spectra correlate at 0.99 are one state as far as any readout
is concerned -- which is exactly how the poly-tap 2w tone turned out to be
leakage rather than a channel.

    python scripts/check_asvi_spectrum.py --device cuda --states \
        macro+/macro+ macro+/macro- vortex_acw/macro+ vortex_acw/vortex_acw
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.spinice import ASVIConfig, ASVIIsland, STATES


@torch.no_grad()
def spectrum(isl, cfg, m0, n_steps, every, amp_mT, fc_ghz, dtype, log=print):
    """Sinc-pulse the island and record per-layer mean dynamic magnetisation."""
    nx, ny, nz = cfg.grid
    unit = torch.zeros(nx, ny, nz, 3, dtype=dtype)
    unit[:, :, :, 2] = isl.mask[:, :, :, 0]          # drive along z, whole island
    amp = amp_mT * 1e-3 / MU_0
    wc = 2 * math.pi * fc_ghz * 1e9
    # Centre the sinc a quarter of the way in so its acausal lobes are captured
    # rather than clipped; a clipped sinc is a different pulse with a different
    # spectrum, and the flat band it is chosen for would not be flat.
    t0 = 0.25 * n_steps * cfg.dt

    def h(t):
        x = wc * (t - t0)
        s = 1.0 if abs(x) < 1e-12 else math.sin(x) / x
        return amp * s

    stepper, graphed = isl.rollout.graph_stepper()
    eager = isl.rollout.rk4_step_fields
    hz = isl.h_zero
    m = m0.clone()
    w = isl.layer_masks
    tot = w.sum(dim=(1, 2, 3)).clamp_min(1e-30)
    base = torch.stack([(m0 * w[k].unsqueeze(-1)).sum(dim=(0, 1, 2)) / tot[k]
                        for k in range(isl.n_layers)])
    rec, t_start = [], time.time()
    for k in range(n_steps):
        tk = k * cfg.dt
        h0 = hz + unit * h(tk)
        hh = hz + unit * h(tk + 0.5 * cfg.dt)
        h1 = hz + unit * h(tk + cfg.dt)
        if graphed:
            try:
                torch.compiler.cudagraph_mark_step_begin()
                m = stepper(m, h0, hh, hh, h1).clone()
            except Exception as e:
                log(f"  compiled step failed ({type(e).__name__}); eager")
                graphed, stepper = False, eager
                m = stepper(m, h0, hh, hh, h1)
        else:
            m = stepper(m, h0, hh, hh, h1)
        if k % every == 0:
            cur = torch.stack([(m * w[j].unsqueeze(-1)).sum(dim=(0, 1, 2)) / tot[j]
                               for j in range(isl.n_layers)])
            rec.append((cur - base).cpu().numpy())
        if (k + 1) % 5000 == 0:
            log(f"    step {k+1}/{n_steps} ({time.time()-t_start:.0f}s)")
    return np.asarray(rec)          # (n_samples, n_layers, 3)


def power_spectrum(sig, dt_sample, fmin_ghz=0.15):
    """|FFT|^2 of each channel, detrended and Hann-windowed, DC discarded.

    DETRENDED, and the first run without it is why. Subtracting the static
    state at t = 0 -- which is what the paper's Methods prescribe -- leaves any
    residual drift during the record as a constant offset, and on the vortex
    states that offset dominated everything: the 0.00 GHz bin came back at 1.00
    with every dynamic peak at 0.00-0.01.

    That is not merely an ugly spectrum, it is a rigged verdict. Every
    DC-dominated spectrum correlates with every other one THROUGH THE DC BIN,
    so the distinguishability matrix would have reported states as identical
    when the modes that distinguish them were sitting two orders down and
    perfectly resolved. A false negative manufactured by the estimator, of the
    same family as the capacity floor whose own spread was larger than the
    products it was scoring.

    So: subtract the time-mean per channel, then window, then drop everything
    below `fmin_ghz`. The floor is set above the 38 MHz resolution and below
    any real mode -- a vortex gyrotropic mode sits at a few hundred MHz, which
    survives at 0.15 GHz.
    """
    n = sig.shape[0]
    sig = sig - sig.mean(axis=0, keepdims=True)
    win = np.hanning(n)[:, None, None]
    F = np.fft.rfft(sig * win, axis=0)
    f = np.fft.rfftfreq(n, d=dt_sample) / 1e9
    keep = f >= fmin_ghz
    return f[keep], (np.abs(F) ** 2)[keep]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--states", nargs="+",
                   default=["macro+/macro+", "macro+/macro-",
                            "vortex_acw/macro+", "macro+/vortex_acw",
                            "vortex_acw/vortex_acw", "vortex_acw/vortex_cw"],
                   help="per-run layer states as hard/soft")
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--ns", type=float, default=26.0, help="record length, ns")
    p.add_argument("--every", type=int, default=33,
                   help="save interval in steps; 33 ps puts Nyquist at 15.15 GHz")
    p.add_argument("--amp-mT", type=float, default=1.0)
    p.add_argument("--fc-ghz", type=float, default=15.0)
    p.add_argument("--length-nm", type=float, default=550.0)
    p.add_argument("--width-nm", type=float, default=140.0)
    p.add_argument("--offset-nm", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--dx-nm", type=float, default=5.0)
    p.add_argument("--fmin-ghz", type=float, default=0.15,
                   help="discard bins below this. A residual static offset puts\n"
                        "all its power in the DC bin, which then dominates every\n"
                        "vortex spectrum and makes them all correlate with each\n"
                        "other through it rather than through their modes.")
    p.add_argument("--dump-b64", action="store_true",
                   help="print each state's summed power spectrum to stdout as\n"
                        "base64 as soon as it is computed, so a run killed\n"
                        "part-way still yields every state it finished.")
    p.add_argument("--clean-ghz", type=float, default=1.0,
                   help="floor for the second correlation matrix. Above the\n"
                        "vortex gyrotropic mode, so the verdict can be\n"
                        "checked without the band the low-frequency floor\n"
                        "is clipping.")
    p.add_argument("--fmax-ghz", type=float, default=15.0)
    p.add_argument("--outdir", default="runs/asvi_spec")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cfg = ASVIConfig(length=a.length_nm * 1e-9, width=a.width_nm * 1e-9,
                     layer_offset=a.offset_nm * 1e-9, alpha=a.alpha,
                     dx=a.dx_nm * 1e-9, dz=a.dx_nm * 1e-9)
    n_steps = int(round(a.ns * 1e-9 / cfg.dt))
    nx, ny, nz = cfg.grid
    isl = ASVIIsland(cfg, timesteps=32, dtype=dtype)
    df = 1.0 / (a.ns * 1e-9) / 1e9
    print(f"ASVI island {a.length_nm:g} x {a.width_nm:g} x {cfg.thickness()*1e9:.0f} nm")
    print(f"mesh {nx} x {ny} x {nz} = {nx*ny*nz:,} cells, alpha {cfg.alpha:g}")
    print(f"sinc {a.amp_mT:g} mT, cutoff {a.fc_ghz:g} GHz, {a.ns:g} ns = "
          f"{n_steps} steps, sample every {a.every} = "
          f"{a.every*cfg.dt*1e12:.0f} ps (Nyquist "
          f"{1/(2*a.every*cfg.dt)/1e9:.2f} GHz), df {df*1e3:.0f} MHz\n", flush=True)

    out, spectra = [], {}
    for spec in a.states:
        states = spec.split("/")
        if len(states) != isl.n_layers or any(s not in STATES for s in states):
            raise SystemExit(f"bad --states entry {spec!r}; want "
                             f"{isl.n_layers} of {STATES} joined by '/'")
        print(f"===== {spec} =====", flush=True)
        t0 = time.time()
        m0 = isl.relax(states, steps=a.relax_steps, dtype=dtype)
        # A spectrum measured on a state that did not survive relaxation is a
        # spectrum of something else. Gate one exists for this reason, and the
        # check is repeated here so a spectroscopy run cannot silently report
        # the wrong element's modes.
        tags = []
        for k in range(isl.n_layers):
            pk, mn = isl.layer_core(m0, k)
            mx, my = isl.layer_moment(m0, k)
            ip = (mx**2 + my**2) ** 0.5
            got = ("vortex" if (ip < 0.5 and pk >= 0.5)
                   else "macro" if ip >= 0.5 else "other")
            want = "vortex" if states[k].startswith("vortex") else "macro"
            tags.append(f"L{k} {got}{'' if got == want else f' (WANTED {want})'}")
        print(f"  relaxed: " + ", ".join(tags), flush=True)

        sig = spectrum(isl, cfg, m0, n_steps, a.every, a.amp_mT, a.fc_ghz,
                       dtype, log=lambda s: print(s, flush=True))
        # How far the state drifted over the record, per layer, against how
        # much it oscillated. A drift comparable to the dynamics means the
        # relax had not converged and the spectrum is of a moving target.
        drift = np.abs(sig.mean(axis=0)).max(axis=1)
        swing = sig.std(axis=0).max(axis=1)
        print("  drift/oscillation: " + "  ".join(
            f"L{k} {drift[k]:.2e}/{swing[k]:.2e} = {drift[k]/max(swing[k],1e-30):.2f}"
            for k in range(isl.n_layers)), flush=True)
        f, P = power_spectrum(sig, a.every * cfg.dt, a.fmin_ghz)
        sel = f <= a.fmax_ghz
        f, P = f[sel], P[sel]
        spectra[spec] = P
        for k in range(isl.n_layers):
            tot = P[:, k, :].sum(axis=1)
            idx = np.argsort(tot)[::-1]
            peaks, seen = [], []
            for i in idx:
                if all(abs(f[i] - g) > 0.3 for g in seen):
                    seen.append(f[i]); peaks.append((float(f[i]), float(tot[i])))
                if len(peaks) >= 4:
                    break
            print(f"  L{k} peaks: " + "  ".join(
                f"{fr:.2f} GHz ({pw/max(peaks[0][1],1e-30):.2f})"
                for fr, pw in peaks), flush=True)
        # DUMP TO THE LOG, per state, not only to a file on the VM.
        # This project has now lost six runs to reclaimed machines, and the
        # last one died after three of six states with every completed spectrum
        # sitting in an npz that went with the container. A log line survives
        # what the filesystem does not.
        if a.dump_b64:
            import base64, io
            buf = io.BytesIO()
            np.save(buf, P.sum(axis=2).astype(np.float32), allow_pickle=False)
            print(f"B64 {spec} {f[0]:.6f} {f[-1]:.6f} "
                  + base64.b64encode(buf.getvalue()).decode(), flush=True)
        out.append({"states": states, "seconds": round(time.time()-t0, 1)})
        np.savez_compressed(outdir / "asvi_spectra.npz", freq_ghz=f,
                            **{s.replace("/", "__"): v for s, v in spectra.items()})
        print(flush=True)

    # ------------------------------------------- distinguishability matrix
    names = list(spectra)
    print("=" * 72)
    print("spectral correlation between microstates (1.00 = indistinguishable)")
    def flat(P, band=None):
        Q = P if band is None else P[band]
        v = Q.sum(axis=2).reshape(-1)      # both layers, summed over components
        return v / max(np.linalg.norm(v), 1e-30)

    # TWO matrices, at two frequency floors, because one is not falsifiable.
    # The vortex states put a large lightly-damped gyrotropic component in the
    # lowest retained bins -- measured drift/oscillation 1.4 with the peak
    # sitting ON the floor -- so a correlation taken over the full band is
    # dominated by a feature the floor is clipping. If the verdict is the same
    # over the clean GHz band it does not rest on that feature; if it differs,
    # the full-band number is an artifact of where the floor was put and the
    # clean-band one is what to believe.
    def matrix(band, label):
        print(f"\n{label}")
        print(f"{'':<24} " + " ".join(
            f"{n.split('/')[0][:6]+'/'+n.split('/')[1][:6]:>14}" for n in names))
        w = (0.0, None, None)
        for i, a_ in enumerate(names):
            row = f"{a_:<24} "
            for j, b_ in enumerate(names):
                c = float(np.dot(flat(spectra[a_], band), flat(spectra[b_], band)))
                row += f"{c:>14.3f}"
                if i < j and c > w[0]:
                    w = (c, a_, b_)
            print(row)
        return w

    clean = f >= a.clean_ghz
    worst_full = matrix(None, f"FULL BAND ({f[0]:.2f}-{f[-1]:.1f} GHz)")
    worst_clean = matrix(clean, f"CLEAN BAND (>= {a.clean_ghz:g} GHz, "
                                f"{int(clean.sum())} bins, gyrotropic excluded)")
    print(f"\nworst pair full band  {worst_full[0]:.3f}"
          f"  ({worst_full[1]} vs {worst_full[2]})")
    print(f"worst pair clean band {worst_clean[0]:.3f}"
          f"  ({worst_clean[1]} vs {worst_clean[2]})")
    agree = (worst_full[0] > 0.99) == (worst_clean[0] > 0.99)
    print("the two bands " + ("AGREE on the verdict" if agree else
          "DISAGREE -- the full-band number depends on the clipped low bins "
          "and\n  the clean-band number is the one to believe"))
    worst = worst_clean
    print()
    if worst[1] is None:
        print("Only one state measured; nothing to distinguish.")
    else:
        print(f"least distinguishable pair: {worst[1]} vs {worst[2]} at {worst[0]:.3f}")
        if worst[0] > 0.99:
            print("Above 0.99 these are one state to any readout. The microstate\n"
                  "space is large on paper and small in the spectrum, which is\n"
                  "the same failure as the 2w tone that turned out to be\n"
                  "leakage correlated 0.99 with the carrier block.")
        elif worst[0] > 0.9:
            print("Distinguishable but not comfortably. Worth checking whether\n"
                  "the separation survives a realistic readout bandwidth before\n"
                  "counting these as independent reservoir states.")
        else:
            print("Every microstate pair is spectrally distinct. The readout can\n"
                  "see the state space, which is the precondition for using it.")
    print(f"\nwrote {outdir / 'asvi_spectra.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
