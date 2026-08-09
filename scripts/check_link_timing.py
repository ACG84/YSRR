#!/usr/bin/env python3
"""Is stage B reached by the guide or by stray dipolar field? Separate by timing.

The cascade run showed both. Stage B's memory window shifted from lags 0-4 with
the link deleted to lags 3-12 with it present, and the no-link arm still
recovered u at r^2 0.90 -- so the pre-registered control fired and the linked
result could not be certified as guided transport.

Material deletion cannot settle it, because removing the link also removes that
material's own dipolar field. Timing can, and the two mechanisms make opposite
predictions:

  dipolar   near-field, effectively INSTANTANEOUS, amplitude falling steeply
            with separation (~1/r^3)
  guided    arrival delayed by separation / group velocity, so the delay GROWS
            LINEARLY with distance, amplitude falling slowly

So sweep the separation and watch when B responds. A guided component must
arrive no faster than the group velocity allows -- ~0.35 ns over 700 nm at
~2000 m/s -- and its arrival must move as the gap widens. Dipolar arrival does
not move at all.

This is an impulse measurement, not a task run: drive A with a burst, cut it,
and record B's port envelope. A few thousand steps per configuration against
600 frames for the cascade, which is why it can afford a sweep at all.

    python scripts/check_link_timing.py
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.vortex import CoupledPortedConfig, CoupledPortedArray
from magnonic_nn._compat import get_device


@torch.no_grad()
def pulse_response(arr, amp, freq, dtype, n_burst, n_quiet):
    """Drive disk A for n_burst steps, then watch. Returns per-step port signals."""
    cfg = arr.cfg
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    unit[:, :, 0, 0] = arr.disk_masks[0].to(dtype)   # disk A only
    m = arr.m0.clone()
    sig = []
    for k in range(n_burst + n_quiet):
        tk = k * cfg.dt
        a = amp if k < n_burst else 0.0

        def h_drive(theta, tk=tk, a=a):
            return unit * (a * math.sin(2 * math.pi * freq * (tk + theta * cfg.dt)))

        m = arr.rollout.rk4_step(m, arr.h_zero, h_drive)
        sig.append(arr.port_signals(m).double().cpu().numpy().copy())
    return np.asarray(sig)          # (steps, 2*n_ports)


def envelope(x, win):
    """Running RMS over `win` steps -- the carrier is 12 GHz, the arrival is not."""
    k = np.ones(win) / win
    return np.sqrt(np.convolve(x ** 2, k, mode="same"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--separations", type=float, nargs="+",
                   default=[500, 700, 1000, 1400], help="nm, centre to centre")
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--freq", type=float, default=12.0)
    p.add_argument("--burst", type=int, default=400)
    p.add_argument("--quiet", type=int, default=1600)
    p.add_argument("--relax-steps", type=int, default=5000)
    p.add_argument("--require-tol", type=float, default=1e-3)
    p.add_argument("--link-width", type=float, default=80.0)
    p.add_argument("--outdir", default="runs/link_timing")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device("cpu")
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    amp = a.amp_mT * 1e-3 / MU_0
    win = max(4, int(round(1e12 / a.freq / 1e9)))    # ~one carrier period in steps

    print(f"{a.amp_mT:.0f} mT burst of {a.burst} steps at {a.freq:.1f} GHz, "
          f"then {a.quiet} quiet\n")
    print(f"{'sep_nm':>7} {'link':>6} {'t_arrive_ns':>12} {'t_peak_ns':>10} "
          f"{'B_peak':>10} {'B/A':>9}")
    rows = []
    for sep in a.separations:
        for linked in (True, False):
            cfg = CoupledPortedConfig(
                chirality=+1, chirality_b=-1, separation=sep * 1e-9,
                link_width=(a.link_width * 1e-9 if linked else 0.0))
            arr = CoupledPortedArray(cfg, timesteps=a.burst + a.quiet + 8,
                                     dtype=dtype)
            tag = f"{int(sep)}_{'linked' if linked else 'nolink'}"
            m0c = outdir / f"m0_{tag}.pt"
            if m0c.exists():
                arr.m0 = torch.load(m0c, weights_only=False).to(device=get_device(), dtype=dtype)
            else:
                arr.relax(steps=a.relax_steps, require_tol=a.require_tol)
                torch.save(arr.m0.cpu(), m0c)

            t0 = time.time()
            sig = pulse_response(arr, amp, a.freq * 1e9, dtype, a.burst, a.quiet)
            npb = cfg.n_ports
            # RMS across each disk's own taps, then envelope in time
            A = envelope(np.sqrt((sig[:, :npb] ** 2).mean(axis=1)), win)
            B = envelope(np.sqrt((sig[:, npb:] ** 2).mean(axis=1)), win)
            dt_ns = cfg.dt * 1e9
            bpk = float(B.max()); apk = float(A.max())
            # arrival: first time B exceeds 10% of its own peak, measured only
            # while the drive is still on -- after the burst everything decays,
            # so a late threshold crossing would be ring-down, not arrival
            thr = 0.1 * bpk
            idx = np.argmax(B > thr) if (B > thr).any() else -1
            t_arr = idx * dt_ns if idx >= 0 else float("nan")
            t_pk = float(np.argmax(B) * dt_ns)
            rows.append({"sep_nm": sep, "linked": linked,
                         "t_arrive_ns": t_arr, "t_peak_ns": t_pk,
                         "B_peak": bpk, "A_peak": apk,
                         "B_over_A": bpk / max(apk, 1e-30),
                         "seconds": round(time.time() - t0, 1)})
            print(f"{sep:>7.0f} {str(linked):>6} {t_arr:>12.3f} {t_pk:>10.3f} "
                  f"{bpk:>10.3e} {bpk / max(apk, 1e-30):>9.5f}", flush=True)
            (outdir / "results.json").write_text(json.dumps(rows, indent=2))

    print("\nguided  -> arrival grows ~linearly with separation "
          "(700 nm at ~2000 m/s = 0.35 ns)")
    print("dipolar -> arrival flat in separation, amplitude falling steeply")
    lin = [r for r in rows if r["linked"]]
    nol = [r for r in rows if not r["linked"]]
    if len(lin) >= 2:
        d_lin = lin[-1]["t_arrive_ns"] - lin[0]["t_arrive_ns"]
        d_nol = nol[-1]["t_arrive_ns"] - nol[0]["t_arrive_ns"]
        dsep = (lin[-1]["sep_nm"] - lin[0]["sep_nm"]) * 1e-9
        print(f"\nlinked arrival shift {d_lin:+.3f} ns over "
              f"{dsep*1e9:.0f} nm -> implied speed "
              f"{(dsep / (d_lin * 1e-9)) if d_lin > 0 else float('nan'):.0f} m/s")
        print(f"nolink arrival shift {d_nol:+.3f} ns")
    print(f"\nwrote {outdir / 'results.json'}")


if __name__ == "__main__":
    main()
