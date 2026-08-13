#!/usr/bin/env python3
"""Record the spin wave crossing the tapped bus, for rendering.

Every number this project has about the delay line is a scalar pulled out of a
cross-correlation: an arrival lag, a peak amplitude, a spacing error. Those are
the right quantities to judge it on, but they are also how four campaigns went
wrong -- a truncated record produced plausible lags for a wave that had never
arrived, and nothing in the numbers looked odd. A picture of the field would
have shown an empty bus immediately.

So this records the field itself: m_z minus the ground state, over the whole
mesh, through a burst and the silence after it.

The ground state is SUBTRACTED, which matters here more than it usually would.
A vortex core is m_z = 1 over a few cells and the propagating wave is order
1e-3, so on raw m_z the four cores would be all anyone saw. The deviation is
the wave.

Frames are quantised to one signed byte against a scale shared by every frame,
so brightness is comparable across the whole sequence and the decay down the
bus is readable rather than auto-levelled away. The scale is a high percentile
rather than the max, because a single hot cell at the injector would otherwise
compress everything else to nothing.

    python scripts/capture_wave.py --device cuda --bus-alpha 0.1 \
        --lags 3 5 7 9 --bus-width 160 --v-g 542.9 --freq 9.0
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from _polytap_probe import make_array, ensure_m0, check_record_length


@torch.no_grad()
def capture(arr, cfg, amp_mT, freq_ghz, n_burst, n_quiet, every, stride,
            dtype=torch.float32, log=print):
    """Drive a burst into the bus and record (m_z - m0_z) every `every` steps."""
    unit = torch.zeros(*arr.mask.shape[:3], 3, dtype=dtype)
    unit[:, :, 0, 2] += arr.inject_mask
    amp = amp_mT * 1e-3 / MU_0
    w = 2 * math.pi * freq_ghz * 1e9
    stepper, graphed = arr.rollout.graph_stepper()
    hz = arr.h_zero
    m = arr.m0.clone()
    mz0 = arr.m0[:, :, 0, 2].clone()
    mask2d = arr.mask[:, :, 0, 0].to(dtype)

    frames, ports, t0 = [], [], time.time()
    for k in range(n_burst + n_quiet):
        tk = k * cfg.dt
        on = k < n_burst
        a0 = amp if on else 0.0
        h0 = hz + unit * (a0 * math.sin(w * tk))
        hh = hz + unit * (a0 * math.sin(w * (tk + 0.5 * cfg.dt)))
        h1 = hz + unit * (a0 * math.sin(w * (tk + cfg.dt)))
        if graphed:
            torch.compiler.cudagraph_mark_step_begin()
            m = stepper(m, h0, hh, hh, h1).clone()
        else:
            m = stepper(m, h0, hh, hh, h1)
        if k % every == 0:
            d = ((m[:, :, 0, 2] - mz0) * mask2d)[::stride, ::stride]
            frames.append(d.detach().cpu().numpy().astype(np.float32))
            ports.append(arr.port_signals(m).double().cpu().numpy().copy())
        if (k + 1) % 4000 == 0:
            log(f"  step {k+1}/{n_burst+n_quiet} ({time.time()-t0:.0f}s)")
    return np.asarray(frames), np.asarray(ports)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-taps", type=int, default=4)
    p.add_argument("--lags", type=float, nargs="+", default=[3, 5, 7, 9])
    p.add_argument("--gap", type=float, default=30.0)
    p.add_argument("--bus-guide-width", type=float, default=80.0)
    p.add_argument("--bus-width", type=float, default=160.0)
    p.add_argument("--bus-alpha", type=float, default=0.1)
    p.add_argument("--tap-alpha", type=float, default=1.0)
    p.add_argument("--v-g", type=float, default=542.9)
    p.add_argument("--freq", type=float, default=9.0)
    p.add_argument("--steps-per-frame", type=int, default=1200)
    p.add_argument("--amp-mT", type=float, default=30.0)
    p.add_argument("--quiet", type=int, default=20000)
    p.add_argument("--every", type=int, default=360,
                   help="record a field frame every N steps. 360 steps is\n"
                        "0.36 ns, over which the wave advances ~195 nm -- about\n"
                        "39 cells, so the front moves visibly between frames\n"
                        "without aliasing the 108.6 nm wavelength away.")
    p.add_argument("--stride", type=int, default=4,
                   help="spatial downsample; 4 keeps 20 nm resolution against a\n"
                        "108.6 nm wavelength, which is 5.4 samples per cycle")
    p.add_argument("--relax-steps", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", default="runs/wave")
    a = p.parse_args()

    mnn.set_precision("float32"); mnn.set_device(a.device)
    dtype = torch.float32
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    steps = a.steps_per_frame + a.quiet + 8
    cfg, arr, run, _ = make_array(
        a.n_taps, a.lags, a.gap, 0.0, a.tap_alpha, steps, dtype,
        bus_alpha_mult=a.bus_alpha, bus_guide_width_nm=a.bus_guide_width,
        bus_width_nm=a.bus_width, steps_per_frame=a.steps_per_frame, v_g=a.v_g)
    check_record_length(cfg, a.steps_per_frame + a.quiet, v_g=a.v_g)
    print(f"{run}\nmesh {cfg.grid[0]}x{cfg.grid[1]}, bus {cfg.bus_length()*1e6:.2f} um, "
          f"{a.freq:g} GHz, burst {a.steps_per_frame} + quiet {a.quiet}", flush=True)

    ensure_m0(arr, outdir, run, relax_steps=a.relax_steps, dtype=dtype,
              log=lambda s: print(f"  {s}", flush=True))
    frames, ports = capture(arr, cfg, a.amp_mT, a.freq, a.steps_per_frame,
                            a.quiet, a.every, a.stride, dtype)
    print(f"captured {frames.shape[0]} frames of {frames.shape[1]}x{frames.shape[2]}",
          flush=True)

    # Shared scale, from a high percentile rather than the max: the injector
    # cell is orders above the propagating wave and would otherwise set the
    # scale for everything, rendering the bus black.
    scale = float(np.percentile(np.abs(frames), 99.5))
    q = np.clip(np.round(frames / max(scale, 1e-30) * 127.0), -127, 127).astype(np.int8)
    meta = {
        "run": run, "shape": list(q.shape), "scale": scale,
        "stride": a.stride, "every": a.every, "dt_ps": cfg.dt * 1e12,
        "dx_nm": cfg.dx * 1e9, "freq_ghz": a.freq, "v_g": a.v_g,
        "bus_um": cfg.bus_length() * 1e6, "bus_width_nm": a.bus_width,
        "bus_alpha": a.bus_alpha, "n_burst": a.steps_per_frame,
        "tap_x_nm": [float(x * 1e9) for x in cfg.tap_x()],
        "inject_x_nm": float(cfg.inject_at * 1e9),
        "lags": list(a.lags[:a.n_taps]),
        "bus_y_nm": float(cfg._bus_y() * 1e9),
        "ny_nm": float(cfg.grid[1] * cfg.dx * 1e9),
    }
    np.savez_compressed(outdir / f"wave_{run}.npz", frames=q,
                        ports=ports.astype(np.float32),
                        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {outdir / f'wave_{run}.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
