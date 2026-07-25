"""Shared CLI plumbing for the training scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import magnonic_nn as mnn  # noqa: E402


def base_parser(description: str, default_preset: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--preset", default=default_preset, choices=sorted(mnn.PRESETS),
                   help="simulation size/duration preset")
    p.add_argument("--geometry", default="freeform", choices=["freeform", "ms", "nanomagnets"],
                   help="how the scatterer is parameterised")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", default=None, help="cpu / cuda / cuda:N (default: preset's)")
    p.add_argument("--precision", default="float32", choices=["float32", "float64"])
    p.add_argument("--threads", type=int, default=None, help="CPU threads for torch")
    p.add_argument("--outdir", default=None, help="where plots and checkpoints go")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nx", type=int, default=None, help="override mesh size (square)")
    p.add_argument("--timesteps", type=int, default=None, help="override rollout length")
    p.add_argument("--Bt-mT", type=float, default=None, dest="bt_mt",
                   help="excitation amplitude in mT (>~20 mT enters the non-linear regime)")
    p.add_argument("--no-demag", action="store_true",
                   help="drop the demagnetisation field: ~3x faster, but the dispersion "
                        "becomes exchange-only and no longer matches a real film")
    p.add_argument("--per-sample", action="store_true",
                   help="backpropagate one sample at a time to keep memory flat")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    return p


def setup(args, task_name: str):
    """Apply global settings and return ``(cfg, outdir)``."""
    if args.threads:
        torch.set_num_threads(args.threads)
    mnn.set_precision(args.precision)

    cfg = mnn.get_preset(args.preset)
    cfg.geometry = args.geometry
    cfg.seed = args.seed

    device = args.device or cfg.device
    cfg.device = str(mnn.set_device(device))

    if args.nx:
        cfg.mesh.nx = cfg.mesh.ny = args.nx
        cfg.material.abc_width = min(cfg.material.abc_width, max(args.nx // 8, 2))
    if args.timesteps:
        cfg.solver.timesteps = args.timesteps
    if args.bt_mt is not None:
        cfg.fields.Bt = args.bt_mt * 1e-3
    if args.no_demag:
        cfg.solver.demag = False

    torch.manual_seed(args.seed)

    outdir = Path(args.outdir or f"runs/{task_name}_{args.preset}_{args.geometry}")
    outdir.mkdir(parents=True, exist_ok=True)
    return cfg, outdir


def report_physics(cfg, freqs=()):
    """Print the numbers that decide whether a run can possibly work."""
    fmr = mnn.kittel_fmr(cfg.fields, cfg.material)
    lo, hi = mnn.usable_band(cfg)
    thickness = cfg.mesh.dz * cfg.mesh.nz

    print(f"  mesh          {cfg.mesh.nx}x{cfg.mesh.ny}x{cfg.mesh.nz} "
          f"@ {cfg.mesh.dx * 1e9:g} nm  ->  "
          f"{cfg.mesh.extent[0] * 1e6:.2f} x {cfg.mesh.extent[1] * 1e6:.2f} um")
    print(f"  rollout       {cfg.solver.timesteps} x {cfg.solver.dt * 1e12:g} ps "
          f"= {cfg.duration * 1e9:.2f} ns")
    print(f"  bias          {cfg.fields.B0 * 1e3:g} mT   drive {cfg.fields.Bt * 1e3:g} mT "
          f"({'non-linear' if cfg.fields.Bt >= 20e-3 else 'linear'} regime)")
    print(f"  FMR           {fmr / 1e9:.2f} GHz")
    print(f"  usable band   {lo / 1e9:.2f} .. {hi / 1e9:.2f} GHz "
          f"(propagating, and >= 6 cells per wavelength)")
    dt_max = mnn.estimate_max_timestep(cfg.mesh, cfg.material, cfg.fields)
    print(f"  dt guide      {dt_max * 1e12:.1f} ps (using {cfg.solver.dt * 1e12:g} ps)")

    for f in freqs:
        lam = mnn.wavelength(f, cfg.fields, cfg.material, thickness)
        if lam == float("inf"):
            print(f"  {f / 1e9:.2f} GHz     BELOW FMR -- evanescent, will not propagate")
        else:
            print(f"  {f / 1e9:.2f} GHz     lambda = {lam * 1e9:.0f} nm "
                  f"({lam / cfg.mesh.dx:.1f} cells)")
    print()
