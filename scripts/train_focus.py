#!/usr/bin/env python3
"""Train a spin-wave lens: steer one tone onto one probe.

The simplest thing the device can be asked to do, and the right first run --
one input, one unambiguous target, and a result you can see directly in the
time-integrated intensity map.

    python scripts/train_focus.py --preset focus --epochs 20
    python scripts/train_focus.py --preset tiny --epochs 5      # ~1 min smoke run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import torch

import magnonic_nn as mnn
from _common import base_parser, report_physics, setup


def main():
    p = base_parser(__doc__, default_preset="focus")
    p.add_argument("--freq", type=float, default=4.0e9, help="drive frequency in Hz")
    p.add_argument("--probes", type=int, default=19)
    p.add_argument("--target", type=int, default=None, help="probe index to focus on")
    args = p.parse_args()

    cfg, outdir = setup(args, "focus")

    print(f"focusing task -> {outdir}")
    report_physics(cfg, freqs=[args.freq])

    task = mnn.build_focusing(cfg, freq=args.freq, n_probes=args.probes, target=args.target)
    model = task.model
    print(model, "\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = mnn.TrainHistory()
    if args.resume:
        _, history = mnn.load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from {args.resume} at epoch {len(history.loss)}")

    def on_epoch(epoch, model, u, loss, history):
        mnn.save_checkpoint(outdir / "checkpoint.pt", model, optimizer, history, epoch)
        mnn.plot_loss(history, outdir / "loss.png", metrics=("contrast_dB",))
        mnn.plot_design(model, outdir / "design.png")
        mnn.plot_probe_outputs(u[0], task.targets[0], outdir / "probes.png")

    mnn.train(
        model, task.signals, task.targets, task.loss_fn,
        epochs=args.epochs, optimizer=optimizer,
        metric_fns=task.metric_fns, history=history,
        on_epoch=on_epoch, grad_clip=10.0,
    )

    # Final diagnostics: where the energy actually ends up.
    with torch.no_grad():
        result = model.run(task.signals[0], record_traces=True,
                           snapshot_every=max(cfg.solver.timesteps // 60, 1))
        m0 = model.equilibrium()

    if result.snapshots is not None:
        mnn.plot_integrated_intensity(model, result.snapshots, outdir / "integrated.png", m0=m0)
        mnn.plot_snapshot(model, result.snapshots[-1], outdir / "snapshot.png", m0=m0)

    u = result.intensities
    contrast = float(mnn.contrast(u.unsqueeze(0), task.targets).mean())
    summary = {
        "target_probe": int(task.targets[0]),
        "argmax_probe": int(u.argmax()),
        "contrast_dB": contrast,
        "normalised_intensities": (u / u.sum()).tolist(),
        "final_loss": history.loss[-1] if history.loss else None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ntarget probe {summary['target_probe']}, "
          f"strongest probe {summary['argmax_probe']}, "
          f"contrast {contrast:.2f} dB")
    print(f"artefacts in {outdir}")


if __name__ == "__main__":
    main()
