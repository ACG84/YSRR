#!/usr/bin/env python3
"""Train a magnonic demultiplexer: each frequency to its own probe.

The paper's frequency-separation experiment. This one works in the linear
regime -- distinct frequencies do not need to interact for the scatterer to
send them different ways, it just needs a design whose interference pattern is
frequency dependent, which any dispersive medium gives you for free.

    python scripts/train_demux.py --preset demux --epochs 30
    python scripts/train_demux.py --preset tiny --epochs 5
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
    p = base_parser(__doc__, default_preset="demux")
    p.add_argument("--freqs", type=float, nargs="+", default=None,
                   help="drive frequencies in GHz (default: 3.5 4.0 4.5)")
    args = p.parse_args()

    cfg, outdir = setup(args, "demux")
    freqs = [f * 1e9 for f in args.freqs] if args.freqs else None

    print(f"demultiplexing task -> {outdir}")
    report_physics(cfg, freqs=freqs or [3.5e9, 4.0e9, 4.5e9])

    task = mnn.build_demux(cfg, freqs=freqs)
    model = task.model
    print(model, "\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = mnn.TrainHistory()
    if args.resume:
        _, history = mnn.load_checkpoint(args.resume, model, optimizer, lr=args.lr)
        print(f"resumed from {args.resume} at epoch {len(history.loss)}")

    def on_epoch(epoch, model, u, loss, history):
        mnn.save_checkpoint(outdir / "checkpoint.pt", model, optimizer, history, epoch)
        mnn.plot_loss(history, outdir / "loss.png", metrics=("accuracy", "contrast_dB"))
        mnn.plot_design(model, outdir / "design.png")

    mnn.train(
        model, task.signals, task.targets, task.loss_fn,
        epochs=args.epochs, optimizer=optimizer,
        metric_fns=task.metric_fns, history=history,
        on_epoch=on_epoch, per_sample=args.per_sample, grad_clip=10.0,
        # Monitor the loss rather than accuracy. Accuracy over three inputs
        # can only take four values and saturates at 1.0 long before the design
        # stops improving -- monitoring it would freeze the "best" checkpoint on
        # the first epoch that got all three right, at half the contrast the run
        # eventually reaches.
        best_path=outdir / 'checkpoint_best.pt', monitor=("loss", "min"),
    )

    # Per-channel diagnostics: one intensity map and one probe spectrum each.
    rows = []
    with torch.no_grad():
        for i, label in enumerate(task.classes):
            result = model.run(task.signals[i], record_traces=True,
                               snapshot_every=max(cfg.solver.timesteps // 60, 1))
            m0 = model.equilibrium()
            tag = label.replace(" ", "").replace(".", "p")

            if result.snapshots is not None:
                mnn.plot_integrated_intensity(
                    model, result.snapshots, outdir / f"integrated_{tag}.png",
                    m0=m0, title=f"Time-integrated intensity, {label}",
                )
            mnn.plot_probe_outputs(
                result.intensities, task.targets[i], outdir / f"probes_{tag}.png",
                labels=task.classes, title=f"Probe outputs, input {label}",
            )
            u = result.intensities
            rows.append({
                "input": label,
                "target_probe": int(task.targets[i]),
                "argmax_probe": int(u.argmax()),
                "contrast_dB": float(mnn.contrast(u.unsqueeze(0), task.targets[i : i + 1])[0]),
                "normalised": (u / u.sum()).tolist(),
            })

    with torch.no_grad():
        u_all = model(task.signals)
    summary = {
        "channels": rows,
        "accuracy": mnn.accuracy(u_all, task.targets),
        "mean_contrast_dB": float(mnn.contrast(u_all, task.targets).mean()),
        "final_loss": history.loss[-1] if history.loss else None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    cm = mnn.confusion_matrix(u_all, task.targets, len(task.classes))
    mnn.plot_confusion_matrix(cm, task.classes, outdir / "confusion.png")

    print(f"\nrouting accuracy {summary['accuracy']:.2f}, "
          f"mean contrast {summary['mean_contrast_dB']:.2f} dB")
    for row in rows:
        print(f"  {row['input']:>10} -> probe {row['argmax_probe']} "
              f"(target {row['target_probe']}, {row['contrast_dB']:+.1f} dB)")
    print(f"artefacts in {outdir}")


if __name__ == "__main__":
    main()
