#!/usr/bin/env python3
"""Train vowel classification on the spin-wave scatterer.

The paper's headline result, and the one that needs the *non-linear* regime.
Vowel classes differ in the envelope of their spectrum, not in which single
frequency is present. A linear medium routes every frequency component
independently of every other, so its probe intensities are a fixed linear
functional of the input power spectrum -- there is no design that separates
classes which superposition maps to overlapping outputs. Driving hard enough
that the precession angle grows past a few degrees breaks that, and the same
film becomes able to compute functions of the joint spectrum.

Run the two regimes and compare:

    python scripts/train_vowels.py --Bt-mT 50 --epochs 30    # non-linear
    python scripts/train_vowels.py --Bt-mT 1  --epochs 30    # linear baseline
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
    p = base_parser(__doc__, default_preset="vowels")
    p.add_argument("--n-per-class", type=int, default=16)
    p.add_argument("--n-train-per-class", type=int, default=4,
                   help="paper protocol: 4 training tokens per vowel, rest held out")
    p.add_argument("--jitter", type=float, default=0.06,
                   help="speaker-to-speaker formant variation")
    p.add_argument("--probe-radius", type=float, default=3.0, dest="probe_radius",
                   help="detector radius in cells; shrink it on a small mesh so "
                        "neighbouring detectors do not overlap")
    args = p.parse_args()

    cfg, outdir = setup(args, "vowels")

    print(f"vowel classification -> {outdir}")
    report_physics(cfg)

    task = mnn.build_vowels(
        cfg,
        n_per_class=args.n_per_class,
        n_train_per_class=args.n_train_per_class,
        jitter=args.jitter,
        probe_radius=args.probe_radius,
        seed=args.seed,
    )
    model = task.model
    print(model)
    print(f"  {task.info['n_train']} training / {task.info['n_test']} test tokens, "
          f"classes {task.classes}")
    print(f"  formants mapped into {task.info['band'][0] / 1e9:.2f} .. "
          f"{task.info['band'][1] / 1e9:.2f} GHz\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = mnn.TrainHistory()
    if args.resume:
        _, history = mnn.load_checkpoint(args.resume, model, optimizer, lr=args.lr)
        print(f"resumed from {args.resume} at epoch {len(history.loss)}")

    def on_epoch(epoch, model, u, loss, history):
        mnn.save_checkpoint(outdir / "checkpoint.pt", model, optimizer, history, epoch)
        mnn.plot_loss(history, outdir / "loss.png", metrics=("accuracy",))
        mnn.plot_design(model, outdir / "design.png")

    mnn.train(
        model, task.signals, task.targets, task.loss_fn,
        epochs=args.epochs, optimizer=optimizer,
        metric_fns=task.metric_fns, history=history,
        on_epoch=on_epoch, per_sample=args.per_sample, grad_clip=10.0,
        batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch,
        # Monitor the loss rather than accuracy. Accuracy over three inputs
        # can only take four values and saturates at 1.0 long before the design
        # stops improving -- monitoring it would freeze the "best" checkpoint on
        # the first epoch that got all three right, at half the contrast the run
        # eventually reaches.
        best_path=outdir / 'checkpoint_best.pt', monitor=("loss", "min"),
    )

    # Held-out evaluation. This is the number that matters -- training accuracy
    # on 12 tokens says nothing about whether the design generalises.
    with torch.no_grad():
        u_train = model(task.signals)
        u_test = model(task.test_signals)

    train_acc = mnn.accuracy(u_train, task.targets)
    test_acc = mnn.accuracy(u_test, task.test_targets)
    cm = mnn.confusion_matrix(u_test, task.test_targets, len(task.classes))

    mnn.plot_confusion_matrix(cm, task.classes, outdir / "confusion_test.png",
                              title=f"Test set, Bt = {cfg.fields.Bt * 1e3:g} mT")

    with torch.no_grad():
        result = model.run(task.signals[0], snapshot_every=max(cfg.solver.timesteps // 60, 1))
        m0 = model.equilibrium()
    if result.snapshots is not None:
        mnn.plot_integrated_intensity(model, result.snapshots, outdir / "integrated.png", m0=m0)

    summary = {
        "Bt_mT": cfg.fields.Bt * 1e3,
        "regime": "non-linear" if cfg.fields.Bt >= 20e-3 else "linear",
        "classes": task.classes,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "chance": 1.0 / len(task.classes),
        "confusion_test": cm.tolist(),
        "final_loss": history.loss[-1] if history.loss else None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ntrain accuracy {train_acc:.3f}   test accuracy {test_acc:.3f}   "
          f"(chance {summary['chance']:.3f})")
    print(f"regime: {summary['regime']} (Bt = {summary['Bt_mT']:g} mT)")
    print(f"artefacts in {outdir}")


if __name__ == "__main__":
    main()
