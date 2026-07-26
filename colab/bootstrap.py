#!/usr/bin/env python3
"""Bring up magnonic-nn on a Colab GPU, and prove the GPU path actually works.

Paste the one-liner from ``colab/README.md`` into a Colab cell; it fetches and
runs this.

The order matters. The CUDA path in this package has never been executed -- it
was written and tested entirely on CPU -- so this verifies before it benchmarks
and benchmarks before it trains. A silent device bug that produces plausible
numbers is worse than a crash, and the test suite is what separates them: it
checks Larmor precession against gamma*H, autograd against finite differences,
and checkpointing invariance, none of which care what device they run on but
all of which fail loudly if tensors land on the wrong one.

    python colab/bootstrap.py --stage verify      # tests only
    python colab/bootstrap.py --stage benchmark   # + timing vs the CPU numbers
    python colab/bootstrap.py --stage train --task vowels-full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/ACG84/YSRR.git"
BRANCH = "claude/magnonic-neural-network-sim-16r541"

# Measured on 4 CPU cores in the development container, for comparison.
CPU_BASELINE = {
    (64, 600): 30.4,
    (100, 600): 47.7,
    (80, 2000): 116.0,   # per token, forward + backward
}


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=kw.pop("check", True), **kw)


def stage_install():
    if not Path("YSRR").exists():
        sh(f"git clone --branch {BRANCH} {REPO}")
    sh("pip install -q magnumnp pytest")
    print("\ninstalled\n")


def stage_verify():
    """Run the full suite on the GPU. This is the part that matters."""
    import torch

    if not torch.cuda.is_available():
        print("NO GPU VISIBLE -- Runtime > Change runtime type > GPU")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__}\n")

    r = sh("cd YSRR && python -m pytest tests/ -q --no-header", check=False)
    if r.returncode != 0:
        print("\nTests FAILED on GPU. Do not trust any training run until this is")
        print("green -- a device bug here produces plausible numbers, not a crash.")
        return r.returncode

    print("\nSuite green on GPU. The physics checks (Larmor frequency, autograd")
    print("against finite differences, checkpoint invariance) all pass, so the")
    print("device port is sound.")
    return 0


def stage_benchmark():
    """Time the same configurations the CPU numbers were measured on."""
    sys.path.insert(0, "YSRR")
    import torch

    import magnonic_nn as mnn

    mnn.set_precision("float32")
    mnn.set_device("cuda")

    print(f"{'config':<22} {'GPU':>10} {'CPU (4 cores)':>15} {'speed-up':>10}")
    for (nx, steps), cpu_seconds in CPU_BASELINE.items():
        cfg = mnn.get_preset("focus")
        cfg.mesh.nx = cfg.mesh.ny = nx
        cfg.solver.timesteps = steps
        cfg.material.abc_width = max(nx // 8, 2)

        task = mnn.build_focusing(cfg, n_probes=5)
        u = task.model(task.signals)              # warm up kernels and the demag tensor
        task.loss_fn(u, task.targets).backward()
        torch.cuda.synchronize()

        t0 = time.time()
        u = task.model(task.signals)
        task.loss_fn(u, task.targets).backward()
        torch.cuda.synchronize()
        gpu_seconds = time.time() - t0

        print(f"{f'{nx}x{nx}, {steps} steps':<22} {gpu_seconds:>9.1f}s "
              f"{cpu_seconds:>14.1f}s {cpu_seconds / gpu_seconds:>9.1f}x")

    print("\nSmall meshes are launch-overhead bound, so the speed-up grows with")
    print("size. The configurations worth moving here are the ones CPU ruled out:")
    print("  - full-batch training over a real dataset (168 tokens)")
    print("  - 4000-step rollouts, for 12.5 MHz resolution on the ei/ih pair")
    print("  - the 200x200 'paper' preset")
    print("  - graded meshes fine enough for vortex cores (dx <= 5 nm)")
    return 0


TASKS = {
    "vowels-full": (
        "python scripts/train_vowels.py --preset vowels --nx 80 --timesteps 4000 "
        "--probe-radius 1 --n-per-class 20 --n-train-per-class 14 --Bt-mT 20 "
        "--epochs 40 --lr 0.015 --outdir runs/vowels_full_gpu --device cuda"
    ),
    "paper-focus": (
        "python scripts/train_focus.py --preset paper --epochs 30 --lr 0.05 "
        "--outdir runs/focus_paper --device cuda"
    ),
}


def stage_train(task):
    if task not in TASKS:
        print(f"unknown task {task!r}; choose from {sorted(TASKS)}")
        return 2
    sh(f"cd YSRR && {TASKS[task]}", check=False)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="verify",
                   choices=["install", "verify", "benchmark", "train", "all"])
    p.add_argument("--task", default="vowels-full", choices=sorted(TASKS))
    args = p.parse_args()

    if args.stage in ("install", "all"):
        stage_install()
    if args.stage in ("verify", "all"):
        if stage_verify():
            return 1
    if args.stage in ("benchmark", "all"):
        stage_benchmark()
    if args.stage in ("train", "all"):
        stage_train(args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
