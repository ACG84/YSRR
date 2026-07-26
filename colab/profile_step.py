#!/usr/bin/env python3
"""Find out what a timestep actually spends its time on.

The T4 benchmark returned 31.9 s for 64x64/600 steps and 31.6 s for
100x100/600 steps -- 2.44x the cells in the same wall clock. Work per cell is
therefore free, and the entire cost is fixed overhead per step. At 600 steps
that is ~53 ms/step, which is far too slow to be kernel-launch latency
(microseconds); it is the price of dispatching hundreds of tiny eager ops per
RK4 stage through Python.

If that is right, the fix is not a bigger GPU -- it is fusion. ``_compat.py``
no-ops ``torch.compile`` before magnum.np imports, because on CPU inductor cost
more than it returned and interacted badly with checkpoint re-entrancy. On GPU
that trade should invert. ``MAGNONIC_NN_COMPILE=1`` leaves compilation in
place, so this runs both and compares.

Compilation is decided at import time, so each variant needs its own process.
Forward and backward are timed separately: the re-entrancy problem lives in
backward, and a forward-only win would still be worth having.

``colab exec`` runs a file as source inside the kernel rather than as a script,
so ``__file__`` is undefined there and the parent cannot find itself to spawn
children. Upload a copy to a known path first and both halves work:

    colab upload -s <session> colab/profile_step.py /content/profile_step.py
    colab exec -s <session> -f colab/profile_step.py --timeout 2700
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

CONFIGS = [(64, 600), (100, 600)]

# Undefined when run as cell source by ``colab exec``; see the module docstring.
SELF = globals().get("__file__") or "/content/profile_step.py"


def measure() -> dict:
    """Time one forward and one backward per config, in this process."""
    sys.path.insert(0, "YSRR")
    import torch

    import magnonic_nn as mnn

    mnn.set_precision("float32")
    mnn.set_device("cuda")

    out = {"compile": mnn.compile_enabled(), "torch": torch.__version__, "runs": []}
    for nx, steps in CONFIGS:
        cfg = mnn.get_preset("focus")
        cfg.mesh.nx = cfg.mesh.ny = nx
        cfg.solver.timesteps = steps
        cfg.material.abc_width = max(nx // 8, 2)
        task = mnn.build_focusing(cfg, n_probes=5)

        row = {"nx": nx, "steps": steps}
        try:
            u = task.model(task.signals)          # warm up: kernels, demag tensor,
            task.loss_fn(u, task.targets).backward()   # and inductor compilation
            torch.cuda.synchronize()

            t0 = time.time()
            u = task.model(task.signals)
            torch.cuda.synchronize()
            row["forward_s"] = time.time() - t0

            t0 = time.time()
            task.loss_fn(u, task.targets).backward()
            torch.cuda.synchronize()
            row["backward_s"] = time.time() - t0
            row["ms_per_step"] = 1e3 * (row["forward_s"] + row["backward_s"]) / steps
        except Exception as exc:                  # compiled backward may not survive
            row["error"] = f"{type(exc).__name__}: {exc}"[:400]
        out["runs"].append(row)

    return out


def child(compile_on: bool) -> dict:
    env = dict(os.environ, MAGNONIC_NN_COMPILE="1" if compile_on else "0",
               MAGNONIC_NN_PROFILE_CHILD="1")
    r = subprocess.run([sys.executable, SELF], env=env,
                       capture_output=True, text=True, timeout=2400)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"error": (r.stderr or r.stdout)[-1200:]}


def main() -> int:
    if os.environ.get("MAGNONIC_NN_PROFILE_CHILD"):
        print("RESULT " + json.dumps(measure()), flush=True)
        return 0

    results = {}
    for label, on in (("eager", False), ("compiled", True)):
        print(f"--- {label} ---", flush=True)
        results[label] = child(on)
        print(json.dumps(results[label], indent=2), flush=True)

    print(f"\n{'config':<20} {'eager':>12} {'compiled':>12} {'speed-up':>10}")
    for i, (nx, steps) in enumerate(CONFIGS):
        rows = [results[k].get("runs", [{}] * len(CONFIGS))[i] for k in ("eager", "compiled")]
        e, c = (r.get("ms_per_step") for r in rows)
        name = f"{nx}x{nx}, {steps} steps"
        if e and c:
            print(f"{name:<20} {e:>10.1f}ms {c:>10.1f}ms {e / c:>9.1f}x")
        else:
            note = next((r["error"] for r in rows if "error" in r), "no result")
            print(f"{name:<20} {'--':>12} {'--':>12}   {note[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
