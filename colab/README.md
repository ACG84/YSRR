# Running on a Colab GPU

Paste this into one Colab cell, with **Runtime → Change runtime type → GPU** set:

```python
!git clone -q --branch claude/magnonic-neural-network-sim-16r541 https://github.com/ACG84/YSRR.git
!pip install -q magnumnp pytest
!cd YSRR && python colab/bootstrap.py --stage verify
```

Then, once that is green:

```python
!cd YSRR && python colab/bootstrap.py --stage benchmark
!cd YSRR && python colab/bootstrap.py --stage train --task vowels-full
```

## Verify before you trust anything

The CUDA path in this package **has never been executed** — everything was
written and tested on CPU. A static audit found no `.numpy()` on device
tensors, consistent generator/device pairing, and no implicit-CPU allocations
that mix with model tensors, but that is not the same as running.

Run the test suite on the GPU first. It is not a formality: it checks Larmor
precession against `ω = γH`, the autograd gradient against central differences
to 0.2 %, and checkpointing invariance. None of those care which device they
run on, and all of them fail loudly if tensors land on the wrong one. A device
bug that survives into training does not crash — it produces plausible numbers,
which is worse.

## What the GPU is actually for

These are the things CPU ruled out, in rough order of value:

| | why CPU couldn't | GPU cost |
|---|---|---|
| **4000-step rollouts** | 12.5 MHz resolution resolves the ei/ih vowel pair (20 MHz apart); at 2000 steps it is marginal by design | 2× a 2000-step run |
| **Full-batch over a real dataset** | 168 tokens × 116 s = 5.4 h *per epoch*; minibatching was a workaround, not a preference | one epoch, not one afternoon |
| **`paper` preset** (200×200, 1200 steps) | ~10 min/epoch | the size the paper actually used |
| **Vortex disks** | cores are 10–20 nm, needing `dx ≤ 5 nm` — 100× the cells and ~100× smaller timesteps | the graded-mesh work, via `DemagFieldNonEquidistant` |

CPU baselines for comparison, measured on 4 cores at float32, forward +
backward: 64×64/600 steps **30.4 s**, 100×100/600 steps **47.7 s**,
80×80/2000 steps **116 s per token**. `--stage benchmark` times the same
configurations and prints the ratio.

Expect a modest speed-up on the small meshes — 10⁴ cells is launch-overhead
bound — and a large one as size grows. That is the shape of the magnum.np
benchmark in the NeuralMag paper, and it is why the interesting configurations
are the big ones.

## What I can and cannot do from here

I **cannot drive Colab from this session**. Colab has no headless CLI: a runtime
has to be started from a browser, and there is no API to submit a job to it.
What is here is everything up to that point — clone, install, verify, benchmark,
train — as a single cell you run.

If you want something genuinely unattended, the alternatives are a plain SSH box
with a GPU, or `runpod`/`vast.ai`-style rentals, both of which do have CLIs I
could drive end to end. Colab Pro's background execution keeps a run alive after
you close the tab, which covers most of the gap.

## Checkpoints come back

Runs write to `YSRR/runs/<name>/checkpoint.pt` every epoch. Colab wipes local
storage when the runtime recycles, so mount Drive first if a run matters:

```python
from google.colab import drive; drive.mount('/content/drive')
!cd YSRR && python scripts/train_vowels.py ... --outdir /content/drive/MyDrive/ysrr_runs/vowels_full
```

Every training script takes `--resume <checkpoint>`, and `train()` numbers
epochs from the restored history, so a resumed run continues its count rather
than restarting at zero.
