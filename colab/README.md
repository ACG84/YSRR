# Running on a Colab GPU

Colab has a headless CLI (`google-colab-cli`, June 2026), so this does not need
a browser tab. From a machine that is already authorised:

```bash
colab run --gpu T4 --keep -s ysrr --timeout 21600 \
    colab/bootstrap.py --stage all --task vowels-full
colab download -s ysrr YSRR/runs/vowels_full_gpu/checkpoint.pt runs/vowels_full_gpu/
colab stop -s ysrr
```

`run` is the one that forwards arguments (`exec` only takes `-f FILE`, no
argv), it uploads the local script itself so the repo need not be pushed
first, and `--stage all` chains install → verify → benchmark → train. Watch
the `--timeout`: it defaults to **30 s**, which silently truncates anything
real. Without `--keep` the VM is torn down when the script exits, taking the
checkpoints with it.

The equivalent in a notebook cell, with **Runtime → Change runtime type → GPU**:

```python
!git clone -q --branch claude/magnonic-neural-network-sim-16r541 https://github.com/ACG84/YSRR.git
!pip install -q magnumnp pytest
!cd YSRR && python colab/bootstrap.py --stage verify
```

## Installing and authorising the CLI

The CLI needs Python ≥ 3.12 and this container's system Python is 3.11, so
install it with its own interpreter rather than into the environment:

```bash
uv tool install --python 3.12 google-colab-cli
```

Authorisation is a copy-paste OAuth flow — not a localhost redirect, and not
the OOB flow Google blocked in 2022 — so it works from a container as long as
something can carry one string in each direction. `colab_cli` reads the code
from a TTY, which a non-interactive session does not have; `auth_relay.py`
splits the same flow in two so the code can be relayed by hand:

```bash
python colab/auth_relay.py url                # open in a browser, approve
python colab/auth_relay.py exchange <CODE>    # writes ~/.config/colab-cli/token.json
```

The scopes are the CLI's own and include `cloud-platform` and `drive.file`,
which is broad for what is being asked. The token lands in a container that is
reclaimed on idle; revoke at https://myaccount.google.com/permissions.

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

## What has to happen by hand

One step, once: approving the OAuth URL and pasting the code back. Everything
after that — starting a runtime, uploading, installing, running the suite,
training, pulling checkpoints down, stopping the VM — is scriptable, and the
refresh token persists for as long as the container does.

Colab reclaims idle runtimes, so an unattended run still wants either Drive
(below) or a `colab download` on a timer. A plain SSH GPU box or a
`runpod`/`vast.ai` rental avoids that, at the cost of not being free.

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
