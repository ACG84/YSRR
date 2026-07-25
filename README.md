# magnonic-nn

A magnonic **physical neural network** trained end to end by backpropagating
through full micromagnetics, built on [magnum.np](https://gitlab.com/magnum.np/magnum.np).

Spin waves are launched into a thin YIG film, scattered by a trainable pattern,
and read out as intensity at a few detector spots. Signal routing and non-linear
activation are both done by the magnetisation dynamics — there are no artificial
neurons, no weight matrices, and no activation functions anywhere in the model.
The only learnable parameters are physical: a field pattern above the film, a
saturation-magnetisation pattern in it, or the up/down states of an array of
nanomagnets.

Reproduces the setup of

> Papp, Á., Porod, W. & Csaba, G. *Nanoscale neural network using non-linear
> spin-wave interference.* Nat Commun **12**, 6422 (2021).
> [doi:10.1038/s41467-021-26711-z](https://doi.org/10.1038/s41467-021-26711-z)

using magnum.np's validated field terms in place of a bespoke solver, and following
the inverse-micromagnetics approach of

> Abert, C. *et al.* *NeuralMag: an open-source nodal finite-difference code for
> inverse micromagnetics.* npj Comput Mater **11**, 193 (2025).
> [doi:10.1038/s41524-025-01688-1](https://doi.org/10.1038/s41524-025-01688-1)

---

## Install

```bash
pip install -e .
```

`magnumnp` sets `torch`'s default dtype to `float64` and grabs a CUDA device at
import time. This package routes every magnum.np import through
`magnonic_nn._compat` so those globals stay under your control — call
`set_precision()` / `set_device()` before building anything.

## Quick start

```python
import magnonic_nn as mnn

mnn.set_precision("float32")

task = mnn.build_focusing(mnn.get_preset("focus"), freq=4.0e9)
mnn.train(task.model, task.signals, task.targets, task.loss_fn,
          epochs=20, lr=0.08, metric_fns=task.metric_fns)
```

Or from the command line:

```bash
python scripts/validate_dispersion.py                 # run this FIRST
python scripts/train_focus.py  --preset focus  --epochs 20
python scripts/train_demux.py  --preset demux  --epochs 30
python scripts/train_vowels.py --preset vowels --epochs 30 --Bt-mT 50
python scripts/plot_convergence.py runs/*/checkpoint.pt -o convergence.html
```

Every script takes `--preset tiny` for a smoke run that finishes in about a minute.

---

## How it works

```
drive waveform ──▶ antenna ──▶ [ YIG film + trainable scatterer ] ──▶ probes ──▶ intensities
                                          ▲                                          │
                                          └────────── ∂L/∂ρ ◀── backprop ────────────┘
```

Each forward pass is a fixed-step RK4 integration of the Landau–Lifshitz–Gilbert
equation over the whole mesh — typically 600–1200 steps of 20 ps. The effective
field comes from magnum.np (`ExchangeField`, `DemagField`) plus the static bias,
the trainable scatterer, and the time-dependent drive. Probe intensities are
accumulated over time, and the gradient with respect to the design comes from
backpropagation through the entire rollout.

### Gradients

Backpropagation through time with **gradient checkpointing**. With `T` steps in
segments of `c`, peak activation memory is `O(T/c + c)` states, minimised at
`c = √T` — for 600 steps that's ~25 stored states instead of 600.

magnum.np's `TorchDiffEqAdjoint` is the memory-lean alternative and is what the
NeuralMag paper advocates, but it reconstructs the trajectory by integrating
*backwards*. A driven, oscillatory wave is exactly the regime where backwards
reconstruction accumulates error, so exact BPTT is the default here. Fixed steps
also mean the graph depth doesn't depend on the design variables, so epoch cost
is predictable.

`tests/test_solver.py::test_gradient_matches_finite_differences` checks the
autograd gradient against central differences to 0.2%.

### Why the drive amplitude is the most important knob

The non-linearity isn't added anywhere — it's intrinsic to the LLG equation, and
it switches on when the precession angle grows past a few degrees.

| `Bt` | regime | what the network can compute |
|---|---|---|
| ~1 mT | linear | superposition holds: each frequency is routed independently, so the output is a fixed linear functional of the input power spectrum |
| 20–50 mT | non-linear | frequency components interact; functions of the *joint* spectrum become reachable |

Frequency demultiplexing works fine in the linear regime — distinct frequencies
never need to interact. Vowel classification does not: the classes differ in
spectral *envelope*, which is precisely what a linear medium cannot separate.
That contrast is the paper's central claim, and
`test_model_train.py::test_strong_drive_leaves_the_linear_regime` pins it down.

---

## Getting the physics right

**Run `scripts/validate_dispersion.py` before anything else.** It drives a strip
of film, measures the wavelength that actually propagates, and compares against
the analytic dipole-exchange dispersion. Agreement means the mesh, field terms,
timestep and absorbing boundary are all doing what they should:

```
   f (GHz)   lambda meas   lambda calc      err
      3.60        890 nm        995 nm    11.9%
      4.00        457 nm        461 nm     0.8%
      4.50        296 nm        301 nm     1.5%
      5.00        221 nm        234 nm     5.9%
```

The band edges are looser for physical reasons — at 3.6 GHz the ~1 µm wavelength
barely fits inside the measurement window, and at 5 GHz it is down to 4.7 cells,
which is where `usable_band()` says the mesh starts to alias. Errors in the
*middle* of the band point at a real problem.

### The dispersion relation actually matters

For a film magnetised in plane with `k ⊥ M` (the Damon–Eshbach geometry the
default source/probe layout sets up):

```
ω²(k) = ω₀(k)·[ω₀(k) + ω_M] + (ω_M²/4)·(1 − e^{−2kd})
ω₀(k) = γ·(H₀ + (2A / μ₀M_s)·k²),      ω_M = γ·M_s
```

An exchange-only estimate is **wrong by a factor of three** here: at 4 GHz in
20 nm YIG it predicts 158 nm where the true wavelength is 461 nm, because it
charges the entire frequency excess above resonance to exchange when most of it
is dipolar. Three times the wavelength is three times the cells you need.

Two consequences worth internalising:

- **There is a hard low-frequency cutoff.** At the default 60 mT bias the FMR
  sits at **3.33 GHz**; below it a drive is evanescent, not a wave. This is why
  the demux default is 3.5/4.0/4.5 GHz rather than the paper's 3.0/3.5/4.0 —
  those were taken at a different bias point. Drop `B0` to 30 mT if you want the
  paper's exact triple.
- **The group velocity is only ~430 m/s.** Crossing a 5 µm film takes ~12 ns,
  which is exactly the 600 × 20 ps rollout the `focus` preset uses. Shorten the
  rollout and the wave simply never reaches the probes.

`build_*` task builders warn when a requested frequency falls outside the usable
band, rather than letting you discover it an hour into a run that could not work.

---

## Results

Focusing, `focus` preset (100×100 cells at 50 nm = 5 µm square, 12 ns rollout,
4 GHz, 1 mT, free-form design):

| | epoch 0 | epoch 7 |
|---|---|---|
| focus loss | 1.013 | −1.121 |
| probe contrast | −0.61 dB | **+16.2 dB** |

+16 dB means the target probe receives ~42× the intensity of its strongest
competitor, starting from a design that was actually losing to one. About 60 s
per epoch on 4 CPU cores.

---

## Layout

```
magnonic_nn/
  config.py       dataclasses + presets (tiny / focus / demux / vowels / paper)
  dispersion.py   analytic dispersion, usable band, numerical validation
  geometry.py     trainable designs: free-form field, Ms pattern, nanomagnet array
  damping.py      absorbing boundary profile
  sources.py      point / line / apodised stripe antennas
  probes.py       disk probes, phase-sensitive intensity readout
  solver.py       differentiable RK4 rollout with checkpointing
  model.py        SpinWaveNetwork (torch.nn.Module)
  signals.py      tones, chirps, and the synthetic vowel dataset
  losses.py       focus loss, cross-entropy, contrast, accuracy
  train.py        training loop, per-sample accumulation, checkpointing
  tasks.py        build_focusing / build_demux / build_vowels
  plotting.py     design, snapshots, integrated intensity, convergence
scripts/          CLI entry points
tests/            physics, gradients and training checks
```

### Trainable designs

| `cfg.geometry` | parameter | notes |
|---|---|---|
| `freeform` | per-cell field offset, `B0 + B1·tanh(ρ)` | most degrees of freedom, easiest to train |
| `ms` | per-cell `M_s` multiplier | patterning the film itself |
| `nanomagnets` | binary up/down PMA dots | the fabricable device; stray field from a Newell kernel at the stand-off height, trained through a straight-through estimator |

`freeform` bounds the design with `tanh` by default. The reference
implementation leaves it unbounded, which lets the optimiser converge on local
fields no real device could apply — pass `bound="none"` to reproduce that, and
check `model.design_field_tesla()` before believing the result.

---

## Deviations from the paper

Deliberate, and each for a stated reason:

1. **Vowel frequency mapping.** The paper scales speech "up to microwave
   frequencies" with a single factor. That cannot work here: formants span more
   than a decade, and mapping F1 into the passband throws F3 into the
   unresolvable exchange regime. The default is an *affine* map onto the
   device's usable band, preserving the relative formant spacing that
   distinguishes the vowels. `mapping="linear"` gives the paper-literal version.
2. **Synthetic vowels.** A source–filter model (harmonic comb × three Lorentzian
   formants, with per-speaker jitter) rather than the Hillenbrand recordings, so
   the dataset is reproducible and dependency-free. Formant centres and
   bandwidths are the published adult-male averages.
3. **Demux frequencies.** 3.5/4.0/4.5 GHz instead of 3.0/3.5/4.0, because
   3.0 GHz is below this film's FMR. See above.
4. **Absorbing boundary.** Monotonic taper to `alpha_max` at the outermost ring.
   The reference profile leaves the outermost cell at the bulk value; the
   difference is one cell deep inside an already-absorbing layer.
5. **Exact BPTT rather than the adjoint method.** Discussed above.

## Performance

Measured on 4 CPU cores, `float32`, forward + backward:

| mesh | steps | per epoch |
|---|---|---|
| 48×48 | 100 | ~2 s |
| 64×64 | 600 | ~30 s |
| 100×100 | 600 | ~48 s |

A GPU is strongly recommended for the `paper` preset (200×200, 1200 steps). Set
`--device cuda`; everything else is unchanged.

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow"     # fast suite
pytest                   # includes the full-stack dispersion check
```

The suite covers Larmor precession against `ω = γH`, norm conservation,
absorbing-boundary effectiveness, autograd versus finite differences,
checkpointing invariance, the linear/non-linear transition, and end-to-end
convergence.

## Licence

GPL-3.0-or-later, matching magnum.np.
