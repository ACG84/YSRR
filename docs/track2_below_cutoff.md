# Track 2 — breaking the readout/memory tradeoff

## The constraint, as measured

Memory capacity is 2.56 frames (absorber 0.35) or 3.11 (absorber 0.12).
NARMA-10 needs ten, so the device ties a shift register there; NARMA-2 needs two
and it wins 2.9×. Decomposing the loss budget:

| channel | frames |
|---|---|
| intrinsic Gilbert damping, 1/(2π α N_cycles) | 8.3 |
| **measured total** | **3.11** |
| residual (port) channel alone | 4.97 |

The ports cost more memory than the material does. That is the constraint, and
four separate levers have failed to move it:

| lever | result |
|---|---|
| absorber 0.35 → 0.12 | +21% memory, prediction slightly worse (reflections → alternation, lag-1 autocorr +0.481 → −0.299) |
| features 12 → 60 | nothing; the extra channels were near-degenerate |
| YIG (α 80× lower) | ~1.6×, because port drain is geometric and ignores α |
| deeper chain | adds delay, but 0.48 amplitude per hop |

## The hypothesis

Port drain has three parts, and only one of them is a real loss that scales
with how open the perimeter is:

1. **propagating loss** — energy carried away as a guided mode. Requires
   f > cutoff. Irreversible.
2. **absorber dissipation** of the evanescent tail. Measured: ≤21% of the port
   drain, since cutting the absorber 3× bought only that.
3. **reactive storage** in the guide volume. Not a loss — it returns.

Every memory measurement so far ran at 12 GHz, above the 80 × 20 nm guide's
11 GHz cutoff, so (1) was active throughout. Below cutoff (1) vanishes: there is
no propagating channel, the guide becomes a reactive load, and an evanescent
stub has no travelling wave to reflect, so it needs no absorber — which also
removes the alternation artefact that spoiled the low-absorber run.

**The readout does not need propagation.** The 3/3 mode decomposition was first
verified at 40 nm width and 8 GHz — below cutoff — where the port is a
near-field tap over 150 nm. That was established early and then forgotten while
a day was spent making the guides propagate.

## Predicted budget

If (1) is ~80% of the port drain:

| | τ_port | τ_damping | τ_total |
|---|---|---|---|
| now (12 GHz, above cutoff) | 4.97 | 8.3 | 3.11 |
| permalloy below cutoff | ~25 | 8.3 | **~6.2** |
| YIG below cutoff | ~25 | 663 | **~24** |

Permalloy alone would roughly double the memory; YIG alone gave 1.6×. The two
are multiplicative because they attack different terms, and ~24 frames puts
NARMA-10 in range. Neither works without the other.

YIG lands in this regime naturally: its measured ladder is 1.77–3.83 GHz and its
80 × 20 nm guide cutoff is ~3–3.5 GHz, so most modes are already below cutoff.
Permalloy has to be deliberately detuned to get there.

## Experiment

One variable, one geometry. Permalloy, absorber 0.12, frequency swept across
the known 11 GHz cutoff:

| condition | f | expectation |
|---|---|---|
| A | 8 GHz | well below; drain minimal |
| B | 10 GHz | just below |
| C | 12 GHz | above — the current operating point, as control |

Then the best of A/B repeated with **absorber 0.0**, which is only sensible
below cutoff.

Each run: 900 frames × 200 steps ≈ 35 min. Four runs ≈ 2.5 h.

Measured per condition, both matter:

- **memory capacity** (Jaeger memory function, `check_task_suite.py`)
- **readout quality** — AB/BA discrimination through the ports
  (`check_readout_comparison.py` at the same frequency). The 24.9× was measured
  only in the propagating regime and there is no guarantee it survives.

## What would falsify it

If capacity does not rise below cutoff, the port drain is **not** propagating
loss — it is near-field coupling into the guide volume, which no frequency
change repairs. The only remaining lever would then be less perimeter coupled:
fewer or narrower ports, which costs readout directly and confirms the tradeoff
is fundamental rather than an artefact of the operating point.

## Decision rule

| outcome | action |
|---|---|
| capacity > 5 **and** AB/BA ratio > 3 | proceed to YIG below cutoff; NARMA-10 becomes reachable |
| capacity > 5 but readout dead | the tradeoff is fundamental. Report it as the result — it is a real constraint on ported reservoirs, not a failure |
| capacity unchanged | mechanism wrong; drain is reactive/near-field. Fall back to fewer ports, or accept the 3-frame envelope and build for it |

## Caveat carried forward

The port-addressed drive (single-port injection, flat in azimuthal order)
changes the excitation basis and is being tested on NARMA now. If it widens the
capability envelope, every number above was measured under uniform drive with
77.9% of the amplitude in one mode, and the Track 2 baselines should be retaken
under port drive before the comparison means anything.
