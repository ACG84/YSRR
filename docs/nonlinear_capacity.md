# Why the chain loses: it has the memory and does not compute

The depth ladder ended on an uncomfortable number. Three stages score **0.2120**
on NARMA-10 against **0.1243** for a 20-lag *linear* filter, while holding r² >
0.5 recall of the input out to lag 15 and a memory capacity of 15.7. A reservoir
that remembers fifteen lags and loses to a filter that only weights past inputs
is not short of memory. It is failing to compute.

This is the measurement of what it is failing at.

## What computing would mean here, precisely

NARMA-10 is

    y[n+1] = 0.3 y[n] + 0.05 y[n] Σ y[n-9..n] + 1.5 · u[n-9] · u[n] + 0.1

The term `u[n-9] · u[n]` is a product of two inputs nine steps apart. **No linear
filter can produce it at any lag count.** Supplying exactly that is the textbook
justification for building a reservoir at all — if the device carried it, the
linear baseline could not keep up.

## Method

Information processing capacity in the sense of Dambre et al.: for each target
function of the input history, ridge-regress it from the device state and record
test r². Over an *orthogonal* family the r² are additive and bounded by the
state's rank, so the split across degrees says where finite capacity is spent.

Orthogonality is the easy thing to get wrong. `u ~ U[0, 0.5]` is non-negative, so
u² correlates with u at ~0.97 and a naive "u² capacity" is mostly linear memory
wearing a hat. Rescaling to `s = 4u − 1` on [−1, 1] makes the Legendre
polynomials orthogonal for uniform input:

| family | target | meaning |
|---|---|---|
| degree 1 | `P1(s[n−k]) = s[n−k]` | linear memory |
| degree 2 | `P2(s[n−k]) = (3s²−1)/2` | same-lag nonlinearity |
| degree 2 | `s[n]·s[n−k]` | **cross-lag product — the task's term at k = 9** |
| degree 2 | `s[n−5]·s[n−k]` | a product not involving the newest sample |
| degree 3 | `P3(s[n−k]) = (5s³−3s)/2` | third order |

Degree 3 and the delayed product are there so that "the device is linear" is a
measurement rather than a consequence of only having looked at degree 2.

**Floor.** With 300 training rows and 180 features this estimator returns
*something* for any target, so every family is also run against itself shifted 37
frames out of reach — the certification protocol's leakage guard. Max floor r²
was 0.042 (3-chain) and 0.044 (4-chain); anything at or below that is nothing.

## Result: 98–99% of the measured capacity is degree 1

Cached features, no simulation. Same seed, frames, splits, and scorer as the
depth ladder.

| family | 3-chain (180 feat) | 4-chain (240 feat) |
|---|---|---|
| degree 1 `P1(s[n−k])` | **15.48** | **17.08** |
| degree 2 `P2(s[n−k])` | 0.11 | 0.00 |
| degree 2 `s[n]·s[n−k]` | 0.03 | 0.02 |
| degree 2 `s[n−5]·s[n−k]` | 0.19 | 0.14 |
| degree 3 `P3(s[n−k])` | **0.00** | **0.00** |
| total measured | 15.82 | 17.24 |
| effective rank (99% var) | 21 | 21 |
| **degree-1 share of measured** | **98%** | **99%** |

**At lag 9 — the exact pairing NARMA-10 needs — the cross-lag product is
0.000.** In both chains. The device does not carry the task's nonlinear term at
all, and degree 3 is flat zero at every lag out to 20.

The one place any product appears is `s[n−5]·s[n−k]` at k = 6 and 7 (0.208,
0.069) — products of samples *one and two steps apart*. The device mixes
adjacent samples slightly and nothing else.

*What this does and does not show.* Measured capacity (15.82) falls short of the
state's rank (21), and it must: this is five families out of an infinite basis.
The claim supported is the **share** — of the capacity found, 98% is degree 1 —
not that the unmeasured remainder is empty.

## What the missing products are worth

Priced in software, by giving a linear filter its own products:

| baseline | dim | NARMA-10 NMSE |
|---|---|---|
| linear, 10 lags | 10 | 0.9901 |
| linear, 11 lags | 11 | 0.4119 |
| linear, 15 lags | 15 | 0.1268 |
| **linear, 20 lags** | 20 | **0.1243** |
| linear, 26 lags | 26 | 0.1506 |
| linear + products, 11 lags | 77 | 0.2771 |
| **linear + products, 15 lags** | 135 | **0.0460** |
| *the device (3 stages)* | *180* | *0.2008* |

The products are worth **2.7×** over the best pure linear filter and **4.4×**
over the device. They are the whole game on this task, and the device has none
of them. That single fact explains the entire deficit — no appeal to noise,
readout loss, or insufficient depth is needed.

## Two incidental findings about the readout

**The 24 GHz tone is the carrier, leaking.** The readout locks in at 2f
specifically to catch second-harmonic generation. It carries 9.20 of degree-1
capacity and ~0 of anything else — and its canonical correlations with the 12
GHz block are **0.992, 0.991, 0.989** (the other three tones sit at 0.54–0.63).
That is a near-perfect linear image of the carrier: spectral leakage from a
200-step lock-in window, not an independently generated harmonic. One of the
five tones is a redundant copy.

**Rank saturates at 21 regardless of readout width.** 180 columns → rank 21; 240
columns → rank 21. The fourth stage added memory capacity (15.7 → 18.4) and
*zero* new state dimensions. Whatever limits the readout to ~21 independent
directions is not the number of disks.

## The readout hypothesis, tested and rejected

The coherent lock-in returns (real, imaginary) — a **linear** functional of the
magnetisation. A linear readout of a linear state is linear in u by construction,
so the nonlinearity has nowhere to enter. A square-law detector is the obvious
alternative and not a compromise: |A|² is what a diode or bolometer returns, and
power detection is *easier* in hardware than maintaining a phase reference.

The algebra even predicts success. If `A[n] = Σ c_k u[n−k]` then

    |A[n]|² = Σ_ij c_i c_j* · u[n−i] u[n−j]

— every cross-lag product, for free, from a cheaper detector. Same cached state,
only the detector changes:

| readout | dim | rank | deg1 | deg2 P2 | cross-prod | prod@lag9 | NMSE |
|---|---|---|---|---|---|---|---|
| coherent (re, im) | 180 | 21 | 15.48 | 0.11 | 0.03 | **0.000** | 0.2008 |
| power (re²+im²) | 90 | **5** | 13.32 | 0.31 | 0.03 | **0.000** | 0.2045 |
| both | 270 | 21 | 15.98 | 0.66 | 0.02 | **0.000** | **0.1905** |

**It does not work.** The cross-lag product stays at 0.000 in every variant, and
the reason is visible in the rank column: squaring collapses the state from 21
independent directions to **5**. |A|² discards phase, and phase is where the lag
information lives — the same defect that killed the raw-waveform and
state-snapshot readouts earlier in this project, arriving from the opposite
direction. What the square law does buy is same-lag quadratic capacity (0.11 →
0.66 combined) and a modest real gain, 0.2008 → **0.1905**, the best device
number so far and still 1.5× short of the linear filter.

## Where this leaves it

**The device operates in linear response.** The drive modulates a 12 GHz carrier
between 10 and 30 mT; the magnetisation's envelope responds linearly to that,
and a linear readout of a linear envelope is a linear filter with extra steps.
Depth gave it fifteen lags of memory and could not make it compute, because
depth composes delays and delays are linear operations.

That reframes the remaining 1.6× as a drive problem, not a geometry problem, and
it is the first hypothesis in this project with a mechanism specific enough to
predict its own failure mode. The nonlinearity in a vortex disk is amplitude-
dependent: larger precession cone, nonlinear frequency shift, and — at 60 mT,
already measured — core annihilation with λ = +5.5, far past useful. If a
nonlinear-but-stable window exists it is between 30 and 60 mT, and it is narrow
or it would have shown up already.

Two things are worth saying plainly about that. Sweeping drive amplitude is
cheap to screen on a single disk and is the obvious next measurement. But the
window may not exist: the same measurements that make 10–30 mT linear make 60 mT
unstable, and "usefully nonlinear yet stable" is a real constraint that many
physical reservoirs fail. A negative there would be a genuine limit of this
device on this task, not a tuning failure.
