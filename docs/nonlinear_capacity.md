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

## The drive hypothesis, tested and REFUTED — the disk is already nonlinear

The obvious reading of everything above is that the device operates in linear
response, and that the fix is to drive it harder. That reading is wrong, and the
measurement that settles it is cheap: drive one disk at constant amplitude, lock
in per port, transform across ports, and watch the response as amplitude rises.

(The cross-port transform is load-bearing. A uniform in-plane drive couples
almost entirely to n = ±1 by symmetry, so averaging the six ports first lands on
n = 0 and cancels the response — reporting the disk's strongest mode as silence,
and the phase of that silence as a frequency shift.)

Across the *exact* window the runs use:

| drive | gain `|A|/a`, normalised | phase shift | mean m_z | vortex |
|---|---|---|---|---|
| 5 mT | 1.000 | — | 0.2466 | intact |
| **10 mT** | **0.980** | **+3.2°** | 0.2262 | intact |
| 15 mT | 0.947 | +8.6° | 0.1992 | intact |
| 20 mT | 0.898 | +15.7° | 0.1712 | intact |
| 25 mT | 0.835 | +24.0° | 0.1455 | intact |
| **30 mT** | **0.758** | **+32.9°** | 0.1255 | intact |
| 35 mT | 0.669 | +41.5° | 0.0910 | intact |
| 40 mT | 0.571 | +49.2° | −0.1581 | **core expelled** |

**The disk is strongly nonlinear over 10–30 mT.** Gain falls 23% across the
modulation range and the response phase swings 30°. This is a large-signal
nonlinearity, sitting squarely inside the operating point every run so far has
used. There is nothing to fix about the drive amplitude; the nonlinearity was
always there. (It also bounds the headroom: the core is expelled by 40 mT, so
the stable window ends at ~35 mT, not the 60 mT annihilation threshold measured
earlier under different conditions.)

## Where this leaves it: the nonlinearity and the memory are in series, in the wrong order

Both facts are now measured and they look contradictory: the disk is strongly
nonlinear, and the chain carries essentially no nonlinear capacity. The
resolution is in *where* each thing lives.

A nonlinearity can only multiply signals that are present in the same state at
the same time. The nonlinear element is the driven disk — stage 1 — and **stage
1 remembers lags 0–6** (measured, depth ladder). So the only products it can
form are between samples inside that span. That is exactly, and only, what was
found:

| product | r² |
|---|---|
| `s[n]·s[n−1]` | 0.054 |
| `s[n]·s[n−5]` | 0.056 |
| **`s[n−5]·s[n−6]`** | **0.208** |
| `s[n−5]·s[n−7]` | 0.069 |
| everything at lag ≥ 8 apart | **0.000** |

Every product the device makes is between samples one or two frames apart. The
deep stages remember out to lag 14 — but they receive no fresh input to mix
against, only a delayed copy of one thing. So the chain nonlinearly mixes
adjacent samples *first*, then delays the result. Depth composes delays, and
delays are linear.

**NARMA-10 needs the opposite order.** `u[n−9]·u[n]` requires a nonlinear element
that sees a 9-frame-old copy and a fresh sample simultaneously. No stage in this
chain ever does.

That is an architectural prescription rather than a tuning knob, and the geometry
already measured supplies the number: **stage 3's impulse response peaks at lag
9.48**. Co-driving stage 3 with the same input gives one nonlinear element both
the chain-delayed copy (~9.5 frames) and the fresh sample — which is `u[n]·u[n−9.5]`,
against the `u[n]·u[n−9]` the task asks for. The delay line was accidentally
built to the right length.

This is now running: `--drive-stages 0 2`, same seed, frames, splits, and scorer,
so it drops straight onto the ladder. The prediction is specific and therefore
falsifiable — cross-lag product capacity near lag 9 should go from 0.000 to
something, or the account above is wrong.

Worth stating what would make it fail even if the reasoning is right: the fresh
drive at stage 3 also injects a large lag-0 signal there, and if it swamps the
delayed copy (which has been attenuated by 0.092 per hop, twice) the product term
is a small correction on a big one. Transfer says the delayed copy arrives ~100×
down on the direct drive, which is the real risk in this experiment.

## Pricing the fix before buying it — and finding the real constraint

The co-drive arm was launched on a specific claim: NARMA-10 needs `u[n−9]·u[n]`,
the chain has 0.000 of it, so manufacture it. Before spending three hours
confirming whether it appears, it is worth pricing what it is worth **if it
does**. That pricing is arithmetic on the task, not a device measurement, so it
bounds what *any* mechanism delivering these terms could buy.

The answer refutes the framing. Adding the exact term to a 20-lag linear filter
changes nothing:

| readout | dim | NMSE |
|---|---|---|
| linear, 20 lags (the bar) | 20 | 0.1243 |
| + `u[n]·u[n−9]` **only** | 21 | **0.1242** |
| + `u[n]·u[n−k]`, k = 8,9,10 | 23 | 0.1284 |
| + `u[n]·u[n−k]`, k = 1..15 | 35 | 0.1375 |
| + all 105 products, 15 lags | 135 | **0.0448** |

The celebrated `u[n−9]·u[n]` term — the one every description of NARMA-10 quotes,
including mine — is worth **nothing** on its own. What makes the task hard is the
*y*-recursion, whose expansion needs products across many lag pairs.

Splitting the 105 products by separation says which:

| product family | dim | NMSE |
|---|---|---|
| squares `u[n−i]²` only | 35 | 0.1245 |
| adjacent `u[n−i]·u[n−i−1]` | 35 | 0.1300 |
| near pairs, \|i−j\| ≤ 2 | 62 | 0.1215 |
| near pairs, \|i−j\| ≤ 4 | 85 | 0.1236 |
| **far pairs, \|i−j\| ≥ 5** | 75 | **0.0401** |

All of the value is in **long-separation** products, and none in short ones. The
device makes exactly and only the worthless family: every product it carries is
between samples one or two frames apart.

### How many dimensions the value needs

The value is not a few well-chosen terms. Projecting the far-pair block onto its
own leading principal components and adding them to the linear filter:

| product dimensions kept | NMSE | ×1.53 readout tax |
|---|---|---|
| 0 | 0.1243 | 0.1901 |
| 3 | 0.1113 | 0.1702 |
| 8 | 0.1123 | 0.1718 |
| 21 | 0.1070 | 0.1637 |
| **40** | 0.0745 | **0.1140** |
| 75 (all) | 0.0404 | 0.0618 |

Nothing useful happens until ~40 independent product directions are present. The
"tax" column applies the device's own measured penalty — the chain scores 0.1905
while holding linear memory of comparable span to the 20-lag filter's 0.1243, so
its readout costs ~1.53× — and says the device would need **≥ 40** product
dimensions to clear the bar.

### What the device has

Effective rank, measured on the training block:

| state | columns | rank |
|---|---|---|
| single disk, port drive | 60 | 13 |
| chain stage 1 (uniform drive) | 60 | 15 |
| chain stage 2 | 60 | 14 |
| chain stage 3 | 60 | 9 |
| **all three stages together** | **180** | **21** |
| four stages together | 240 | 21 |

Stages of rank 15, 14 and 9 combine to 21, not 38: they are largely redundant
with each other. And degree-1 capacity alone is 15.5 of that 21, leaving roughly
**5 directions** for everything nonlinear — against the ~40 required.

That also explains why depth stopped paying and why a drive-mode change will not
rescue it. For a device whose state is a near-linear functional of input history,
**effective rank is bounded by memory span**: each remembered lag contributes one
direction, and a delayed copy of a signal already in the state is a linear
combination of directions already counted. Adding disks in series adds delayed
copies. It cannot add rank, and measurement agrees — 180 columns and 240 columns
both give 21.

Rank and nonlinearity are therefore not two problems but one. Genuinely new
directions come only from computing genuinely new functionals of the history.

## Is NARMA-10 feasible? Not on this architecture

The gap is now quantified rather than guessed. The device needs ~40 independent
long-separation product dimensions and has ~5, and the deficit cannot be closed
by more depth (rank pinned at 21), more readout width (pinned at 21), a
square-law detector (collapses rank to 5), or a harder drive (the disk is
already nonlinear at the operating point; the core is expelled by 40 mT).

What the measurements *do* support is a different architecture. Every ingredient
has been separately demonstrated:

- the link carries a genuine guided wave, 1173 m/s, 90× the dipolar background
- delay is tunable and linear in separation, and a 3-hop delay lands at lag 9.5
- the disk is strongly nonlinear at 30 mT and stable to ~35 mT
- identical parallel disks are perfectly redundant (CC = 1.000) — but only when
  fed identically

Which points at a **delay-line bus tapped in parallel**: one guided line carrying
the input, tapped at many points, each tap feeding its own nonlinear disk that
*also* receives the fresh sample. Each disk then mixes a different delay against
the present, producing long-separation products at a different separation each —
many independent nonlinear functionals rather than one, in parallel rather than
in series. That is the shape the capacity accounting asks for, and it is
precisely not the serial chain that was built.

Whether it clears 0.1243 is unproven and should not be assumed: it needs ~8×
more nonlinear dimensions than anything measured here has produced, and the
redundancy that killed parallel disks before would have to be broken by the
differing delays rather than merely assumed to be.

## The co-drive arm: stopped early, and it failed the way it was warned it might

The co-drive run (`--drive-stages 0 2`) was stopped at 260 of 600 frames once its
partial data was decisive. Scored against the single-drive control **at matched
sample size** — same 260 frames, same reduced splits, same scorer, because a
smaller training block lowers every capacity estimate and the comparison is
worthless otherwise:

| | single drive | co-drive (0 and 2) |
|---|---|---|
| degree-1 capacity | 8.91 | **4.95** |
| r² at lag 9 | 0.857 | **0.152** |
| r² at lag 11 | 0.548 | 0.126 |
| r² at lag 14 | 0.298 | 0.005 |

Products were unmeasurable either way at this sample size — the noise floor for
`s[n]·s[n−k]` is 0.289 with 30 test points — so the arm never got to test its own
prediction. What it did show is that **co-driving actively destroyed the long
memory**: the horizon pulled in from ~lag 12 to ~lag 7 and degree-1 capacity
nearly halved.

This is the failure mode that was flagged when the arm was launched: the fresh
sample injected at stage 3 arrives at full amplitude, the chain-delayed copy
arrives ~100× down after two hops at 0.092 per hop, and the tap stops being a
delay tap and becomes another input disk. It is an amplitude-balance failure, not
a refutation of mixing fresh against delayed — and it makes the balance
requirement quantitative for anything built next.

## The bus: transport measured, and it is 16× better than the chain

A poly-tap architecture needs one thing the chain could not provide — a delayed
copy that arrives strong enough to be mixed rather than drowned. That is a
property of the guide, not of the disks, and it had never been measured over
micron distances. A bare 80 nm strip, 5 µm long, absorbing at both ends, driven
near one end and tapped every 150 nm:

| | |
|---|---|
| attenuation length | **951 nm** |
| group velocity | **945 m/s** (chain fit gave 1173) |
| one frame of delay | **189 nm** |

| lag (frames) | tap at | amplitude | vs lag-5 tap |
|---|---|---|---|
| 5 | 900 nm | 1.454e−02 | 1.000 |
| 8 | 1500 nm | 7.860e−03 | 0.541 |
| 10 | 1950 nm | 4.743e−03 | 0.326 |
| 12 | 2250 nm | 3.507e−03 | 0.241 |
| 15 | 2850 nm | 1.841e−03 | **0.127** |

Over the ~9.5 frames of delay that cost the chain two hops, the bus loses a
factor of ~7 where the chain lost ~118 (0.092²). **The bus is ~16× better
transport over the same delay**, which is the whole reason to route around the
disks rather than through them. Tap positions follow directly: lags 5–15 sit
between 0.9 and 2.9 µm, so the entire delay line is a ~3 µm strip.

*Measurement note.* The first run of this used 4000 steps and averaged over the
last third — a window beginning at 2.67 ns while the 2.7 µm tap was still
arriving at 2.6 ns. Far taps were measured mid-transient and the fitted
attenuation length came out 951 → 544 nm, nearly 2× too pessimistic, with
arrival times that fell with distance. Converged at 12000 steps the profile is
clean and exponential beyond the near field.

### What this does and does not settle

Settled: the transport half. A bus carries a usable delayed copy across the full
range of separations the task needs, which the serial chain demonstrably cannot.

Not settled, and it is the larger half: whether N tap disks each mixing fresh
against delayed actually yield the ~40 independent long-separation product
dimensions the capacity accounting demands. Nothing measured so far has produced
more than ~5.

The specific constraint the numbers now impose is **dynamic range**. The delayed
amplitude spans 8× across lags 5–15, while the disk's usefully-nonlinear-and-
still-stable drive window is roughly 15–35 mT — a factor of ~2.3. Those do not
fit without per-tap compensation. Compensation is available in principle, since
tap coupling strength is a geometric design parameter (stub width and length)
and each tap's fresh feed is its own line, so near taps can couple weakly and far
taps strongly. But that is an assumption about a design not yet built, and the
honest position is that the bus removes the objection that killed the chain
without yet establishing that the replacement works.

## The coupling-gap sweep: the trade has no operating point at this tap geometry

The zero-gap build worked in one respect and failed in another. Tap 1 arrived at
lag **4.90** against a designed 5.0 — the bus delay calibration transfers
exactly, so tap placement is a solved problem. But delivered amplitude fell 314×
across the four taps, and probing the bus showed why: each tap drains the line,
so the loaded bus falls 566× end to end where the bare strip falls 17×.

`coupling_gap` was added to test the obvious fix — couple each tap weakly so it
passes the wave on. Two points, 0 and 30 nm, on the identical geometry:

| | gap 0 (galvanic) | gap 30 nm |
|---|---|---|
| tap 1 received | 1.495e−03 | 7.309e−04 |
| tap 2 | 1.371e−04 | 3.875e−05 |
| tap 3 | 1.821e−05 | 7.972e−06 |
| tap 4 | 4.767e−06 | 4.049e−06 |
| spread tap1/tap4 | 314× | 181× |
| balance vs fresh drive | 0.104 → 0.002 | 0.056 → 0.002 |
| **tap 1 measured lag** | **4.90 (designed 5.0)** | **1.50 — wrong** |

**The gap does exactly what it was designed to do, and it is not enough.** Bus
through-loss at each tap, against what the bare strip would lose over the same
span:

| | gap 0 | gap 30 | bare bus |
|---|---|---|---|
| tap 1 | 0.431 | **0.912** | 0.729 |
| tap 2 | 0.288 | 0.473 | 0.810 |
| tap 3 | 0.551 | 0.549 | 0.810 |
| tap 4 | 0.606 | 0.421 | 0.729 |

At tap 1 the draining is *fixed*: 0.431 → 0.912, essentially no excess loss over
the bare guide. The mechanism is confirmed. But tap 1's received signal halves in
the process, and its measured arrival collapses from lag 4.90 to lag 1.50 —
meaning the tap is now reading stray field rather than the guided wave that
arrives four frames later.

### Why no gap fixes this

**Guided coupling and stray coupling are both near-field, and they scale together
with distance.** Widening the gap attenuates the intended evanescent coupling and
the unintended dipolar pickup at comparable rates, so the contrast between them
barely moves while both fall. There is no gap that favours one over the other,
which is why the sweep improves the spread (314× → 181×) and destroys the delay
signature at the same time. A third intermediate point would interpolate between
two failures rather than find a window between them.

The taps beyond the first were never receiving a guided signal in either
configuration — tap 4 changes by only 1.18× across a gap that halves tap 1,
which is the signature of a stray-dominated reading that the coupling geometry
does not control at all.

### What the measurements point at instead

A perpendicular stub is the wrong coupler. The standard way to discriminate
guided from stray is a **directional coupler**: run the tap's feed guide
*parallel* to the bus over a coupling length, phase-matched, instead of butting
it in perpendicular. Guided coupling then accumulates coherently along that
length while stray pickup does not, so coupling strength becomes a function of
length — a parameter that trades against nothing — rather than of proximity,
which trades against contrast.

That is a genuinely different tap geometry, not a parameter change, and it is the
honest next step rather than a third gap. What survives from this build and
should be carried into it: the bus itself (951 nm attenuation length, 945 m/s,
16× better transport than the chain), the tap-position calibration (189 nm per
frame, confirmed to 2% at tap 1), the checkpointed relax, and the geometry
checker, which caught two build-breaking bugs before any physics ran.

What this build does not support is the claim it was made to test. Four taps
were meant to supply long-separation products at four different delays; three of
them never received a delayed signal at all.

## Correction: tap position never set tap delay, in any poly-tap build

Two earlier sections of this document report that the zero-gap build validated
the delay calibration — "tap 1 arrives at lag 4.90 against a designed 5.0, so
the bus delay calibration transfers exactly" and "tap placement is a solved
problem." **Both statements are wrong.** They rest on an arrival estimator that
cannot support them, and the correct measurement says the opposite.

The estimator was the first crossing of 10% of a tap's own envelope peak. It
fires on whatever arrives first, so a small instantaneous precursor reads as
"no delay" even when most of the energy arrives late — and conversely, a tap
whose precursor happens to sit just under 10% reads as a clean delayed arrival.
The 4.90 was the second case: a coincidence, not a measurement.

Cross-correlating each tap's envelope against the injected burst removes the
ambiguity. Envelope rather than carrier, because the 12 GHz period is 0.42
frames and a carrier-level correlation is ambiguous modulo the period. Ring-up
still biases the absolute number late, but that bias is common to all taps, so
the trustworthy quantity is the **difference** between taps — designed 3 frames
apart throughout.

| tap | designed | galvanic stub | 30 nm gap | directional coupler |
|---|---|---|---|---|
| 1 | 5.0 | 12.87 | 12.96 | 12.31 |
| 2 | 8.0 | 12.13 | 13.00 | 13.00 |
| 3 | 11.0 | 13.00 | 1.17 | 12.87 |
| 4 | 14.0 | 12.43 | 3.83 | 3.33 |
| **spacing errors** | **+3.0 each** | −3.73, −2.13, −3.57 | −2.96, −14.84, −0.33 | −2.31, −3.13, −12.54 |
| within tolerance | | **0 / 3** | **1 / 3** | **0 / 3** |

One of nine spacing deltas across three geometries lands within tolerance, and
that one (the 30 nm gap's 3→4 pair) sits between two taps whose absolute lags
are 1.17 and 3.83 — both outliers on weak, broad correlations — so it is best
read as chance rather than as a tap pair that worked.

*Caveat on the estimator, since this section exists because of estimator
trouble.* Cross-correlating a 200-step burst against a response that rings for
~1600 steps gives a broad correlation whose argmax is noisy, and it is biased
toward the tail's centroid rather than the arrival. That is why some taps
return outliers. The robust part of the argument is not any single absolute
number but the DIFFERENCES: those are designed to cancel whatever common bias
the estimator carries, and they miss 3.0 frames by −0.33 to −14.84 in every
geometry.

Every tap in both geometries responds at the same ~12.5 frames regardless of
where it sits on the bus. Tap position is not setting tap delay and never was.
All three poly-tap builds were assessed with an instrument incapable of
detecting the thing under test.

This does **not** show the architecture fails. It shows the gate was
uninformative. The chain's stage delays were real and were resolved — by memory
functions on a 600-frame task run, not by an impulse.

## The limit that explains all three failures at once

The ~12.5 frames is the disks' own ring-up and ring-down, not bus transit, and
that is what makes the delay differences invisible: a detector that rings for
12 frames cannot timestamp a 3-frame spacing.

Which turns into a hard constraint once the two measured bus constants are put
side by side, because they are not independent — and neither is separable from
the disk's ring-down, since the guide's decay time and the resonator's are both
~1/(alpha*omega) in one material at one frequency:

    attenuation length  L = 951 nm
    group velocity      v = 945 m/s
    so the guide's own decay time is L / v = 1.01 ns

**A wave decays over exactly the distance it travels in one decay time.** That
is not a coincidence of this material — it is what an attenuation length *is*.
So the distance a tap must be separated by to be resolvable (at least one
detector response time of travel) is necessarily comparable to the distance
over which the signal decays.

With the measured 12.5-frame tap response, resolvable spacing is 2363 nm = 2.5
attenuation lengths, so **each additional resolvable tap costs e^2.5 ≈ 12× in
amplitude**:

| resolvable taps | dynamic range across the array |
|---|---|
| 4 | 1.7 × 10³ |
| 6 | 2.5 × 10⁵ |
| 10 | 5.1 × 10⁹ |

Against a usefully-nonlinear-and-still-stable drive window of ~2.3× per disk.

That is the whole story of the three coupling failures, and it was never about
the coupler. Galvanic stub, 30 nm gap, and 400 nm directional coupler all land
in the same place — spreads of 314×, 181× and 157× — because the spread is set
by the ratio of two material constants, not by how the taps attach. No coupler
geometry moves it, which is why the gap sweep and the coupler sweep produced
the same numbers by different routes.

**A delay line tapped at many points is not viable in a medium this lossy.** The
requirement was ~40 independent long-separation product dimensions; the array
cannot deliver even ten resolvable taps without a 10⁹ dynamic range.

### The fix is differential damping, not lower damping

The obvious response to that table — use a lower-damping film — **does not
work, and the reason is worth stating because it is exactly counter to the
instinct this project has been following.** Both quantities in the ratio are
1/(alpha*omega): the guide's decay time and the disk's ring-down come from the
same damping at the same frequency. Lowering alpha stretches both equally and
the cost per tap does not move at all:

| alpha | tau (ns) | attenuation length | resolvable spacing | cost per tap |
|---|---|---|---|---|
| 8e-3 (permalloy, here) | 1.66 | 1.57 um | 1.57 um | **2.72** |
| 1e-3 | 13.3 | 12.5 um | 12.5 um | **2.72** |
| 1e-4 | 133 | 125 um | 125 um | **2.72** |
| 1e-5 | 1326 | 1253 um | 1253 um | **2.72** |

A hundredfold better film buys a hundredfold bigger device and not one extra
tap. That is the correction to the recommendation made above, which named a
lower-damping film as the way out; it is not.

What breaks the tie is making the two times DIFFERENT — a low-loss bus with
deliberately lossy taps. The tap's job is to multiply a fresh sample against a
delayed one, not to remember; the memory lives in the bus. So a tap that rings
briefly is not a compromised tap, it is the correct tap, and a short ring-down
is what lets taps sit close enough to be resolved:

| alpha_tap / alpha_bus | resolvable spacing | cost per tap | taps within a 10x spread |
|---|---|---|---|
| 1 | 125 um | 2.72 | 3 |
| 5 | 25 um | 1.22 | 12 |
| **10** | **12.5 um** | **1.11** | **24** |
| 30 | 4.2 um | 1.03 | 70 |

At alpha_tap = 10 x alpha_bus, twenty-four resolvable taps fit inside a 10x
amplitude spread — against the four taps and 314x spread measured here. That is
the first configuration in this project whose arithmetic clears the ~40
independent long-separation product dimensions the task needs.

Locally raising damping is standard practice rather than an exotic ask: a heavy
metal cap (Pt, Pd) on the tap disks raises alpha by spin pumping, typically
several-fold, while the uncapped bus keeps its own. The bus wants a low-damping
film; the taps want that film plus a cap.

Two caveats before this is treated as a plan. Everything above is arithmetic on
measured constants, not a simulation of such a device -- and this project's
record is that architectures which look sound on paper fail on a mechanism the
paper omitted, three times so far. And the spacings are microns: 24 taps at 12.5
um is a 300 um bus, which is enormous next to the 4 um structures simulated here
and would take a mesh this approach cannot afford. Verification would need a
coarser cell size or a different solver, not more of the same runs.

## Differential damping, measured: the mechanism is real and does not reach the design

The design equation says tap spacing scales with alpha_tap/alpha_bus, which means
the principle is testable at whatever spacing already exists rather than needing
the 300 um device. At the 567 nm spacing already built, a tap must respond within
0.60 ns to be resolved -- alpha_tap > 0.022, only 2.8x permalloy's. So raise the
damping on the tap disk bodies alone and re-measure the delays.

| tap alpha | tap 1 | tap 2 | tap 3 | tap 4 | measured spacings (designed +3.0) |
|---|---|---|---|---|---|
| 1x | 12.87 | 12.13 | 13.00 | 12.43 | −0.73, +0.87, −0.57 |
| **10x** | **10.84** | **11.95** | **13.00** | 1.09 | **+1.11, +1.05**, −11.91 |
| 30x | 10.14 | 10.91 | 1.09 | 1.08 | +0.77, −9.81, −0.01 |

**At 10x the first three taps come out monotonically ordered in delay — the first
time any build in this project has done so.** Every earlier geometry returned the
same ~12.5 frames at every tap regardless of position. The mechanism is real and
it points the way the theory says it should.

It also does not reach the design. The measured spacing is ~1.1 frames against a
designed 3.0, and pushing to 30x makes it *worse*, not better: tap 3 falls out at
30x and the ordering survives only across two taps. That is not the theory
failing but a second constraint biting — the tap response gets shorter and the
tap signal gets weaker together, and by 30x taps 3 and 4 are at 6.1e-6 and 2.7e-6,
too weak for the estimator to place at all.

### The two knobs are not independent

This is the part worth carrying forward. Damping sets whether a tap can *resolve*
a delay; coupling sets whether it *receives* enough to be measured. Testing them
one at a time — which is what the gap sweep, the coupler, and this damping sweep
each did — cannot find the window, because the useful region is where a tap rings
briefly AND still receives usable amplitude, and the 314x spread at galvanic
coupling means the far taps were never measurable at any damping.

The next experiment is therefore two-dimensional, over coupling geometry against
tap damping, which is 20-30 runs. That is a poor fit for this machine and a
natural one for a GPU, now that the array runners take the graph-capturable path
rather than breaking the CUDA graph on a Python callable every substep.

What should not be run yet is a 600-frame task benchmark on this array. An array
whose taps cannot be told apart will produce a clean, meaningless number, and
this project has already paid for that lesson three times.

## The 2D sweep, on GPU: the delay mechanism works, the bus cannot feed it

Twenty points on a T4 -- five geometries (galvanic stub, 15 nm gap, 30 nm gap,
300 nm and 450 nm directional couplers) against four tap-damping multipliers
(1, 3, 10, 30). Bar fixed in advance: all three consecutive tap pairs monotonic,
spacing near the designed 3.0 frames, and the weakest tap above ~1e-5 where the
estimator can place it.

**No point clears it. Two of the three criteria fail everywhere.**

| | result across all 20 points |
|---|---|
| all three pairs monotonic | **never** (best 2 of 3) |
| weakest tap above 1e-5 | **never** (range 2.3e-06 to 4.9e-06) |

### What did work, and it is the thing the design rested on

The FIRST tap pair resolves correctly, and only when both knobs are turned:

| point | first-pair spacing (designed 3.0) | tap damping |
|---|---|---|
| gap 15 | 0.11 | 1x |
| gap 15 | 2.62 | 10x |
| **gap 15** | **3.07** | **30x** |
| gap 30 | 0.04 | 1x |
| gap 30 | 2.64 | 10x |
| gap 30 | 2.65 | 30x |

**3.07 against a designed 3.0 is the delay mechanism working, to 2%.** Undamped,
the same geometries give 0.11 and 0.04 -- no resolution at all. So differential
damping does exactly what the design equation said: shorten the tap's ring-down
below the delay being resolved, and tap position starts setting tap delay. That
claim was made from arithmetic on two measured constants and is now measured
directly.

It needs a gap as well as damping. The galvanic stub never exceeds 1.11 at any
damping, because a stub that drains the line leaves nothing to time.

### Why four taps still fail

Every configuration collapses at the SECOND pair, which sits at ~-11.9 in every
resolving point: tap 3 cannot be placed at all. The reason is in the last
column and it does not move -- the weakest tap is 2.3e-06 to 2.8e-06 in all
twenty configurations, against the ~1e-5 the estimator needs.

Damping does compress the spread, and on the directional couplers it does so
hard: 300x -> 49x at 300 nm, 163x -> 47x at 450 nm. But it compresses from the
TOP. The weakest tap gets slightly *worse* as damping rises (2.6e-06 -> 2.3e-06),
so the array levels down rather than up. A flat array of unreadable taps is not
progress.

### Verdict

**Two taps are feasible on this material; four are not.** The delay mechanism is
confirmed and the reception limit is exactly the one derived from the two bus
constants -- each resolvable tap costs e^(d/L) in amplitude, and past tap 2 that
puts the signal under the noise.

Two taps supply two product separations. The task needs roughly forty
independent long-separation product dimensions. **The poly-tap architecture is
finished for NARMA-10**, not because the idea was wrong but because the material
cannot carry a delay line far enough to tap it more than twice.

What would change it is the one thing this project cannot fix by geometry: a
larger ratio of attenuation length to tap response time. That is a material and
frequency question, and it is the same question the handoff brief already asks.

## Testing the materials answer in simulation: half the prediction, and the wrong half is the informative one

The sweep's verdict handed the reception failure to materials -- a longer
attenuation length relative to tap response time. That is simulable rather than
only askable, by lowering the BUS damping while leaving the taps where they are.
The prediction was recorded before the run: a tenth the bus damping lifts the
weakest tap about 12x, from 2.7e-06 to ~3e-05, and collapses the tap-to-tap
spread from ~400x to roughly 1.2x.

| bus damping | weakest tap, predicted | measured | spread |
|---|---|---|---|
| x1 | 2.7e-06 | 2.8e-06 | 405x |
| x0.3 | 1.9e-05 | 7.7e-06 | 458x |
| **x0.1** | 3.3e-05 | **1.19e-05** | 408x |
| **x0.03** | 4.0e-05 | **1.40e-05** | 388x |

**Reception is fixable.** At a tenth and a thirtieth the bus damping the weakest
tap clears the 1e-5 bar for the first time in the project, at 1.19e-05 and
1.40e-05. It arrives at about a third of the predicted size, but it arrives.

**The spread does not move at all** -- 405x, 458x, 408x, 388x across a
thirtyfold change in bus damping. That refutes the model the prediction came
from, and the correction matters more than the confirmation.

### What that invalidates

The limit derived earlier -- that each resolvable tap costs e^(d/L) in
amplitude, so N taps need e^(N-1) of dynamic range -- assumes the tap-to-tap
ratio is set by propagation loss along the bus. If it were, a thirtyfold longer
attenuation length would have flattened the array. It did not, so **propagation
loss is not what sets the spread in this array**, and that derivation does not
describe it. The likely candidate is that taps 2-4 read near-field from the
injection rather than the guided wave -- which falls with distance geometrically
and is indifferent to damping -- consistent with those taps never having shown a
delayed arrival in any geometry.

The derivation may still be right about resolvable spacing in general. It is
wrong as the explanation for THIS array's spread, and it was committed as the
explanation.

### The verdict is unchanged and now better supported

Resolution and reception remain anti-correlated, and the low-loss bus does not
break the tie:

| configuration | first-pair spacing (want 3.0) | weakest tap |
|---|---|---|
| bus x0.1, no tap damping | 0.09 (no resolution) | **1.19e-05** (clears) |
| bus x0.1, tap damping 10x | 2.13 (resolves) | 5.88e-06 (fails) |
| bus x0.03, no tap damping | 0.09 (no resolution) | **1.40e-05** (clears) |
| bus x0.03, tap damping 10x | 2.12 (resolves) | 6.53e-06 (fails) |

Turning on the damping that buys resolution halves the weakest tap and puts it
back under the bar. No point in twenty-eight now reaches 3/3 monotonic. The
architecture is still finished for this task -- but the reason is not the one
committed a few hours ago, and a collaborator sent the earlier explanation would
have been sent the wrong question.

## The near-field test, predictions registered before the run

The replacement hypothesis -- that taps 2-4 read injection near-field rather
than the guided wave -- is testable without touching the code: move every tap
ten frames further from the injection, keep the 3-frame tap spacing identical,
and run at bus damping x0.03 where the guide is nearly lossless over the extra
1890 nm. Whatever is feeding the taps has to survive that shift or not.

Tap distances are lag x 189 nm, so lags 5/8/11/14 put the array at
945-2646 nm and lags 15/18/21/24 put it at 2835-4536 nm. At bus x0.03 the
attenuation length is 951/0.03 = 31.7 um.

| feeding mechanism | tap 1 amplitude, shifted / baseline | array spread |
|---|---|---|
| guided wave | 0.94 | unchanged (~1.05) |
| dipolar near field, 1/r^3 | 0.037 | 22x -> 4.1x |
| evanescent pickup, kappa = 285 nm | 0.0013 | unchanged |

The third row is the exponential fit that would ACCOUNT for the measured 388x
spread: 1701 nm across the array needs kappa = 1701/ln(388) = 285 nm. It is
listed because an exponential pickup is shift-invariant in spread, which is the
one model that explains both the 388x and its indifference to bus damping. Note
that neither near-field model is a comfortable fit -- dipolar 1/r^3 predicts
only 22x where 388x was measured -- so the spread is not yet explained by the
replacement hypothesis either. What the shift discriminates cleanly is tap 1's
ABSOLUTE amplitude, where the three models are separated by a factor of 25 and
by 700.

Baseline tap 1 measured 5.43e-03 (min amp 1.40e-05 at spread 388x). Guided
predicts 5.1e-03 after the shift, dipolar 2.0e-04, evanescent 7e-06.
