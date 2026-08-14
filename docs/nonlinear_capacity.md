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

### Result: the bus is not delivering a propagating wave

| model | predicted tap 1 after the shift | miss |
|---|---|---|
| guided wave | 5.11e-03 | **562x** |
| dipolar near field, 1/r^3 | 2.01e-04 | 22x |
| evanescent pickup, kappa = 285 nm | 7.20e-06 | **0.79x** |
| MEASURED | **9.09e-06** | -- |

The pre-registered evanescent number is a hit, and it is not a fitted one: its
kappa came from the SPREAD ACROSS TAPS inside the baseline array, and it then
predicted an absolute amplitude 1890 nm away, in a separately built and
separately relaxed mesh, to within 26%. The guided prediction is wrong by nearly
three orders of magnitude.

Two more things fall out of the same two runs.

**The two arrays lie on one curve.** Baseline tap 4 at 2646 nm reads 1.398e-05
and shifted tap 1 at 2835 nm reads 9.087e-06, a 438 nm decay length across the
189 nm that separates them -- continuous with the 354 and 386 nm measured
between baseline taps. The meshes differ in size (127,836 against 187,938
cells) and were relaxed independently, so that continuity is a real check, not
bookkeeping.

| span (nm) | 945-1512 | 1512-2079 | 2079-2646 | 2646-2835 | 2835-3402 | 3402-3969 | 3969-4536 |
|---|---|---|---|---|---|---|---|
| kappa (nm) | 196 | 354 | 386 | 438 | 638 | 539 | 1283 |

A single exponential over all eight points gives kappa = 426 nm with a 3.7x
worst residual, so the decay is not one clean exponential -- it softens with
distance, as a short-range field plus a weak long-range remainder would. What is
not in doubt is the scale: a few hundred nm, against the 31.7 um the bus damping
implies for a propagating wave.

**No tap shows a propagation delay.** Across the baseline array the arrival lags
are 12.91, 13.00, 13.00, 12.95 -- flat to 0.04 frames over 1701 nm, where the
measured 945 m/s group velocity demands tap 4 lag tap 1 by **9.0 frames**. This
is the argument that closes the case, because it separates the two remaining
explanations. If the taps were loading a propagating bus, the wave would arrive
attenuated but still late. It arrives on time everywhere, so it did not travel.

(The shifted array's lags -- 13.00, 12.96, 8.73, 4.59, running backwards -- are
the estimator failing below the placement bar, not physics: every tap there is
at or under 9.1e-06.)

### What this costs, and what it is worth

The poly-tap architecture is dead for a reason unrelated to anything the last
several sweeps were adjusting. It needs a delay line; the bus is not one at 12
GHz in this geometry. Coupling gap, directional couplers, tap damping ratio,
bus damping -- all four were tuning how much of a signal that never propagated
reached the taps.

It also puts the two earlier bus-transport constants in question. The 951 nm
"attenuation length" on the bare strip is the same kind of measurement on the
same kind of field, and the 945 m/s "group velocity" came from arrival-vs-
distance under the threshold estimator that has already produced one false
delay in this project. Note the direction of the inconsistency: the decay here
is ~400 nm at a THIRTIETH the damping, shorter than the 951 nm measured at full
damping. Propagation loss cannot do that.

The open question is therefore dispersion, not materials: whether 12 GHz sits
inside the propagating band of a 20 nm x 80 nm permalloy strip in this magnetic
configuration at all, and if not, what bias field, width, or drive frequency
puts a genuinely propagating mode on the bus. That is measurable here -- a
dispersion sweep on the bare strip, frequency against wavevector -- and it is
the prerequisite for any delay-line architecture, not a refinement of this one.

## The dispersion question, and what geometry predicts before it is measured

The near-field result left one prior question: whether the bus supports a
travelling wave at 12 GHz at all. Geometry gives a sharp prediction, and it is
worth writing down before the measurement rather than after.

The bus is a bare 80 x 20 nm permalloy strip with NO applied field. Shape
anisotropy puts the ground state along the strip (measured: mean m_x = 0.998),
so the drive wavevector runs ALONG the magnetisation. That is the
backward-volume geometry, whose magnetostatic band lies BELOW the k = 0
resonance rather than above it. Note this is not the geometry
`magnonic_nn/dispersion.py` implements -- that module is Damon-Eshbach, k
perpendicular to M, and does not apply to the bus.

For an infinite bar of rectangular cross-section w x t magnetised along its
length, the transverse demagnetising factors are about Ny = t/(w+t) and
Nz = w/(w+t), and Kittel with no applied field gives

    f0 = (gamma/2pi) * sqrt((H + Ny*Ms)(H + Nz*Ms))

| bias along x | k=0 resonance |   | strip width | k=0 resonance |
|---|---|---|---|---|
| 0 mT | **11.27 GHz** | | 80 nm | **11.27 GHz** |
| 20 mT | 11.96 GHz | | 160 nm | 8.85 GHz |
| 50 mT | 12.98 GHz | | 240 nm | 7.51 GHz |
| 100 mT | 14.62 GHz | | 400 nm | 6.00 GHz |

**The drive sits 0.7 GHz above the band top.** If this is right, 12 GHz has no
real wavevector to couple to on this strip and can only produce a localised,
non-propagating response -- which is exactly what the tap array measured, at a
frequency chosen years of runs ago for reasons that had nothing to do with the
bus.

Three consequences follow, and each is a prediction the measurement can kill:

1. There should be a propagating band BELOW ~11.3 GHz, with a real ridge in the
   (k, f) map and decay lengths of microns rather than hundreds of nm.
2. Bias along the strip RAISES the band top, so ~20 mT should bring 12 GHz
   inside it and ~50 mT should put it comfortably in.
3. A WIDER strip moves the band the wrong way -- 240 nm drops the top to 7.5
   GHz. Widening the bus, which was on the list of things to try, would have
   made this worse.

### Measured: the band is ABOVE the resonance, and 12 GHz is inside it

The prediction got the edge FREQUENCY right and the SIDE wrong. There is a
propagating band, its edge sits where Kittel says the k = 0 resonance sits, and
the band lies above that edge rather than below it -- a forward branch, not the
backward-volume one the geometry argument assumed.

| configuration | predicted k=0 resonance | measured band bottom | decay at 12 GHz |
|---|---|---|---|
| 80 nm, no bias (as built) | 11.27 GHz | ~10.5 GHz | **3252 nm** |
| 80 nm, 50 mT along x | 12.98 GHz | ~13 GHz | 371 nm |
| 240 nm, no bias | 7.51 GHz | ~5.5 GHz | (fits degenerate) |

The band edge is unmistakable in the map contrast rather than in any fitted
number: on the as-built strip it runs 24, 36, 75 at 7, 8, 9 GHz and then 2113 at
10 GHz. Below the edge the response is smeared across all k, which is what a
field localised at the source looks like in a spatial transform; above it there
is a ridge, and the ridge's k rises smoothly from 4.6e7 at 10 GHz to 2.1e8 at 40
GHz.

**At 12 GHz the as-built strip carries a wave 3.25 um, at k = 7.6e7 rad/m
(lambda 82.6 nm) and v_g 585 m/s.** That is enough to feed a 1.7 um tap array,
and it flatly contradicts what the array measured.

Two things follow immediately. Bias is the wrong direction: at 50 mT the band
bottom moves up past the drive and the decay at 12 GHz collapses to 371 nm,
which is the evanescent scale again. And widening the bus, which was on the list
of things to try, moves the edge DOWN and away -- so it would have helped, but
for the opposite reason to the one that would have been given.

### A retraction: the 1000-step run that "confirmed" the evanescent reading

Before the 8192-step run there was a 1000-step CPU run reporting a 275 nm decay
at 12 GHz. It was quoted as an independent confirmation of the near-field
result. It is an artifact, and it should have been checked before it was quoted.

With `--steps 1000` and the default `--t0-ps 1000`, the sinc source's main lobe
lands on the LAST SAMPLE of the record, where the Hann window is essentially
zero. The transform saw the pre-pulse and nothing else. Separately, 1 ns is too
short for a 585 m/s wave to cross the 1.9 um analysis window at all, so the far
half of the fit was measuring field that had not yet arrived -- the same
mid-transient error already on record in `check_bus_transport.py`, which fitted
an attenuation length 2x too short for exactly that reason.

`check_bus_dispersion.py` now refuses to run when the record is shorter than
three times the source delay.

That the wrong number agreed so well with the array measurement (275 nm against
the array's ~285 nm) is worth stating plainly: a broken estimator reproduced the
expected answer, which is the circumstance in which a broken estimator is least
likely to be caught.

### What is actually left to explain

The bare strip propagates at 12 GHz. The tap array does not. The difference has
to be in the array, and there are two candidates:

  the launcher   a uniform 100 nm segment has its first spectral zero at
                 k = 2*pi/100nm = 6.28e7, and the mode wants k = 7.6e7 --
                 just past the null, at sinc = 0.162 in amplitude, 2.6% in
                 power. Suppressed, but not obviously by the ~400x observed.
  the loading    each tap disk is 200 nm across and the mode's wavelength is
                 82.6 nm, so every tap is a 2.4-wavelength obstacle sitting a
                 15 nm gap from the guide. The zero-gap build already measured
                 the bus falling 566x across the array where a bare strip fell
                 17x over the same span.

These separate cleanly: run the identical measurement on the loaded bus driven
by its own injector, and on the bare strip driven by a 100 nm source.

## The fault is the tap aperture, and the fix is geometric

Three suspects were measured out in turn. The band: 12 GHz is inside it, bare
strip decay 3252 nm. The launcher: fine -- the real 100 nm injector gives 3095
nm against 3252 nm for a 15 nm probe line. The loading: costs a factor of 2.4
and no more, and on the LOADED bus driven by its own injector the 12 GHz phase
slope has R^2 = 1.000, k within 1% of the map, decay 1354 nm.

So the delay line works. Recording the bus field and the disk readouts in one
rollout says where the signal is lost:

| tap | x | bus field | rel tap 1 |
|---|---|---|---|
| 1 | 1545 nm | 1.81e-02 | 1.000 |
| 2 | 2112 nm | 1.39e-02 | 0.768 |
| 3 | 2679 nm | 8.31e-03 | 0.459 |
| 4 | 3246 nm | 3.56e-03 | 0.196 |

**The bus delivers the wave to every tap within 5.1x while the disks report
388x.** A fivefold variation in what arrives cannot produce a 388-fold variation
in what is read, so the loss is in the bus-to-disk coupling.

There is a specific reason it should be. The coupling guide is 80 nm wide
measured along the bus, which makes it a receiving aperture, and a uniform
aperture of width W samples a wave with weight |sinc(kW/2)|, nulling when W
equals one wavelength. At 12 GHz the measured k is 7.63e7, so lambda = 82.3 nm
against a guide of 80 nm: **W/lambda = 0.97, three percent from the null.**
Every sweep in this project held the drive at 12 GHz, so every one of them sat
on that null and adjusted something else.

### Predictions, registered before the sweep

Coupled amplitude goes as W*sinc(kW/2), not sinc alone -- a narrower aperture
cancels less but intercepts less too -- so the optimum is near W = lambda/2
rather than at W -> 0.

| W (nm) | W/lambda | sinc | W*sinc | vs 80 nm |
|---|---|---|---|---|
| 80 (as built) | 0.97 | 0.029 | 2.35 | 1.0x |
| 60 | 0.73 | 0.329 | 19.7 | 8.4x |
| 50 | 0.61 | 0.495 | 24.7 | 10.5x |
| **40** | 0.49 | 0.655 | **26.2** | **11.2x** |
| 25 | 0.30 | 0.855 | 21.4 | 9.1x |

And one prediction that will look like a failure if it is read against the old
design. Tap positions were laid out at 189 nm per frame, from a group velocity
fitted across 12-20 GHz. That broadband figure is vindicated -- a linear fit to
the loaded bus's phase slope over 12-20 GHz gives 948 m/s against the 945 m/s in
the config -- but the envelope travels at the LOCAL group velocity, and at 12
GHz that is 706 m/s, or 141 nm per frame. The taps are 567 nm apart, so a tap
array actually fed by the wave should read

    567 nm / 141 nm per frame = 4.02 frames of spacing, not the 3.00 designed.

A measured spacing near 4.0 is the wave arriving. A measured spacing near 3.0
would mean something else is setting the timing. `score_point` scores against
the designed 3.0 and will report this as a +1.0 error; that error is the result,
not a fault.

### Aperture sweep: the shape confirms, the magnitude does not

| W (nm) | W/lambda | W*sinc predicted | spread | min amp | spacings |
|---|---|---|---|---|---|
| 80 (as built) | 0.97 | 2.3 | 405x | 2.83e-06 | 0.11, 0.00, -8.87 |
| 50 | 0.61 | 24.7 | 197x | 3.09e-06 | 3.31, -11.77, 2.46 |
| **40** | 0.49 | **26.2** | **141x** | **3.25e-06** | 3.35, -11.79, 2.44 |
| 25 | 0.30 | 21.4 | 223x | 3.02e-06 | 2.18, -11.85, 2.51 |

**The 11x amplitude prediction is refuted.** The weakest tap rose 15%, not
1100%. A separate out-of-sample number from a two-point decomposition (bw25
should land near 2.4e-06) also missed, at 3.02e-06.

**The predicted SHAPE holds.** W*sinc(kW/2) peaks at W = lambda/2 rather than at
W -> 0, and 40 nm is the best point on both spread and weakest tap, with 25 nm
worse than 50 nm on both. A coupling-area model -- coupling simply proportional
to W -- predicts monotonic improvement as W shrinks and is excluded by the 25 nm
point. So the aperture is real and correctly located; it is not the dominant
term.

What actually moved was the spread, 405x -> 141x, and it moved because tap 1
FELL 2.5x rather than because the far taps rose. Tap 1 is dominated by injection
near field, which scales with aperture area, so narrowing the guide cuts the
near field roughly in proportion while the wave contribution goes as
W*sinc(kW/2). The two terms pull opposite ways at tap 4, which is why it barely
moved.

Spacings went from garbage (0.11, 0.00, -8.87) to two of three positive and
near 2.4-3.4. Still not a delay line: the middle pair is stuck at -11.8 in every
narrowed configuration, which is tap 3 sitting at the noise floor and the
envelope estimator picking a spurious peak.

### The wave is damping-limited; the near field never was

    v_g / (alpha * omega) = 706 / (0.008 * 2pi * 12e9) = 1170 nm
    measured decay on the loaded bus                   = 1354 nm

Within 16%. This matters more than it looks, because it resolves what seemed
like a contradiction with the earlier bus-damping sweep, where a thirtyfold
change in alpha left the tap spread at 405, 458, 408, 388x. At W = 80 nm the
taps were reading NEAR FIELD, which is geometric and indifferent to damping --
so of course the sweep found nothing. With the aperture off its null the taps
can see the wave, and the wave does respond to damping. At bus alpha x0.1 the
decay would reach 11.7 um.

### The configuration all of this points at

Three measured facts now compose. The near field decays over ~400 nm and
dominates tap 1. The wave decays over 1354 nm at full damping, 11.7 um at a
tenth. The aperture wants W near lambda/2.

So: move the array out until the near field has died, narrow the aperture, and
lower the bus damping so the wave survives the longer path.

  lags 15/18/21/24   puts tap 1 at 2835 nm, where the injection near field is
                     down exp(-1890/400) = 0.9% of its value at tap 1 today
  bus_guide_width 40 the measured optimum
  bus alpha x0.1     decay 11.7 um, so the 1701 nm array span costs
                     exp(1701/11700) = 1.16x rather than 3.5x

Predictions, registered before the run:

  spread     should collapse from 141x to of order 2x -- near field gone, wave
             nearly lossless across the span
  spacings   all three positive and near 4.0 frames, not the 3.0 designed
  min amp    order 1e-5, uncertain to a factor of about 2

The falsifier is sharp: if the spread stays above ~20x or the spacings stay
broken, then the injection near field is not what has been keeping taps 2-4
dark, and this whole line of reasoning is wrong.

## The rollout ended before the wave arrived

Every poly-tap delay run in this project used `--burst 200 --quiet 2600`, a
record of 2808 steps = **2.81 ns**. At the local group velocity measured from
the loaded bus's phase slope, 706 m/s:

| tap | distance from injection | transit | within a 2.81 ns record? |
|---|---|---|---|
| lag 5 | 945 nm | 1.34 ns | yes |
| lag 8 | 1512 nm | 2.14 ns | yes |
| lag 11 | 2079 nm | 2.94 ns | **no** |
| lag 14 | 2646 nm | 3.75 ns | **no** |
| lag 15 | 2835 nm | 4.02 ns | **no** |
| lag 24 | 4536 nm | 6.42 ns | **no** |

In the shifted array the wave reached **no tap at all** before the simulation
stopped. What every one of those runs measured was the injection near field,
which is instantaneous. That is why all four taps reported the same arrival
time, and why the time was ~13 frames: it is the disk's own response, identical
at every tap because it has nothing to do with distance.

Nothing in the output looked wrong. Amplitudes were plausible, the estimator
returned finite lags, and those lags were reproducible run to run and geometry
to geometry -- which read as reliability rather than as the signature of a
quantity that did not depend on the thing being varied.

This is the third instance of one error class here:

  check_bus_transport.py   fitted its far taps mid-transient at 4000 steps and
                           got an attenuation length 2x too short
  check_bus_dispersion.py  a 1000-step record put the source's main lobe on the
                           last sample and reported 275 nm where 8192 steps give
                           3252 nm
  sweep_polytap.py         this one

`check_record_length()` now refuses any rollout that ends before the wave
reaches the farthest tap, and the default `--quiet` is 12000 steps.

### What this does and does not invalidate

Unaffected, because they are steady-state quantities rather than delay ones:
the whole dispersion campaign, the bus-versus-tap comparison (bus field varying
5.1x across the taps while the taps report 405x), and the aperture sweep's
amplitude and spread results.

Now unproven: the claim that the tap aperture is WHY the taps showed no delay.
The record length is a sufficient explanation on its own, and it was present in
every run the aperture argument was built on. The aperture null is still a
measured fact -- W/lambda = 0.97, and the non-monotonic optimum at W = lambda/2
was confirmed out of sample -- but its role in the delay failure is no longer
established.

## The as-built array works. The failure was the record length.

Same geometry, same 80 nm aperture, same full damping, same gap. Only the
rollout is long enough for the wave to arrive.

| | 2.81 ns record | 12.2 ns record |
|---|---|---|
| arrival lags (frames) | 12.91, 13.00, 13.00, 12.95 | **13.13, 16.59, 17.80, 21.55** |
| spacings | 0.11, 0.00, -8.87 | **3.46, 1.20, 3.76** |
| monotonic pairs | 1/3 | **3/3** |
| amplitude spread | 405x | 29x |
| weakest tap | 2.83e-06 | **3.93e-05** |
| amps | -- | 1.15e-03, 1.42e-04, 7.06e-05, 3.93e-05 |

Arrival advances monotonically with distance for the first time in the project,
and the weakest tap clears the 1e-5 placement bar by a factor of four against a
previous best of 2.83e-06.

Two conclusions of mine that this overturns.

**The aperture was not the cause.** W/lambda = 0.97 is a measured fact and the
non-monotonic optimum at W = lambda/2 was confirmed out of sample, so the
aperture effect is real. But the failure it was invoked to explain -- taps
reading no delay -- was the record length. This run sits the aperture squarely
on its null and resolves all three pairs anyway. The aperture is a real second-
order effect that was promoted to first-order because the first-order cause was
invisible.

**Four campaigns were aimed at a device that was not broken.** Coupling gap,
directional couplers, differential damping, bus damping: every one of them was
tuning a working delay line whose output was being truncated before it arrived.
The bus-damping sweep's headline finding -- spread immovable across a thirtyfold
change in alpha -- now has a mundane explanation. It was measuring near field,
which damping cannot touch, because the wave had not landed yet.

### The corrected group velocity is supported

Taps are laid out at 189 nm per frame, from the broadband 12-20 GHz fit. The
envelope travels at the LOCAL group velocity, 706 m/s or 141 nm per frame, so
taps 567 nm apart should read 567/141.2 = 4.02 frames rather than the 3.00
designed. Measured: 3.46, 1.20, 3.76. Two of the three sit between the two
figures and nearer the prediction; the middle pair is an outlier that the
amplitude ordering does not explain.

Re-laying the taps at 141 nm per frame would put the designed 3.0 back on the
measured delay, and is the obvious next geometry change.

### Damping trades amplitude against timing, and the trade is now visible

All three at a 12.2 ns record, gap 15, no tap damping.

| run | aperture | lags | bus alpha | mono | spacings | spread | weakest tap |
|---|---|---|---|---|---|---|---|
| A as built | 80 nm | 5-14 | x1 | **3/3** | 3.46, 1.20, 3.76 | 29x | 3.93e-05 |
| B | 40 nm | 5-14 | x0.1 | 2/3 | 14.15, -0.11, 19.35 | **2.8x** | **5.50e-04** |
| C composed | 40 nm | 15-24 | x0.1 | 2/3 | 12.35, 0.84, -5.55 | 3.1x | 2.84e-04 |

Lowering bus damping does exactly what the damping-limited-wave measurement said
it would: the array gets fourteen times brighter and ten times flatter. It also
destroys the timing, and damping cannot change group velocity, so those 12-19
frame spacings are not transit. C's absolute lags make it plainest -- 46.7,
59.0, 59.9, 54.3 frames, where the shifted tap 1 should land near 20 frames of
transit plus ~13 of disk response.

Two candidate mechanisms, both consistent with the size of the effect:

  ring-down     `bus_alpha_mult` damps the coupling GUIDES as well as the bus,
                so at x0.1 they ring about ten times longer and drag the
                envelope peak late. B's lags sit ~11 frames behind A's.
  termination   at x0.1 the wave crosses the 4 um bus at exp(-4000/11700) =
                0.71, so 400 nm absorbers sized for full damping stop being
                good terminations and reflections become comparable to the
                direct signal.

A damping ladder at x0.3 separates them: ring-down degrades smoothly with alpha,
a termination failure stays fine until the bus stops attenuating and then breaks.
The missing configuration -- the narrowed aperture AT FULL damping -- is in the
same ladder, and on present evidence is the most likely best point of any tried.

### Correction: the 189 nm/frame layout was right, and 141 nm/frame is wrong

Twice above this record says the taps are laid out 34% too far apart, because
the envelope should travel at the LOCAL group velocity at 12 GHz (706 m/s) and
not the 12-20 GHz broadband fit (948 m/s). The measured arrivals say otherwise.

Subtract the predicted transit from each measured lag; what remains is the
disk's own response, which must be the same at every tap:

| tap | distance | residual at 706 m/s | residual at 948 m/s |
|---|---|---|---|
| 1 | 945 nm | 6.44 | 8.15 |
| 2 | 1512 nm | 5.88 | 8.62 |
| 3 | 2079 nm | 3.08 | 6.83 |
| 4 | 2646 nm | 2.81 | 7.59 |

At 706 m/s the residual falls systematically, which means the assumed velocity
is too slow. At 948 m/s it is constant to +-0.9 frames around 7.8 -- a genuine
common response time. The mean measured spacing is 2.81 frames against the 3.00
designed, implying 1010 m/s: a 6% error, not 34%.

The reason is bandwidth. The burst is one frame long, so it spans roughly 5 GHz,
and a packet that wide travels at about the average group velocity across its
spectrum rather than the value at the carrier. A NARMA input changes every frame
and has the same bandwidth, so the broadband figure is the right one there too.

`frame_nm = 189e-9` stays as it is.

### The full ladder: as built is the best point, and the aperture theory is dead

All at lags 5-14, gap 15, no tap damping, 12.2 ns record.

| aperture | bus alpha | mono | spacings | mean | spread | weakest tap |
|---|---|---|---|---|---|---|
| **80 (as built)** | **x1** | **3/3** | 3.46, 1.20, 3.76 | **2.81** | 29x | 3.93e-05 |
| 80 | x0.3 | 2/3 | 8.98, 1.32, -3.53 | 2.26 | 13x | 2.89e-04 |
| 40 | x1 | 3/3 | 9.51, 2.32, 9.71 | 7.18 | 35x | 1.32e-05 |
| 40 | x0.3 | 3/3 | 6.07, 7.71, 4.99 | 6.26 | 5x | 2.28e-04 |
| 40 | x0.1 | 2/3 | 14.15, -0.11, 19.35 | 11.07 | 2.8x | 5.50e-04 |

**The sinc-aperture theory is refuted.** At matched damping the 80 nm aperture
beats 40 nm at every tap -- 3.93e-05 against 1.32e-05 at x1, 2.89e-04 against
2.28e-04 at x0.3 -- and the per-tap ratios at x1 are 2.5, 1.1, 1.6, 3.0 for a
width ratio of exactly 2. Coupling is proportional to aperture width, with no
null, which is the plain coupling-area model this record earlier claimed was
"excluded by the 25 nm point". It was excluded on a record that was measuring
near field. W/lambda = 0.97 remains arithmetic; the wave does not care.

The earlier out-of-sample "confirmation" -- 40 nm beating both 50 and 25 -- was
confirming how NEAR-FIELD pickup varies with aperture width, which is a real
curve and the wrong one.

**Bus damping buys amplitude and costs timing.** Lower alpha raises the weakest
tap sevenfold at bw80 (3.93e-05 to 2.89e-04) and flattens the array (29x to
13x), exactly as the damping-limited-wave measurement said it would. It also
scatters the spacings and drops monotonicity. Only the as-built point gives
both 3/3 and a mean spacing near the designed 3.0.

**Nothing tried beats the device as built** on the quantity the architecture
needs, which is resolvable delays. Every modification is a loss on timing, a
gain on brightness, or both.

Not explained, and worth saying so rather than papering over: why the narrowed
aperture inflates the mean spacing to 6-7 frames when its amplitudes are healthy
(bw40 ba0.3 has a weakest tap of 2.28e-04 and still reads 6.26 frames per tap
step against a designed 3.0). Dispersion is a candidate -- the packet is ~5 GHz
wide and v_g runs 706 m/s at 12 GHz to 1267 at 20, so arrivals smear by about 3
frames across the array -- but that should not depend on aperture width, and it
does. This is the next thing to measure, not the next thing to assert.

## NARMA-10 on the poly-tap bus: best memory in the project, zero nonlinearity

As built (80 nm aperture, gap 15, bus alpha x1, lags 5/8/11/14), 600 frames,
splits 100/300/100, co-driven with the fresh sample scaled per tap to the
measured delayed amplitude.

| readout | dim | NARMA-10 test NMSE |
|---|---|---|
| input_only | 1 | 1.3366 |
| linear_10lag | 10 | 0.9661 |
| linear_15lag | 15 | **0.1822** |
| linear_20lag | 20 | 0.1901 |
| **poly-tap, co-driven** | **200** | **0.3006** |

The device loses, 0.3006 against 0.1822. In relative terms that is where the
3-stage chain stood (0.2008 against 0.1243, a ratio of 1.62 against 1.65 here),
so resolved delays did not by themselves change the answer.

The decomposition says why, and it is stark.

| family | capacity |
|---|---|
| deg1 P1(s[n-k]) | **17.36** |
| deg2 P2(s[n-k]) | 0.00 |
| deg2 s[n]*s[n-k] | 0.00 |
| deg2 s[n-5]*s[n-k] | 0.00 |
| deg3 P3(s[n-k]) | 0.00 |
| **degree-1 share** | **100% of measured** |

Effective rank 17, and every last unit of it is linear. Recall holds above 0.93
out to lag 10 and 0.74-0.83 out to lag 19 -- **the best memory this project has
produced**, against the chain's 15.62 with a horizon of 17. At lag 9, the exact
pairing NARMA-10 needs, the linear term reads 0.939 and the product reads 0.000.

The delay line works. The architecture built on it does not, and it now has
LESS nonlinear capacity than the chain it was meant to replace (100% degree-1
against 98-99%).

### A cause I introduced

The fresh sample is scaled per tap to match the delayed copy: 1.0000, 0.1235,
0.0616, 0.0343. At 10-30 mT drive that puts taps 2, 3 and 4 at roughly 1-4 mT.
The drive-nonlinearity sweep measured gain 0.980 at 10 mT with compression only
appearing above it, so **three of the four disks were sitting in linear
response**.

That scaling came from the chain's failure, where a full-amplitude fresh sample
drowned a delayed copy ~100x smaller and cut degree-1 capacity from 8.91 to
4.95. Applying that lesson here produced the opposite failure: operands that
balance, in an element too weakly driven to multiply them.

The test is `--fresh-scale 1 1 1 1`, which puts every disk at 10-30 mT where the
nonlinearity is measurable. It risks exactly what killed the chain -- at tap 4
the fresh sample would be 29x the delayed copy -- but there is far more memory to
spend now: 17.36 of degree-1 capacity with a horizon of 20, against the chain's
8.91 falling to 4.95. Losing half of it would still leave more than the chain
ever had.

### The bus-only control: an excellent delay line, and nothing else

| readout | dim | NARMA-10 test NMSE |
|---|---|---|
| bus only | 200 | 0.8322 |
| co-driven | 200 | 0.3006 |
| best linear (15 lags) | 15 | **0.1822** |

Bus-only recall, which is the cleanest picture of what the delay line does:

| lag | 0 | 1 | 2 | 3 | 4 | 5 | 9 | 14 | 20 |
|---|---|---|---|---|---|---|---|---|---|
| r^2 | 0.000 | 0.122 | 0.224 | 0.633 | 0.879 | 0.868 | 0.984 | 0.972 | 0.949 |

Blind to lags 0-2 and then **0.87-0.98 from lag 4 to lag 20**. The array holds
the input for twenty frames with almost no loss, which is what a delay line is
supposed to do and better than anything else in this project has managed.
Capacity 16.78, rank 29, and again **100% degree-1** with every product family
at 0.00.

Co-driving fills the short lags the delay line cannot see -- lag 0 goes 0.000 to
0.953 -- and that accounts for the whole 0.8322 to 0.3006 improvement. It is
purely linear information, not products. It also collapses effective rank from
29 to 17, because driving all four disks with the SAME fresh sample makes them
partly redundant: the identical-disks-are-redundant result from the parallel
input layer (CC = 1.000), showing up again in a different architecture.

So the two arms agree on the only thing that matters. **Zero nonlinear capacity,
at every lag, in both modes.**

The honest summary of the architecture: it is an excellent linear delay line and
not a reservoir computer. Memory was never the binding constraint -- the chain
had 15.62 and this has 16.78 with a far better lag profile -- and neither
delivers the products the task needs.

### Full-amplitude co-drive: the hypothesis is refuted

Driving every disk at 10-30 mT, where the drive sweep measured 24% compression
and 33 degrees of phase shift, produces no products at all.

| | bus only | balanced co-drive | full amplitude |
|---|---|---|---|
| NMSE | 0.8322 | **0.3006** | 0.6463 |
| effective rank | 29 | 17 | **40** |
| deg1 capacity | 16.78 | **17.36** | 11.09 |
| recall at lag 17 | 0.978 | 0.717 | **0.009** |
| **all deg2/deg3 families** | **0.00** | **0.00** | **0.00** |
| degree-1 share | 100% | 100% | 100% |

Weak drive was not the cause. What full amplitude did do is reproduce the
chain's failure exactly -- degree-1 capacity 17.36 -> 11.09 and the memory
horizon collapsing from 20 to 16, against the chain's 8.91 -> 4.95 and 12 -> 7.
The same lesson, on a completely different architecture.

### The self-product is missing too, and that is the clue

`P2(s[n-k])` is 0.00 at every lag in all three configurations. Not merely the
CROSS-lag products -- the disk makes no u[n]^2 either, while a constant-amplitude
sweep on the same disk over the same 10-30 mT range measures gain falling 0.980
to 0.758 and phase advancing 3.2 to 32.9 degrees. A nonlinearity that large
should be visible as degree-2 capacity, and it is not.

The likely reason is a timescale mismatch rather than a missing nonlinearity.
Ring-down is 1/(alpha*omega) = 1.66 ns at alpha 0.008 and 12 GHz, which is **8.3
frames**. The envelope the disk's nonlinearity acts on is therefore an average
over the last ~8 input samples, not a single one. A nonlinear function of a
heavily low-passed signal distributes its products over many lag pairs with a
small coefficient each, and thin enough spreads sit under the noise floor
everywhere -- which is what a uniform 0.00 across every family looks like.

That is testable and the prediction is specific: damp the tap disks until they
respond inside one frame. tau < 0.2 ns needs alpha > 0.066, so
`tap_alpha_mult >= 8.3`. The ta=10 point already exists in this codebase, built
during the coupling sweeps for what turns out to have been the wrong reason.

The cost is known and measured: ta=10 halves what each tap receives, which
mattered when reception was marginal. It is much less marginal now -- the
weakest tap reads 3.93e-05 against a 1e-5 bar.

## The fast-disk test: the timescale explanation is confirmed, and insufficient

tap_alpha_mult = 10 drops ring-down from 8.3 frames to under 1, at the cost of
Q = 1/(2*alpha) falling 62.5 to 6.25. Measured first, on a valid record: the
ta=10 array resolves 2/3 rather than 3/3, spacings 5.83, 6.42, -1.23, and the
weakest tap falls 3.93e-05 to 1.68e-05 -- the halving the old coupling sweep
predicted, now confirmed without the truncated record.

| config | NMSE | rank | deg1 | deg2 P2 | cross-lag | deg1 share |
|---|---|---|---|---|---|---|
| ta=1 bus only | 0.8322 | 29 | 16.78 | 0.00 | 0.00 | 100% |
| ta=1 balanced | **0.3006** | 17 | 17.36 | 0.00 | 0.00 | 100% |
| ta=1 full amp | 0.6463 | 40 | 11.09 | 0.00 | 0.00 | 100% |
| ta=10 balanced | 0.7837 | 13 | 17.59 | 0.18 | 0.00 | 99% |
| **ta=10 full amp** | 0.5196 | 14 | 16.41 | **1.17** | 0.00 | **93%** |
| best linear (15 lags) | **0.1822** | | | | | |

**The explanation was right.** A disk ringing 8.3 frames averages its input and
shows no degree-2 capacity at all; a disk settling inside one frame shows
P2(s[n-1]) = 0.772 and P2(s[n-2]) = 0.474. The nonlinearity was always present
-- the drive sweep measured 24% compression and 33 degrees of phase shift over
this range -- it simply could not reach the readout through an 8-sample average.

**And it is not enough.** The products are strictly local, lags 1 and 2 only,
and **cross-lag products remain exactly 0.000 at every lag in all five
configurations**, including lag 9. Short-separation products are the family this
project already priced at nothing: they leave NARMA-10 at the linear baseline
while the value sits in |i-j| >= 5.

### The remaining fault is a balance error of mine

The disk's nonlinearity acts on its instantaneous internal state, and that state
is dominated by the fresh drive. I set `fresh_scale` by matching the fresh DRIVE
amplitude to the measured delayed READOUT amplitude, and those are not
commensurable: the fresh drive is a uniform field over the whole disk body,
while the delayed copy arrives through a 15 nm gap and an 80 nm guide. Their
coupling efficiencies differ by orders of magnitude.

The cross term is bilinear in the two operands, so if the delayed copy is 1% of
the internal state its product with the fresh sample is 1% of a term that is
itself 7% of measured capacity -- comfortably under the floor, which is what
0.000 at every lag looks like.

The measurement that fixes this already exists and was not used.
`check_polytap_impulse.py` runs bus-only, fresh-only and both arms and reports a
per-tap balance ratio between the delayed copy and the fresh drive AT THE
READOUT, which is the commensurable comparison. Setting `fresh_scale` from that
ratio is the principled version of what was guessed at here.

It needs its record lengthened first: burst 400 + quiet 2000 is 2.4 ns, and the
wave needs 3.75 ns to reach the lag-14 tap. The impulse script carries the same
too-short-record bug that invalidated the delay probe.

### A caveat on the absolute numbers

Total measured capacity exceeds the effective rank in several rows -- 17.77 of
rank 13, 17.58 of rank 14 -- which cannot be literally true and means the
per-target r^2 are upward-biased at 200 features against 300 training rows. The
shifted-target noise floor absorbs some of that and evidently not all. The
CONTRAST between configurations is measured identically throughout and is the
part to trust: 0.00, 0.00, 0.00, 0.18, 1.17.

## The balance measurement, and the bracket it closes

`check_polytap_impulse.py` measures the commensurable quantity: what a tap
reports from the bus alone against what the same tap reports from a UNIT fresh
drive alone, both at the readout. Three fixes were needed first. Its record was
burst 400 + quiet 2000 = 2.4 ns against a 3.75 ns transit -- the same
too-short-record bug, found for the fourth time, meaning every balance ratio it
had ever produced was a ratio of injection near field. It now takes --tap-alpha,
because the balance has to be measured where the disk's nonlinearity actually
reaches the readout. And its fresh arm now drives every disk at unit scale, so
the ratio is a measurement rather than a check on a previous guess.

Measured at ta=10, ba=1, 30 mT:

    recommended --fresh-scale   0.044832  0.008058  0.002274  0.001548

Normalised to tap 1 that is 1, 0.18, 0.051, 0.035 -- close in SHAPE to the
1, 0.134, 0.050, 0.023 used before. The absolute scale is the finding:
**0.0448, not 1.0. Every previous run drove the fresh sample 22x too hard**, so
the delayed copy was a few percent of each disk's internal state and a bilinear
cross term in a 4% operand sits far under the floor.

### Correcting it closes the bracket the other way

| ta=10 configuration | fresh drive | deg2 P2 | cross-lag | deg1 share | NMSE |
|---|---|---|---|---|---|
| full amplitude (22x unbalanced) | 10-30 mT | **1.17** | 0.00 | 93% | 0.5196 |
| measured balance | ~1.3 mT | **0.00** | 0.00 | 100% | 0.3839 |

Balancing the operands requires 0.045 x 30 mT = 1.3 mT, and the drive sweep puts
the disk in linear response below ~10 mT. So the disk can have equal operands or
a nonlinearity, not both, and the two requirements are separated by about a
factor of eight.

This is not a tuning failure. It is a statement about the geometry: the delayed
copy arriving through a 15 nm coupling gap is ~22x weaker than the drive a disk
needs before it compresses.

### The one way out, with numbers

Make the delayed copy stronger so the balance point moves up into the nonlinear
range. The valid-record ladder measured the weakest tap at 3.93e-05 at bus alpha
x1 and 2.89e-04 at x0.3, a gain of 7.4x, with x0.1 higher again. Roughly 22x
would put the balanced fresh drive near 30 mT, where the disk compresses 24%.

The cost is known and measured: lower bus damping scattered the tap timing badly
at ta=1 (spacings 8.98, 1.32, -3.53 at x0.3 against 3.46, 1.20, 3.76 at x1).
Whether it still does at ta=10 is untested -- that scatter was attributed to
guide ring-down, and `bus_alpha_mult` damps the guides while `tap_alpha_mult`
damps only the disk bodies, so a fast disk on a slow guide is a combination this
project has not run.

That is the experiment: balance re-measured at ta=10 with bus alpha x0.1 and
x0.03, then NARMA at whichever puts the balanced fresh drive above 10 mT while
keeping the delays resolved. If both cannot be had, the tapped bus cannot make
long-separation products in this geometry, and that is the architecture's
answer.

## The bus-damping suite: the balance saturates, and it saturates short

Balance measured at ta=10 for four bus dampings, valid record, unit fresh drive:

| bus alpha | tap-1 balance scale | fresh drive at balance |
|---|---|---|
| x1 | 0.0448 | 0.45-1.34 mT |
| x0.3 | 0.1057 | 1.1-3.2 mT |
| x0.1 | 0.1341 | 1.3-4.0 mT |
| **x0.03** | **0.1455** | **1.5-4.4 mT** |

**A 33x reduction in bus damping buys 3.25x in balance, and the curve is
asymptoting to ~0.15.** Even a lossless bus would top out near 4.5 mT against
the ~10 mT the disk needs before it compresses -- short by a factor of about
2.3, with no damping left to spend.

The saturation is structural rather than a limit of the sweep. `bus_alpha_mult`
damps the bus AND the guides, and the readout guides carry the fresh arm out too,
so lowering it lifts both sides of the ratio. Once the bus stops attenuating,
what remains is the geometric coupling through the 15 nm gap, which damping
cannot touch. The delayed copy's share of a disk's internal state is a property
of the COUPLER, not of the line.

### And the timing cost does transfer to a fast disk

That was the one thing genuinely untested: the scatter at low bus damping was
attributed to guide ring-down, and `tap_alpha_mult` damps only the disk bodies,
so a fast disk on a slow guide might have escaped it. It does not. At ta=10,
ba x0.03 the delay probe gives spacings 13.88, -0.04, 24.95, mono 2/3 -- the
same scatter the ta=1 ladder showed.

So both requirements fail at low bus damping, independently: the balance never
reaches the nonlinear range, and the delays stop resolving on the way.

### What has never actually been measured

The coupling-strength axis. Every gap and coupler result in this project --
"galvanic tap drains the bus, 314x spread", "30 nm gap, 181x", "directional
coupler, 157x" -- was measured on the 2.81 ns record, which is to say on
injection near field. None of them is valid, and none of them says what stronger
coupling does to the delayed copy's share of the disk state.

That is the remaining lever, and it is the right one: the balance ceiling is set
by the coupler, so the coupler is what has to change. A gap sweep on a valid
record would say whether ~2.3x more coupling is available before the tap starts
draining the line -- which is the trade the original gap sweep was built to
measure and never validly did.

### Phase 3: the low-damping branch fails outright

| | ta=10, ba x0.03, balanced |
|---|---|
| NMSE | **2.1598** (worse than the input alone, 1.3366) |
| effective rank | 64 |
| deg1 capacity | **1.71** |
| deg1 noise floor | 0.291 |
| all deg2/deg3 | 0.00 |

Rank 64 and 1.71 of capacity: a near-lossless bus driven at 1.5-4.4 mT produces
a state that is mostly long-lived reverberation uncorrelated with the recent
input. High dimension, almost no information. The noise floor at 0.291 says the
estimator is fitting that reverberation as readily as the target.

## The architecture's answer

Eight configurations, all on valid records:

| config | NMSE | deg1 | deg2 P2 | cross-lag |
|---|---|---|---|---|
| ta1 bus only | 0.8322 | 16.78 | 0.00 | 0.00 |
| **ta1 balanced** | **0.3006** | 17.36 | 0.00 | 0.00 |
| ta1 full amp | 0.6463 | 11.09 | 0.00 | 0.00 |
| ta10 balanced (guessed scale) | 0.7837 | 17.59 | 0.18 | 0.00 |
| ta10 full amp | 0.5196 | 16.41 | **1.17** | 0.00 |
| ta10 measured balance | 0.3839 | 15.32 | 0.00 | 0.00 |
| ta10 ba0.03 measured balance | 2.1598 | 1.71 | 0.00 | 0.00 |
| best linear (15 lags) | **0.1822** | | | |

Three things are established and none of them is a tuning failure.

**The delay line works.** 3/3 delays resolved, 0.87-0.98 recall from lag 4 to
lag 20, the best memory this project has produced.

**The disk's nonlinearity can be made to reach the readout.** It takes a disk
that settles inside one frame (ta=10) driven above ~10 mT, and it yields
P2(s[n-1]) = 0.772 with the degree-1 share falling to 93%.

**The two cannot be combined.** Balancing the operands -- which the cross term
requires, being bilinear -- caps the fresh drive at 4.4 mT even on a lossless
bus, a factor of 2.3 below where the disk compresses. The cap is set by the
coupling through the 15 nm gap, and the damping sweep shows it saturating.

**s[n]*s[n-k] is 0.000 in all eight configurations, at every lag.**

The remaining lever is the coupler, and it is the one axis never validly
measured: every gap and directional-coupler result in this project was taken on
the 2.81 ns record, which measured injection near field. Whether ~2.3x more
coupling is available before the tap drains the line is the open question, and
it is exactly the trade the original gap sweep was built for.

## The coupler sweep: the last axis, and it is saturated

Measured at ta=10, ba x1 -- the only damping that resolves the delays. There the
balance is 0.0448 and the target is 0.33, so the coupler must supply **7.4x**.

| gap | aperture | balance scale | vs baseline | drive at balance |
|---|---|---|---|---|
| 15 nm | 80 nm | 0.0448 | 1.00x | 0.4-1.3 mT |
| 5 nm | 80 nm | 0.0492 | 1.10x | 0.5-1.5 mT |
| 0 nm (galvanic) | 80 nm | 0.0492 | 1.10x | 0.5-1.5 mT |
| 0 nm | 160 nm | **0.0521** | **1.16x** | 0.5-1.6 mT |

**The entire coupler axis buys 1.16x against a target of 7.4x.** Going from a
15 nm evanescent gap to direct metallic contact changes the balance by 10%, and
gap 5 and gap 0 are identical to three figures. Proximity was never the
bottleneck.

The aperture prediction also failed. The valid-record width sweep measured
coupling proportional to width and implied 2x for 80 -> 160 nm; the measurement
gives 1.06x. That sweep compared 80 against 40, below saturation. Above ~80 nm
the guide already spans most of a 200 nm disk's edge and widening adds nothing.
Coupling is saturated on both axes, which is what an overlap-limited junction
looks like: a 200 nm disk and an 82 nm wave are mode-mismatched, and making the
junction more intimate does not fix a mismatch.

And the strongest coupler is worse where it counts. Its delay probe gives
spacings 3.6, 8.97, -1.1, mono 2/3, spread 102x, weakest tap 5.84e-06 -- against
3/3, 29x and 3.93e-05 for the as-built array. Its recommended per-tap scale,
0.0521 / 0.0136 / 0.0007 / 0.0005, shows why: a stronger tap drains the line and
taps 3 and 4 collapse. That is exactly the trade the original gap sweep existed
to find, now measured on a valid record, and it runs the wrong way.

Its NARMA arm: NMSE 0.3048, rank 32, deg1 **18.79** -- the highest memory in the
project -- deg2 P2 0.02, cross-lag 0.00, 100% degree-1.

## The architecture is answered

Nine configurations, all on valid records, spanning tap damping, bus damping,
drive amplitude, drive balance, coupling gap and aperture width:

**s[n]*s[n-k] is 0.000 at every lag in all nine.**

The reason is a single measured inequality. The cross term is bilinear, so the
delayed copy must be a comparable fraction of a disk's state. Balancing it caps
the fresh drive at 1.6 mT with the best coupler and 4.4 mT on a near-lossless
bus; the disk does not compress below ~10 mT. That is a factor of 6 in drive,
36 in power, and no knob in this design closes it:

  bus damping   33x reduction buys 3.25x and saturates at 0.15
  coupler       galvanic contact plus a doubled aperture buys 1.16x
  tap damping   makes the nonlinearity visible (P2 1.17) but only at a drive
                that unbalances the operands 22x, and costs delay resolution

What the device is, on the evidence: **an excellent linear delay line.** It holds
the input for twenty frames at 0.87-0.98 recall, resolves four taps, and carries
18.79 of degree-1 capacity -- more than the chain ever had. It is not a
reservoir computer, because the element meant to do the mixing cannot be reached
by the signal it is meant to mix.

Closing the gap needs something outside this design: a nonlinear element that
saturates far below 10 mT, gain on the bus, or a nonlinear readout -- and the
square-law readout was already measured and collapses rank 21 to 5.

## Introducing a nonlinear element: what the survey found

The tapped bus is limited by one inequality -- balancing the operands caps the
fresh drive at ~1.6 mT, and the permalloy vortex disk does not compress below
~10 mT. So the question became whether any element is strongly nonlinear at
1.6 mT. Four routes, measured.

### Drive frequency: real structure, not enough of it

The disk's azimuthal spectrum, measured by impulse ring-down: n=+-3 at 10.3 GHz,
n=+-4 at 11.0, n=-1 at 11.3, n=+-5 at 11.7/11.8, n=+-6 at 12.6, n=0 at 12.7. A
uniform in-plane drive couples by symmetry to n=+-1 only, so 11.3 GHz was the
predicted sweet spot.

| drive | threshold |
|---|---|
| 10.3 GHz | **8 mT** (below the bus band) |
| 11.0 GHz | 10 mT |
| 11.3 GHz | 20 mT |
| 11.7 GHz | 20 mT |
| 12.0 GHz | 15 mT |
| 12.6 GHz | 15 mT |

**The n=-1 prediction is refuted** -- 11.3 GHz is the WORST point measured. The
actual trend is downward toward lower frequency, which is the disk softening as
the drive approaches the bottom of its own spin-wave band rather than any
identified mode. Best inside the bus's band is 10 mT at 11.0 GHz, a 1.5x gain,
still 6.2x short.

### Radius and moment: both dissolve the vortex

| element | circulation | verdict |
|---|---|---|
| 100 nm, 800 kA/m | **+0.963** | vortex; threshold 15 mT |
| 60 nm, 800 kA/m | -- | not a vortex; threshold 30 mT |
| 40 nm, 800 kA/m | -- | not a vortex; never nonlinear |
| 100 nm, 300 kA/m | **-0.036** | **not a vortex**; "0.50 mT" is a different element |
| 100 nm, 140 kA/m | **+0.005** | **not a vortex**; same |

The low-moment result was the most promising number in this survey and it is an
artifact. The exchange length l_ex = sqrt(2A/(mu0 Ms^2)) is 5.7 nm at 800 kA/m,
15.2 at 300 and 32.5 at 140, so a fixed 100 nm radius takes R/l_ex from 17.5 to
6.6 to 3.1 -- through the point where flux closure stops paying for itself.
Lowering the moment without rescaling the disk dissolves the vortex.

### The pattern both routes share

**The disk softens as it approaches the conditions where it stops existing.**
Lower drive frequency softens it toward the band bottom, which is where the bus
stops propagating (~10.5 GHz). Lower moment softens it toward the exchange
length, which is where the vortex stops being the ground state. Twice, the
nonlinearity arrives exactly as the element or its delay line dissolves.

That is an argument for a different element rather than this one pushed harder.
The one variant that separates softness from instability is a low-moment disk
with the radius scaled to hold R/l_ex fixed -- 270 nm at 300 kA/m, 580 nm at
140 -- and the open question there is whether its modes stay above the bus band.

### Four diagnostics were wrong first

Worth recording, because three of them produced plausible numbers:

  CUDA conversion   the script was CPU-only until --device was added
  circulation (i)   a (nx,ny) integrand against a (nx,ny,1) mask broadcast to
                    rank 3 and returned +23.168 for a quantity bounded by 1
  circulation (ii)  averaged over disk AND guides, whose radial magnetisation
                    contributes nothing, diluting a perfect vortex to
                    31400/103400 = 0.30 -- the baseline read +0.337 and was
                    flagged as not a vortex
  bias equilibrium  relaxed at zero field then driven with the bias applied, so
                    the core started displaced and moved to equilibrium during
                    the run. Its signature was in the output and missed: the
                    biased run reported mean m_z = 0.25486, identical to the
                    unbiased baseline to five figures.

The circulation metric is bounded by construction -- exactly 1 for a perfect
vortex, 0 for any uniform state -- which is the only reason the first version
was caught immediately. A diagnostic with no known bound would have passed.

### A note on the hybrid architecture

A reversing first layer feeding non-reversing reservoir disks is structurally
sound, and for a specific reason: a memoryless nonlinearity ahead of a linear
reservoir is a Wiener system, which fills the P2(s[n-k]) column and leaves
s[n]*s[n-k] at zero -- the failure already measured nine times. A HYSTERETIC
node is different: core polarity persists, the gyrotropic sense flips with it,
so the response to a fresh sample is p[n]*u[n] with p[n] set by input history.
That is a current-times-past product. The nonlinearity has to sit AFTER the
memory, and a bistable node puts it there.

`check_drive_nonlinearity.py` now reports a three-way core verdict --
ok / REVERSED / lost -- because the previous test used mean m_z and read a
reversal as an instability, discarding the exact event such a layer would need.

## The material swap, done properly, and what it actually shows

Holding R/l_ex at permalloy's 17.6 while lowering the moment keeps the vortex:

| element | relax | converged | circulation | verdict |
|---|---|---|---|---|
| permalloy, 100 nm | 6k @ a=0.5 | |dm| = 0.0000 | **+0.963** | vortex, 15.00 mT |
| Ms 450 kA/m, 178 nm | 60k @ a=1.0 | |dm| = 0.0000 | **+0.968** | vortex, threshold NOT MEASURABLE |

So the scaled-radius reasoning was right about the STATE -- a low-moment disk at
matched R/l_ex relaxes to a clean vortex, better than the reference's own
circulation. The earlier "not a vortex at 300 and 140 kA/m" was entirely the
fixed radius.

**The threshold, however, is still not measurable on this element**, and the
0.50 mT it reports is an artifact for the second time, now for a different
reason. |A| does not scale with drive -- 6.07e-02 at 0.25 mT, 3.21e-02 at 1,
1.05e-02 at 4 -- so |A|/a falls as 1/a and the criterion fires on that. The
first time this happened the ground state was still settling (|dm| = 0.419);
this time it is converged to 0.0000, so the non-stationarity is in the DRIVEN
state.

The cause is in the core column: **the core is REVERSED at essentially every
amplitude at or above 2 mT.** The element is not responding smoothly and
becoming nonlinear, it is switching, and a lock-in over a fixed 600-step window
on a state that flips mid-window does not measure a response amplitude at all.

### Which is the interesting result

A converged, confirmed vortex reverses its core at ~2 mT at 12 GHz -- within a
factor of 1.25 of the 1.6 mT the tapped-bus balance delivers. That is the first
evidence here that a REVERSING element is reachable at the drive the coupling
permits, even though a smooth one is not.

It bears directly on the hybrid architecture, and it favours it. A memoryless
nonlinearity ahead of a linear reservoir is a Wiener system: it fills the
P2(s[n-k]) column and leaves s[n]*s[n-k] at zero, which is the failure already
measured nine times. A hysteretic node is different -- polarity persists, the
gyrotropic sense flips with it, so the response to a fresh sample is p[n]*u[n]
with p[n] set by input history, and that is a current-times-past product.

What this measurement cannot yet say is whether the reversals are USABLE:
deterministic against input amplitude, repeatable, and retaining polarity
between frames. A core that flips chaotically every frame is a noise source,
not a threshold element. That is the next measurement, and it needs a different
instrument from this one -- drive a single frame at a given amplitude, read the
polarity, repeat, and check the map from amplitude to final polarity is a step
function rather than a scatter.

### Two more diagnostics were wrong

  mean m_z        the same (nx,ny) against (nx,ny,1) broadcast as the
                  circulation, fixed in one place and not the other. It showed
                  as |mean m_z| = 2.71, which the mean of a unit-vector
                  component cannot reach.
  relax budget    12000 steps at alpha 0.5 left the 178 nm disk at |dm| = 0.419
                  against a 0.02 tolerance. 60000 at alpha 1.0 reaches 0.0000.
                  The budget had been scaled by radius rather than measured, and
                  magnum.np's warning went to stderr under a printed threshold.

`check_drive_nonlinearity.py` now measures convergence rather than assuming it
-- 200 further relax steps, report max |dm| -- and the verdict refuses to say
USABLE on an unconverged state.

## The element survey, complete

| route | threshold | state | verdict |
|---|---|---|---|
| permalloy 100 nm, 12 GHz | 15.00 mT | vortex, converged | the reference |
| static bias 10 mT | **15.00 mT** | vortex, converged | **no change** |
| drive frequency 11.0 GHz | 10 mT | vortex, converged | 1.5x, best in the bus band |
| radius 60, 40 nm | 30 mT / none | **not a vortex** | closed |
| Ms 300/140, fixed radius | -- | **not a vortex** | closed |
| Ms 450, radius 178 nm | not measurable | vortex, converged | **switches at ~2 mT** |
| Ms 300, radius 267 nm | not measurable | **diverged** -- see below | to be re-run |

### Correction: the low-moment non-convergences were divergences

Two rows above are recorded as states that would not settle. That reading was
wrong, and the cause is the integrator rather than the landscape.

The stiffest field RK4 has to resolve is exchange, `h_ex = 2A/(mu0 Ms dx^2)`,
and it gets **faster** as the moment falls -- the one scaling that runs opposite
to intuition here, since a softer magnet is in every other respect slower:

| Ms (kA/m) | h_ex (A/m) | f_max (GHz) | dt*omega at 1 ps | |
|---|---|---|---|---|
| 800 | 1.035e6 | 36.4 | 0.23 | stable |
| 450 | 1.839e6 | 64.8 | 0.41 | stable |
| 300 | 2.759e6 | 97.2 | 0.61 | marginal |
| 140 | 5.911e6 | 208.2 | 1.31 | **unstable** |

So `|dm| = 1.0033 after 120,000 steps` was not a flat energy surface, and
`circulation +0.657, |dm| = 1.0078` at Ms 300 / 100 nm was not the vortex-uniform
crossover being indecisive. Both were the timestep. dt now scales with Ms --
562 fs at 450, 375 at 300, 175 at 140 -- and settle, measure, relax and the
convergence-probe windows scale with it, since all four are physical durations
rather than step counts.

What this does **not** overturn: the permalloy reference, the bias null, the
frequency sweep and the Ms 450 / r178 switching map all ran at dt*omega <= 0.41
and stand as measured. What it reopens is the low-moment corner of the survey,
which is exactly where the single-domain feeder lives.

**No route gives a smooth nonlinearity at 1.6 mT.** The bias measurement is the
cleanest in the survey and settles that route definitively: |A| scales with
drive over a 160x range (2.65e-03 at 0.25 mT to 2.46e-01 at 40), compression and
phase are smooth and monotonic, the core never leaves "ok", and the threshold
lands at exactly the unbiased 15.00 mT.

### Every 0.50 mT in this survey was the same artifact

Three elements reported a 0.50 mT threshold and none of them meant it. It is the
second point on the amplitude grid, and the metric returns it whenever |A| is
drive-INDEPENDENT, because norm = (|A|/a)/ref collapses immediately when the
numerator does not scale. Three different causes, one signature:

  Ms 300/140 fixed radius   not a vortex -- and, per the correction above,
                            integrated past the stability edge as well
  Ms 450 r178, first try    ground state still settling, |dm| = 0.419
  Ms 450 r178, converged    the driven state switches, so no stationary
                            response exists to lock into
  Ms 300 r267               diverged: dt*omega = 0.61 at 1 ps

A threshold of 0.50 mT in this instrument means the measurement failed, not that
the element is soft. The bias table is the control that proves it: a
well-behaved element produces |A| proportional to drive and a threshold at
15 mT.

The convergence gate now blocks this automatically -- Ms 300 r267 was refused
with "NOT USABLE despite 0.50 mT" rather than reported.

### Where that leaves the architecture

The only accessible nonlinearity at the drive the coupling permits is a
REVERSAL. The Ms 450 / r=178 nm disk is a confirmed, converged vortex
(circulation +0.968, |dm| = 0.0000) whose core flips at ~2 mT against a 1.6 mT
budget -- and its smooth response is unmeasurable precisely BECAUSE it switches.

That favours the hybrid rather than counting against it. A memoryless
nonlinearity ahead of a linear reservoir is a Wiener system and cannot make
cross-lag products at any strength; a hysteretic node can, because polarity
persists and multiplies the current sample. `check_core_switching.py` is the
measurement that decides whether these reversals are usable -- deterministic
against amplitude, persistent through silence, and reversible -- or whether the
element is a noise source with a threshold.

## The switching gate: not a comparator, but still the only nonlinearity available

`check_core_switching.py` drives one frame from the converged ground state, then
holds ten frames of silence, and reads core polarity after each. Every amplitude
starts from the same state, so the map from amplitude to outcome is a function
of amplitude alone.

**Ms 450 kA/m, radius 178 nm** (circulation +0.968, |dm| = 0.0000):

| amp mT | 0.5 | 1.0 | **1.5** | 2.0 | 2.5 | 3.0 | 4.0 | 5.0 | **6.0** | **8.0** | 10 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reversed | - | - | **yes** | - | - | - | - | - | **yes** | **yes** | - | - |

**SCATTERED.** Reversals at 1.5, 6 and 8 mT and not at 2, 2.5, 3, 4, 5, 10 or
15. The amplitude-to-polarity map is not a step, so the clean hysteretic story
-- a bit set by the input, multiplying the current sample to give p[n]*u[n] --
does not hold for this element.

Polarity also fails to persist. At 0.5 mT the core reads +0.999 after the drive
and **-1.000 after ten frames of silence**, and the same happens at 1.0, 2.0,
2.5, 3.0 and 5.0 mT. That is not autonomous chaos: the hold follows a drive and
ring-down is 8.3 frames, so a 0.5 mT pulse deposits enough energy to flip the
core WHILE ringing down. Deterministic, but violently non-monotonic.

The permalloy control on the same probe reversed at no amplitude up to 30 mT and
held polarity through every hold, so the probe detects switching where it exists
and stability where it exists. The scatter is the element, not the instrument.

### What that does and does not rule out

It rules out the comparator reading of the hybrid: there is no reliable p[n] to
carry history. It does not rule out the element. A deterministic, strongly
nonlinear response with memory is what a reservoir wants even when it is not a
clean threshold, and this is still the only nonlinearity in the project
reachable at the drive its coupling delivers -- reversing from 1.5 mT where
permalloy needs 15 mT merely to compress 5%.

So the NARMA run is worth its hour, with the mechanism relabelled. The number
that decides it is not the score but whether s[n]*s[n-k] is finally nonzero. It
has read exactly 0.000 in all nine tapped-bus configurations and in every chain
measured. If a strongly nonlinear element with memory in the first stage does
not move it, the obstruction is not the strength or reachability of the
nonlinearity but where it sits relative to the memory -- which is a more useful
thing to know than another incremental score.

Drive window 1-3 mT rather than the usual 10-30: the element responds from
~1.5 mT, so the standard window would flip it every frame regardless of input,
and a node that always switches carries no information. At 0.383 transfer per
hop that also gives the capped architecture for free -- stage 2 sees 0.4-1.1 mT
and stage 3 sees 0.15-0.4 mT, so the head switches and the tail does not.

## NARMA-10 on the switching-vortex chain: chaos, not computation

Three stages of Ms 450 kA/m / 178 nm disks, driven 1-3 mT so the head straddles
its ~1.5 mT reversal threshold while stages 2 and 3 (0.383 transfer per hop) sit
at 0.4-1.1 and 0.15-0.4 mT and do not switch -- the capped architecture, built
by amplitude gradient rather than by per-disk material.

| readout | dim | NARMA-10 test NMSE |
|---|---|---|
| input alone | 1 | 1.0153 |
| linear 10-lag | 10 | 0.6737 |
| **switching chain** | 180 | **4.4981** |

**The device scores worse than the input by itself**, and the decomposition says
why in one line: effective rank **142 of 180**, and **every capacity family
reads 0.00 -- including degree-1.**

Near-full rank with zero measured capacity is not richness, it is noise. The
state is high-dimensional and decorrelated from the input: no recall at any lag,
no products at any separation, nothing for a linear readout to fit. Noise floors
rose accordingly, with s[n-5]*s[n-k] at max r^2 0.103 against 0.02-0.04 in every
previous run.

That is the echo-state property failing. A reservoir needs FADING memory -- the
state must depend on recent input history and forget the rest. Core reversals
scattered across amplitude make the state depend on the whole sequence, so
nothing generalises.

The integrator agreed. A first attempt at 600 frames ran to frame 525 at a
steady 0.37 s/frame and then hung for 30 minutes with the GPU busy; 500 frames
of the same seed and window completed cleanly in 192 s. The hang is
input-dependent and localised, which is what a trajectory wandering into a state
the stepper cannot handle looks like.

## The two ends are now measured, and nothing sits between them

| element | nonlinearity | capacity |
|---|---|---|
| permalloy vortex | needs 15 mT; coupling delivers 1.6 | 16-18 units, **100% degree-1** |
| switching vortex | reverses from 1.5 mT | **0.00, every family** |

Too little nonlinearity gives an excellent linear delay line -- 0.87-0.98 recall
to lag 20, 18.79 of degree-1 capacity, the best memory in the project -- and no
products. Enough nonlinearity to be reachable at the available drive destroys
fading memory outright.

That is the classic reservoir tension, and this device has no window between the
horns of it. The gap is not small: permalloy needs 9.4x more drive than the
coupling delivers, and the element that closes that gap overshoots into chaos
rather than landing in the useful middle.

What would be needed is an element whose nonlinearity saturates GENTLY at ~1.6
mT -- strong enough to mix, bounded enough to forget. A vortex core does not do
this: it is stiff until it reverses, and reversal is a discrete event with no
graded regime in between. Something with a soft continuous saturation, rather
than a bistable core, is the shape of element this architecture wants, and no
configuration of the vortex disk provides it.

## The invariant: threshold times memory is a constant, and alpha cancels

The trap above was recorded as a damping conflict -- threshold goes as `alpha`,
ring-down as `1/alpha`, so no single value serves both. That framing is true but
it hides the stronger statement. Write the two out for a resonantly driven mode:

```
h_th  ~  alpha * omega / gamma      threshold: drive tips the cone to order 1
tau   ~  1 / (alpha * omega)        ring-down: the linewidth is alpha*omega
--------------------------------------------------------------------------
h_th * tau  =  1 / gamma  =  5.7 ps*T
```

**Both alpha and omega cancel.** The product of the drive a resonant element
needs and the memory it then carries is fixed by the gyromagnetic ratio alone --
not by material, moment, radius, damping, thickness or drive frequency.

This is why the element survey closed every route it tried. Ms, radius, bias and
frequency are all variables the invariant is blind to. The survey was searching a
space the answer does not live in, and its uniform failure was the invariant
being measured five times rather than five independent dead ends.

It also predicts a number already in hand. Permalloy at `alpha` = 0.008 and
12 GHz gives `tau = 1/(alpha*omega)` = 1.66 ns, and the switching runs measured
an 8.3-frame ring-down at 200 ps frames -- 1.66 ns. The model reproduces a
measurement that was made before it was written down.

### What the invariant leaves free

The frame. It is the one parameter this project has never varied, fixed at
`FRAME_STEPS = 200` and 1 ps since the first chain run. Requiring a reachable
threshold and a forgetful element together:

```
h_th <= h_max   and   tau <= 2 * T_frame   =>   T_frame >= 1 / (2*gamma*h_max)
```

At the 1.6 mT the balance currently allows, that is **1.78 ns against the 200 ps
in use -- short by 8.9x**. Both sides of the inequality are movable, and the
second one is already measured: the bus-damping suite found 3.25x of headroom on
the fresh drive, which puts `h_max` at 5.2 mT and brings the frame requirement
down to 0.83 ns, a factor of 4.2.

So the pair (3.25x more drive, 4.2x longer frames) closes a gap that no material
substitution could, and neither number is speculative.

The cost is bus length. Taps sit at `lag * 189 nm` because 189 nm at
v_g = 706 m/s is one 200 ps frame; at 0.83 ns frames the spacing becomes 586 nm
and a 20-lag bus grows from 3.8 to 11.7 um, with the attenuation that implies.
That is a real price and it is the next thing to measure -- but it is an
engineering cost on a known axis, not another closed route.

### Why the perpendicular-bias feeder was the wrong shape

The feeder disk was designed to escape by installing a tunable anisotropy field:
static bias with the AC drive perpendicular, so saturation would be set by
H_bias and memory by alpha, independently. The invariant says that cannot work,
and the FMR arithmetic says why concretely.

At Ms 140 kA/m with 2 mT bias the Kittel frequency is
`28 GHz/T * sqrt(H(H+mu0*Ms))` = 0.53 GHz, while the bus drives at 12. Driving
at 23x the resonance suppresses the response by `(omega_0/omega)^2` and the
element would have read as weakly linear -- another null, and a misleading one.
Raising the bias until FMR reaches 12 GHz requires 350 mT, which then IS the
threshold. The same trap wearing different clothes.

The measurement that is worth making instead is the falsifiable half of the
invariant: `h_th ~ alpha*omega/gamma` says threshold falls in PROPORTION to drive
frequency. The survey already hints at it -- 11.0 GHz gave 10 mT against 15.00 at
12, a 1.5x fall for a 1.09x change, steeper than proportional, which is what
approaching a mode from above looks like.

### Frame length, first attempt: a confounded design

The invariant leaves the frame free, and the frame has a second effect besides
memory: the burst is one frame wide, so the frame sets the packet BANDWIDTH.
At 200 steps the packet spans ~5 GHz, over which v_g runs 706 to 1267 m/s, and
dispersive smearing was the standing explanation for the spacing scatter --
3.46, 1.20, 3.76 frames against 3.00 designed on an array placed to 1 part in
500. A narrower packet should smear less.

Measured, with tap spacing scaled to keep designed lags in frames:

| frame | packet | spacings (designed 3.00) | mono | spread | weakest tap |
|---|---|---|---|---|---|
| 200 steps | ~5.0 GHz | 3.46, 1.20, 3.76 | 3/3 | 29x | 3.93e-05 |
| 400 steps | ~2.5 GHz | 5.43, -0.05, 3.37 | 2/3 | 225x | **1.33e-06** |

Mean spacing error went 1.01 -> 1.95 frames. The prediction was that it would
FALL, so on its face this refutes dispersion.

It does not, because the design confounded two things. Scaling tap spacing with
the frame also scaled the BUS: 2.6 um to 5.3 um, against a 1354 nm propagation
length. The weakest tap fell to 1.33e-06, **below the 1e-5 placement bar**, and
under that bar an arrival estimate is reading near field and noise rather than a
delayed wave. That is the same failure that cost this project four campaigns,
reappearing in a new place: a starved tap mis-times, so the timing degradation
is a plausible CONSEQUENCE of the amplitude collapse rather than evidence about
dispersion at all.

The 200-step control reproduced the recorded numbers exactly -- 3.46, 1.20,
3.76, spread 29x, weakest tap 3.93e-05 -- so the build is sound and the
comparison is real. It is the interpretation that does not follow.

The clean version holds the bus FIXED and varies only the burst, restating the
lags in the new frame units so the taps sit in the same physical places:

| frame | frame_nm | lags | tap x (nm) | designed spacing |
|---|---|---|---|---|
| 200 | 189.0 | 5, 8, 11, 14 | 945, 1512, 2079, 2646 | 0.60 ns |
| 400 | 378.0 | 2.5, 4, 5.5, 7 | 945, 1512, 2079, 2646 | 0.60 ns |
| 600 | 567.0 | 1.667, 2.667, 3.667, 4.667 | 945, 1512, 2079, 2646 | 0.60 ns |

Attenuation, coupling geometry and the ground state are then identical and only
the packet changes. amp_min is the control on the control: same bus, so it
should land near 3.9e-05 at every frame, and if it does not the geometry is not
actually matched and nothing in the comparison counts.

Errors get compared in NANOSECONDS, not frames. The designed spacing is 3.00
frames at 200 steps and 1.50 at 400 -- the same 0.60 ns -- so a fixed
frame-count tolerance silently tightens as the frame grows.

Whatever it returns, note what is NOT at stake: the frame was introduced for the
memory argument, where its effect is arithmetic rather than empirical. A
dispersion null would make longer frames timing-neutral, not useless.

### Frame length, controlled: dispersion refuted, reception improved

Same bus, same taps at 945/1512/2079/2646 nm, same ground state; only the burst
width changes. Errors in ns, because the designed separation is a fixed 0.60 ns
while its frame count is not.

| frame | packet | designed | err (frames) | err (ns) | mono | spread | weakest tap |
|---|---|---|---|---|---|---|---|
| 200 steps | ~5.0 GHz | 3.00 fr | 1.007 | 0.201 | 3/3 | 29x | 3.93e-05 |
| 400 steps | ~2.5 GHz | 1.50 fr | 1.057 | 0.423 | 2/3 | 22x | 6.81e-05 |
| 600 steps | ~1.7 GHz | 1.00 fr | 0.527 | 0.316 | 3/3 | 12x | 9.57e-05 |

Dispersive smearing predicts 0.201 -> 0.100 -> 0.067 ns. Measured: 0.201 ->
0.423 -> 0.316. **Refuted**, and this time with the amplitude confound removed --
the weakest tap is above the 1e-5 placement bar at every point, so all three
are interpretable.

The timing numbers show no trend in the predicted direction and no clean law in
the other direction either. Three points do not support one, and an earlier
reading of the first two as "about one frame regardless of frame length" does
not survive the third (0.527 frames at 600 steps). What can be said is that the
error stays in the 0.2-0.4 ns band across a 3x change in packet width, so
whatever sets the spacing scatter, it is not the bandwidth of the packet.

**Reception is the clean result, and it runs opposite to the prediction.**
Spread fell 29x -> 22x -> 12x and the weakest tap rose by 2.4x, both monotonic
in burst width, which is what more injected energy per burst does.

So a longer frame is timing-neutral and reception-positive AT FIXED BUS LENGTH.
The confounded run is the counter-case and stays in the record for it: scaling
the bus with the frame cost 30x of reception (3.93e-05 -> 1.33e-06) against a
1.7x burst-energy gain. Bus length is what hurts; frame length is not.

### What this does to the frame-length plan

The memory argument wanted 600-step frames. That is now measured as free on
timing and favourable on reception, with one constraint attached: the bus must
not grow with the frame. Holding the bus at 2.6 um puts the taps 1.00 frame
apart with 0.527 frames of error, which is too tight to resolve cleanly.

The intermediate design is the one to test -- 600-step frames with taps at lags
2, 4, 6, 8, a 4.5 um bus, and 2.00 frames of designed separation against 0.527
of error, a relative error of 0.26 against the control's 0.34. It buys the
memory the invariant demands without either starving the taps or crowding them.

Note what is NOT resolved: the handoff brief attributes the spacing scatter to
dispersion and offers a ~3 frame smear estimate. That explanation is now
refuted, and no replacement has been measured. The scatter is real, reproducible
across four independent runs of the control (3.46, 1.20, 3.76 every time), and
unexplained.

## The frequency sweep: the disk is soft at 5 GHz, and the bus cannot go there

The invariant predicts h_th ~ alpha*omega/gamma, so at fixed damping the
threshold should be proportional to drive frequency. Measured on the permalloy
100 nm vortex -- the one element with a trusted reference, a converged ground
state and clean amplitude scaling over a 160x range:

| f (GHz) | linear \|A\|/a | h_th (mT) | on-resonance floor | ratio |
|---|---|---|---|---|
| 1.5 | 1.29e-04 | 40.00 | 0.43 | 93.4 |
| 2.5 | 2.55e-04 | 24.00 | 0.71 | 33.6 |
| 3.5 | 5.78e-04 | 12.00 | 1.00 | 12.0 |
| **5.0** | 7.59e-03 | **1.50** | 1.43 | **1.05** |
| 7.0 | 2.30e-03 | 16.00 | 2.00 | 8.0 |
| 9.0 | 2.10e-02 | 6.00 | 2.57 | 2.3 |
| 12.0 | 1.44e-02 | 16.00 | 3.43 | 4.7 |

The log-log slope is -0.62, so the simple proportionality is NOT what happens.
What happens instead is mode structure: the linear response has two peaks, at
5 and 9 GHz, with an anti-resonance between them at 7 GHz. Read against the
on-resonance floor rather than against a power law, the sweep says the disk
sits 4.7x off its mode at 12 GHz and essentially ON one at 5 GHz, where the
ratio is 1.05.

**At 5 GHz the threshold is 1.50 mT, below the 1.6 mT the tap balance
delivers.** That is the first element in this project whose nonlinearity is
reachable at its own operating point without switching, and it is graded rather
than bistable -- compression runs 1.000, 0.943, 0.905, 0.820, 0.597 with phase
rising 0, 1.21, 2.12, 4.44, 10.46 degrees and the core intact throughout.

It is not the artifact. The tell registered in advance was that |A| must keep
SCALING while `norm` falls, since a flat |A| is what produced three spurious
0.50 mT thresholds. Here |A| runs 1.90e-03 to 3.53e-02, a factor of 18.6, while
norm falls to 0.290. At 12 GHz and the same 1.5 mT drive, norm is 0.999 and the
phase shift is 0.09 degrees -- flatly linear.

### Response magnitude and nonlinearity are not the same quantity

Worth stating because it was assumed otherwise. The LARGEST linear response is
at 9 GHz (2.10e-02, three times the 5 GHz peak), but its threshold is 6 mT, not
1.5. So locating a mode by its response amplitude does not locate the useful
operating point, and the two have to be measured separately.

### The constraint this hits

The 80 nm bus has a band bottom near 10.5 GHz. Both reachable-threshold points
-- 5 GHz at 1.50 mT and 9 GHz at 6 mT -- are BELOW it, where the delay line
carries nothing. The only measured point inside the propagating band is 12 GHz
at 16 mT, which is the wall.

So the problem is no longer "no element is nonlinear at 1.6 mT". One is. The
problem is that **the disk's soft modes and the bus's passband are in different
places**, which is a two-sided tuning problem rather than a closed route.

### Caveat, unresolved at time of writing

The settle window was 7.2 drive cycles, which at 5 GHz is 1.44 ns against a
ring-down of 3.98 ns -- the lock-in read a state 36% of the way to steady.
Because --cycles makes settle and tau both scale as 1/omega, that fraction is
identical at every frequency and the COMPARISON is fair; but the absolute
numbers are mid-transient, and settling in a nonlinear system is itself
amplitude-dependent. A drive-dependent transient reading as compression is
exactly how this project produced three false thresholds.

A verification at 40 cycles (2.0 tau at 5 GHz, 2.4 at 12) is running. The 12 GHz
control is the one that matters: it has been quoted at 15-16 mT throughout this
project, and if it MOVES then the frequency structure above is an artifact of
the window rather than a property of the disk.

## Widening the bus: 9 GHz is reachable, 5 GHz is not

If the disk's soft modes and the bus's passband are in different places, either
can move. The disk side is constrained -- 60 and 40 nm radii both lost the
vortex -- so this measures the bus. Lateral confinement sets the band bottom
(the lowest mode has k_y = pi/w), so a wider strip should lower it.

| bus width | band bottom | carries 5 GHz? | carries 9 GHz? |
|---|---|---|---|
| 80 nm (as built) | 11 GHz | no | no |
| 160 nm | 8 GHz | no | **yes** |
| 240 nm | 7 GHz | no | **yes** |
| 320 nm | 6 GHz | no | **yes** |

Widening 4x buys 5 GHz of band bottom, with clear diminishing returns as the
dipolar terms take over from exchange. Extrapolating, 5 GHz would need roughly
500-640 nm -- reachable in principle and probably not usefully, because at that
width the strip stops being a single-mode waveguide and becomes a film with
many lateral modes, which would scramble the delay structure the architecture
depends on.

**9 GHz is inside the band at 160 nm already**, and that is the more useful of
the two soft points anyway: 6 mT rather than 16, against a budget of 1.6 mT
base times the bus-damping suite's measured 3.25x = 5.2 mT. Short by 1.15x,
where the 12 GHz operating point was short by 9.4x.

The band edge is sharp rather than gradual. On the 160 nm strip, below 8 GHz the
phase fit is poor (R^2 0.06 to 0.74) with decay under 1.8 um; at 8 GHz it snaps
to R^2 = 1.000, contrast 1370, decay unmeasurable.

### What 9 GHz on a 160 nm bus looks like

| quantity | 12 GHz / 80 nm (as built) | 9 GHz / 160 nm |
|---|---|---|
| wavelength | 82.3 nm | 108.6 nm |
| group velocity | 706 m/s | 542.9 m/s |
| nm per 200-step frame | 189 (measured) | 108.6 |
| phase R^2 | 1.000 | 1.000 |

The slower group velocity helps rather than hurts: the same delay fits in a
shorter bus, which is the axis the tap array is actually limited on.

**Caution on the decay column.** The "inf" figures are bare-strip fits, and the
80 nm strip reads inf at 12 GHz too while the LOADED bus measures 1354 nm. Tap
loading dominates the real attenuation, so this says the intrinsic propagation
is better, not that the reception problem goes away. That has to be measured on
a loaded 160 nm bus before any of it is designed around.

### Verified settled: the frequency structure is real

The sweep's settle window was 7.2 drive cycles, 36% of the ring-down, so every
number in it was read mid-transient. Rerun at 40 cycles -- 2.0 tau at 5 GHz,
2.0 at 12 -- on the same disk and the same grid:

| amp (mT) | 5 GHz @ 0.36 tau | 5 GHz @ 2.0 tau | 12 GHz @ 0.36 tau | 12 GHz @ 2.0 tau |
|---|---|---|---|---|
| 0.25 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.50 | 0.995 | 0.995 | 1.000 | 1.000 |
| 1.00 | 0.974 | 0.976 | 1.000 | 1.000 |
| 1.50 | 0.943 | 0.949 | 0.999 | 0.999 |
| 2.00 | 0.905 | 0.918 | 0.999 | 0.999 |
| 12.0 | 0.373 | 0.267 | 0.962 | 0.955 |

**Threshold 1.50 mT at 5 GHz and 16.00 mT at 12 GHz, both unchanged.** The
control is the one that mattered -- it has been quoted at 15-16 mT throughout
this project -- and it reproduces to three decimals at 5.6x the settle. So the
sweep was not measuring settling, and at 1.5 mT the same disk compresses 50x
more at 5 GHz than at 12.

One correction to how this should be quoted. The criterion fires at norm 0.950
and the settled 1.5 mT row is 0.949, so the "1.50 mT" label clears its own
cutoff by 0.001 and overstates the precision. The defensible statement is the
curve: at 5 GHz this disk compresses ~5% at 1.5 mT, ~8% at 2, ~15% at 3 and
~35% at 6, settled, with the core intact throughout.

## Where the architecture actually stands

The element problem is solved, and it was never the problem it was taken for.
Five materials routes were closed, an invariant was derived to explain why, and
the answer turned out to be that **12 GHz sits 4.7x off the disk's own mode**.
The disk was always capable of graded nonlinearity at reachable drive; it was
being driven in the wrong place.

| | as built | proposed |
|---|---|---|
| bus width | 80 nm | 160 nm |
| carrier | 12 GHz | 9 GHz |
| disk threshold | 16.00 mT | 6.00 mT |
| drive available | 1.6 mT | 5.2 mT (bus damping, measured) |
| shortfall | **9.4x** | **1.15x** |
| group velocity | 706 m/s | 542.9 m/s |
| nm per 200-step frame | 189 | 108.6 |

5 GHz is softer still at 1.50 mT, but the bus cannot be widened to carry it
without becoming multi-mode, so 9 GHz is the operating point.

What is NOT established, and should not be assumed from the above:

  9 GHz threshold      measured at 7.2 cycles only. The 5 and 12 GHz points
                       both survived the settled rerun, so it probably holds,
                       but it has not been checked.
  loaded 160 nm bus    every propagation number here is a BARE strip. The 80 nm
                       strip reads decay inf at 12 GHz while the loaded bus
                       measures 1354 nm, so tap loading dominates and the
                       reception budget is unmeasured at the new width.
  coupling geometry    the tap aperture was tuned against the 82.3 nm wave at
                       12 GHz. At 9 GHz on 160 nm the wave is 108.6 nm and the
                       bus is twice as wide; neither the aperture nor the gap
                       has been re-optimised.
  the products         no capacity decomposition has been run at any of this.
                       s[n]*s[n-k] has read 0.000 in nine configurations, all
                       of them driven at 12 GHz on an element that was linear
                       there. That is the run this makes worth its hours, and
                       it is the only thing that settles whether any of it
                       matters.

## First nonzero cross-lag product, at 9 GHz

The run the frequency result exists to justify. Poly-tap array at the retuned
9 GHz operating point -- 160 nm bus, gap 30 nm, aperture 80 nm, 4 taps at lags
2/3/4/5, 1200-step frames, co-driven with the fresh scale measured at those
lags (1.000, 0.372, 0.206, 0.078).

**s[n]*s[n-k] = 0.20 against exactly 0.000 in all nine previous
configurations**, and the degree-1 share falls from 98-99% to 92%.

| family | capacity |
|---|---|
| deg1 P1(s[n-k]) | 8.73 |
| deg2 P2(s[n-k]) | 0.55 |
| **deg2 s[n]*s[n-k]** | **0.20** |
| deg2 s[n-5]*s[n-k] | 0.00 |
| deg3 P3(s[n-k]) | 0.00 |
| TOTAL | 9.47 of rank 14 |

### What it is not

**The product sits at lag 1.** Of the 0.20, lag 1 contributes 0.134 (r^2 =
0.241 against a 0.107 noise floor); lags 18 and 20 contribute the rest at 0.122
and 0.156, barely over the floor. The reservoir has NO linear memory past lag 9
-- P1 reads 0.000 from lag 10 on -- so a product against s[n-18] when s[n-18]
is not remembered at all is noise, not capacity. Short-separation products are
the family this project already measured as worthless on NARMA-10: every one
sits at the 0.1243 linear baseline while far pairs score 0.0401.

**The NARMA scores are not usable at this run length.** 150 training frames
against 200 feature columns is rank-deficient, and the LINEAR baseline shows it
-- 0.5088 here against 0.1243 on the longer runs. The device read 1.4419, worse
than input-only at 1.0323. That is a statistics artifact of a run shortened to
survive the harness, not a measurement of the device.

**Effective rank is 14**, down from 21 in earlier configurations.

### What behaved as designed

Memory horizon is ~9 frames at 1.2 ns per frame, about 10.8 ns, against ~15-20
frames at 0.2 ns -- roughly 3-4 ns -- before. Far more memory in absolute time
and fewer frames of it, which is the delay bus supplying memory independently of
disk ring-down.

The retuned array is also the best delay line measured in this project: spacings
0.83, 0.79, 0.68 against 1.00 designed, 3/3 monotonic, 0.279 ns error, spread
13x, weakest tap 4.13e-04 -- against 29x spread and 3.93e-05 at 12 GHz.

### Unexplained

Reservoir lag-1 autocorrelation is **-0.393**, reproduced exactly across two
independent runs. With tau/frame = 1.84 the state should be POSITIVELY
correlated frame to frame. Nothing here explains the sign.

### What the next run has to fix

600 frames with splits 100/300/100, which is the configuration the earlier runs
used and the only way these magnitudes become trustworthy. Lags should include
9, since NARMA-10's own term is u[n]*u[n-9] and the lag set IS the separation
set. At ~16.5 s/frame that is a ~2.8 hour rollout, which the harness has not yet
survived -- so it needs either the chunked-checkpoint approach or a fix to the
graph-capture path, which currently fails inside magnum.np's tensor wrapper and
leaves the run dispatch-bound at 13.6 ms/step.

### Four harness and configuration errors, recorded

| error | cost | fixed by |
|---|---|---|
| `--timeout 3600` read as a gap timeout; it is a TOTAL limit | 2 sessions | 14400 |
| session reclaimed on CLI disconnect, `--keep` notwithstanding | checkpoints lost | shorter runs |
| splits summing to `frames` leaves an EMPTY test set -> silent nan | 1 completed 82-min run | guard added |
| `--n-ports 6` where n_readout is 5 (the sixth guide is the bus coupler) | assertion | 5 |

The third is the one worth remembering: three runs carried it from the start, so
none of the sessions that died would have produced a usable number either.

## Building the memory in: a low-loss bus

The first nonzero product sat at lag 1 because the reservoir had no linear
memory past lag 9 and cannot multiply against a sample it does not retain. The
memory had to come from the delay line -- not from disk ring-down, since
lengthening tau is what destroys products in the first place -- and the delay
line had a measured wall.

Two points on the loaded 160 nm bus at 9 GHz, same gap and aperture:

    far tap 3255 nm   amplitude 4.13e-04
    far tap 5859 nm   amplitude 1.79e-05
    -> loaded-bus decay length 830 nm

At that decay a lag-14 tap receives 3.5e-07, two orders under the placement
bar. Damping-limited decay scales as 1/alpha, so lowering bus damping buys
reach directly. Physically a hybrid: a low-loss bus feeding permalloy disks,
with x0.1 (8e-4) between permalloy at 8e-3 and YIG at ~1e-4.

| lags | bus alpha | mono | err (ns) | weakest tap | usable |
|---|---|---|---|---|---|
| 3,5,7,9 | x1.0 | 3/3 | 0.463 | 1.79e-05 | yes |
| 3,5,7,9 | x0.3 | 3/3 | 0.803 | 1.30e-03 | yes |
| **3,5,7,9** | **x0.1** | **3/3** | **0.537** | **5.28e-03** | **yes** |
| **5,8,11,14** | **x0.1** | **3/3** | 1.472 | 3.17e-03 | **yes** |
| 5,8,11,14 | x0.03 | 2/3 | 4.987 | 6.81e-03 | no |

**The timing degradation did not recur.** At 12 GHz this same move cost the
ordering -- 2/3 monotonic at both x0.3 and x0.1 where x1 gave 3/3 -- and that
reach-versus-ordering trade defeated four campaigns. At 9 GHz with 1200-step
frames it does not bite until x0.03, where reception improves again (6.81e-03)
while the ordering collapses and the FIRST tap arrives last (21.19 against
12.75). x0.1 is an optimum, not merely a point that worked.

What that buys, measured rather than designed:

| | 12 GHz array | 9 GHz, x0.1 |
|---|---|---|
| delivered lags | 13.1, 16.6, 17.8, 21.6 | 2.87, 5.70, 8.08, 10.21 |
| separations | 3.5, 1.2, 3.8 | 2.8, 2.4, 2.1 |
| fresh-drive balance | 1.000, 0.196, 0.043, 0.007 | 1.000, 0.766, 0.523, 0.447 |
| tap-to-tap spread | 29x | 2x |

The balance is the part that matters for products. A product needs two
comparable operands, and on the old array balancing them meant scaling the far
taps to a hundredth -- which is exactly how the chain's co-drive failed. Every
tap now receives within a factor of 2.2.

The 5,8,11,14 array reaches further still (delivered 4.49, 8.47, 13.18, 17.17)
and was the first choice for the capacity rerun, but its 2099-cell mesh runs at
48.6 s/frame against 16.2 for the 1447-cell one -- 1.45x the cells for 3.0x the
time. There is a cliff between those two sizes that 840 -> 1447 does not show
(1.72x the cells at 1.01x the time, purely dispatch-bound). Not diagnosed.

## The field: two velocities, and the lag discrepancy explained

Every delay-line number here is a scalar out of a cross-correlation, which is
how four campaigns went wrong -- a truncated record gave plausible lags for a
wave that had never arrived. `capture_wave.py` records the field itself.

It shows the bus carrying **two velocities at once**:

    leading edge                          1258 m/s
    envelope centroid, tap 1 -> tap 4      444 m/s
    bare strip at 9 GHz                    543 m/s   <- what frame_nm assumed

A one-frame burst has sharp edges and is therefore broadband, so its leading
edge runs at the fast high-frequency components -- v_g reaches 1267 m/s at
18-20 GHz -- while its envelope, which is what a cross-correlation times and
therefore what sets a tap's lag, crawls at 444 on the loaded bus.

That is a candidate mechanism for the delivered-versus-designed discrepancy
recorded earlier as unexplained. Tap spacing was calibrated on the BARE strip
at 543 m/s; the loaded bus carries the envelope at ~470, so every tap lands
later in time than designed. It also predicts the damping dependence that came
with it: damping eats the packet's tail, pulling the centroid earlier and
making the bus look faster, which is the measured direction (0.77-0.81 at
x1.0 against 1.22-1.41 at x<=0.3).

Not yet confirmed. It predicts the centroid should shift with BURST WIDTH at
fixed damping, and that has not been measured.

### Two analysis errors on the way, both mine

Reading the capture, I first averaged the field over the x axis instead of y
and concluded the wave was not propagating at all -- a spectacular false
negative on a bus that demonstrably works -- and separately dropped a factor of
1000 in the arrival-time units. Both were caught by the numbers being absurd
rather than by the plot looking wrong, which is the same reason the field
capture is worth having.

## The memory works; the run that tested it was linear by construction

NARMA-10 and the capacity decomposition on the low-loss array -- lags 3,5,7,9 at
bus alpha x0.1, delivered lags 2.87/5.70/8.08/10.21, 350 frames.

**The memory went in exactly as intended:**

| | before (bus alpha x1) | after (x0.1) |
|---|---|---|
| P1 at lag 10 | 0.000 | **0.709** |
| P1 at lag 15 | 0.000 | 0.367 |
| P1 at lag 20 | 0.000 | 0.126 |
| degree-1 capacity | 8.73 | **11.91** |
| effective rank | 14 | 17 |

The horizon went from stopping dead at lag 9 to still carrying 0.126 at lag 20.

**The products did not follow.** s[n]*s[n-k] fell from 0.20 to 0.09 and the
degree-1 share ROSE from 92% to 97%. The only above-floor entry is lag 19 at
r^2 0.161 against a 0.075 floor, on 50 test frames.

### Why that is not evidence about memory

The drive never reached the element's threshold. Input amplitude ran 1-5 mT and
the per-tap fresh scale is 1.000 / 0.766 / 0.523 / 0.447, so the taps saw:

    tap 1   5.00 mT    83% of threshold
    tap 2   3.83 mT    64%
    tap 3   2.62 mT    44%
    tap 4   2.23 mT    37%

against the measured 6.00 mT threshold at 9 GHz. **Nothing in the run was ever
nonlinear**, so no product could form whatever the memory horizon was. The
5 mT ceiling came from the 5.2 mT coupling budget; the threshold is 6.00.

This was flagged when the run was designed and then wrongly waved off as
"largely dissolved" once the tap balance improved. The balance is a RATIO --
all four taps within 2.2x instead of 140x -- and it says nothing about whether
the absolute drive crosses threshold. It does not.

So the run measures one thing cleanly: a low-loss delay bus with linear
elements is an excellent linear filter. Degree-1 capacity 11.91, rank 17,
memory past lag 20. On NARMA-10 it scores 1.0880 against the 20-tap linear
filter's 0.3182 -- still a loss, and the linear baseline improved too (0.5088
-> 0.3182) because 200 training frames beat 150.

### What the next run has to change

Drive above threshold. At 2-12 mT input the taps see 12, 9.2, 6.3 and 5.4 mT,
putting three of four over the 6.00 mT threshold with the crossing mid-range,
which is where a threshold nonlinearity is most informative. That is
physically safe: the settled sweep at 9 GHz held the core intact at every
amplitude to 40 mT.

Whether it is BUILDABLE is a separate question that also needs redoing. The
1.6 mT cap came from balancing the fresh drive against a delayed copy on the
12 GHz / 80 nm bus. On this bus the delayed copy is 134x stronger (weakest tap
5.28e-03 against 3.93e-05), so the balanced fresh drive scales with it and the
budget is almost certainly no longer 5.2 mT. That arithmetic has not been
redone at this operating point.

## Driving the elements through threshold: nonlinearity at the cost of the reservoir

Both issues cleared first, which changed the configuration materially.

**Threshold in situ.** The 6.00 mT figure was a LONE disk with six readout
guides. A tap disk has five plus a bus coupler and sits a gap from a loaded
160 nm waveguide. Measured in the array it compresses at 10, 12, never and
12 mT -- about 2x stiffer. Passes the artifact check (|A| scales x47-61 over a
drive x60). Tap 3 is anomalous and unexplained: its normalised gain RISES to
1.021 and it never compresses in range.

**Balance.** The bus at 30 mT delivers a copy worth 0.201 / 0.159 / 0.106 /
0.089 of direct drive. At exact balance the taps top out at 8.0 / 6.3 / 4.2 /
3.6 mT -- under threshold everywhere, and tap 4 would need 135 mT on the bus.
**Balance and threshold cannot both be satisfied on this bus.** The run used
2.5x balance at 8-40 mT, giving a 2.5:1 operand imbalance against the 100:1
that killed the chain co-drive, with the drive spanning the threshold.

### The result

| quantity | sub-threshold (1-5 mT) | above-threshold (8-40 mT) |
|---|---|---|
| effective rank | 17 | **44** |
| degree-1 capacity | 11.91 | 3.00 |
| deg2 P2 | 0.27 | 0.11 |
| **deg2 s[n]*s[n-k]** | 0.09 | **0.22** |
| deg2 s[n-5]*s[n-k] | 0.00 | 0.05 |
| **deg3 P3** | 0.00 | **0.21** |
| TOTAL measured | 12.26 | **3.59** |
| degree-1 share | 97% | **83%** |
| P1 at lag 5 | 0.991 | 0.158 |
| P1 at lag 10 | 0.709 | 0.021 |
| NARMA-10 NMSE | 1.088 | **5.625** |

**The nonlinearity is real.** Degree-1 share fell 97% -> 83%, the cross-lag
family more than doubled, and degree-3 appeared from nothing. Driving the
elements through threshold does what it was supposed to do.

**And it destroys the reservoir.** Memory collapsed -- P1 at lag 5 went 0.991
to 0.158 and the horizon from past lag 20 to about lag 4. Total measured
capacity fell 12.26 to 3.59 while effective rank ROSE 17 to 44, so roughly 40
of 44 dimensions carry nothing attributable to the input at degrees 1-3. That
is the chaos signature, the same one the switching-vortex chain produced at
rank 142/180 with all capacity 0.00. NARMA went 1.088 to 5.625, worse than
input-only at 0.986.

The noise floor moved with it: the s[n]*s[n-k] floor rose 0.055 -> 0.132,
because features that can predict targets shifted out of reach are overfitting.
Of the three above-floor entries, only lag 2 (r^2 0.289, P1 0.814) sits where
the reservoir has memory; lag 10 (0.138) and lag 15 (0.193) have P1 of 0.021
and 0.015 and are the same artifact called out in the earlier run.

### What is now bracketed

The classic reservoir tension, measured at one operating point with a working
delay line on both sides of it:

    1-5 mT    sub-threshold   excellent linear filter, memory past lag 20,
                              capacity 12.26, products 0.09
    8-40 mT   above threshold nonlinear and incoherent, memory to lag ~4,
                              capacity 3.59, products 0.22, NARMA 5.6

Both horns are now measured rather than inferred. The untested middle is a
drive that crosses threshold only at the very top of the input range -- at
2.5x balance, an 8-24 mT input puts tap 1 at 12.1 mT against its 10 mT
threshold and leaves the other three under, so most samples stay in the graded
region and only the peaks compress. That is the narrowest version of the
window, and if it is empty the tension is not resolvable by drive alone at
this operating point.

Features from the above-threshold run are committed under data/narma_9ghz/ so
the decomposition can be redone without the GPU.

### Five runs lost to infrastructure

This result took six attempts. The fifth died with the VM wiped mid-flight at
frame 330/350 -- checkpoint, ground state and repo clone all gone while the
session still reported BUSY. Completed runs cluster at 81, 85 and 100 minutes;
the wiped one was 105. The sixth was sized to 83 minutes and dumps its feature
matrix to the log as base64 the moment the rollout ends, so the result now
survives the machine.

## Can the reservoir be preserved? Splitting the readout says not this way

The above-threshold run saturated taps 1 and 2 (20.1 and 15.9 mT against
thresholds of 10 and 12) and left taps 3 and 4 below (10.6 and 8.9 mT against
none-in-range and 12). Disks couple to the bus BIDIRECTIONALLY, so a saturated
tap re-radiates its distorted response into the bus and downstream. The array
may therefore already contain a saturating feeder followed by a linear
reservoir, summed together in one readout -- which splitting the readout tests
for free.

| readout | cols | rank | floor | deg1 | product | total | deg1 share | horizon |
|---|---|---|---|---|---|---|---|---|
| all four taps | 200 | 44 | 0.132 | 3.00 | 0.22 | 3.59 | 83% | 4 |
| saturated (taps 1,2) | 100 | 29 | 0.113 | 2.82 | 0.12 | 3.40 | 83% | 4 |
| sub-threshold (taps 3,4) | 100 | 29 | 0.183 | 2.58 | **0.26** | 2.85 | 90% | 3 |

**The nonlinearity does propagate.** The unsaturated taps carry more than twice
the product capacity of the saturated ones, 0.26 against 0.12, so a saturated
disk's distortion does reach its neighbours.

**But the memory is gone at the unsaturated taps too.** Horizon 3 frames,
against 20+ when the same array ran sub-threshold. Taps 3 and 4 are
individually linear -- neither was driven past its own threshold -- so the
corruption did not happen in those disks. It arrived through the BUS, which
every tap shares.

So a saturating element placed in line with the delay line pollutes the delay
line. Separating the READOUT is not enough. The feeder needs a path into the
reservoir that does not also carry the memory.

Caveat on the 0.26: it is measured against a 0.183 floor on 100 columns and 40
test frames, where the saturated view's floor is 0.113. A higher floor makes
the estimate less trustworthy, not more, and the difference between 0.26 and
0.12 is not large against that.

### Three routes, by cost

  off-centre coupling   attach the feeder's guide away from the disk's
                        symmetry axis, or tangentially rather than radially,
                        so it drives an azimuthal mode that overlaps poorly
                        with the bus's propagating mode. Coupling goes as mode
                        overlap, so this buys asymmetry cheaply -- but it is
                        mode selectivity, not isolation, and the leakage back
                        into the bus falls rather than vanishing.
  absorbing barrier     a lossy segment of bus between feeder and reservoir.
                        Directional only in the sense that it attenuates
                        everything crossing it, so it costs the delayed copy
                        as much as the pollution unless the feeder sits
                        downstream of every reservoir tap.
  non-reciprocity       the physically correct answer. In-plane films with
                        k perpendicular to M (Damon-Eshbach) are genuinely
                        one-way: +k and -k localise on opposite surfaces. That
                        is a true isolator rather than an attenuator. It needs
                        the bus magnetised transverse to propagation, which
                        means a bias field and a different dispersion from the
                        one measured here -- the band bottom, v_g and the
                        108.6 nm wavelength would all have to be re-measured.

The first is a modest geometry change and testable with the existing tooling;
the third is a redesign that changes every transport constant this project has
measured.

## Route 1: off-centre coupling

### What the knob is

`PolyTapConfig.bus_guide_offset` displaces the bus coupling guide's axis
laterally from the disk centre. At zero it is radial and drives the disk's
radial response; at 60-80 nm against the 100 nm radius it meets the disk close
to tangentially and drives azimuthal modes instead. Verified on the mask before
any physics: offsets of 40/60/80 nm move the guide by exactly +40/+60/+80 nm at
constant 80 nm width, on an unchanged 1447x178 grid.

One thing had to be fixed first. `polytap_alpha` cuts a corridor out of the
absorbing taper directly below each disk so the coupling guide is not damped,
and that cut was pinned to the disk axis at the READOUT guide width. An offset
guide would have run straight through the taper -- at 60 nm offset, 60 of its
80 nm inside the ramp -- and its coupling would have collapsed for a reason
that has nothing to do with mode overlap, in exactly the direction that reads
as "the offset bought isolation". The corridor now tracks `bus_guide_width` and
`bus_guide_offset`, and the coupling guide measures 8.00e-04 alpha (the bus
value, undamped) at every offset tested.

### Registered prediction

Reciprocity says the linear transfer bus->disk equals disk->bus for the same
mode pair. Nothing in this geometry breaks reciprocity: no bias asymmetry, no
gyrotropic term the offset touches. So the expectation on the record is that
the offset scales forward and backward TOGETHER and the isolation ratio comes
out flat. If it does, route 1 is closed on a measurement rather than on an
argument, for ten minutes against a NARMA run's ninety.

Two ways it could still earn a place, and both are measured:

  directionality  a chord-coupled disk re-radiates into the bus with a phase
                  gradient along x, which can favour one direction. That is
                  reciprocal and legal, and it is enough: a feeder that dumps
                  its distortion downstream of every reservoir tap is as good
                  as one that does not dump it at all.
  distortion      forward coupling is at the carrier; the pollution is at 2w
                  and 3w. Mode overlap is frequency dependent, so an offset
                  can in principle keep the carrier and shed the harmonics.
                  Both are inside the bus band -- 18 and 27 GHz against a band
                  bottom of 8 GHz at 160 nm -- so they propagate if launched.

### Method

Three lock-in arms per offset, on the built array (4 taps, lags 3/5/7/9, 160 nm
bus, bus alpha x0.1, 9 GHz):

  F     bus injector at 3 mT -> each disk's readout ports at w. Forward
        coupling. If this collapses the offset merely decoupled the disk.
  Blin  disk 1's body at 3 mT -> bus probes at w. Backward coupling in the
        linear regime. F/Blin is both the isolation figure and the
        reciprocity check.
  Bsat  disk 1's body at 20 mT -- what tap 1 actually saw in the run whose
        reservoir died, against a 10 mT in-array threshold -> bus probes at
        w, 2w, 3w. What a saturated feeder puts into the shared line.

Bus probes are 20 nm windows on the bus centreline, upstream of tap 1 and
between each tap pair. Short on purpose: the guided wavelength here is 108.6
nm, so a probe approaching that length averages the carrier to nothing and
would report a dead bus.

Forward and backward are read on different observables -- a guide-end m_z
average against a bus-segment m_z average -- so their ratio has no absolute
meaning. Every number is therefore also reported relative to the offset-0 row,
and only the relative numbers carry the verdict.

### Result: route 1 is closed

| offset | forward | rel | back-leak | rel | isolation | rel | dir | 2w/w | 3w/w |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3.10e-06 | 1.00 | 3.87e-05 | 1.00 | 8.03e-02 | 1.00 | 0.41 | 0.018 | 0.007 |
| 40 nm | 4.82e-06 | 1.55 | 1.15e-04 | 2.97 | 4.20e-02 | 0.52 | 0.94 | 0.023 | 0.016 |
| 60 nm | 4.76e-06 | 1.53 | 1.15e-04 | 2.96 | 4.16e-02 | 0.52 | 0.87 | 0.032 | 0.014 |
| 80 nm | 4.56e-06 | 1.47 | 1.30e-04 | 3.36 | 3.51e-02 | 0.44 | 0.96 | 0.025 | 0.013 |

The prediction was that reciprocity would hold and the ratio would come out
flat. It came out WORSE. Off-axis coupling raises forward transfer about 1.5x
and the back-leak about 3x, so isolation falls by roughly half at every offset
tested. The offset does not separate the two directions; it opens both, and it
opens the wrong one faster.

Both escapes are closed with it. The harmonic fraction does not fall -- it
drifts slightly UP, 1.8% to 2.5-3.2% -- so there is no frequency selectivity to
exploit. And directionality moves the wrong way.

That last one is the finding worth keeping, because it is the opposite of the
hypothesis. Splitting the leak by direction:

| offset | upstream | downstream | up rel | down rel |
|---|---|---|---|---|
| 0 | 9.48e-05 | 3.87e-05 | 1.00 | 1.00 |
| 40 nm | 1.22e-04 | 1.15e-04 | 1.28 | 2.97 |
| 60 nm | 1.31e-04 | 1.15e-04 | 1.38 | 2.96 |
| 80 nm | 1.36e-04 | 1.30e-04 | 1.43 | 3.36 |

The offset barely changes the upstream leak and opens the downstream channel
threefold. So the ASYMMETRY BELONGS TO THE RADIAL GEOMETRY: a radial guide
couples preferentially back toward the injector, 2.4x at equal distance, and a
chord coupling merely radiates both ways. The un-offset disk was already the
better element and route 1's knob is what ruins it.

Twenty minutes of GPU against a ninety-minute NARMA run. That is what
registering the prediction bought, and the prediction was still wrong in an
informative direction -- flat was expected, worse was measured.

### The 0.41 is not yet a result

It was read at ONE probe separation, and the two sides of a disk sit in
different interference environments: downstream has the next tap as a reflector
703 nm away, upstream has the injector and the end absorber. A standing wave
produces exactly that signature. A radiation asymmetry is the same at every
distance and a standing wave oscillates with it, so the ratio is now measured
at 400/600/800 nm and the spread reported. Until that comes back, the 2.4x is a
candidate and not a number to design on.

### On the leak's decay, which decides whether route 2 can work at all

Downstream of the driven tap, at offset 0:

| distance | 651 nm | 1954 nm | 3257 nm | 4033 nm |
|---|---|---|---|---|
| leak | 5.06e-05 | 8.18e-07 | 2.40e-07 | 7.65e-08 |

The 62x collapse over the first interval looked at first like a near field
dying, which would have closed route 2 immediately -- a leak that does not
travel along the bus is a leak no bus barrier can reach. It is not that. Beyond
the first tap spacing the fall is 3.4x and 3.1x per interval, decay lengths of
1065 and 685 nm, which is this project's own MEASURED loaded-bus figure of 830
nm rather than the 8300 nm the bare bus gives at bus alpha x0.1. The tap disks
are the dominant loss and lowering bus damping does not lower tap loading. So
the far-field leak is guided, a barrier can attenuate it, and the 62x is the
near-field-to-guided transition at 651 nm rather than a decay rate.

This also means the route-2 layout is already worth something before any
barrier: with the feeder at tap 4, the leak reaching taps 3, 2 and 1 travels
1.3, 2.6 and 3.9 um, worth about 3.4x, 11x and 39x on this decay. A 400 nm
barrier at alpha 0.2 multiplies each by a further 15.9x.

### The 0.41 survives the control

Downstream/upstream leak from tap 1, at three probe half-separations:

| 400 nm | 600 nm | 800 nm | spread |
|---|---|---|---|
| 0.50 | 0.41 | 0.60 | 1.46x |

Never near 1.0 at any distance. So the upstream preference is a genuine
radiation asymmetry -- between 1.7x and 2.5x depending where it is read -- with
standing-wave modulation of about +-20% riding on top. It is not a standing
wave masquerading as directionality, which was the failure mode the single-
distance measurement could not exclude.

A radial-guide disk therefore couples preferentially back toward the injector,
and that is free directionality the array already has. It is not large. The
array that lost its memory had it, so 2x on its own settles nothing -- but it
multiplies with the loaded decay and the barrier rather than adding to them.

That number cost one crash to get. --sym-nm became a list when the control went
in and one legend line still formatted it with :g, so the summary raised AFTER
all three arms had run and every number had been computed. The point paid its
full GPU time and threw the result away, and because the driver copied its
result file only on success, the next point overwrote it. Arm F's bus probes
are now printed rather than only saved: a log that does not carry every
quantity the verdict needs is a log that can lose them again.

### The mechanism, quantified

Arm F's bus probes were being computed and discarded until this run. With them,
the saturated feeder's leak can be read against the injected signal ON THE SAME
STRETCH OF BUS, which is the only form in which a leak amplitude means
anything. Feeder = tap 1, per mT of each drive:

| stretch | injected (F) | leak (Bsat) | leak/signal |
|---|---|---|---|
| up (before tap 1) | 9.52e-04 | 6.91e-05 | 0.073 |
| 1-2 | 5.06e-06 | 6.84e-05 | **13.5** |
| 2-3 | 1.36e-06 | 6.62e-07 | 0.49 |
| 3-4 | 7.25e-07 | 2.57e-07 | 0.35 |
| after | 4.37e-07 | 1.04e-07 | 0.24 |

On the stretch immediately downstream of the saturated disk its own
re-radiation is 13.5x the injected signal, and it is still 35-50% of it two and
three taps away. That is the memory collapse in one table: the delay line
downstream of a saturated tap carries mostly that tap's distortion rather than
the delayed input it exists to carry. It also explains the shape of the
split-readout result -- taps 3 and 4 sat at leak/signal around 0.4, enough to
destroy a memory horizon without either of them ever crossing its own
threshold.

Two caveats held on this. It is normalised per mT and the operating run drove
the bus and the disk bodies at different amplitudes, so the operating ratio
needs that scaling before it is quoted as an absolute. And the steep fall over
the first stretch -- 153 nm decay length -- is the injector's own evanescent
near field rather than bus transport; the propagating decay settles to
1000-2000 nm per tap spacing further out.

### What the barrier actually trades

Worth stating before route 2's numbers arrive, because it decides which row of
the table is the good one.

Of the two orderings, the barrier only pays in one. In
`injector -> feeder -> barrier -> reservoir` the reservoir's own signal crosses
the barrier too, so the barrier cancels out of the leak-to-signal ratio
entirely and only the bare 2x directionality is left. In
`injector -> reservoir -> barrier -> feeder` the reservoir's signal never
crosses and the leak does, so the barrier does not cancel -- at the cost of a
factor of 2, because the leak travels upstream, which is the disk's preferred
direction. A 15.9x barrier nets about 8x. The second ordering wins and the
directionality finding halves its benefit rather than overturning it.

But it is not free, and this is the part that decides how to read the table:

  reservoir signal      unchanged, never crosses the barrier
  leak into reservoir   proportional to 1/B, so reservoir SNR improves as B
  feeder's operand      also crosses, proportional to 1/B, and the product goes
                        as fresh x delayed -- so product amplitude degrades as
                        1/B

The barrier trades reservoir cleanliness against product amplitude one for one,
and raising the feeder's fresh drive cannot buy it back because the feeder is
already driven above threshold on purpose. So the row to want is NOT the
largest leak drop. It is the WEAKEST barrier that restores enough cleanliness,
because everything past that is paid out of the product the architecture exists
to make. If 2x suffices, route 2 is cheap. If it takes 16x, the product may not
survive its own protection.

### Result: route 2's barrier is bypassed, but its LAYOUT is worth 29x

Two things have to be separated here, because the run measures both and only
one of them works.

**The barrier does not isolate.** Local suppression of the injected signal,
measured at the barrier's own centre against the same probe with no barrier:

| barrier | at its centre | beyond it |
|---|---|---|
| 400 nm, alpha 0.05 (designed 2.0x) | 1.88x | 0.93x |
| 400 nm, alpha 0.2 (designed 15.9x) | 6.25x | 0.87x |

The field is suppressed inside the barrier and recovers completely past it. Tap
4's forward drive is unchanged across all three points -- 6.942e-07, 6.981e-07,
7.019e-07 -- so the barrier costs nothing, and it buys nothing either. The leak
behaves the same way. Against the clean baseline, leak/signal per stretch:

| | up | 1-2 | 2-3 | 3-4 (barrier) | after |
|---|---|---|---|---|---|
| no barrier | 4.9e-05 | 0.0340 | 0.461 | 139.3 | 322.7 |
| alpha 0.05 | 4.2e-05 | 0.0298 | 0.487 | 131.1 | 297.9 |
| alpha 0.2 | 3.6e-05 | 0.0283 | 0.590 | 119.8 | 282.9 |

A barrier designed for 15.9x delivers at most 1.34x on any reservoir stretch,
and makes the NEAREST one worse -- 0.461 to 0.590 -- which is what a reflector
does. A 400 nm damped segment attenuates what propagates through it and does
nothing to the stray field, and demag is long-range and switched on in these
simulations. Both the signal and the leak simply step over it.

This closes route 2 as a mechanism, and it closes it in the manner registered
in advance: "if the leak does not fall with the barrier's attenuation, it is
not arriving along the bus and a bus barrier cannot touch it." The prediction
named the leak; the measurement found the SIGNAL doing it too.

It also makes the cleanliness-against-product-amplitude trade moot. There is no
trade, because there is no effect. The analysis was right about what a working
barrier would cost and wrong to assume one would work.

**The ordering works, and it does not need the barrier at all.** What each
RESERVOIR tap sees, leak against injected signal on its own stretch:

| feeder position | nearest reservoir tap | middle | farthest |
|---|---|---|---|
| tap 1 (the array that died) | 13.5 | 0.487 | 0.354 |
| tap 4 (this layout) | 0.461 | 0.034 | 4.9e-05 |

Moving the feeder to the far end takes the worst reservoir stretch from 13.5 to
0.461, a factor of 29, from ordering alone. The two far taps land at 0.034 and
4.9e-05, clean by any standard. The nearest reservoir tap still sits at 0.461,
comparable to the 0.35-0.49 that destroyed the memory before, so this buys a
GRADED reservoir -- two clean taps and one compromised -- rather than a
uniformly poisoned line.

### And the directionality is near-field only

Tested directly rather than assumed. Probe 2-3 sits 1955 nm downstream of tap 1
in the control and 1954 nm upstream of tap 4 in the baseline -- equal distances,
opposite directions:

| | distance | direction | leak |
|---|---|---|---|
| feeder tap 1 | 1955 nm | downstream | 6.62e-07 |
| feeder tap 4 | 1954 nm | upstream | 6.27e-07 |

Ratio 0.95. The prediction on the record was 1.32e-06 if the 2x upstream
preference held at that range. It does not: the 1.7-2.5x measured at 400-800 nm
is a near-field effect, gone by 2 um, and the reservoir taps it was supposed to
protect sit 1.3-3.9 um away. Isolation by directionality does not work at the
distances that matter, and the 29x above comes from DISTANCE -- the loaded-bus
decay -- rather than from any radiation pattern.

Three probe pairs proved the asymmetry was real and a fourth measurement at
four times the range proved it was useless. Both were needed; neither would
have been enough alone.

## Route 3: DMI, not Damon-Eshbach

### The target was wrong, and structurally rather than budgetarily

Classical DE cannot serve this design. Its asymmetry is surface localisation
through the film thickness -- +k on one face, -k on the other -- and every mesh
in this project is nz = 1, so it is not merely unmeasured but unrepresentable.
And resolving it in 3D would not fix the geometry: omega(k) is reciprocal for a
symmetric film, so the asymmetry only reaches a transducer that couples to ONE
face, while these coupling guides are lateral and in-plane and meet both faces
alike. Route 3 as sketched needed a 3D mesh AND vertically stacked coupling.
The earlier estimate -- "re-measure the transport constants" -- was wrong in
kind, not in size.

Interfacial DMI adds a term linear in k to the dispersion itself, so
omega(+k) != omega(-k) in the plane already simulated. solver.py has said so
since before this branch -- "the only non-reciprocal term available here" --
and LLGRollout has taken a Di argument all along. The vortex path never passed
it. Threading it through was a few lines.

### The bound is the spiral, not the cost

| Di (mJ/m^2) | helix period | vs 160 nm bus width | vs 7.3 um bus |
|---|---|---|---|
| 0.02 | 8168 nm | 51x | 1.12x |
| 0.05 | 3267 nm | 20x | 0.45x |
| 0.10 | 1634 nm | 10x | 0.23x |
| 0.20 | 817 nm | 5.1x | 0.11x |
| 1.00 | 163 nm | 1.0x | 0.02x |

With no anisotropy the ground state becomes a helix of period 4*pi*A/D. Shape
anisotropy on a 160 x 20 nm strip raises that bound by an amount this project
has not calculated, so the bus <m_x> is printed at every point and a point that
spiralled is reported rather than quietly used. A spiralled bus is not a
waveguide and its transport numbers would look perfectly finite while meaning
nothing -- the same shape as the truncated-record failure that cost four
campaigns here.

### The feeder is tap 2, and that is the measurement

Taps sit 1303 nm apart, so probes placed +-d about tap 2 are equidistant from
tap 1 and tap 3 at EVERY d: at 1600 nm both sit 297 nm from a neighbouring tap,
at 1200 nm both sit 103 nm from one. The two sides carry identical loading by
construction. That is precisely what the route-1 directionality measurement
lacked, where downstream had a tap 703 nm away and upstream had the injector
and an end absorber.

Registered prediction, and it is the one that separates a real result from a
repeat of route 1's artifact: a dispersion asymmetry ACCUMULATES along the
path, because the two directions travel at different k and therefore attenuate
at different rates per micron. So the ratio should GROW with probe separation.
The near-field asymmetry did the opposite -- 2.4x at 600 nm, 0.95 by 1954 nm --
which is how it was caught. Flat-in-distance would mean this is that artifact
wearing a different hat, and would close route 3 the same way.

### Result: DMI moves the coupling, not the line

| Di (mJ/m^2) | 400 nm | 800 nm | 1200 nm | 1600 nm |
|---|---|---|---|---|
| 0 (reciprocal) | 2.339 | 1.596 | 0.923 | 0.762 |
| 0.02 | 2.303 | 1.564 | 0.923 | 0.764 |
| 0.05 | 2.198 | 1.545 | 0.901 | 0.742 |
| 0.10 | 2.062 | 1.486 | 0.897 | 0.741 |

The Di = 0 row earns its keep before any DMI is read off. In a RECIPROCAL
medium, with probes equidistant from the neighbouring taps by construction, the
downstream/upstream ratio is 2.34 at 400 nm and 0.76 at 1600 nm -- not 1.0
anywhere. So the raw ratio measures environment, and only deviation from this
row measures non-reciprocity. It also settles the route-1 leftover: that
measurement gave 0.41 for tap 1 and this one gives 2.34 for tap 2, the same
phenomenon with opposite sign, which is local loading rather than any radiation
pattern.

Normalised to the reciprocal baseline, the DMI effect is 11.9% at 400 nm and
2.8% at 1600 nm. IT SHRINKS WITH DISTANCE. The registered prediction was the
opposite -- a dispersion asymmetry accumulates along the path, so it should
grow -- with flat-or-shrinking called in advance as the signature of the same
near-field artifact route 1 produced. By that criterion this is that artifact:
DMI is modifying the disk's LOCAL coupling, not making the LINE one-way.

The coefficient is linear and clean, about 119% per mJ/m^2 at 400 nm across the
top two points, so extrapolation is trustworthy. Di = 0.25 would give roughly
30% at 400 nm and 7% at 1600 nm. Order-of-magnitude isolation at the 1.3-3.9 um
where the reservoir taps sit would need several mJ/m^2 -- past the spiral bound
and past any real material.

Two things are still worth the next twenty minutes, and neither is an
extrapolation:

  band edge   differential attenuation is what the linear coefficient
              describes. A separate mechanism appears if the DMI shift pushes
              one direction out of the propagating band entirely -- the carrier
              is 9 GHz against a band bottom of 8 GHz, so that needs about
              1 GHz, near Di = 0.25. Below it the effect is invisible and above
              it the bus is genuinely one-way, so a linear fit from 0.10 cannot
              see it in either direction.
  a loose end tap 4's forward response jumped 2.6x at Di = 0.05 and returned to
              normal at 0.10, while the bus stayed uniform at <m_x> = 0.996
              throughout. A non-monotonic hop is a disk changing STATE rather
              than coupling, which the bus check cannot see. Peak and mean m_z
              per disk body now go in the log for exactly this.
