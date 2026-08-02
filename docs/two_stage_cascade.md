# Does a two-stage cascade compose its memory?

**Written before any cascade task data exists.** Predictions, controls and the
decision rule are fixed here, including the conditions under which the run
answers nothing.

## Why this is the only hypothesis left

Three mechanisms for extending the disk's ~8-frame memory have been measured and
all three are dead:

| mechanism | measurement | result |
|---|---|---|
| damping | α 0.008 → 0.002 | MC 8.27 → 8.61 (+0.34 frames) |
| radiation | free decay vs port count | τ 1–2 ns everywhere, non-monotonic |
| readout | whole state vs six rim taps, equal width | 0.97×, same cliff |

NARMA-10's recursion needs `u[n-10]`; the disk's r² crosses 0.5 at lag 8.

Depth is untouched by any of those measurements and is the standard route to
multiple timescales in reservoir computing: if stage B integrates stage A's
output, and each holds ~8 frames, B's dependence on `u` reaches further than
either alone. That is the claim under test.

## Topology, and why it is not the obvious one

`--drive one`: **only disk A is driven**. Disk B receives the input solely
through the link. That is a cascade.

The previous coupled script drove both disk bodies, which makes two parallel
reservoirs sharing a substrate. That tests whether coupling helps a wider
readout — a real question, and not this one. Composition requires that B's only
path to `u` runs through A.

## What had to be built first

`CoupledPortedArray` carried no readout at all — no `port_signals`, no taps — so
every task run through it was impossible rather than merely wrong. Taps are now
built with the single disk's geometry (just inside the absorbing taper) at each
disk's own centre, restricted to cells carrying material so a window overhanging
empty space cannot read low. Twelve taps, 48–49 cells each, none empty.

The readout is otherwise identical to `run_narma_modal`: same five tones, per-disk
cross-port DFT, signed modes, real and imaginary parts. It has to be, or the
single-disk numbers this is compared against measure a different instrument.

Checkpointing was added for a blunt reason: the mesh is 248×108 against a single
disk's 108×108, so 600 frames is ~55 minutes and this container reboots every
10–30. The previous all-or-nothing cache would have banked nothing, ever.

## Predictions

Measured on the same memory function as everything else, `max_lag` 28.

**Composition holds.** Stage B's r² curve extends past stage A's: cliff at 11 or
beyond, MC materially above 8. Combined A+B features beat A alone. This is the
outcome that makes the layered architecture worth building out.

**Composition fails.** B carries usable signal — it can reconstruct `u` at some
lag with r² > 0.5 — but its cliff is at 8 or below, i.e. B is a delayed copy of
A rather than an integrator of it. Depth then buys nothing on this task and the
NARMA-10 line closes for the architecture, not just for the single disk.

## The inconclusive case, stated in advance

Measured on a 6-frame smoke test, **B/A amplitude ratio is 0.0068**. The link
delivers under 1% of the driven stage's amplitude.

If stage B's features are dominated by numerical noise rather than transferred
signal, MC(B) will be ~0 at every lag — and that would look exactly like
"composition fails" while actually meaning "the measurement had no signal to
work with". So: **if B cannot reconstruct `u` at ANY lag with r² > 0.5, the run
is inconclusive, not negative.** The fix in that case is a stronger link or a
larger drive, not a conclusion about depth.

This is the same failure mode that killed three readout designs — an arm with no
signal reporting a confident verdict — and it is written down here rather than
discovered afterwards.

## Controls

- **`--no-link`** (link material deleted). Stage B must show NO memory of `u`.
  If it does, the coupling is stray dipolar field rather than guided transport,
  and any linked result is measuring the wrong mechanism. This is the control
  that decides whether the link is doing anything.
- **Single disk**, already measured: MC 8.04, cliff 8.
- Both cascade arms use identical readout, splits, seed and frame count, so the
  linked/unlinked difference is the link alone.

## Decision rule, fixed now

1. If B fails the signal gate (no lag with r² > 0.5) → **inconclusive**, report
   the transfer ratio and stop.
2. Else if `no-link` B shows memory of `u` → **invalid**, the link is not the
   mechanism; report and stop.
3. Else if cliff(B) ≥ 11 or cliff(A+B) ≥ 11 → **composition holds**, and the
   next step is the six-seed certification protocol on the cascade.
4. Else → **composition fails**; depth does not extend memory on this geometry.

Stability is not a gate here: λ at the operating point for a four-port disk is
−0.09 full-run and +0.08 late-half, i.e. ~0, the edge of chaos. It is recorded
because a cascade stage spends two of its six ports on links, and that budget is
what earlier work got wrong by measuring at 8 GHz, below the guides' own 11 GHz
cutoff, where they cannot radiate at all.

---

# Result: the control fired, and the timing says why

Both arms, 600 frames, one seed, splits (150, 300, 75), max_lag 28.

| run | arm | dims | MC | window (r² > 0.5) | width | peak @ lag |
|---|---|---|---|---|---|---|
| linked | A | 60 | 5.79 | 0 – 6 | 7 | 1 |
| linked | **B** | 60 | 8.88 | **3 – 12** | 10 | **6** |
| linked | **A+B** | 120 | **10.58** | **0 – 11** | **12** | 1 |
| no-link | A | 60 | 4.59 | 0 – 4 | 5 | 1 |
| no-link | **B** | 60 | 6.30 | **0 – 4** | 5 | **1** |
| no-link | A+B | 120 | 8.65 | 0 – 8 | 9 | 2 |
| *single disk* | *ports* | *60* | *8.04* | *0 – 7* | *8* | *0* |

## The pre-registered rule fires: not certified

Rule 2 says that if no-link stage B shows memory of `u`, the coupling is not
purely guided and the linked result measures a mixed mechanism. No-link B peaks
at r² 0.90. **The rule fires, so this run does not certify composition**, and
that is the ruling regardless of what follows.

It fires for a real reason. The drive is applied only to disk A's cells, so with
the link deleted, disk B can only be reached by stray dipolar field across the
700 nm gap. It is reached: no-link B recovers `u` at r² 0.90.

## What the rule did not anticipate: the two mechanisms separate by timing

No-link B peaks at **lag 1** and dies by lag 4. Linked B peaks at **lag 6** and
holds past lag 12.

Dipolar coupling is a near-field effect and is effectively instantaneous — it
cannot produce a five-frame delay. A frame is 0.20 ns, so linked B's peak sits
~1.0 ns behind the drive, which over 700 nm is ~700 m/s, comfortably below the
~2000 m/s magnon group velocity. A guided component must arrive no faster than
the group velocity allows, and this one does not.

So both mechanisms are present and they are distinguishable: an instantaneous
dipolar background in both arms, and a delayed component that exists only when
the link is there. The memory extension rides on the delayed one — A+B reaches
lag 11 linked against lag 8 unlinked, and the single disk reaches lag 7.

This is the discriminator `run_narma_coupled.py`'s own header identified before
any of this ran: *"What distinguishes them is timing: a guided component must
arrive no faster than the group velocity allows."* The control was built to
subtract the dipolar contribution and cannot, because deleting the link also
deletes that material's own dipolar field. Timing separates them; material
deletion does not.

## A metric that misreported its own arm

`cliff` — the first lag whose r² falls below 0.5 — assumes memory decays
monotonically from lag 0. Stage B does not: it *rises* to a peak at lag 6. The
metric returned `cliff = 1` for linked B, which reads as "no memory" for an arm
that actually spans lags 3 – 12.

Every memory number earlier in this project came from a single driven disk,
where the assumption held and the metric was sound. It broke the first time the
readout sat downstream of a delay. The window (first and last lag above 0.5)
is reported here instead, and `cliff` should not be used on a cascade.

## What would settle it

Vary the separation. Dipolar coupling falls off steeply with distance while a
guided delay grows linearly with it, so measuring stage B's peak lag against
separation separates the two cleanly — no material deletion, no confound. If the
peak lag scales with distance and the dipolar background falls away, the guided
cascade is doing the work and the six-seed certification is worth its compute.

Until that runs, the honest statement is: **a two-stage geometry reaches lag 11
where one disk reaches 7, the extension appears only with the link present and
arrives with a propagation-consistent delay, and the pre-registered control
cannot yet rule out a mixed mechanism.**

---

# Follow-up: separation settles what deletion could not

The control fired because no-link stage B still recovered `u`. Deleting the link
removes that material's own dipolar field along with the guide, so it cannot say
whether the link *is* the mechanism — only that something couples. Separation
can, because the two candidates scale oppositely: a guided arrival grows with
distance, a near-field one does not.

Impulse measurement, 30 mT burst then silence, envelope of each disk's own taps
(`check_link_timing.py`). Cheap enough to sweep — thousands of steps per point
against 600 frames for a task run.

| separation | link | arrival (ns) | implied speed | B/A |
|---|---|---|---|---|
| 500 nm | **linked** | 0.237 | 2110 m/s | **0.396** |
| 700 nm | **linked** | 0.430 | 1628 m/s | **0.383** |
| 1000 nm | **linked** | 0.666 | 1502 m/s | **0.248** |
| 500 nm | no-link | 0.093 | — | 0.056 |
| 700 nm | no-link | 0.014 | — | 0.0043 |
| 1000 nm | no-link | 0.014 | — | 0.0013 |

**Linked arrival grows linearly with separation.** A least-squares fit over the
three points gives **1173 m/s** with a −0.181 ns intercept — the intercept being
the time disk A needs to build up before it launches anything, which is why the
per-point "implied speed" falls from 2110 toward the true slope as the gap
widens. 1173 m/s sits below the ~2000 m/s magnon group velocity, which is the
requirement: a guided component may arrive no *faster* than the group velocity
allows.

**No-link arrival does not move at all** — 0.093, 0.014, 0.014 ns — and its
amplitude collapses by 42× between 500 and 1000 nm. Flat arrival with steeply
falling amplitude is a near-field signature, not propagation.

**At the cascade's own 700 nm separation, the guided path is 90× the dipolar
one** (B/A 0.383 against 0.0043).

## What this does and does not change

It does not retroactively pass the cascade run: the pre-registered control fired
and that ruling stands. What it changes is the interpretation. The control
established that *something* couples without the link; it could not establish
whether that something carried the result. The sweep shows it does not — the
dipolar background is real, physically distinct (instantaneous rather than
propagating), and two orders of magnitude weaker at the operating separation
than the guided path the cascade actually ran on.

So the memory extension measured earlier — stage B spanning lags 3–12 with a
peak at lag 6, A+B reaching lag 11 against 8 unlinked and 7 for a single disk —
is attributable to guided transport through the link. The delay that shifted B's
window is the propagation time this sweep measures directly.

The 1400 nm point is missing: its relaxation failed the drift guard at 5000
steps (8.8e-3 against a 1e-3 tolerance) and is being re-run longer. A wider gap
needs a longer relax, and running a stability-sensitive measurement on a still-
settling ground state is exactly the error the guard exists to prevent.

## Standing conclusion

Depth works, and it is the only one of the four mechanisms tested that does.
Damping, radiation and readout each failed to extend the single disk's ~8-frame
memory; a second stage fed through a guide reaches lag 11. The next step is the
six-seed certification protocol on the cascade — same arms, same three tiers,
same leakage guard — which is what would turn this from a measured mechanism
into a certified result.

---

# How deep can it go? Measured, and shallower than extrapolation predicted

`ChainPortedConfig` / `ChainPortedArray` generalise the two-disk build to N
stages in a line with alternating chirality, so every adjacent pair is opposite
— the only assignment avoiding the Bloch wall that same-chirality neighbours
nucleate in the link. Verified at N = 3, 4, 5: grids 388/528/668 × 108, no empty
taps, and the five-disk ground state passes the drift guard (1593 s to relax).

One impulse into stage 1 measures every stage at once.

| stage | arrival (ns) | peak lag (frames) | amplitude | rel. to stage 1 | **predicted** |
|---|---|---|---|---|---|
| 1 | 0.088 | 1.98 | 8.97e-02 | 1.0000 | 1.0000 |
| 2 | 0.430 | 4.48 | 3.43e-02 | **0.3822** | **0.3830** |
| 3 | 1.253 | 9.48 | 3.66e-03 | 0.0408 | 0.1467 |
| 4 | 2.089 | 13.77 | 4.61e-04 | 0.0051 | 0.0562 |
| 5 | — | 13.62 | 4.99e-06 | 0.0001 | 0.0215 |

## The extrapolation was wrong, and why

Stage 2 lands on prediction to three digits (0.3822 against 0.3830). Stages 3–5
fall progressively far below it: the fitted transfer is **0.092 per hop**, not
the 0.383 measured across a single hop.

The single-hop number does not extrapolate because a terminal stage and an
interior stage are different objects. In the two-disk build, stage B is an
endpoint and the arriving wave is fully deposited. In a chain, every interior
stage is a scattering node carrying **four free radiating ports, each with an
absorbing taper** — the very ports that make a stage readable. They drain the
signal passing through. Depth and readability trade against each other per
stage, and no two-disk measurement can show it.

The delay per hop also grows: **0.652 ns (3.26 frames)** fitted across depth,
against 0.430 ns for a single hop, consistent with each interior disk holding
and re-radiating rather than simply relaying.

## Where the floor actually is

Comparing each stage against the dipolar crosstalk reaching it *directly* from
the driven disk, using the separation sweep's own falloff — which fits
**1/r^3.3**, against 1/r³ for an ideal dipole, an independent confirmation that
the no-link coupling is what it was called:

| stage | gap | guided | dipolar floor | margin |
|---|---|---|---|---|
| 2 | 700 nm | 0.382 | 0.00427 | 89× |
| 3 | 1400 nm | 0.0408 | 0.00044 | **92×** |
| 4 | 2100 nm | 0.0051 | 0.00012 | **44×** |
| 5 | 2800 nm | 0.00006 | 0.000046 | **1.2×** |

**Depth 4 is the ceiling.** Stage 5 sits at its own crosstalk floor — whatever
arrives through four hops is no stronger than what leaks straight across the
substrate, so a fifth stage is not part of the cascade in any meaningful sense.

Absolute level is the other limit and this simulation cannot rule on it: stage 4
carries 0.5% of stage 1 and stage 5 carries 0.006%, with no thermal noise
modelled. A real detector's noise floor may bind before the crosstalk floor
does.

## What this means for the architecture

**Three stages is the operating point for NARMA-10.** Stage 3 peaks at lag
**9.48** — essentially on `u[n-10]`, the term the whole task turns on — while
holding a 92× crosstalk margin. Stage 4 extends coverage to lag 13.8 at 44×, so
a four-stage chain spans lags 0–14 across its stages and is worth building if
the task needs the tail.

And the input stage can be treated as expendable, which was the architectural
point. Stage 1 carries 200× the amplitude of the deepest usable stage, so it can
be driven hard enough to annihilate its own state each frame without starving
what is downstream. Its job is transduction; the lags that matter live in stages
3 and 4, which never see the drive directly. What the measurement adds is the
budget: that division of labour works over three or four stages, not over the
six that a geometric extrapolation from one hop would have promised.

---

# Can the input stage annihilate? Not while staying usable

The architecture's appeal is a division of labour: if the deep stages hold the
memory, the input stage is free to destroy its own state every frame, because
its job is transduction. Stage 1 carries 200× the amplitude of the deepest
usable stage, so there is signal budget for it. The question is whether the disk
has a drive that both **fires** (reverses its core) and leaves the dynamics
**usable** (λ < 0).

Measured on a single ported disk, sweeping amplitude at two bands:

| band | amp | core (end) | flips | λ/ns | verdict |
|---|---|---|---|---|---|
| 12 GHz | 30 mT | +0.347 | 0 | **−0.596** | stable, no fire |
| 12 GHz | 60 mT | −0.277 | **7** | +5.545 | fires, chaotic |
| 12 GHz | 100 mT | −0.003 | **4** | +9.417 | fires, chaotic |
| 12 GHz | 150 mT | +0.042 | 0 | +9.771 | scrambled |
| 12 GHz | 220 mT | −0.021 | 0 | +9.730 | scrambled |
| 0.5 GHz | 5 mT | +0.883 | 0 | **−1.441** | stable, no fire |
| 0.5 GHz | 10 mT | +0.868 | 0 | **−1.429** | stable, no fire |
| 0.5 GHz | 20 mT | +0.764 | 0 | +0.349 | no fire |
| 0.5 GHz | 40 mT | −0.178 | 0 | +2.092 | no fire |

**No amplitude in either band both fires and stays stable.** At the 12 GHz
carrier the core reverses from 60 mT, but λ is +5.5/ns there — over a 0.20 ns
frame that is ~3× divergence *per frame*, so stage 1's output stops being a
reproducible function of the input. Nothing downstream can recover information
that sensitivity to initial conditions has already destroyed, however much
memory the deep stages have.

## What this does not settle

The 0.5 GHz arm was a **guess** at the gyrotropic band, not a measurement. The
gyrotropic frequency of this specific disk has never been measured here, and the
0.5 GHz sweep shows the core barely moving (0.883 → 0.764 up to 20 mT) which is
consistent with driving off-resonance. So this rules out core annihilation at
the operating carrier, and does **not** rule out a cheap reversal at the true
gyrotropic resonance. Measuring that frequency — ring down the core position
after a small in-plane pulse and take its spectrum — is the prerequisite for
any further attempt, and it is cheap.

## The reframing the data supports

Core reversal is not the only annihilating nonlinearity available, and it may
not be the one worth chasing. This disk already has a *threshold* nonlinearity
that does not destroy state: three-magnon splitting above its power threshold,
which is what produces the 23.4× AB/BA discrimination this project measured. It
redistributes energy among modes rather than erasing the core, so mode
populations — and therefore memory — survive it.

That preserves the architectural idea while dropping the mechanism. The input
stage can be driven hard into strong nonlinearity, be lossy, and hold almost no
memory of its own, with the lags that matter living in stages 3 and 4. What it
cannot be, on this device at this carrier, is literally spiking.

---

# How wide, and does width pay?

Width has an obvious confound: more disks means more features, and more features
fit better whether or not they carry more information. Ridge with a fixed budget
is the control — hold the feature count constant and change only how it is
spent.

## A single disk saturates its own readout

Effective rank, dimensions carrying 99% of variance:

| features | effective rank |
|---|---|
| one stage, 60 features | **16** of 60 |
| two stages, 120 features | 22 of 120 |

One disk read through six ports and five tones is a **16-dimensional** object.
The shipped 60-feature readout is roughly four times oversampled on it, which is
consistent with the earlier finding that two tone pairs sit at 0.993 and 0.999
canonical correlation. Extra feature budget spent on one disk buys nothing,
because there is nothing left there to read.

## The same budget spent across disks does pay

Sixty features, allocated differently, each stage reduced on its own so a quiet
stage is never out-voted by a loud one:

| allocation | dims | MC | window |
|---|---|---|---|
| all from stage 1 | 60 | 5.79 | 0 – 6 |
| 30 stage 1 + 30 stage 2 | 60 | 7.36 | 1 – 7 |
| 20 stage 1 + 40 stage 2 | 60 | **8.41** | **1 – 10** |
| all from stage 2 | 60 | **8.88** | **3 – 12** |
| *(120, both stages)* | *120* | *10.58* | *0 – 11* |

Monotonic in how much budget moves to the deeper stage, at constant parameter
count. The gain is information.

A plain PCA of the 120 down to 60 does NOT show this — it scores 6.66, barely
above stage 1 alone — because PCA ranks directions by variance and stage 2's
signal is 0.38× stage 1's amplitude, so its directions are discarded for being
quiet rather than uninformative. That is the same trap that made a 45-tone
superset score below the five tones it contained. Reduce each stage separately
and the effect is plain.

## What this says about width, and what it does not

The measured gain comes from **diversity of lag coverage**, not from disk count:
stage 1 covers lags 0–6, stage 2 covers 3–12, and they are different because the
link delays one relative to the other. Serial stages are diverse automatically.

That is exactly what **parallel** width would lack. Two disks at the same depth,
driven by the same input, sit at the same point in the delay line and see the
same history; their states are copies up to geometry. This project has already
paid for that lesson once — the Thiele array carried "twelve hand-tuned dampings
that `alpha_spread` was faking", i.e. diversity had to be manufactured because
identical parallel nodes supplied none.

So the prediction is that parallel width is nearly free of benefit unless the
disks are made genuinely different — detuned radius, different port angles, or
different coupling phase — while serial depth buys diversity for nothing. It is
a prediction, not a result: the parallel case has not been run. The cheap test
is the canonical correlation between two same-depth disks' feature blocks under
a shared drive; near 1 means redundancy and no width benefit, and it costs an
impulse rather than a task run.

## Practical shape

Combining this with the depth budget — four stages before the guided signal
reaches the crosstalk floor, three stages putting the deepest peak on lag 9.5 —
the indicated device is **narrow and serial**: three or four stages of one disk
each, with the feature budget weighted toward the deeper stages, rather than a
wide bank at any single depth.

## Is a parallel input layer worth it? Only if the feeds are diverse

Two reasons a wide input layer might pay, and they need separating because one
is quantifiable from measurements already in hand.

**As a power combiner: no.** Depth is amplitude-limited — transfer is 0.092 per
hop, an ~11× loss per stage. Parallel input disks summing into stage 2 buy depth
only logarithmically, so **one extra usable stage costs 11 parallel input disks**
if they add coherently and 121 if they do not. That is not a trade worth making.

**As diversity: yes, but only with diverse feeds.** Measured on two ported disks
with the link deleted, both driven, 30 mT impulse at 12 GHz:

| drive | canonical correlations, disk A vs B | effective rank (12 features) |
|---|---|---|
| same phase | 1.00 1.00 1.00 1.00 1.00 1.00 | **5** |
| 90° offset | 0.90 0.87 0.63 0.60 0.39 0.14 | **8** |
| *(one disk, 6 features)* | — | *5* |

Identically driven, the two disks are **exactly** redundant: every canonical
correlation is 1.000, B/A rms is 1.0000, and the pair carries the same five
dimensions one disk does. Doubling the input layer adds nothing. Note this holds
even though the disks have opposite chirality — the mirror asymmetry is
invisible through this readout.

Offsetting the feed phase by 90° breaks it: the pair carries **8** effective
dimensions against 5 for a single disk. Same disks, same drive amplitude, same
geometry — only the feed line length differs, which is free.

**Caveat on what this measures.** Effective rank is a necessary condition, not a
sufficient one: redundant features certainly cannot help, but new dimensions are
not automatically *useful* dimensions for a task. Whether the extra three carry
lag information NARMA-10 needs, or merely carry the phase offset itself, needs a
task run. The prediction is that they help less than depth does, because they
sit at the same point in the delay line and depth is what supplied the lag
coverage.

**Practical reading.** Spread the input layer only with phase-diverse feeds, and
expect a modest widening of the state rather than the extra memory that depth
buys. The indicated device remains narrow and serial, with any width spent on
the input stage rather than the deep ones — the deep stages are where lag
coverage lives, and lag coverage is what the task was short of.
