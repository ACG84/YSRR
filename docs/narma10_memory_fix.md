# Can NARMA-10 be made to work? A damping sweep

**Written before any data at the new damping values exists.** Predictions are
numeric and falsifiable, and the decision rule for what happens next is fixed
here rather than after seeing the numbers.

## The diagnosis this follows from

The certification failed, and failed with a specific cause rather than a vague
one. From `docs/narma10_certification.md`:

- device test NMSE **0.625**, best linear filter on the same input **0.157**
- Jaeger MC **8.12**, r² crossing 0.5 at lag **8.0**
- equivalent depth **11.0** — the device performs like an 11-lag filter while
  remembering 8, so its nonlinearity does real work, just not enough
- a linear filter goes **0.71 → 0.37** between 10 and 11 lags

NARMA-10's recursion contains `u[n-10]`. The device's memory ends at 8. It is
one lag short of the only place on the curve where performance lives.

## The lever

Memory in frames is set by ring-down against frame duration:

    MC ≈ τ / T_frame = 1 / (2 π α N_cycles)

At the shipped α = 0.008 with N_cycles = 2.40 that predicts **8.29** against
**8.12** measured — the model is good to 2%, which is why it is worth trusting
one extrapolation.

Two knobs move `α N_cycles`. Lowering **α** is the clean one: it buys memory
without touching the readout. Lowering **N_cycles** (shorter frame, or lower
carrier) buys the same memory and makes runs cheaper, but the single-frame
lock-in already integrates only 2.4 carrier cycles, and fewer would degrade the
readout further. So this sweep varies α alone.

This varies damping and nothing else. A real material change would move `Ms` and
`A` as well, so these are idealised points on one axis, not four materials.
For scale: 0.008 is permalloy-like, 0.004 CoFeB-like, 0.002 needs an optimised
low-damping alloy.

## Prediction 1 — memory (the model check)

| α | predicted MC (frames) | predicted cliff | reaches `u[n-10]`? |
|---|---|---|---|
| 0.008 (control, measured) | 8.29 | ~8 | no — **measured 8.12, cliff 8** |
| 0.006 | 11.05 | ~11 | marginal |
| 0.004 | 16.58 | ~17 | yes |
| 0.002 | 33.16 | ~33 | yes |

Falsified if measured MC departs from `1/(2π α N_cycles)` by more than ~20% at
any point, or if the ratio drifts systematically with α — either would mean the
ring-down model stops governing memory once damping is low, and the whole
premise of the fix is wrong.

## Prediction 2 — the part that actually decides

Enough memory is necessary. Whether it is *sufficient* is the real question, and
there are two honest answers that predict different numbers.

**H1, the additive-offset reading.** Equivalent depth tracks memory at a fixed
offset (+3, from 8 → 11 today). At α = 0.004 that gives equivalent depth ≈ 20,
and a 20-lag linear filter scores ≈ 0.165. The device would land near **0.16**,
which is a tie with `linear_best` (0.157), not a win. Tier 2 still fails, and
the conclusion becomes "this device cannot beat a linear filter on this task at
any damping" — a much stronger negative than the one we have.

**H2, the unlocked-nonlinearity reading.** The +3 offset is not a constant; it
is what the nonlinearity can contribute when the product term is *out of reach*.
Once `u[n-10]` is inside the memory window the device can form NARMA-10's
`u[n-10]·Σy` product, which no linear filter of any depth can. The offset should
then jump rather than creep, and the device should go **well below 0.157**.
Tier 2 passes.

H1 and H2 differ by roughly a factor of two in NMSE at α = 0.004. This sweep
separates them, and that is its point — not "does it get better".

## Risks that would invalidate the run rather than answer it

Lower damping is not free, and each of these is checked rather than assumed.

- **Washout.** A reservoir needs its initial state forgotten before scoring. At
  MC 33 the shipped 150-frame washout is only ~4.5 τ. Washout is held at ≥ 5×
  measured MC; if α = 0.002 needs more, its splits change and that is recorded.
- **Echo state property.** If the state stops responding to input, memory is
  infinite and useless. Flagged by lag-1 feature autocorrelation → 1.
- **Energy accumulation.** The drive is continuous and the ring-down now spans
  many frames. Watch for feature magnitudes growing without bound, or the core
  `mz` moving — the vortex being expelled is a different device, not a
  longer-memory one.
- **Chaos.** This project has already seen coupled disks produce chaos rather
  than computation near a threshold. Divergence between seeds sharing an input
  would show it.

## Decision rule, fixed now

Screen first: one seed, 700 frames, α ∈ {0.006, 0.004, 0.002}, against the
existing α = 0.008 run as control. Measure MC and cliff position only — no
NARMA-10 scoring, so there is nothing to tune against.

Then commit to **one** α for the full six-seed certification: the largest α
(least exotic material) whose measured cliff exceeds 12, provided its lag-1
autocorrelation stays below 0.98 and its core `mz` matches the control. Largest,
not best-performing — performance is not measured at screening time, so it
cannot influence the choice.

If no α clears the cliff, that is the answer: the geometry cannot reach
NARMA-10's product term at any physically sensible damping, and the certification
stands as a property of the design rather than of one parameter.

The full run then re-uses `docs/narma10_certification.md`'s protocol unchanged —
same arms, same three tiers, same leakage guard, same six seeds. A second
certification is a second pre-registration, not a re-scoring of this one.

---

# Result: the model is falsified. Memory is not damping-limited.

One seed, 800 frames, α ∈ {0.006, 0.004, 0.002}, against the certification's
α = 0.008 run as control. Measured with `screen_memory.py` at max_lag 28.

| α | predicted MC | measured MC | meas/pred | cliff | reaches `u[n-10]`? |
|---|---|---|---|---|---|
| 0.008 (control) | 8.29 | **8.27** | 1.00 | 8 | no |
| 0.006 | 11.05 | **8.29** | 0.75 | 8 | no |
| 0.004 | 16.58 | **8.41** | 0.51 | 8 | no |
| 0.002 | 33.16 | **8.61** | 0.26 | 8 | no |

Both pre-registered falsification conditions fired: the departure exceeds 20%,
and the ratio drifts systematically with α (1.00 → 0.75 → 0.51 → 0.26). A
fourfold reduction in Gilbert damping bought **0.34 frames** of memory where the
model demanded a fourfold increase.

The r² curves are nearly identical across the sweep — same plateau to lag 7,
same collapse at lag 8, same partial revival around lag 11 — so this is not a
small effect being missed. Nothing about the memory changed.

The damping did change. The run banners record ring-down 1.66 → 2.21 → 3.32 →
6.63 ns, and the damping field's minimum tracks α exactly. So the parameter
reached the physics; the physics did not care.

## Why the control agreed, and what actually sets the memory

The model predicted 8.29 against 8.12 measured at α = 0.008, and that 2%
agreement is what made the extrapolation look safe. It was a coincidence. The
same field that carries α has a **fixed maximum of 0.5** in every run — the
absorbing tapers — and `absorb_alpha` was never a swept variable.

That points at the mechanism. Above the guides' 11 GHz cutoff the ports
propagate, so a driven disk radiates its state into the guides, where the tapers
absorb it. That loss channel is set by geometry and taper strength, both
untouched by α. Bulk Gilbert damping is a minority contributor, so scaling it
moves the total decay hardly at all.

The independent evidence lines up. Below cutoff (8 GHz) the guides are
evanescent and cannot shed energy, and the renormalised Lyapunov exponent is
positive at every port count; above cutoff (12 GHz) it drops sharply. Radiation
into the ports is doing the work in both measurements.

## What this retires, and what it costs

The certification's diagnosis — "8 frames of memory against a task needing 11,
so the fix is longer ring-down (lower damping, or fewer carrier cycles per
frame)" — named the right deficit and the wrong cure. Lower damping is not a
route to longer memory in this geometry at any physically sensible value.

The pre-registered decision rule settles the rest: *"If no α clears the cliff,
that is the answer: the geometry cannot reach NARMA-10's product term at any
physically sensible damping, and the certification stands as a property of the
design rather than of one parameter."* No α cleared it. **NOT CERTIFIED stands,
and it is now a statement about the design.**

## The trade-off this exposes

If memory is radiation-limited, the ports that make the device readable are the
same channel that drains its state. Memory and readout are then in direct
competition, and that is a sharper and more useful constraint than damping ever
was: six ports were chosen for mode selectivity and for the aperture the
stability argument wanted, and each one is a leak.

That is a measurement, not a conclusion: it predicts memory should rise as port
count or taper absorption falls, and it should be tested by measuring the free
decay directly rather than by inferring it from another 800-frame task run.

## The radiation hypothesis fails too

Tested directly by free decay — drive, cut the drive, fit the deviation energy
inside the disk (`check_ringdown.py`), 20 mT at 12 GHz:

| α | ports | τ (ns) | τ (frames) | model τ | meas/model |
|---|---|---|---|---|---|
| 0.008 | 6 | 1.29 | 6.5 | 1.66 | 0.78 |
| 0.002 | 6 | 1.92 | 9.6 | 6.63 | 0.29 |
| 0.008 | 2 | 0.96 | 4.8 | 1.66 | 0.58 |
| 0.008 | 1 | 1.62 | 8.1 | 1.66 | 0.98 |

Damping-limited predicts τ ∝ 1/α and no port dependence. Radiation-limited
predicts τ flat in α and rising as ports are removed. **Neither happens.**
Fourfold less damping lengthens τ by 1.49×, not 4×; and removing ports moves τ
non-monotonically (6 → 1.29, 2 → 0.96, 1 → 1.62), which is not a trend at all.

τ is 1–2 ns — 5 to 10 frames — in every configuration tested, and MC is pinned
near 8.3 in every configuration tested. Something sets that scale, and it is
neither of the two knobs.

## The remaining candidate: the readout, not the disk

What has never varied across any of these runs is the readout. Every MC number
in this project comes from 60 features: six ports × five lock-in tones ×
(re, im), one vector per frame. MC measures memory *visible in those features*,
not memory in the magnetisation, and the two are only the same if the readout
projects the state faithfully.

Three independent observations now point the same way. The 60 features carry
only ~20 dimensions of variance, and two tone pairs are near-degenerate
(12↔24 GHz at 0.993, 10.3↔13.7 GHz at 0.999). Tier 3 of the certification found
that appending all 60 to a linear filter improves nothing. And MC is invariant
at 8.3 across a fourfold damping change and a sixfold change in port count —
which is what a saturated measurement looks like when the instrument, not the
sample, is the limit.

This is a hypothesis with a cheap decisive test, and it is not yet evidence.
Run a short sweep that saves the raw per-frame port waveform rather than its
five lock-in projections, and compute MC from both. If the waveform carries
substantially more memory than the 60 features, the 8-lag ceiling belongs to the
readout and is fixable without touching the physics. If it carries the same, the
ceiling is in the magnetisation and the device genuinely forgets at 8 frames.

Until that is measured, the honest statement is narrower than either failed
hypothesis: **the device's usable memory is ~8 frames, it is not set by Gilbert
damping, it is not set by radiation into the ports, and the mechanism is
unidentified.**
