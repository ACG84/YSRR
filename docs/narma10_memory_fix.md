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
