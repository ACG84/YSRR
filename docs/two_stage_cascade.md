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
