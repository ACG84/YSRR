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
