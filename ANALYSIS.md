# Analyses

One-off studies that are not runs and not code changes. Newest first.
Runs live in [`RUNS.md`](RUNS.md), datasets in [`COLLECTIONS.md`](COLLECTIONS.md),
source edits in [`CHANGES.md`](CHANGES.md).

---

## 2026-08-31 — Learned state equivalence: measured, and closed

Logged retroactively on 2026-08-31. The probe shipped in commit `b3b458e` (08-27 late) and
the encoder sweep in `ad994ec` (08-31); **neither was recorded in the logbook at the time.**
Code in [`CHANGES.md`](CHANGES.md) items 37-38. Raw numbers:
`outputs/analysis/state_equivalence.json`, `outputs/analysis/state_equivalence_training.json`,
figure `outputs/plots/state_equivalence_tradeoff.png`.

**The proposal under test.** The 08-27 post-mortem concluded that masking embodiment identity
cannot work, and named as the surviving option "sharing only the already-aligned parts —
shared object/goal encoder, per-embodiment proprioception and action heads". This is the
measurement of that option. It trains no policy.

### Step 1 — the three preconditions (`probe_state_equivalence.py`, size 400k)

| | grasp | rotation |
|---|---|---|
| task subspace (auto-selected, per-term sep < 0.70) | `object_pos`, `object_goal_pos` (6 d) | `object_quat`, `object_angvel`, `goal_quat` (11 d) |
| **Q1** NN median, Allegro→LEAP (cross) | 0.486 | 1.466 |
| **Q1** NN median, Allegro→Allegro (self, disjoint episodes) | 0.554 | 0.838 |
| **Q1** cross/self ratio | **0.88** | **1.75** |
| "unrelated" scale √d | 2.45 | 3.32 |
| **Q2** embodiment sep. from task subspace (MLP bal. acc) | 0.623 | **0.964** |
| **Q2** from hand-picked pure-object terms | 0.719 | **0.964** |
| **Q2** from full observation | 1.000 | 1.000 |
| **Q3** R² Allegro action → matched LEAP action | +0.370 | +0.065 |
| **Q3** R² task state → matched LEAP action | **+0.606** | **+0.784** |
| **Q3** R² action + task state → LEAP action | +0.549 | +0.567 |

**Q1 passes.** The equivalence classes are populated. For grasp, cross-embodiment matches are
*tighter* than the within-embodiment floor (ratio 0.88) — for any Allegro state there is a
LEAP state at essentially the same object pose. Rotation is looser (1.75) but still far below
the unrelated scale. So there is no shortage of pairs to align. This is the only one of the
three that passes.

**Q2 fails, and fails differently per family.** Rotation's object/goal terms are individually
at chance (0.49-0.53, from item 30) yet **jointly 96% separable**. Invariance therefore cannot
be obtained by *selecting* terms — it has to be enforced against the joint distribution. Even
grasp's hand-picked pure-object subspace leaks at 0.719. Anyone tempted to build a "shared
object encoder" by picking the object-referenced terms should read this row first: that
construction does not produce a shared representation.

**Q3 fails, and it is the one that closes the idea.** The question is not whether the Allegro
action predicts the LEAP action in absolute terms, but whether it beats the matched *task
state* — i.e. whether the action correspondence carries information the state does not. It
does not, in either family, and rotation is not close (+0.065 vs +0.784). At matched object
pose, knowing what Allegro did tells you nothing extra about what LEAP did. A shared latent
action space has nothing to be shared.

*Honest caveat on the third row:* `action + task state` scoring **below** `task state` alone
(0.549 vs 0.606; 0.567 vs 0.784) cannot be a real information effect — extra inputs cannot
destroy information. It is finite-sample: the regression runs on the tightest quartile
(n≈2000, split 50/50) with a 128-unit MLP, so the wider input overfits. The conclusion rests
on `action < task state`, which is robust; do not quote the combined column as a finding.

### Step 2 — can an encoder be forced to be invariant? (`train_state_equivalence.py`)

Encoder trained with task-decode + gradient-reversal adversary (strength λ) + cross-embodiment
alignment on matched pairs. λ ∈ {1, 10, 100}, z-dim 32, 400 steps, size 400k, seed 0.
**Invariance scored by a probe trained after the encoder froze**, on held-out episodes.

| family | λ | adversary's OWN train acc | **fresh probe bal. acc** | task R² |
|---|---|---|---|---|
| grasp | 1 | 0.754 | **0.9993** | 0.960 |
| grasp | 10 | 0.741 | **0.9965** | 0.883 |
| grasp | 100 | 0.741 | **0.9993** | 0.835 |
| rotation | 1 | 0.728 | **0.9837** | 0.947 |
| rotation | 10 | 0.670 | **0.9767** | 0.884 |
| rotation | 100 | **0.595** | **0.9857** | 0.783 |

**Adversarial invariance does not work on this pair.** Raising λ a hundredfold buys no
invariance whatever — the fresh probe stays at 0.98-0.999 balanced accuracy, and the movement
that does occur is not even monotone in λ — while costing real task information (grasp
0.960 → 0.835, rotation 0.947 → 0.783). The trade-off curve is essentially **vertical**: you
pay and get nothing. Alignment on matched pairs, which Q1 showed are plentiful and tight, does
not rescue it either.

**And the trap is right there in the table.** The adversary's own accuracy *does* fall with λ,
most visibly rotation 0.728 → 0.595, heading toward chance. Read that column and you would
report invariance working. The fresh probe on the same encoder says 0.986. This is the third
time in this project that the honest evaluator and the convenient one disagreed, and it is
why the fresh-probe protocol is non-negotiable — see also the linear-vs-nonlinear removability
error in the 08-27 entry below.

### Where this leaves cross-embodiment transfer

The 08-27 entry left two surviving options. **Measurement has now closed the first one.**

- ~~Shared object/goal encoder with per-embodiment heads~~ — closed. Q2: the object/goal
  subspace is not embodiment-invariant (rotation 0.964), and invariance cannot be enforced
  (fresh probe ≥ 0.977 at λ=100). Q3: there is no action correspondence to share even given
  perfectly matched task states.
- **Sequential rather than joint transfer** — still untested, and now the only option on the
  table that has not been ruled out by a measurement. Pretrain on one body, fine-tune on the
  target; nothing measured here bears on it, because it never requires a shared representation
  to exist at one instant.

Net: every route that depends on a *simultaneously shared* representation between Allegro and
LEAP has been measured and found closed — separability (AUC 1.000), non-removability by affine
maps (MLP 0.999), timestep gating (source → exactly 0.000), term selection (0.964), and
adversarial invariance (0.98-0.999). That is five independent closures pointing the same way.
Stop proposing masking, alignment and invariance variants; the next experiment worth compute is
sequential transfer.

---

## 2026-08-27 (evening) — Ambient diffusion: the hypothesis is falsified, and why

Design, job accounting and the pre-registered predictions are in the morning entry below
and in [`RUNS.md`](RUNS.md). This is the result.

### Prediction vs outcome

Predicted: an interior peak in sigma*, strongest at N=400k. Observed: **the interior is a
valley — worse than BOTH endpoints.** Grasp, Allegro, seed 0, all points at 100 rollouts:

| N | sigma*=0 (naive mix) | 50 | 75 | 90 | sigma*=100 (Allegro-only) |
|---|---|---|---|---|---|
| 10k | **0.370** | 0.080 | 0.080 | 0.060 | **0.370** |
| 50k | **0.980** | 0.610 | 0.480 | 0.460 | **0.970** |
| 100k | 0.980 | 0.820 | 0.860 | 0.850 | **0.990** |
| 400k | 0.960 | 0.950 | 0.960 | 0.960 | **0.990** |

Both endpoints ~0.97; the interior bottoms out at 0.46. mean(interior) - sigma*=0:

| family | seed 0 | seed 1 | best interior beat naive mixing |
|---|---|---|---|
| grasp | **-0.225** | **-0.130** | **0/4 budgets, both seeds** |
| rotation | +0.067 | **+0.054** | 2/4 and 2/4 |

> [!IMPORTANT]
> **The rotation row was corrected on 2026-08-31.** It originally read
> `seed 1 = -0.143`, `3/4 and 1/3 — signs disagree`, with the conclusion "rotation is
> unresolved, the two seeds disagree in sign". That was an artifact of an **incomplete row
> set**, not a real disagreement: seed 1 was averaged over 3 budgets because
> `N=400k, sigma*=0` had not finished, *and* every seed-1 `sigma*=100` cell was blank
> because of the eval-row schema bug fixed in [`CHANGES.md`](CHANGES.md) item 36. With the
> grid complete at 40/40 the missing cell lands at 1.500 against an interior mean of 2.147
> (+0.647), moving the seed-1 average to **+0.054** — the same sign as seed 0.
>
> Corrected reading: **grasp is decisively negative and replicated; rotation is null to
> slightly positive**, well inside its 0.196 per-run noise sd. Neither family supports the
> pre-registered interior peak.
>
> This is the second time this project drew a conclusion from a partially-populated grid.
> Both times the error was in the same direction: reporting a seed disagreement that the
> completed grid did not contain.

**Grasp is decisively negative.** The 50k effects (-0.46, -0.22) are 4-8x the measured
per-run sd of 0.056, and no interior point beat naive mixing in any of the 8 budget-seed
cells.

**Rotation is null, not contradictory.** Both seeds land slightly positive (+0.067, +0.054)
against a per-run sd of 0.196, so the interior is statistically indistinguishable from naive
mixing. The "best interior beat naive mixing in 2/4 budgets" at both seeds is what a null
looks like at this noise level, not evidence of an effect. A third seed would sharpen the
interval but is not needed to reject the pre-registered peak.

### The control is the real finding

LEAP (the source embodiment), both seeds, every budget:

| sigma* | 0 | 50 | 75 | 90 |
|---|---|---|---|---|
| grasp LEAP | 1.000 | 0.82-0.94 | **0.000** | **0.000** |
| rotation LEAP | 2.3-2.6 | **0.00-0.02** | 0.000 | 0.000 |

Exactly zero, at 100 rollouts, across 8 runs per family and both seeds. So the gating
mechanism works precisely as designed, and the lesson is not about ambient diffusion at
all:

**The low-noise denoising steps are not "polish" — they are where embodiment identity is
realised.** An embodiment trained only on the coarse end of the schedule is not degraded,
it is completely non-functional. That is a much stronger statement than the coarse/fine
hypothesis assumed, and it is the one result here needing no replication (floor effect).

### Why the target degrades is still open

Allegro retains full-schedule data at every sigma*, so its collapse from 0.98 to 0.46 is
not explained by the control. The obvious candidate fails: Allegro's share of gradient
weight is 11% at sigma*=0, 55% at sigma*=90, 100% at sigma*=100, while performance goes
0.98 -> 0.46 -> 0.97. Non-monotone in the share, so it is not a weighting effect.

A plausible but **untested** mechanism: in a diffusion trajectory the high-noise steps
select which mode the sample lands in and the low-noise steps only refine within it. At
sigma*=90 the high-noise function is roughly half LEAP-weighted, so Allegro sampling may
be pushed toward LEAP's action mode with no authority left to redirect it. At sigma*=0
that does not happen because LEAP is present at every t, so the network conditions on
observation-identity consistently across the whole schedule. Testing this needs a probe of
where along the trajectory the two hands' predictions diverge.

---

## 2026-08-27 (evening) — Why cross-embodiment sharing cannot work by masking

Prompted by the question "perhaps the state distributions are so different that the
diffusion easily distinguishes which agent is executing." They are, and the consequences
run deeper than the sweep.

### The observation identifies the embodiment perfectly, at every timestep

`scripts/measure_state_separability.py`. Splits are by **episode**, never by step —
consecutive frames are near-duplicates and a step-level split inflates everything toward
1.0. 200 episodes per hand, ~95-100k steps.

| | linear acc | linear AUC | MLP acc |
|---|---|---|---|
| grasp obs | **1.0000** | 1.0000 | 1.0000 |
| rotation obs | **1.0000** | 1.0000 | 1.0000 |
| grasp action | 0.9980 | 1.0000 | 1.0000 |
| rotation action | 0.9871 | 0.9984 | 0.9995 |

A *linear* probe on a single observation vector is perfect. And the observation is **never
noised** — it enters `global_cond` at full fidelity at every reverse step. Hypothetically
noising it, versus the action:

| t | 0 | 25 | 50 | 75 | 90 | 99 |
|---|---|---|---|---|---|---|
| grasp obs | 1.000 | 1.000 | 0.995 | 0.856 | 0.628 | 0.500 |
| grasp action | 1.000 | 0.910 | 0.715 | 0.586 | 0.522 | 0.499 |
| rotation obs | 1.000 | 0.999 | 0.945 | 0.751 | 0.580 | 0.499 |
| rotation action | 0.999 | 0.779 | 0.627 | 0.548 | 0.513 | 0.501 |

Actions become ambiguous by t~75-90; observations would need t~99. In practice the
observation stays at 1.000 forever.

**This voids the ambient premise outright.** Ambient Diffusion's guarantee requires the
sample to be unidentifiable at high noise. Here identity is perfectly recoverable at every
timestep including t=99. The morning entry called the required condition "weaker and
empirically checkable but not guaranteed". It is now checked and it fails maximally — no
timestep is ever in the regime the theory needs, so the sweep's interior had no benefit
available to buy and could only lose.

### It explains the one-hot null result

The conditioning ablation (2026-08-26) found one-hot embodiment labels capacity-neutral in
both seeds with no mechanism offered. There is one now: the observation already identifies
the hand linearly and perfectly, so an explicit label was **strictly redundant**. Two
separate findings, one cause.

Grasp `hand_dof_pos[9]` is a de facto embodiment ID sitting in the observation:

    Allegro: +1.1973 +/- 0.1401      LEAP: -0.5192 +/- 0.1282     (12.8 within-std gaps)

### The difference is NOT removable by normalisation — and the linear probe lied

`scripts/characterize_state_difference.py`. Per-hand alignment using train-only statistics:

| alignment | linear AUC | MLP AUC | MLP acc |
|---|---|---|---|
| raw | 1.000 | 1.000 | 1.000 |
| centred (per-dim mean) | **0.509** | **1.000** | **0.9993** |
| standardised (mean+std) | **0.503** | **1.000** | **0.9989** |
| whitened (per-hand ZCA) | 0.524 | — | — |

The linear column says the difference is a pure rigid offset and that per-embodiment
normalisation would make the data shareable. **That conclusion is false.** A small MLP
still separates the hands at 0.999 accuracy after full standardisation. The offset carried
only the *linear* part; underneath is a genuine joint-distribution difference no affine map
touches.

Consequence: **you cannot construct the indistinguishable regime by preprocessing.** The
natural follow-up — per-embodiment normalisation plus an explicit one-hot, so identity is a
clean label instead of an entangled offset — was about to be proposed and is dead on
arrival.

This also qualifies the earlier claim in this file, "Allegro vs LEAP: the shift is almost
all rigid offset" (centred per-dimension W1 falling to 0.31-0.44). That statement remains
true *about per-dimension marginals*, but it does **not** imply the distributions become
interchangeable after alignment. Per-dim W1 is blind to cross-dimension structure, which is
exactly where the surviving signal lives.

### How concentrated the difference is

- **One dimension suffices.** Top-1 discriminative dim alone: grasp AUC 0.9998
  (`hand_dof_pos[9]`), rotation 0.9993 (`object_pos[2]`).
- **Disjoint support:** 6/115 grasp dims and 3/69 rotation dims have <1% marginal overlap;
  `hand_root_quat[2]` and `keypoint_pos_rel[1,2,7,9]` are at exactly 0.0000.
- Rotation `object_pos[2]` reaching 0.9993 means the *object's* position betrays the hand —
  the ~9 cm palm offset already noted in this file.
- Least informative terms sit at chance: grasp `object_goal_pos` 0.482, rotation
  `goal_quat` 0.493. The task is shared; the body is not.

### Two methodology traps worth remembering

1. **Below-chance accuracy is a bug signal, not a result.** The first removability run
   reported 0.386 accuracy after centring. Episode-level splits do not balance step counts,
   so a model with no signal that predicts one class scores the majority fraction. Switched
   to balanced accuracy and AUC, which are imbalance-invariant, plus class weights in the
   fit.
2. **Never conclude "removable" from a linear probe alone.** The linear result was clean,
   interpretable, and wrong. It would have produced a confident recommendation to change the
   normaliser. One nonlinear check reversed it.

### Where this leaves cross-embodiment transfer here

Any scheme that works by *hiding* which body produced a sample is unavailable — the
observation encodes the body too well, in a way that survives affine alignment, and the
fine denoising steps are indispensable to each body (LEAP -> 0.000 without them). Schemes
that remain plausible either share only the parts that already align (a shared object/goal
encoder with per-embodiment proprioception and action heads — the design the term-level data
supported all along) or transfer sequentially rather than jointly (pretrain, then finetune).

---

## 2026-08-27 — Ambient diffusion: the premise check, before the sweep lands

The mixed grid measured a sign flip that data mixing alone does not explain. Adding
LEAP@400k to Allegro (rotation, mean of seeds 0+1, from
`mixed_matrix_in-hand_rotation_unconditioned_seed01.png`):

| Allegro budget | effect on Allegro |
|---|---|
| 100k | **+0.23 … +0.60** (helps) |
| 400k | **-0.41 … -0.65** (hurts) |

One-hot conditioning does not fix it (capacity-neutral in both seeds — see the 2026-08-26
entry below). The hypothesis under test: cross-embodiment data is **valid signal for the
coarse, high-noise end of denoising** (gross trajectory structure, shared across hands) and
**corrupting signal for the fine, low-noise end** (precise joint targets, kinematics-
specific). That predicts the flip directly: scarce target data -> coarse gain dominates;
plentiful target data -> only the fine-detail bias is left.

### The premise is testable without training anything, so it was tested first

`scripts/measure_ambient_threshold.py` noises both embodiments' actions on the policy's own
cosine schedule and measures W1 per dimension over pooled std, as a function of timestep t.

| | t=0 | t=50 | t=75 | t=99 |
|---|---|---|---|---|
| grasp, raw | 1.13 | ~0.63 | ~0.40 | 0.06 |
| grasp, mean-centred | **0.31** | ~0.05 | ~0.05 | 0.05 |
| rotation, raw | 1.24 | ~0.72 | ~0.42 | 0.07 |
| rotation, mean-centred | **0.34** | ~0.05 | ~0.06 | 0.07 |

The t=0 centred values (0.31 / 0.34) reproduce the independently-measured action distances
already in this file (grasp actions 0.337, rotation actions 0.361 — "Allegro vs LEAP: the
shift is almost all rigid offset"), from a different script and a different normalisation.
That agreement is the reason to trust the rest of the curve.

**The two curves behave completely differently, and that is the finding.** Mean-centred
distance — genuine distribution *shape* — collapses by t≈25 and is flat thereafter. Raw
distance, which includes the rigid per-embodiment offset, decays almost **linearly** and is
still ~0.4 at t=75, reaching the near-identical band only around t≈95.

### This moved the sigma* grid before a single sweep job ran

The planned grid was {25, 50, 75}. The model sees uncentred actions, so the raw curve is the
operative one: a rigid offset is exactly what makes LEAP data wrong for Allegro, and noise
does not mask it until very high t. sigma*=25 would therefore have been a near-duplicate of
naive mixing, and the interesting region sits **between 75 and 100** where the plan had no
interior point. Grid changed to **{50, 75, 90}** — same 60-run budget, better placed.

Worth being explicit that this is a *marginal* analysis while the model is *conditional*. A
rigid offset that is predictable from the observation can in principle be absorbed by the
network, and one-hot conditioning already tested a version of that and found nothing. So the
raw curve is an upper bound on how hard the offset is to handle, not a proof that it is
handled badly.

### Predictions, recorded before the results exist

- **N=400k:** non-monotone with an interior peak. Naive mixing sits 0.4-0.65 below the
  Allegro-only reference; intermediate sigma* should recover it, and beating it is the
  strong result.
- **N=10k/50k:** flatter and above the reference across most of the range — the coarse gain
  dominates when target data is scarce.
- **Control (bottom row of the sweep figure):** LEAP performance must *fall* as sigma* rises.
  If it does not, the gating is not doing what it claims and nothing above is interpretable.
- **A flat curve falsifies the coarse/fine story** — the interference would then not be
  organised by noise level at all. That is a real negative result for this line of attack,
  and cheaper to learn now than after building an architecture around it.

### Honest limit on the theory borrowed here

Ambient Diffusion (Daras et al., NeurIPS 2023) and the Ambient-o / data-scaling follow-ups
prove unbiasedness for data whose *corruption* makes it unidentifiable at high noise. Here
only the action is noised — the observation is always clean, so the network can always
identify the embodiment. The condition actually required is that `p(a_sigma | o)` converge
between embodiments, which is weaker and empirically checkable but **not** guaranteed by
those results. This work borrows the schedule-dependent admission mechanism; it does not
inherit the theorem. Any writeup must say so.

### What this does and does not address from the 2026-08-26 post-mortem

Addresses: headroom (rotation at N<=100k is far from saturated), noise floor (100-rollout
evals, 2 seeds, per-run sd 0.056 grasp / 0.196 rotation now measured rather than assumed).

Does **not** address: the missing `A@100k + A@900k` control that separates "more data" from
"more *foreign* data". The ambient sweep tests a *mechanism* for using foreign data, not
whether foreignness per se is the operative variable. Both are still worth running.

---

## 2026-08-26 — Mixed-embodiment one-hot conditioning: result and post-mortem

Arrays `750464` / `750465` / `750612`, 40/40 cells complete. Figures
`outputs/plots/mixed_matrix_{grasp,in-hand_rotation}.png`; run registry in `RUNS.md`.

### Result: mixing is close to capacity-neutral

| | overall mean delta | median | sd |
|---|---|---|---|
| Grasp | **-0.009** | 0.000 | 0.107 |
| Rotation | **-0.017** | 0.000 | 0.356 |

Own data does roughly 8-20x more work than partner data. Linear fit on log10(samples),
both axes in one regression:

| | own slope /decade | partner slope /decade | R2 own | R2 partner | R2 both |
|---|---|---|---|---|---|
| Grasp | **+0.434** | +0.055 | 0.840 | **0.014** | 0.854 |
| Rotation | **+1.209** | +0.061 | 0.868 | **0.035** | 0.870 |

Swapping a 10k partner for a 1M partner — 100x more foreign data — moves in-domain
performance by about +0.11 (grasp) and +0.14 (rotation), against own-data ranges of 0.83
and 2.41. Both endpoints sit inside the noise band.

Note the marginal average *by partner size* runs slightly **downward** and non-monotonically.
That is a confound signature, not an effect: cells with a small partner are
disproportionately cells where the evaluated hand happens to be data-rich.

### The one effect that is probably real

Rotation, Allegro at 100k own data — monotone in partner budget across four independent runs:

| LEAP partner | 10k | 50k | 400k | 1M |
|---|---|---|---|---|
| delta | +0.344 | +0.406 | +0.562 | +0.625 |

and its mirror, Allegro at 400k, negative in all four (-0.312, -0.594, -0.344, -0.531).
Transfer helps a data-poor hand, interferes with a data-rich one, crossover between 100k and
400k of own data. A strict ordering across four runs is unlikely from noise. It does not
survive aggregation because the two bands cancel. The LEAP side shows no such structure —
signs flip with no ordering — and that asymmetry is unexplained.

Consistent with the space analysis above: object/goal features are near-identical across the
two hands (centred W1 < 0.07), but everything the policy acts *through* — `hand_dof_pos`,
`keypoint_pos_rel`, the action head — stays far apart. A one-hot on a monolithic trunk lets
the network partition into two independent policies, which is what the null result looks like.

### Post-mortem: this design could not have answered the question

Worth recording, because the failure is in the design rather than the outcome.

**Measured noise.** Grasp per-eval jitter late in training is **sd ~ 0.053** (n=285
consecutive-eval differences). The binomial floor for a 32-episode eval at p=0.9 is
**0.053**. Essentially all the run-to-run wobble is the eval being too small.

**Power required**, 80% power at alpha 0.05:

| true effect | runs/arm (grasp, sd 0.107) | runs/arm (rotation, sd 0.356) |
|---|---|---|
| +0.05 | 72 | 795 |
| +0.10 | 18 | 199 |
| +0.20 | 5 | 50 |

We ran **one** per cell. The correct reading of the null is "no effect larger than ~0.2 would
have been detectable", not "no effect".

**Eval episodes needed** for a single run to resolve an effect at p=0.9: +0.10 needs 142
episodes, +0.05 needs 565. We used 32.

Three structural problems, ranked by cost:

1. **Eval too small.** The binding constraint. Eval costs ~65s; 256 episodes is ~8x that and
   negligible against a 4.5h run. Cheapest fix by a wide margin.
2. **Half the compute went to a task with no headroom.** Grasp saturates at 1.000 by 400k.
   Every grasp cell at or above 100k was structurally incapable of showing a positive effect.
3. **The architecture was never forced to share.** The design tested whether the network
   *would* exploit shared structure, not whether it *could*.

**Missing control.** `A@100k + L@1M` beats `A@100k` solo — but the mixed model saw 1.1M
unique samples. The comparison that separates "more data" from "more *foreign* data" is
`A@100k + A@900k`, which was never run.

### Proposed replacement experiment

Rotation only, Allegro as target, fixed at 100k own data — the one regime with both a real
effect and headroom. Five arms, matched gradient steps, **3 seeds each, 256-episode eval**:

| Arm | What it isolates |
|---|---|
| A@100k solo | baseline |
| A@100k + L@1M, one-hot | replicate of what was run |
| **A@100k + A@900k** | the missing control: data volume vs data foreignness |
| **Pretrain L@1M -> finetune A@100k** | sequential transfer, a stronger mechanism than joint training |
| **Shared object/goal encoder, per-embodiment proprioception + action heads** | forces sharing where the distributions already align |

15 runs, ~4.5h each, ~70 GPU-hours — about a third of the 40-cell grid's cost with roughly 3x
the power on the question that matters.

**Prerequisite.** Rotation BC caps at ~27% of expert with `drop_rate` ~1.0, argued in `RUNS.md`
to be the 8-step open-loop action chunk rather than data. If so, every effect is compressed
against an architectural ceiling. Run `action_horizon` in {2, 4, 8} on Allegro@1M first —
3 runs, ~14 GPU-hours — and only then the transfer study.

### Transferable lessons

- **Compute the noise floor before sizing a grid.** A 32-episode eval has a 0.053 floor; the
  effects of interest were 0.05-0.15. The grid was unanswerable from the start and that was
  knowable in advance from one line of arithmetic.
- **Do not spend compute on a saturated task.** Headroom is a precondition for measuring an
  intervention, not a detail.
- **Breadth traded against power.** 40 cells at n=1 answered nothing; 15 cells at n=3 with a
  bigger eval would have. Prefer depth when the expected effect is near the noise floor.

---

## 2026-08-26 — Observation / action space structure across embodiments

Scripts: `slurm_jobs/dump_schemas.sh` (job `749870`) → `outputs/analysis/schemas.json`;
`scripts/analyze_spaces.py`; `scripts/compare_matched_embodiments.py`.
Outputs: `outputs/analysis/spaces_summary.md`, `matched_pairs_summary.md`,
`spaces_stats.json`, `matched_pairs.json`; figures `space_term_matrix.png`,
`space_action_ranges.png`, `space_pca.png`, `space_matched_pairs.png`.

### Which embodiments are comparable at all

| obs | act | tasks |
|-----|-----|-------|
| 115 | 22 | **Grasp-Allegro, Grasp-LEAP** |
| 69 | 16 | **InHand-Rotation-Allegro, InHand-Rotation-LEAP** |
| 189 | 26 | Grasp-Shadow |
| 136 | 28 | Grasp-Sharpa |
| 130 | 26 | Grasp-Wuji |
| 89 | 20 | InHand-Rotation-Shadow |
| 87 | 22 | InHand-Rotation-Sharpa |
| 81 | 20 | InHand-Rotation-Wuji |

Only **Allegro and LEAP** match — and they match exactly, same dims *and* same term
layout, so dimension *j* is the same physical quantity in both. Every other hand is
unique. Shadow is the outlier: its `keypoint_pos_rel` is 66-d (22 keypoints) against
12-15 for the others, which is most of why its observation is 189-d.

Grasp has 16 observation terms, rotation 9. Grasp adds the 6-DoF floating wrist
(`hand_root_{pos,quat,linvel,angvel}`), hand-relative object pose, goal position/distance,
and keypoints. **Grasp action dim = rotation action dim + 6** for every hand (22 vs 16,
30 vs 24, ...), i.e. exactly the floating base. Note grasp calls proprioception
`hand_dof_{pos,vel}` while rotation calls it `joint_{pos,vel}` — same quantity, different
name, which any cross-family alignment has to handle.

Terms present in **all ten** tasks: `object_pos`, `object_quat`, `object_linvel`,
`object_angvel`, `actions`.

### Intrinsic dimensionality

PCA components for 95% variance on standardised features (`const` = dims dropped as
constant, std < 1e-6):

| Task | obs dim | const | obs pc_95 | act dim | act pc_95 | act saturation |
|------|---------|-------|-----------|---------|-----------|----------------|
| Grasp-Allegro | 115 | 0 | 35 | 22 | 12 | 52.2% |
| Grasp-LEAP | 115 | 0 | 49 | 22 | 14 | 32.3% |
| Grasp-Shadow | 189 | 27 | 43 | 26 | 13 | 25.8% |
| Grasp-Sharpa | 136 | 0 | 46 | 28 | 12 | 40.4% |
| Grasp-Wuji | 130 | 0 | 45 | 26 | 12 | 36.7% |
| InHand-Rotation-Allegro | 69 | 0 | 41 | 16 | 12 | 47.7% |
| InHand-Rotation-LEAP | 69 | 0 | 41 | 16 | 12 | 33.4% |
| InHand-Rotation-Shadow | 89 | 0 | 47 | 20 | 12 | 51.0% |
| InHand-Rotation-Sharpa | 87 | 0 | 52 | 22 | 14 | 69.4% |
| InHand-Rotation-Wuji | 81 | 0 | 49 | 20 | 14 | 49.4% |

Observations use roughly a third of their nominal dimension; actions about half. Shadow
grasp carries 27 genuinely constant observation dims.

### Actions are stored pre-clip, and the tails are long

`act saturation` above is the fraction of stored action entries with |a| >= 1, and it is
**26-69%**. The reason: `collect_demos` records the raw policy output, while
`RslRlVecEnvWrapper` clamps to `clip_actions=1.0` *inside* `env.step`. Measured range is
about **±15**:

```
Grasp-Allegro:           |a| p50=1.13  p90=2.87  p99=4.23  max=13.76  frac|a|>1=0.544
InHand-Rotation-Sharpa:  |a| p50=1.57  p90=3.14  p99=4.42  max=13.04  frac|a|>1=0.694
```

Not a correctness bug — a BC policy's output is clipped identically at rollout, so expert
and student pass through the same nonlinearity. But `LinearNormalizer` fits **min/max**, so
the informative ±1 band is compressed into roughly 1/15th of the normalised range while
rare outliers occupy the rest. Percentile-based or clip-aware normalisation is an untested
improvement; most likely to matter at small data scales, since grasp at 1M already
saturates its metric.

### Allegro vs LEAP: the shift is almost all rigid offset

Wasserstein-1 per dimension over pooled std. Rule of thumb: <0.2 near-identical, 0.2-0.5
mild, >1.0 barely overlapping.

| | raw | mean-centred |
|---|-----|--------------|
| grasp obs | 1.296 | **0.438** |
| grasp actions | 1.200 | **0.337** |
| rotation obs | 1.052 | **0.310** |
| rotation actions | 1.255 | **0.361** |

Raw distances say the two hands barely overlap; after removing each embodiment's own
per-feature mean they fall to the mild-shift band. Most of what separates them is a
**constant offset**, which per-embodiment normalisation removes for free.

Grasp, by term — the world looks the same, the body does not:

| term | raw | centred |
|------|-----|---------|
| `keypoint_pos_rel` | 3.773 | 0.628 |
| `object_pos_rel_hand` | 2.781 | 0.345 |
| `hand_root_quat` | 2.369 | 0.695 |
| `hand_dof_pos` | 2.132 | 0.643 |
| `actions` | 1.175 | 0.598 |
| ... | | |
| `object_quat` | 0.116 | 0.050 |
| `object_pos` | 0.100 | 0.064 |
| `object_goal_pos` | 0.043 | 0.024 |

Same pattern in rotation: `object_quat` 0.020, `goal_quat` 0.021, `joint_vel` 0.172 versus
`joint_pos` 2.140 and `actions` 1.314.

**Implication for a shared policy.** Allegro↔LEAP is the one pair needing no padding or
re-encoding. Residual centred shift concentrates in `hand_dof_pos`, `keypoint_pos_rel`,
`hand_root_quat` and the action head — the kinematics-dependent parts. A shared
object/goal encoder plus per-embodiment proprioceptive input and action-output adapters is
the design the data supports. This is what motivated the one-hot mixed-embodiment grid
(`RUNS.md`, arrays `750464`/`750465`).

### Caveat worth remembering

A large *raw* W1 on a low-variance feature is a systematic offset, not a distribution-shape
difference. Rotation `object_pos` scores 4.417 raw — the largest of anything measured — on a
feature with std 0.034, because the two palms hold the object ~9 cm apart. Centred it is
0.312. Reading raw distances alone would have flagged the most trivially-correctable feature
as the biggest obstacle.
