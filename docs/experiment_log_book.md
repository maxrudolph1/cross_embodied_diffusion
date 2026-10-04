# Experiment log book

> **How to update this log book.** When the user says "update the log books", do this for every
> experiment that produced results in the session: a final or reporting training run, a new idea
> being tried, or a comparison of conditions. Do it in the same turn and without asking. (1) Add
> an entry at the **top** of [Entries](#entries), because entries are listed newest first. Number it
> one higher than the current highest `E<N>` (never renumber), date it `YYYY-MM-DD` (the date the
> results came in), and start it with `<a id="eN"></a>` and the heading
> `### EN — YYYY-MM-DD — <title>`. If an experiment is still running, add the entry with
> **Status: running** and fill it in later, then change the date in the TOC row to the
> results date. (2) Fill in every field of the [entry template](#entry-template):
> question/hypothesis; status; **data** (stores and sizes, linked to [COLLECTIONS.md](COLLECTIONS.md));
> **policy inputs/outputs** (obs layout and width, horizons, action chunk and width, normalizer);
> training and eval protocol; results as tables **and embedded plots**; takeaways; and links to the
> [RUNS.md](RUNS.md) job IDs, to the [agent log book](agent_log_book.md) entry `A<N>` that holds the
> lower-level implementation changes, and to [CHANGES.md](CHANGES.md) items. (3) Save every figure
> into `docs/plots/` under a descriptive name, with its summary CSV next to it when one exists, and
> embed it as `![caption](plots/<file>.png)`. Do not link to `outputs/` or `$SCRATCH`, because those
> are gitignored or purged. (4) Add a row at the top of the [Table of contents](#table-of-contents):
> number, date, and a **single-sentence** statement of the finding (the result, not just the topic).
> (5) If a later result overturns or qualifies an older entry, keep the old entry. Add a dated
> `> **Update YYYY-MM-DD:**` note to it that points to the new entry. Report only verified numbers
> and give the eval protocol (episodes, seeds, checkpoint rule) for every number. The agent log
> book's own update steps also apply. Entries E1–E6 were rebuilt on 2026-10-04 from the old
> `ANALYSIS.md` ([archive](archive/ANALYSIS_2026-08-26_to_2026-09-28.md), with full tables and
> derivations), and E7–E16 from the old journal. Their figures from the old cluster (E1–E6) were
> never copied to Vista, so those entries have tables only.

**Metrics.** Grasp is scored by **success rate**. InHand-Rotation is scored by **average successes
before drop**: the number of goal orientations reached before the object falls, higher is better, and
the RL expert reaches roughly 2–8. "σ" (sigma) is the ambient gate: the other hands' data trains only
diffusion timesteps t ≥ σ (out of 100). σ=0 is full co-training and σ=100 is target-only.

---

## Table of contents

| # | Date | Finding |
|---|---|---|
| [E16](#e16) | 2026-10-04 | Fine-tuning at lr 1e-4 forgets every non-target hand (0.00). lr 1e-5 keeps most of grasp (0.82) but not rotation (0.25). Co-trained grasp generalists score 0.94–1.00 on the four hands they saw at 1M. |
| [E15](#e15) | 2026-10-04 | Co-training on all 5 hands and then fine-tuning on the 50k target beats both co-training and target-only training: grasp 0.91 vs 0.76/0.38, rotation 1.34 vs 0.66/0.70. |
| [E14](#e14) | 2026-10-03 | With term-aligned padding, co-training gives a large grasp gain (+0.38..+0.44) and a small rotation loss (−0.04..−0.15) against target-only 50k. |
| [E13](#e13) | 2026-10-01 | Tail-padded ambient rotation sweep: co-training works only for Allegro/LEAP, whose columns line up, so the layout is the problem. |
| [E12](#e12) | 2026-10-01 | Ambient minmax, Allegro target: the target improves with σ, and the gated LEAP control drops to 0.00 already at σ=10. |
| [E11](#e11) | 2026-09-30 | Pooled runs scored 0 because of the padded path's Gaussian normalizer and clamp. A min/max padded run matches plain training. |
| [E10](#e10) | 2026-09-28 | Vista GH200 throughput per node is flat from 4 to 16 packed runs, and multi-node jobs scale linearly. |
| [E9](#e9) | 2026-09-27 | Single-hand specialist baselines: grasp at 1M is about 1.0 for all hands; the rotation specialists trained, but their scores were never tabulated. |
| [E8](#e8) | 2026-09-13 | At batch 256 one network saturates an A40, so vmap and bf16 give nothing. `torch.compile` gives 1.34× on training and 2.0× on sampling. |
| [E7](#e7) | 2026-09-02 | Rotation BC at 400k reaches 2.31 (LEAP) and 1.62 (Allegro) successes before drop. |
| [E6](#e6) | 2026-08-31 | A shared embodiment-invariant state encoder is ruled out: a fresh probe still identifies the hand at ≥0.977 balanced accuracy. |
| [E5](#e5) | 2026-08-27 | The observation identifies the embodiment perfectly (AUC 1.000) at every timestep, and normalization does not remove it. |
| [E4](#e4) | 2026-08-27 | Ambient sweep (Allegro+LEAP grasp/rotation, data-first sampler): the interior σ is a valley, and the gated source drops to 0. |
| [E3](#e3) | 2026-08-27 | Premise check: the noised Allegro and LEAP action distributions converge only at t≈95, which moved the σ grid to {50,75,90}. |
| [E2](#e2) | 2026-08-26 | One-hot mixed-embodiment training (Allegro+LEAP) is roughly capacity-neutral, and the design was underpowered. |
| [E1](#e1) | 2026-08-26 | Only Allegro and LEAP share obs/action widths and term layout. The hands differ mostly by a rigid offset, and actions are stored pre-clip (±15). |

---

## Entry template

```md
<a id="eN"></a>
### EN — YYYY-MM-DD — <title>

**Question.** ...  **Status.** done | running | abandoned
**Data.** stores, sizes, target/source split (COLLECTIONS.md links)
**Policy inputs/outputs.** obs (layout, width, horizon) -> action chunk (horizon, width, executed prefix); normalizer
**Protocol.** training (epochs/steps, lr, seeds, sampler, gating) and eval (envs x steps, seed, checkpoint rule)
**Results.** table(s) + ![caption](plots/<file>.png)
**Takeaways.** ...
**Links.** RUNS: `<job ids>` · Agent log: [AN](agent_log_book.md#aN) · CHANGES: items ...
```

---

## Entries

<a id="e16"></a>
### E16 — 2026-10-04 — Cross-embodiment generality of the generalists (co-trained vs fine-tuned)

**Question.** Each co-trained or fine-tuned policy was selected for one target hand. How well does it still
do on the other four hands of its family?  **Status.** done.

**Data / policy.** Same policies as [E14](#e14) (co-trained σ0) and [E15](#e15) (fine-tuned). Each policy is
evaluated on all 5 hands of its family through the term-aligned padded interface (`pad: true`).

**Protocol.** Checkpoint `last0`, 100 envs × 1500 steps, eval seed 1234, mean of 2 seeds.
`scripts/eval_cross_embodiment.py` → `<run>/cross_eval.jsonl`. The target (diagonal) cell is reused from
`final_eval.jsonl`. 300 scores in total.

**Results.** Mean over the 5 policies of each condition. "Own" is the diagonal; "others" is the mean
off-diagonal.

| | Grasp own | Grasp others | Rotation own | Rotation others |
|---|---|---|---|---|
| co-trained | 0.76 | 0.98 | 0.66 | 1.45 |
| FT lr 1e-4 | 0.84 | **0.00** | 1.29 | **0.00** |
| FT lr 1e-5 | 0.91 | 0.82 | 1.34 | 0.25 |

![Grasp: 5x5 cross-embodiment heatmaps per condition](plots/cross_embodiment_grasp.png)
![Rotation: 5x5 cross-embodiment heatmaps per condition](plots/cross_embodiment_rotation.png)

Rows are the policy's target hand, columns the hand it was evaluated on. Raw values:
[`plots/cross_embodiment_summary.csv`](plots/cross_embodiment_summary.csv).

**Takeaways.**
- A co-trained generalist is near-perfect on the four hands it saw at 1M (grasp 0.94–1.00) and weakest on its
  own 50k target.
- Fine-tuning at lr 1e-4 forgets catastrophically: almost every off-diagonal cell is exactly 0.00 in both
  families. lr 1e-5 keeps most of grasp (0.82) but loses most of rotation (0.25).
- For grasp, FT lr 1e-5 is both the best target policy and a reasonable generalist. For rotation, no single
  policy is good at both.

**Links.** RUNS: `1043385`, `1045839`, idev cross-eval row · Agent log: [A37](agent_log_book.md#a37) · CHANGES 68.

<a id="e15"></a>
### E15 — 2026-10-04 — Fine-tuning co-trained policies on the target hand

**Question.** Does a short target-only fine-tune of a co-trained policy beat both co-training and
target-only training on the scarce target?  **Status.** done.

**Data.** Same pooled store as the co-training run: `<Family>_pad5_scarce<Hand>_K50k` (target 50k + four hands at
1M; [COLLECTIONS](COLLECTIONS.md)). It is gated to the target only (tmin 0 for the target, 100 for the others,
noise-first), so in effect it trains on the 50k target data alone.

**Policy inputs/outputs.** Same as [E14](#e14). Initialized from the co-trained σ0 run's `policy_best_val.pt`,
whose normalizer comes with it (`--init-checkpoint`).

**Protocol.** 10 epochs (~156k steps) with a fresh optimizer, lr ∈ {1e-4, 1e-5}, 2 families × 5 targets × co-train
seeds {0,1} = 40 runs, eval + val every epoch. Re-scored with `rescore_selected.py` (best_rollout / best_val /
last0; 100 envs × 1500 steps; seed 1234).

**Results.** Target-hand score, mean over 5 targets × 2 seeds (best_rollout / best_val / last0):

| | Grasp (success) | Rotation (successes before drop) |
|---|---|---|
| target-only 50k (σ100) | 0.37 / 0.38 / 0.38 | 0.73 / 0.73 / 0.70 |
| co-trained (σ0) | 0.81 / 0.76 / 0.76 | 0.66 / 0.58 / 0.66 |
| co-trained + FT lr 1e-4 | 0.87 / 0.84 / 0.84 | **1.38** / **1.26** / 1.29 |
| co-trained + FT lr 1e-5 | **0.92** / **0.93** / **0.91** | 1.25 / 1.20 / **1.34** |

![Selection rule x training condition, both families](plots/conditions_overview.png)
![Grasp per target hand](plots/conditions_grasp.png)
![Rotation per target hand](plots/conditions_rotation.png)

Raw values: [`plots/conditions_summary.csv`](plots/conditions_summary.csv).

**Takeaways.**
- Co-training then fine-tuning wins in both families. For rotation it roughly doubles both baselines,
  although co-training alone slightly hurt there.
- The checkpoint selection rule matters much less than the training condition.
- The cost is generality ([E16](#e16)).

**Links.** RUNS: `1045839` · Agent log: [A36](agent_log_book.md#a36), [A37](agent_log_book.md#a37) · CHANGES 67, 68.

<a id="e14"></a>
### E14 — 2026-10-03 — Co-training vs target-only on term-aligned stores (Bundle recipe)

**Question.** With 4M source transitions from four other hands plus 50k on the target hand, does co-training
beat 50k target-only data? Bundle reported grasp +0.46 and rotation −0.27.  **Status.** done.

**Data.** Per target hand and family: `data/mjlab_hand_demos/padded_ta/<Family>_pad5_scarce<Hand>_K50k.zarr`
(target 50k + other four at 1M, ~4.0M steps). Val store: `val/<family>_val_20k.zarr` (fresh expert rollouts,
~20k steps per hand).

**Policy inputs/outputs.** Diffusion policy (ConditionalUnet1D, down_dims 256/512/1024, cosine schedule with 100
train steps, DDIM with 16 inference steps).
- **Input:** the last 2 observations, each **term-aligned padded** to the family width: grasp 191, rotation 91.
  Each obs term has a block as wide as its widest version, the hand's values sit at the block start, and the
  padding is 0.
- **Output:** an 8-step action chunk at the padded width (grasp 28, rotation 22, front-packed). Only the hand's
  native prefix is executed.
- **Normalizer:** one frozen per-family min/max normalizer, `configs/norm_{grasp,rotation}_minmax.json`, with
  x0 clamp 1.0.

**Protocol.**
- **Training:** noise-first ambient sampler, σ ∈ {0 (co-train), 100 (target-only)}, seeds {0,1}, ~784k steps
  (49–51 epochs), lr 1e-4, batch 256, keep-last 3. In-training eval on the target only: 32 envs × 1500 steps,
  10 times per run.
- **Reporting:** `rescore_selected.py` with best_rollout / best_val / last0, 100 envs × 1500 steps, seed 1234.
- 40 runs.

**Results.** Mean over 5 targets of (co-train − target-only), 2 seeds:

| | best in-training eval | best val loss | last epoch |
|---|---|---|---|
| Grasp (success rate) | +0.441 | +0.380 | +0.376 |
| Rotation (successes before drop) | −0.066 | −0.147 | −0.041 |

- Grasp: co-training wins for all 5 targets under all 3 rules. Per target, last0 co-train vs target-only:
  Allegro 0.83/0.41, LEAP 0.61/0.36, Shadow 0.64/0.42, Sharpa 0.80/0.30, Wuji 0.90/0.42.
- Rotation: slightly negative for 4 of 5 targets (Wuji is worst, −0.18..−0.35). LEAP is positive (+0.05..+0.31).

![Co-train vs target-only, both families, last0](plots/cotrain_vs_target_overview.png)
![Grasp per target, three checkpoint rules](plots/cotrain_vs_target_grasp.png)
![Rotation per target, three checkpoint rules](plots/cotrain_vs_target_rotation.png)

Raw values: [`plots/cotrain_vs_target_summary.csv`](plots/cotrain_vs_target_summary.csv).

**Takeaways.**
- This replicates Bundle's grasp gain (+0.46) and the sign of its rotation loss, which is smaller here.
- Compared with the tail-padded sweep ([E13](#e13)), where co-training collapsed to ~0.05 for Shadow, Sharpa and
  Wuji, term-aligned padding puts every hand on par with or above target-only.
- Caveat: our normalized action std (0.24/0.26) differs from Bundle's quoted 0.077 (CHANGES 63).

**Links.** RUNS: `1043385` + idev rescore row · Agent log: [A32](agent_log_book.md#a32)–[A36](agent_log_book.md#a36) ·
CHANGES 63, 65, 66.

<a id="e13"></a>
### E13 — 2026-10-01 — Ambient rotation sweep on tail-padded stores (old layout)

**Question.** For a 50k rotation target co-trained with four 1M hands, how does target performance depend on
the ambient gate σ?  **Status.** done; provisional, because it was never re-scored with fresh seeds.
Superseded by the migration ([E14](#e14)).

**Data.** Per target: target 50k + other four 1M, pooled in memory, **tail-padded**: each hand's obs is
front-packed and zero-padded at the end. Columns therefore mean different things for different hands, except
for Allegro/LEAP.

**Policy inputs/outputs.** 2 obs (rotation family max 89, tail-padded) → 8-step action chunk (22, tail-padded).
Per-source min/max normalization (`--source-norm minmax`, CHANGES 58), data-first ambient sampler.

**Protocol.** σ ∈ {0,1,2,3,4,5,6,8,10,12,14,16,18,20,25,100} × 4 seeds × 5 targets = 320 runs, 50 epochs. In-training
evals: 32 envs, target only. Checkpoints reported are best_eval, best_val and latest.

**Results.** Target score rises with σ for every hand. At σ=0 (full co-training, last epoch): LEAP 0.58 (≈ its
σ=100 score of 0.52), Allegro 0.36, Shadow/Sharpa/Wuji ~0.05.

![Score vs sigma per target hand](plots/ambient_rot_by_hand.png)
![All hands, raw and normalized to sigma=100](plots/ambient_rot_hands.png)

Raw values: [`plots/ambient_rot_summary.csv`](plots/ambient_rot_summary.csv).

**Takeaways.** Co-training works where the columns align (Allegro/LEAP have identical widths and term layout)
and fails where tail padding misaligns them. This is the evidence that motivated adopting Bundle's term-aligned
layout.

**Links.** RUNS: `1038730` · Agent log: [A25](agent_log_book.md#a25), [A31](agent_log_book.md#a31) · CHANGES 57, 62.

<a id="e12"></a>
### E12 — 2026-10-01 — Ambient minmax test: Allegro target, σ ∈ {0,10,25,100}

**Question.** Does the min/max padded ambient pipeline work end to end, and which way does σ move the target?
**Status.** done (1 seed).

**Data / policy.** Allegro rotation 50k + four 1M, tail-padded, per-source min/max, data-first. The same policy I/O
as [E13](#e13).

**Protocol.** 50 epochs, 1 run/node. Reporting eval with `eval_checkpoints.py`: 100 envs, env seed 1000.

**Results** (successes before drop; best_eval / best_val / latest):

| σ | Allegro (target, 50k) | LEAP (control, 1M, gated) |
|---|---|---|
| 0 | 0.31 / 0.19 / 0.38 | 1.84 / 1.78 / 1.76 |
| 10 | 0.41 / 0.48 / 0.42 | 0.00 / 0.00 / 0.00 |
| 25 | 0.56 / 0.65 / 0.66 | 0.00 / 0.00 / 0.00 |
| 100 | 0.81 / 1.11 / 0.90 | 0.00 / 0.00 / 0.00 |

**Takeaways.** The target improves monotonically with σ, so under this layout co-training hurts. LEAP is fully
functional when co-trained at every timestep, and exactly 0.00 when it is withheld from only the 10 lowest-noise
steps. This matches [E4](#e4)'s control.

**Links.** RUNS: `1036217`, `1038574` · Agent log: [A28](agent_log_book.md#a28), [A30](agent_log_book.md#a30).

<a id="e11"></a>
### E11 — 2026-09-30 — Why pooled/padded runs scored 0: normalization, not pooling

**Question.** The 09-28 scarce sweep scored ~0 on every hand, including the data-rich ones. Is pooling hands
the cause, or the padded code path?  **Status.** done.

**Data / policy.** Allegro rotation 1M and 50k, trained plain (native obs 69 → action 16, LinearNormalizer
min/max) and through the padded path (`--pool-sources`, Gaussian per-source normalizer or min/max).

**Results** (fresh eval: 100 envs, seed 1000, successes before drop; best_eval / best_val / latest):

| run | best_eval | best_val | latest |
|---|---|---|---|
| 1M plain | 1.74 | 2.10 | 1.91 |
| 1M padded **minmax** | 1.85 | 1.83 | 2.11 |
| 1M padded Gaussian | 0.00 at every eval (killed) | | |
| 50k plain | 0.87 | 0.66 | 0.92 |
| 5-hand pool co-train minmax (Allegro 50k + 4×1M) | 0.15 | 0.10 | 0.40 |

**Takeaways.**
- The padded path itself caused the zeros: the sampler's hard [-1,1] clamp, and the Gaussian normalizer's data
  scale. Padded min/max equals plain training within noise.
- Selection by in-training eval is optimistic: the pool's best_eval scored 0.44 during training and 0.15 on the
  fresh eval.

**Links.** RUNS: diag-rot-idev, diag-rot-idev2 · Agent log: [A26](agent_log_book.md#a26), [A27](agent_log_book.md#a27) · CHANGES 58.

<a id="e10"></a>
### E10 — 2026-09-28 — Vista job shape: throughput per node and across nodes

**Question.** What gets the most diffusion-BC runs through Vista: 1-node jobs that pack N runs each, or multi-node
jobs?  **Status.** done.

**Setup.** Grasp-Allegro 50k, production flags (compiled at the time), 4-node `gh-dev` idev.

**Results.**
- Per-node throughput is flat at ~0.37 run-epochs/s from 4 to 16 packed runs, with GPU utilization at 99%.
- A 4-node job ran 32/32 runs, with each node within 4% of a single node.
- Up to ~16 nodes, a job did not wait noticeably longer than a 1-node job (09-28 snapshot; see [A29](agent_log_book.md#a29)
  for a later, different snapshot).
- SU per run is unchanged. Storage on `$WORK` runs out long before scheduling limits do.

**Takeaways.** Use multi-node jobs (≤16 nodes). Packing only sets wall time, not cost. Measure the queue before
each large submission (AGENTS.md procedure). Full tables are in the
[archive](archive/ANALYSIS_2026-08-26_to_2026-09-28.md#2026-09-28--vista-job-shape-how-to-run-the-most-diffusion-bc-runs-at-once).

**Links.** Agent log: [A22](agent_log_book.md#a22) · CHANGES 54.

<a id="e9"></a>
### E9 — 2026-09-27 — Single-hand specialist baselines (old cluster)

**Question.** What does one hand's own data give at 50k and 1M?  **Status.** partial: rotation evals were never
tabulated.

**Data / policy.** Native per-hand obs → 8-step native action chunk, LinearNormalizer min/max, ~780k gradient
steps (200 epochs at 1M, 4000 at 50k), 2 seeds, val fraction 0.1.

**Results.**
- Grasp specialists at 1M: all 5 hands × 2 seeds ≈ 1.0 success.
- Grasp at 50k: 4 of 5 hands available; Allegro is missing, and two seed-1 runs were cut short.
- Pooled AllHands, tail-padded, Gaussian: ~0 everywhere (explained by [E11](#e11)).
- Rotation specialists (`97798`): Allegro, LEAP, Shadow and Sharpa × {50k,1M} × 2 seeds completed with all three
  checkpoints. Wuji seed 1 was still running at the last check (2026-09-27), and its outcome was not recorded.
  None of the scores were tabulated. The outputs are on the old cluster (`outputs/diffusion/specialist/`).

**Links.** RUNS: `89262`, `90599`, `97798` · Agent log: [A14](agent_log_book.md#a14)–[A19](agent_log_book.md#a19).

<a id="e8"></a>
### E8 — 2026-09-13 — More networks per GPU: vmap over seeds, bf16, torch.compile (A40)

**Results.**

| batch | n_seeds 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| 16 | 1.16× | 1.54× | 1.76× | 1.84× |
| 256 | 0.87× | 0.95× | 0.97× | 1.00× |

- vmap speed-up over sequential seeds is in the table above. bf16 autocast is *slower* (29.7 → 32.3 ms/step).
- `torch.compile` speeds up `compute_loss` 1.34× and `predict_action` 2.02×.

**Takeaways.** At batch 256 one network already saturates an A40. Only compile is worth using. It was later wired
in (CHANGES 53) and then removed for numerics parity (CHANGES 63).

**Links.** Agent log: [A16](agent_log_book.md#a16) · CHANGES 46.

<a id="e7"></a>
### E7 — 2026-09-02 — Rotation BC at 400k (LEAP, Allegro)

**Data / policy.** `InHand-Rotation-{LEAP,Allegro}_expert_400k.zarr`; native obs 69 × 2 → action chunk 8 × 16;
min/max normalizer; 500 epochs.

**Results.** Final eval, 32 rollouts: LEAP **2.31** successes before drop, Allegro **1.62**. Drop rate is 100% for
both, so policies keep rotating until the object falls.

**Links.** RUNS: `80853`, `80854` · Agent log: [A6](agent_log_book.md#a6).

<a id="e6"></a>
### E6 — 2026-08-31 — Learned state equivalence (shared invariant encoder)

**Question.** Can Allegro/LEAP share an object/goal encoder with per-hand heads?
**Results.**
- Matched task states exist: the cross-hand/self nearest-neighbour ratio is 0.88 for grasp and 1.75 for rotation.
- The object/goal subspace is still hand-separable (rotation 0.964).
- At a matched task state, the other hand's action adds nothing (R² 0.065 vs 0.784 for rotation).
- With a gradient-reversal adversary at λ up to 100, a fresh probe still identifies the hand at ≥0.977, while
  task R² drops.

**Takeaways.** Closed. Always score invariance with a fresh probe, not the adversary's own accuracy. Sequential
transfer (pretrain → fine-tune) was the one idea left untested, and [E15](#e15) tests a version of it.
**Links.** CHANGES 37–38 · [archive](archive/ANALYSIS_2026-08-26_to_2026-09-28.md).

<a id="e5"></a>
### E5 — 2026-08-27 — The observation identifies the embodiment at every timestep

**Results.**
- A linear probe on one observation separates Allegro from LEAP at accuracy 1.000.
- The observation is never noised, so identity is recoverable at t=99.
- After per-hand centring or standardization, the linear AUC drops to 0.50, but an MLP still scores 0.999.
- One dimension is enough: grasp `hand_dof_pos[9]` gives AUC 0.9998.

**Takeaways.** Masking or invariance schemes that rely on high-noise unidentifiability cannot work here. This
also explains why one-hot labels were redundant ([E2](#e2)). Never conclude "removable" from a linear probe alone.
**Links.** CHANGES 30–31 · [archive](archive/ANALYSIS_2026-08-26_to_2026-09-28.md).

<a id="e4"></a>
### E4 — 2026-08-27 — Ambient diffusion sweep, Allegro target + LEAP source (data-first)

**Data / policy.** Allegro N ∈ {10k,50k,100k,400k} + LEAP 400k, matching widths (grasp 115/22, rotation 69/16),
one-hot. Ambient gate σ* ∈ {0,50,75,90,100}, 2 seeds, 100-rollout evals.

**Results.**
- Grasp: the interior is a valley; at 50k it falls 0.98 → 0.46.
- Rotation: null (+0.067 / +0.054 against an sd of 0.196).
- LEAP, the gated control: exactly 0.000 at σ* ≥ 75 (grasp) and ≥ 50 (rotation).

> **Update 2026-09-01:** this sweep used the tuple-first (data-first) sampling order, which starves low-noise
> training (A5). Bundle retracted its "falsified" reading for the same reason. The control result (a hand
> trained only at coarse noise is non-functional) was reproduced in [E12](#e12) with the fixed code.

**Links.** CHANGES 24–28, 32, 36 · Agent log: [A3](agent_log_book.md#a3), [A5](agent_log_book.md#a5).

<a id="e3"></a>
### E3 — 2026-08-27 — Ambient premise check: when do noised action distributions converge?

**Results.**
- Raw (uncentred) W1 between noised Allegro and LEAP actions decays almost linearly, from 1.13 at t=0 to ~0.40 at
  t=75. It reaches the near-identical band only at t≈95.
- Mean-centred W1 collapses by t≈25.

**Takeaways.** Moved the planned σ* grid from {25,50,75} to {50,75,90}.
**Links.** `scripts/measure_ambient_threshold.py` · [archive](archive/ANALYSIS_2026-08-26_to_2026-09-28.md).

<a id="e2"></a>
### E2 — 2026-08-26 — Mixed-embodiment one-hot grid (Allegro + LEAP)

**Data / policy.** Own budget × partner budget grid, one-hot embodiment label appended to the native obs (115 / 69).
40 cells, 1 seed, 32-episode evals.

**Results.**
- The mean effect of adding the partner is about 0 (grasp −0.009, rotation −0.017).
- Own data does 8–20× more work than partner data.
- One plausible real effect: partner data helps Allegro rotation at 100k own data (+0.34 → +0.63, monotone in
  partner size) and hurts it at 400k.

**Takeaways.** The grid was underpowered (the 32-episode eval floor is 0.053), and grasp saturates. Compute the
noise floor before sizing a grid.
**Links.** CHANGES 13–22 · [archive](archive/ANALYSIS_2026-08-26_to_2026-09-28.md).

<a id="e1"></a>
### E1 — 2026-08-26 — Observation/action spaces across embodiments

**Results.**

| | Allegro | LEAP | Shadow | Sharpa | Wuji |
|---|---|---|---|---|---|
| Grasp obs/act | 115/22 | 115/22 | 189/26 | 136/28 | 130/26 |
| Rotation obs/act | 69/16 | 69/16 | 89/20 | 87/22 | 81/20 |

- Only Allegro and LEAP match in widths and term layout. Their joint order differs, though (corrected
  2026-09-30, `outputs/analysis/spaces_reference.md`).
- Actions are stored pre-clip, so |a| reaches ~15.
- Allegro vs LEAP differ mostly by a rigid offset (per-dimension W1 1.2 → 0.34 after centring). [E5](#e5) shows
  that this does not make them interchangeable.

**Links.** CHANGES 19–21, 59 · [archive](archive/ANALYSIS_2026-08-26_to_2026-09-28.md).
