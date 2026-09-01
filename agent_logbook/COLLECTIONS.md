# Data collections

Status snapshot: **2026-08-24**. Datasets under `data/` are gitignored.

All 10 task+embodiment datasets have been collected — see the section below. The two rows
here are from the 2026-08-22/23 session; those paths no longer resolve and are kept only for
provenance.

| ID | Task | Dataset path | Expert checkpoint | Episodes | Steps | Success | Status | Notes |
|----|------|--------------|-------------------|----------|-------|---------|--------|-------|
| demo-allegro-smoke | Grasp-Allegro | `data/demos/grasp_allegro_expert.zarr` | `logs/rsl_rl/allegro_grasp/2026-08-22_10-37-45_initial/model_5600.pt` | 100 | 50,000 | 100/100 | artifacts lost | 64 envs, collection_steps=1000. obs_dim=115, action_dim=22 |
| demo-allegro-full | Grasp-Allegro | `data/demos/grasp_allegro_expert_full.zarr` | `logs/rsl_rl/allegro_grasp/2026-08-22_10-37-45_initial/model_5700.pt` | 2000 | 998,360 | 1979/2000 | artifacts lost | 256 envs, collection_steps=4000. Used for `dp-allegro-full` |

## COMPLETE: one dataset per task + embodiment (all 10 combos)

Slurm array **`749475`** — all 10 `COMPLETED`, zero errors. Ran 2026-08-24 19:05 -> ~19:45 PDT,
released automatically by `--dependency=afterany:749423`. 2000 episodes / 256 envs /
`max_episode_steps=500` each. Total on disk **5.7 GB**.

| Task | Dataset (`data/demos/`) | Episodes | Steps | obs_dim | act_dim | Size |
|------|-------------------------|----------|-------|---------|---------|------|
| Grasp-Allegro | `Grasp-Allegro_expert.zarr` | 2000 | 995,872 | 115 | 22 | 619M |
| Grasp-LEAP | `Grasp-LEAP_expert.zarr` | 2000 | 992,220 | 115 | 22 | 631M |
| Grasp-Shadow | `Grasp-Shadow_expert.zarr` | 2000 | 999,380 | 189 | 26 | 903M |
| Grasp-Sharpa | `Grasp-Sharpa_expert.zarr` | 2000 | 999,511 | 136 | 28 | 713M |
| Grasp-Wuji | `Grasp-Wuji_expert.zarr` | 2000 | 997,102 | 130 | 26 | 680M |
| InHand-Rotation-Allegro | `InHand-Rotation-Allegro_expert.zarr` | 2000 | 968,945 | 69 | 16 | 429M |
| InHand-Rotation-LEAP | `InHand-Rotation-LEAP_expert.zarr` | 2000 | 971,770 | 69 | 16 | 439M |
| InHand-Rotation-Shadow | `InHand-Rotation-Shadow_expert.zarr` | 2000 | 919,871 | 89 | 20 | 481M |
| InHand-Rotation-Sharpa | `InHand-Rotation-Sharpa_expert.zarr` | 2000 | 938,503 | 87 | 22 | 473M |
| InHand-Rotation-Wuji | `InHand-Rotation-Wuji_expert.zarr` | 2000 | 949,550 | 81 | 20 | 473M |

All ten report `n_success == n_episodes == 2000`. **This is expected and is not evidence of a
broken filter** — see below. Logs: `slurm_jobs/collect/logs/collect_749475_<idx>.{out,err}`.
Reproduce with `sbatch slurm_jobs/collect_all.sbatch` (env overrides `NUM_EPISODES`,
`NUM_ENVS`, `MAX_EP_STEPS`).

### Expert selection: best seed per combo

`scripts/select_experts.py` -> `outputs/experts.json`. Ranks the 3 seeds by headline metric and
keeps only the winner, so no averaging and no losing seed is ever collected from. Regenerated
by array index 0 at collection time, so these reflect final iteration-9999 metrics:

| Task | Seed used | Headline | Seeds rejected |
|------|-----------|----------|----------------|
| Grasp-Allegro | 42 | 0.2581 | 0.2550, 0.2578 |
| Grasp-LEAP | 42 | 0.2531 | 0.2371, 0.2522 |
| Grasp-Shadow | 43 | 0.2791 | **0.0650**, 0.2609 |
| Grasp-Sharpa | 44 | 0.2599 | 0.2576, 0.2582 |
| Grasp-Wuji | 44 | 0.2523 | 0.2463, 0.2486 |
| InHand-Rotation-Allegro | 43 | 0.6920 | 0.6704, 0.6753 |
| InHand-Rotation-LEAP | 44 | 0.5912 | 0.5491, 0.5693 |
| InHand-Rotation-Shadow | 42 | 0.5162 | 0.4332, 0.4812 |
| InHand-Rotation-Sharpa | 43 | 0.5322 | 0.5114, 0.5237 |
| InHand-Rotation-Wuji | 44 | 0.5732 | 0.5374, 0.5644 |

The bolded 0.0650 is `shadow_grasp` seed 42, the sweep's one permanently dead seed —
correctly excluded.

### Bug fixed before collecting: rotation success was never filtered

`collect_demos` gated success detection on `"pose" in command_manager.active_terms`, else
`succ = True`. Verified on GPU (`slurm_jobs/check_terms.sh`, job `749474`) what the terms
actually are — note the grasp env cfg literal only lists `grasp_metrics`, so reading the
config alone would have been misleading:

- `Grasp-*`            -> `['grasp_metrics', 'pose']`  — `has_pose` True, distance filter worked.
- `InHand-Rotation-*`  -> `['rotation']`               — `has_pose` **False**, no filtering.

Every rotation episode was therefore written and labelled successful, dropped objects
included, and since `DiffusionDataset` defaults to `success_only=True` that filter was a
silent no-op for half the sweep. Patched `src/mjlab_hand/diffusion/collect.py` to use
`RotationCommand.metrics["episode_success"]` (already latched per episode, reset on goal
resample) for rotation, and to warn loudly if neither term is present.

### How the filter was actually verified

An earlier note in this file claimed `n_success == n_episodes` would indicate a broken
filter. **That test is invalid.** With `keep_failures=False`, `collect_demos` only writes an
episode when `succ` is true, so `n_success == n_episodes` holds by construction whatever the
filter does. It carries no information.

The valid test is `--keep-failures`, which writes every episode with its real flag.
`slurm_jobs/verify_filter.sh` (job `749532`), 200 episodes of InHand-Rotation-Allegro:

```
n_episodes: 200,  n_success: 197,  n_steps: 94816
```

3 episodes correctly flagged as failures -> the filter discriminates. The high acceptance
rate (98.5%) is expected rather than suspicious: a 500-step episode spans 2-3 goal resamples,
so a policy at ~69% per-goal success almost always reaches at least one goal. Corroborating
check — the "no pose or rotation command term" warning appears in **zero** collection logs,
confirming `has_rotation` resolves True on rotation tasks.

### Note for cross-embodiment work

Obs/action dims differ per embodiment **and** per task family, more than expected — grasp
obs ranges 115-189 and actions 22-28; rotation obs 69-89 and actions 16-22. Only
Allegro/LEAP happen to match each other (115/22 grasp, 69/16 rotation). The ten datasets
cannot be naively concatenated; a cross-hand policy needs padding, per-embodiment
encoders, or a shared observation/action schema.

## Grasp datasets at two scales — 1M and 10M transitions, array `749534`

Submitted 2026-08-24 23:01. Best checkpoint per hand, two dataset sizes each = 10 jobs.

- Submit: `sbatch slurm_jobs/grasp_collect.sbatch`
- Job ID `749534`, array `0-9`; `hand = idx % 5`, `size = ['1M','10M'][idx / 5]`
- Resources: `gpu:h200:1`, 16 CPUs, 192GB, 24h
- Logs: `slurm_jobs/grasp_collect/logs/collect_749534_<idx>.{out,err}`
- Output: `data/demos/<Task>_expert_{1M,10M}.zarr`

`collect_demos` takes `--num-episodes`, not a transition target. Grasp episodes run to the
500-step cap (measured 497.9-499.7 steps/episode), so episodes are sized as
`ceil(target / 498)` — 2010 and 20100 — to land at or just above the requested count.

### 1M datasets — COMPLETE (all 5, 6-9 min each)

| Task | Dataset | Episodes | Transitions |
|------|---------|----------|-------------|
| Grasp-Allegro | `Grasp-Allegro_expert_1M.zarr` | 2010 | 1,004,278 |
| Grasp-LEAP | `Grasp-LEAP_expert_1M.zarr` | 2010 | 996,434 |
| Grasp-Shadow | `Grasp-Shadow_expert_1M.zarr` | 2010 | 1,002,967 |
| Grasp-Sharpa | `Grasp-Sharpa_expert_1M.zarr` | 2010 | 1,000,347 |
| Grasp-Wuji | `Grasp-Wuji_expert_1M.zarr` | 2010 | 998,659 |

All within 0.4% of the 1M target.

### 10M datasets — COMPLETE (all 5, 57-82 min each)

| Task | Dataset | Episodes | Transitions |
|------|---------|----------|-------------|
| Grasp-Allegro | `Grasp-Allegro_expert_10M.zarr` | 20100 | 10,042,701 |
| Grasp-LEAP | `Grasp-LEAP_expert_10M.zarr` | 20100 | 9,980,735 |
| Grasp-Shadow | `Grasp-Shadow_expert_10M.zarr` | 20100 | 10,037,996 |
| Grasp-Sharpa | `Grasp-Sharpa_expert_10M.zarr` | 20100 | 10,012,163 |
| Grasp-Wuji | `Grasp-Wuji_expert_10M.zarr` | 20100 | 10,006,814 |

All within 0.2% of the 10M target. Array `749534` finished 10/10 `COMPLETED`, zero errors.
`data/demos` is now **45 GB** total (38 TB still free on the filesystem).

### Best-checkpoint selection (frozen manifest)

`scripts/select_experts.py` gained `--pick best`, which snaps to the checkpoint nearest the
peak of the headline curve smoothed over a 25-point window, instead of just taking the last
checkpoint. This is not cosmetic — `Grasp-Sharpa` peaks at **iteration 2500** (0.2706) and
settles to 0.2599 by 9999, so last-checkpoint would have used measurably worse weights.

Frozen to `outputs/experts_grasp.json` so the concurrent 40k rotation retrain cannot change
the grasp selection mid-flight:

```bash
.venv/bin/python scripts/select_experts.py --only grasp --pick best --out outputs/experts_grasp.json
```

| Task | Seed | Checkpoint iter | Peak headline | Other seeds |
|------|------|-----------------|---------------|-------------|
| Grasp-Allegro | 42 | 7900 | 0.2689 | 0.2653, 0.2672 |
| Grasp-LEAP | 43 | 7700 | 0.2649 | 0.2431, 0.2624 |
| Grasp-Shadow | 43 | 9900 | 0.2829 | **0.0654**, 0.2659 |
| Grasp-Sharpa | 44 | 2500 | 0.2706 | 0.2668, 0.2689 |
| Grasp-Wuji | 42 | 6500 | 0.2626 | 0.2587, 0.2594 |

`--pick last` (the default) reproduces the original iteration-9999 behaviour;
`--only {grasp,rotation}` restricts the manifest to one family.

Note these differ slightly from the earlier `<Task>_expert.zarr` datasets, which used
last-checkpoint experts. The `_1M` / `_10M` sets supersede them for grasp.

## In-hand rotation datasets at two scales — 1M and 10M, array `749662` (2026-08-25)

Best checkpoint per embodiment, two dataset sizes each = 10 jobs. Mirrors the grasp setup.

- Submit: `sbatch slurm_jobs/rotation_collect.sbatch`
- Job ID `749662`, array `0-9`; `hand = idx % 5`, `size = ['1M','10M'][idx / 5]`
- Resources: `gpu:h200:1`, 16 CPUs, 192GB, 24h
- Output: `data/demos/InHand-Rotation-<Hand>_expert_{1M,10M}.zarr`
- Logs: `slurm_jobs/rotation_collect/logs/collect_749662_<idx>.{out,err}`

### Experts are from the 10k-iteration schedule, deliberately

```bash
.venv/bin/python scripts/select_experts.py --only rotation --pick best \
    --variant none --out outputs/experts_rotation.json
```

`--variant none` is load-bearing. `select_experts.py` scores every run dir in an experiment
directory and takes the max, and the in-progress 40k re-runs (array `749533`) already
outscore the finished 10k sweep — Allegro 40k is at ~0.75 vs 0.729 for 10k. Without the
filter these datasets would silently have been collected from partially-trained 40k
checkpoints. **These datasets come from 10k-training-step experts.**

| Task | Seed | Ckpt iter | Peak episode_success | Other seeds |
|------|------|-----------|----------------------|-------------|
| InHand-Rotation-Allegro | 43 | 7500 | 0.7286 | 0.7035, 0.7195 |
| InHand-Rotation-LEAP | 44 | 9700 | 0.6240 | 0.5920, 0.6021 |
| InHand-Rotation-Shadow | 42 | 9500 | 0.5417 | 0.4756, 0.5025 |
| InHand-Rotation-Sharpa | 43 | 9300 | 0.5801 | 0.5231, 0.5454 |
| InHand-Rotation-Wuji | 44 | 9700 | 0.6074 | 0.5652, 0.6067 |

Note Allegro's best checkpoint is iteration **7500**, not the final 9999 — the smoothed
success curve peaks mid-run and drifts down slightly after.

### Episode counts are per-hand, unlike grasp

Grasp episodes always run to the 500-step cap (497.9-499.7 steps/ep), so one constant worked.
Rotation episodes terminate early on `object_dropped`, and steps/episode varies by hand:

| Hand | steps/ep (measured, array 749475) | episodes for 1M | for 10M |
|------|-----------------------------------|-----------------|---------|
| Allegro | 484.5 | 2065 | 20650 |
| LEAP | 485.9 | 2059 | 20590 |
| Shadow | 459.9 | 2175 | 21750 |
| Sharpa | 469.3 | 2132 | 21320 |
| Wuji | 474.8 | 2107 | 21070 |

Caveat: those rates were measured from *last-checkpoint* experts. A better policy drops the
object less often, so real episodes may run slightly longer and the datasets may overshoot
target a little. Actual `n_steps` is reported per job.

Supersedes the earlier `InHand-Rotation-*_expert.zarr` (array `749475`, ~0.92-0.97M
transitions), which used last-checkpoint rather than best-checkpoint experts.

## Data-scaling ablation subsets — 10k and 100k transitions (2026-08-25)

20 datasets: 5 hands x {10k, 100k} x {grasp, rotation}. Built by **subsampling the 1M
datasets**, not by re-collecting:

```bash
python scripts/subsample_dataset.py --source data/demos/<Task>_expert_1M.zarr \
    --output data/demos/<Task>_expert_100k.zarr --target-steps 100000
```

Subsampling was chosen deliberately over fresh rollouts:

- The subsets are strictly **nested** (10k subset of 100k subset of 1M), so any difference
  between scales is purely quantity, not a different draw of episodes.
- Expert checkpoint, env seed and collection settings are identical by construction, so
  nothing can drift between scales.
- Seconds of CPU instead of a GPU rollout per scale.

Episodes are copied **whole** and never truncated mid-episode, since `DiffusionDataset`
builds obs/action windows within episode boundaries; a partial trailing episode would
produce windows that run off the end. The greedy selector stops at whichever boundary lands
closest to target, so all 20 came in within 0.4% of target.

| Task | 10k | 100k |
|------|-----|------|
| Grasp-Allegro | 20 ep / 10,000 | 200 ep / 100,000 |
| Grasp-LEAP | 21 ep / 10,039 | 201 ep / 100,039 |
| Grasp-Shadow | 20 ep / 10,000 | 200 ep / 100,000 |
| Grasp-Sharpa | 20 ep / 10,000 | 200 ep / 100,000 |
| Grasp-Wuji | 20 ep / 10,000 | 200 ep / 100,000 |
| InHand-Rotation-Allegro | 26 ep / 10,038 | 206 ep / 100,038 |
| InHand-Rotation-LEAP | 24 ep / 10,027 | 204 ep / 100,027 |
| InHand-Rotation-Shadow | 33 ep / 9,959 | 213 ep / 99,936 |
| InHand-Rotation-Sharpa | 28 ep / 10,013 | 208 ep / 100,013 |
| InHand-Rotation-Wuji | 28 ep / 9,956 | 208 ep / 99,956 |

Rotation needs more episodes per scale than grasp because its episodes are shorter
(`object_dropped` terminations). Note how small the 10k sets are in episode terms — 20-33
demonstrations. Provenance is recorded in each store's `extra` attrs
(`subsampled_from`, `target_steps`, `source_n_steps`, `source_n_episodes`).

## Ambient-sweep diagonal datasets — `A400k_L400k` (2026-08-27)

Two datasets, built by `slurm_jobs/build_ambient.sbatch` (array `751406`, 45s each).

| Dataset | steps | episodes | obs | act |
|---|---|---|---|---|
| `data/mixed_noc/Grasp_A400k_L400k.zarr` | 800,356 | 1609 | 115 | 22 |
| `data/mixed_noc/InHand-Rotation_A400k_L400k.zarr` | 800,054 | 1663 | 69 | 16 |

**Why a separate job script.** `build_mixed.sbatch` enumerates ordered pairs of *distinct*
sizes and skips `a == b`, so the diagonal is unreachable through it. The ambient sweep needs
`N=400k` specifically because that is where naive mixing does the most damage
(-0.41 … -0.65 on rotation) and therefore where the coarse/fine hypothesis makes its
sharpest prediction.

Unconditioned (`--no-onehot`), matching the rest of `data/mixed_noc/`: observation stays at
native width, so the ambient sweep does not re-litigate conditioning (shown capacity-neutral
in both seeds).

Source provenance, used by the ambient timestep gating and verified on build:

```
Grasp:    [(0, 400195, 'Grasp-Allegro'), (400195, 800356, 'Grasp-LEAP')]
Rotation: [(0, 400155, 'InHand-Rotation-Allegro'), (400155, 800054, 'InHand-Rotation-LEAP')]
```

`TrajectoryStore.source_step_bounds()` recovers these from `extra.sources[i].n_steps` and
**raises** if they do not sum to `n_steps` — so a truncated or partially-written mixture
fails loudly instead of silently mis-labelling which embodiment a sample came from. This is
also why the 6 pre-existing `A{10k,50k,100k}_L400k` mixtures needed **no rebuild** for the
ambient work: the provenance was already on disk.

---

## Mixed-embodiment datasets (Allegro + LEAP, one-hot conditioned) — 2026-08-26

**40 datasets, 14 GB**, in `data/mixed/<Family>_A<sizeA>_L<sizeB>.zarr`. Built by
`slurm_jobs/build_mixed.sbatch` (arrays `750408` grasp / `750410` rotation), which wraps
`scripts/build_mixed_dataset.py`.

Every **ordered** distinct-size pair from {10k, 50k, 100k, 400k, 1M}, per family = 20 each.
Both orderings so hand identity is not confounded with data budget.

Structure: `obs = [base_obs, onehot]`, actions unchanged.

| Family | base obs | mixed obs | act | onehot order |
|--------|----------|-----------|-----|--------------|
| Grasp | 115 | **117** | 22 | `[Allegro, LEAP]` |
| InHand-Rotation | 69 | **71** | 16 | `[Allegro, LEAP]` |

Examples: `Grasp_A400k_L1M.zarr` = 2811 episodes / 1,396,629 steps;
`InHand-Rotation_A1M_L10k.zarr` = 2089 episodes / 1,010,250 steps.

Verified on disk: exactly one hot per row, per-source step counts match the inputs, and the
`extra` attrs record `mixed`, `onehot_dim`, `onehot_order`, `base_obs_dim` and per-source
paths/steps for provenance.

Only Allegro and LEAP are mixable — the sole pair sharing both dimensionality and term layout
(see `ANALYSIS.md`). The builder **refuses** mismatched spaces rather than zero-padding, which
would put unrelated physical quantities in the same column.

## Allegro / LEAP focused scales — 50k and 400k (2026-08-25)

Eight datasets subsampled from the 1M stores for the four Allegro/LEAP embodiment-task pairs,
completing a 10k / 50k / 100k / 400k / 1M / 10M grid for those pairs. All within 0.1% of
target and nested with the existing subsets.

| Task | 50k | 400k |
|------|-----|------|
| Grasp-Allegro | 100 ep / 50,000 | 801 ep / 400,195 |
| Grasp-LEAP | 101 ep / 50,039 | 808 ep / 400,161 |
| InHand-Rotation-Allegro | 106 ep / 50,038 | 834 ep / 400,155 |
| InHand-Rotation-LEAP | 104 ep / 50,027 | 829 ep / 399,899 |

Note how few episodes 50k is: **100-106 demonstrations**.

## Still not collected

- Multi-embodiment combined dataset for cross-hand diffusion — blocked on deciding how to
  reconcile the mismatched obs/action dims (see note above).
- Failure-inclusive datasets (`--keep-failures`), if negative examples turn out to be wanted.

## Template (copy for new collections)

```md
| id | Task | path | checkpoint | n_episodes | n_steps | n_success | status | notes |
```
