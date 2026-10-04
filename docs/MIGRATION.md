# Migrating Branch to match Bundle: diffusion training, model and eval code

Written 2026-10-01. The goal is to change Branch so its diffusion training, model and eval code behaves like Bundle's. No files in either tree were changed to produce this doc.

| Name | Location | Notes |
| --- | --- | --- |
| Branch (to be changed) | `~/ced-other`, branch `origin/vista-ambient-rotation` (head `46e3ada`) | git |
| Bundle (the reference) | `~/cross_embodied_diffusion` | Snapshot of the cluster checkout `cross_embodied_diffusion-main` (2026-10-01), uncommitted changes included. No `.git`. |

**What "match" means here:**

1. Same code paths and same defaults as Bundle for everything under `src/mjlab_hand/diffusion/`, `cli/train_diffusion.py`, `cli/eval_diffusion.py` and `eval/rotation.py`.
2. Same checkpoint and output formats.
3. Bundle's data-prep scripts and artifacts brought over.
4. Branch's job scripts and manifests updated to pass Bundle's experiment recipe (§5.2). Bundle's code defaults are not what its experiments ran (§5.1).

Compute and scheduling material is out of scope, except where it changes numerics (`torch.compile`, DataLoader seeding).

**Evidence tags:**

- **[code]**: read from source.
- **[runs]**: from Bundle's 1,923 `outputs/**/train_config.json` files.
- **[log]**: from Bundle's logbooks.
- **[inferred]**: my reasoning.

**Bundle log citation keys:**

- J = `archive/JOURNAL_2026-08-22_to_2026-10-04.md`
- R = `RUNS.md`
- C = `CHANGES.md` (followed by the item number)
- A = `ANALYSIS.md`
- CO = `COLLECTIONS.md`
- F = `FINDINGS.md`

---

## 1. What changes

| Area | Branch now | After migration (= Bundle) |
| --- | --- | --- |
| Multi-hand obs layout | per-hand normalize, then zero-pad the end of the obs vector (`pooling.py`) | each obs term padded in place to the family max, using `schemas.json` (`padding.py`); data stored raw |
| Multi-hand data source | pre-built padded zarr or in-memory pooling (`--dataset A B …`) | pre-built term-aligned zarr only (`scripts/build_padded_dataset.py` via `padded_grid.py`) |
| Normalization | per hand, before training; policy normalizers are identity (`GaussianNormalizer`) | one shared `LinearNormalizer` in the policy: `--norm-mode shared` (default), `pad-aware`, `zscore` or `frozen` |
| Padded action loss | always masked (`action_mask`); padded channels free during sampling | unmasked by default; `--mask-pad-loss` masks and holds the padded channels fixed during sampling |
| x0 clamp | per-dim buffers `action_clip_low/high` | scalar `cfg.x0_clamp` (default 1.0) |
| Ambient sampling | noise-first only (`sample_ambient_batch`, main process) | `--ambient-sampler data-first` (default) or `noise-first` (`AmbientNoiseFirstBatchSampler` in the DataLoader) |
| Validation | `--val-fraction` split of the training store, optional `--val-embodiment` | `--val-dataset` (a separate store from `build_val_split.py`), `DenoisedValidator`, per hand on its own dims plus pooled |
| Checkpoints | `latest`, `epoch_N`, `best_val`, `best_eval` + `best_*.json`, `train_done.json` | `latest`, `best` (train loss), `last{k}_epoch{N}` (`--keep-last 3`), `best_rollout`, `best_val`, `selection.json` |
| In-training eval | no forced final eval; row has no `pad` / `val` | forced eval on the last epoch; row has `pad` and `val` |
| Eval wrapper | `embodiment_stats` (normalize, pad end, un-normalize) | `pad_task` (scatter into the padded layout; slice the action prefix; hold padded channels fixed if the checkpoint has `mask_pad_loss`) |
| Rotation metrics | `avg_successes_before_drop`, `drop_rate`, … | also `success_rate_any`, `per_target_success_rate`, `per_target_success_rate_incl_truncated`, `avg_targets_offered`, `per_episode` |
| Re-score | `scripts/eval_checkpoints.py` (100 episodes, 800 steps, seed 1000+seed) | `scripts/rescore_selected.py` (100 episodes, 1500 steps, seed 1234) |
| CUDA missing | silently falls back to CPU | raises |
| Removed | `torch.compile`, wandb, balanced sampling, seeded DataLoader generators, `--embodiment` eval flag | – |

---

## 2. Already identical (no change needed)

- Every `src/mjlab_hand` file not listed in §3.1 is byte-identical (`diff -rq`). That includes `diffusion/model.py` (the U-Net), the RL tasks, robots, objects and `eval/` other than `rotation.py`. [code]
- **Policy architecture and schedule defaults:**
  - `down_dims=(256,512,1024)`, `diffusion_step_embed_dim=128`
  - `obs_horizon=2`, `action_horizon=8`
  - `num_train_timesteps=100`, `num_inference_steps=16`
  - DDIM (eta=0) sampler
- **Optimizer:** AdamW lr 1e-4 / wd 1e-6, batch 256, grad-clip 1.0, `drop_last=True`.
- **Single-hand normalization:** `LinearNormalizer.fit`. Single-hand training goes through the same code path in both trees, apart from the checkpoint and validation differences below.
- **`collect.py`:** functionally identical. Only an import moved and a comment was added; copy Bundle's for byte parity.
- **torch:** both lockfiles pin 2.10.0. Keep Branch's `pyproject.toml` CUDA-index pin for aarch64; it changes how torch is installed, not what the code does.
- **Data history:** the demo stores share a lineage. Branch's subsets were made with the same `subsample_dataset.py` (prefix of the 1M store).

---

## 3. Migration plan

### 3.1 `src/` files

Do these as file replacements from Bundle, then re-add anything from §3.5 you decide to keep. Item by item:

**`diffusion/policy.py`: replace with Bundle's**

- Remove:
  - `normalizer_type` config field and the `GaussianNormalizer` branch in `__init__`
  - `LEGACY_GAUSSIAN_CLIP`
  - `action_clip_low/high` buffers and `set_action_clip`, plus the legacy-clip block in `load()`
  - `action_reconstruction_loss`
  - the `action_mask=` parameter
- Add:
  - config fields `mask_pad_loss: bool = False` and `x0_clamp: float = 1.0`
  - `compute_loss(..., timesteps=None, t_min=None, weight=None, action_valid=None)`:
    - data-first: t ~ U[t_min, T), weight-normalized mean;
    - noise-first: `timesteps` given;
    - passing both raises;
    - mask built from `action_valid`.
  - `predict_action(obs, action_valid=None)`: holds the padded channels fixed at `sqrt(ᾱ)·c + sqrt(1−ᾱ)·z` before each network call, sets them to `c` at the end, and clamps with `x0.clamp(-x0_clamp, x0_clamp)`.

**`diffusion/normalizer.py`: replace with Bundle's**

- Remove `GaussianNormalizer`.
- Add:
  - `LinearNormalizer.fit_masked(data, groups, clip_pct=None)`: fits only over the rows that use each column; percentile option.
  - `LinearNormalizer.fit_standardized(data, groups=None)`: z-score in affine form.

**`diffusion/dataset.py`: replace with Bundle's**

- Remove:
  - `source_real_dims`
  - `_episode_source_ids`, `_train_val_split`
  - `split` / `val_fraction` / `val_seed` / `only_source`
  - `action_mask_per_episode`, `source_id_per_window`, `source_sample_weights`
  - the `_win_*` arrays, `_build_ambient_index`, `sample_ambient_batch`
- Change: `source_step_bounds()` returns `[]` for a non-mixed store (Branch raises).
- Add:
  - `mask_pad_loss` constructor argument, `episode_tmin` / `window_tmin`, `episode_act_dim` / `window_act_dim`
  - `__getitem__` accepts `(idx, t)` tuples and returns `t_min` or `t` (+ `action_valid`)
  - the `AmbientNoiseFirstBatchSampler` class

**`diffusion/train.py`: replace with Bundle's**

- `TrainConfig` after migration:
  - `dataset: Path`
  - `ambient_tmin`, `ambient_sampler="data-first"`, `mask_pad_loss=False`
  - `norm_mode="shared"`, `norm_clip_pct=None`, `norm_artifact=None`, `x0_clamp=1.0`
  - `val_dataset=None`, `val_windows=2048`, `keep_last=3`
  - the shared fields as in §4
- Removed fields: `val_fraction`, `val_seed`, `val_embodiment`, `val_every_epochs`, `val_max_batches`, `source_sample_mode`, `compile_mode`, `wandb_*`, `pool_sources`, `task_family`, `source_norm`.
- Behavior:
  - CUDA check raises.
  - Loader is a plain shuffle or the noise-first batch sampler, with no explicit `torch.Generator`: shuffle order then comes from `torch.manual_seed(cfg.seed)`. Branch passes seeded generators, which changes batch order.
  - Normalizer chosen by `norm_mode`.
  - Checkpoints and `selection.json` as in §1. Forced final eval.
  - `train_config.json` gains `norm_digest`.
  - No `source_stats.json`, `train_done.json`, `best_eval.json`, `best_val.json` or `val_metrics.jsonl`.
- Note: Bundle's `save_every_epochs` field exists, but the loop never reads it.

**`diffusion/pooling.py`: delete**

Bundle has no in-memory pooling. Multi-hand runs use a pre-built term-aligned zarr.

**`diffusion/evaluate.py`: replace with Bundle's**

- Remove `EmbodimentStats` and the `embodiment=` arguments of `evaluate_diffusion_policy` / `render_diffusion_rollout` / `DiffusionActionChunkPolicy`.
- Add `pad: bool` / `pad_task`:
  - scatter through `padding.build_plan(...).obs_index(task)`;
  - hard error if the native width doesn't match;
  - slice the action prefix;
  - hold padded channels fixed when `cfg.mask_pad_loss`;
  - call the old `predict_action(hist)` signature when no fixing is needed.

**New files from Bundle**

- `diffusion/padding.py` (term-aligned layout). Reads `outputs/analysis/schemas.json`, resolved from the package location (`parents[3]`), or from `MJHAND_SCHEMAS`.
- `diffusion/frozen_norm.py` (load / select / build the frozen per-family normalizer).
- `diffusion/validate.py` (`DenoisedValidator`).

These three are untracked in Bundle's own git (BUNDLE_NOTES), so the bundle snapshot is the only copy.

**`eval/rotation.py`: replace with Bundle's**

Adds the targets-offered counter, per-episode records and the new rotation metrics (§1).

**`cli/train_diffusion.py`: replace with Bundle's**

Flags change as in §4. `--dataset` takes one path. Bundle rejects bad flag combinations up front:

- `frozen` requires `--norm-artifact`, and an artifact is only accepted with `frozen`;
- `zscore` requires `--x0-clamp > 1`;
- `--norm-clip-pct` requires `pad-aware`.

**`cli/eval_diffusion.py`: replace with Bundle's**

Drops `--embodiment`. Bundle's CLI has no way to evaluate a padded checkpoint; that goes through `evaluate_diffusion_policy(pad=True)` or the in-training eval specs.

**`diffusion/collect.py`: copy Bundle's**

Byte parity only; no behavior change.

### 3.2 Scripts and artifacts to bring over from Bundle

| Bundle file | Purpose | Notes for Branch |
| --- | --- | --- |
| `outputs/analysis/schemas.json` | per-task obs term names and widths; required by `padding.py` | Copy the file (it's in the bundle). Regenerate with `slurm_jobs/dump_schemas.sh` if the envs change. Branch's `outputs/` is gitignored and symlinked per machine (Branch item 23/55), so either track this file or point `MJHAND_SCHEMAS` at it. |
| `outputs/analysis/norm_{grasp,rotation}_{minmax,zscore}.json` | frozen normalizer artifacts (digests rotation `sha256:3e4a9afc…`, grasp `sha256:28bd9175…`) | Copy as-is. Rebuilding needs the five 10M single-hand stores per family, which Branch's HF mirror doesn't have (README "Data layout" lists 1M, subsets and padded only). Same storage caveat as above. |
| `scripts/build_padded_dataset.py` | term-aligned builder (replaces Branch's tail-pad wrapper) | Writes `extra.pad_scheme="term_aligned"`; concatenates sources in `HANDS` order |
| `scripts/check_padded_dataset.py` | post-build check | |
| `scripts/padded_grid.py` | names, sizes and source order for every padded config; per-source `t_min` vectors; epochs for the ~784k-step budget | Hardcodes `data/demos/<Task>_expert_<size>.zarr` and `data/padded/`. Branch keeps demos at `data/mjlab_hand_demos/<Task>_expert_1M.zarr` and `…/subsets_50k/…`. Either symlink a flat `data/demos/` or edit `DEMOS` / `PADDED`. Branch also needs whatever 100k/500k subsets the configs reference (`subsample_dataset.py`). |
| `scripts/build_val_split.py` | `data/val/<family>_val_20k.zarr` (10M episodes deduplicated against 1M) | Needs the 10M stores. Run it on the old cluster, where `data/demos/*_10M.zarr` exist (Branch COLLECTIONS "1M and 10M"), or copy the val zarrs from there. |
| `scripts/build_family_normalizer.py`, `scripts/check_frozen_norm.py` | build and verify frozen artifacts | Only needed to rebuild; `--stat` defaults to `zscore`, but the runs used `minmax` |
| `scripts/rescore_selected.py` | fresh-seed re-score → `final_eval.jsonl` | Replaces Branch's `eval_checkpoints.py` |
| `scripts/check_ambient_sampler.py`, `scripts/check_pad_fixes.py`, `scripts/check_sampler.py` | verification | `check_sampler.py` may already exist in Branch: compare it |
| `slurm_jobs/train_cotrain_ambient.sbatch` (and `_fine`, `train_cotrain_regimes.sbatch`) | reference for the exact recipe (§5.2) | Use them as the spec for Branch's manifests; their cluster settings don't apply on Vista |

### 3.3 Branch files that will break and need updating or deleting

From `git grep` on Branch for the APIs being removed:

| Branch file | Uses | Action |
| --- | --- | --- |
| `scripts/build_padded_dataset.py` | `pooling.pool_padded`, `--source-norm` | replace with Bundle's |
| `scripts/check_pooling.py` | `pooling`, `GaussianNormalizer`, `source_stats.json`, `val_fraction` | delete (it tests the removed pooling path) |
| `scripts/eval_checkpoints.py` | `embodiment=`, `policy_best_eval.pt`, `val_embodiment` | delete; use `rescore_selected.py` |
| `scripts/build_ambient_rotation_manifest.py` | `--dataset` list, `--source-norm`, `--val-fraction/-embodiment`, `--wandb-*`, eval `embodiment`, `--eval-num-steps 800` | rewrite to emit Bundle-recipe args (§5.3) |
| `scripts/build_scarce_specialist_manifest.py` | `--val-fraction`, `--source-sample-mode`, `--wandb-*` | rewrite or delete |
| `scripts/prototype_torch_compile.py` | `action_reconstruction_loss` | delete |
| `scripts/run_manifest_task.py` | `compile_mode` passthrough | drop compile |
| `scripts/promote_outputs.sh` | `train_done.json`, `policy_epoch_*` | switch the completion marker to `selection.json`; drop the `--no-epoch-ckpts` logic |
| `slurm_jobs/vista_train_manifest.sbatch` | `COMPILE_MODE=reduce-overhead` default, staging of list-valued `dataset` | default compile off; stage the single padded zarr plus `data/val/*` and the norm artifact |
| `slurm_jobs/vista_eval_checkpoints.sbatch` | `eval_checkpoints.py`, `policy_best_eval` | point at `rescore_selected.py --which best_rollout best_val last0 --envs 100 --steps 1500 --eval-seed 1234` |
| `slurm_jobs/train_cross_embodiment{,_50k}.sbatch`, `train_ambient_400k_20k_sweep.sbatch`, `train_rotation_specialist.sbatch`, `train_scarce_specialist.sbatch` | `--source-norm`, wandb, `policy_epoch_*` | update or retire |
| `scripts/hf_sync.py` | mentions `source-norm` | check; probably comment-only |
| `AGENTS.md`, `CHANGES.md`, logbooks | document the removed features | update after migrating, per Branch's logbook protocol |

### 3.4 Data to rebuild or obtain

1. **Padded datasets must be rebuilt.** Branch's `data/mjlab_hand_demos/padded/*.zarr` are tail-padded and per-hand normalized: grasp 189/28, rotation 89/22, `extra.padded=True` plus mean/std. Bundle's are term-aligned and raw: grasp 191/28, rotation 91/22, `pad_scheme="term_aligned"`.
   - Build with `padded_grid.py --family <f> --index <i> --build`.
   - Branch's ambient sweep pools (target 50k + four 1M partners) are Bundle's `scarce<Hand>_K50k` configs (`<Family>_pad5_scarce<Hand>_K50k.zarr`).
2. **Val stores:** `data/val/{grasp,rotation}_val_20k.zarr`, from the old cluster (they need the 10M stores).
3. **Artifacts:** `schemas.json` and the `norm_*` JSONs (§3.2).
4. **Old Branch checkpoints won't load afterwards.** `DiffusionPolicyConfig(**payload["cfg"])` raises on `normalizer_type`, and padded runs need `source_stats.json` and the `GaussianNormalizer` state. Re-score anything you need before migrating, or keep the old Branch commit around to evaluate them.

### 3.5 Branch-only features: remove, or keep without changing behavior?

Bundle has none of these. To match exactly, remove them. Each one's effect on results:

| Feature | Affects results? | Recommendation |
| --- | --- | --- |
| In-memory pooling (`pooling.py`, `--dataset A B`, `--pool-sources`, `--task-family`, `--source-norm`) | yes (different layout and normalization) | Remove. If wanted later, rebuild it on top of `padding.py` so it produces the same arrays as Bundle's builder. |
| `torch.compile` (`--compile-mode`, `_restore_backend_flags`) | yes (numerics change; C notes 1.34× faster) | Remove, or keep with default off and never use it for runs compared against Bundle |
| Seeded `torch.Generator` on the DataLoaders | yes (batch order) | Remove to match Bundle's shuffle stream |
| `--source-sample-mode balanced` | yes when used (default uniform = Bundle) | Remove |
| `--val-fraction`, `--val-embodiment`, `action_reconstruction_loss` | selection only | Remove; replaced by `--val-dataset` and `DenoisedValidator`. In Bundle's latest runs the validator follows the eval spec, so a target-only spec gives target-only validation (the same effect `--val-embodiment` had). |
| Per-dim action clamp | yes for padded runs | Remove; replaced by the scalar `x0_clamp` |
| wandb logging | no | Optional; harmless if kept |
| `train_done.json` | no | Optional. Bundle uses `selection.json` as its completion marker; harmless to keep writing both. |
| `--embodiment` on `eval-diffusion` | no (CLI only) | Remove to match |

---

## 4. Defaults: Branch now vs after migration

Code defaults (dataclass = CLI) after migration. The "after" column is Bundle's code default; the recipe Bundle actually ran is in §5.2.

| Knob | Branch now | After (Bundle default) |
| --- | --- | --- |
| `dataset` | path or list | single path |
| `num_epochs` / `batch_size` / `lr` / `weight_decay` | 50 / 256 / 1e-4 / 1e-6 | 50 / 256 / 1e-4 / 1e-6 |
| `num_workers` | 4 | 4 |
| `success_only` / `seed` | True / 0 | True / 0 |
| `save_every_epochs` | 10 (writes `policy_epoch_N`) | 10 (unused) |
| `latest_every_epochs` | 1 | 1 |
| `keep_last` | – | 3 |
| `eval_every_epochs` / `eval_num_envs` / `eval_num_steps` | 10 / 32 / 1500 | 10 / 32 / 1500 (+ forced final eval) |
| `render_*` | 0 / 400 / 1 | 0 / 400 / 1 |
| `ambient_tmin` | None | None |
| `ambient_sampler` | (noise-first, implicit) | `data-first` |
| `mask_pad_loss` | (masked, implicit, for padded data) | False |
| `norm_mode` | (`linear` / identity, implicit) | `shared` |
| `norm_clip_pct` / `norm_artifact` | – | None / None |
| `x0_clamp` | (per-dim, ±1 or data range) | 1.0 |
| `val_dataset` / `val_windows` | – | None / 2048 |
| `val_fraction`, `val_*`, `source_sample_mode`, `compile_mode`, `wandb_*`, `pool_sources`, `task_family`, `source_norm` | various | removed |
| Policy config fields | …, `normalizer_type="linear"` | …, `mask_pad_loss=False`, `x0_clamp=1.0` |
| `evaluate_diffusion_policy` | `num_envs=16, num_steps=2000, seed=0, embodiment=None` | `num_envs=16, num_steps=2000, seed=0, pad=False` |
| Re-score | `eval_checkpoints.py`: 100 / 800 / 1000+seed | `rescore_selected.py`: `--envs 100 --steps 1500 --eval-seed 1234 --which best_rollout` (default) |

**Watch out: matching Bundle's defaults changes ambient behavior.** Branch currently always samples noise-first. After migration, the default is data-first, which Bundle's own logs call confounded (A 2026-09-01). Every ambient run must pass `--ambient-sampler noise-first` (Bundle's job scripts do). The alternative is to deviate from Bundle and flip the default; Bundle's `train.py:165` then needs adjusting, because it raises on noise-first without `--ambient-tmin`.

---

## 5. Bundle's experiment recipe (what Branch's manifests should pass)

### 5.1 Code defaults are not the recipe

Bundle's latest 887 runs [runs] used `frozen`, `noise-first` and `num_workers=0`, none of which are code defaults. Matching the code alone won't reproduce Bundle's results; the manifests must pass these flags.

### 5.2 The recipe

From `slurm_jobs/train_cotrain_ambient.sbatch`; confirmed in all 887 `outputs/cotrain_ambient/*/train_config.json` files [runs]:

```bash
train-diffusion \
  --dataset data/padded/<Family>_pad5_<config>.zarr \
  --num-epochs <E: ~784k steps / (n_windows/256)> --batch-size 256 --lr 1e-4 \
  --obs-horizon 2 --action-horizon 8 --num-workers 0 --seed <s> \
  --ambient-tmin <t_min per source, in the store's HANDS order; 0 for the target> \
  --ambient-sampler noise-first \
  --norm-mode frozen --norm-artifact outputs/analysis/norm_<family>_minmax.json \
  --x0-clamp 1.0 \
  --val-dataset data/val/<family>_val_20k.zarr --val-windows 2048 \
  --keep-last 3 --save-every-epochs <E> --latest-every-epochs <per padded_grid> \
  --eval-spec '[{"task": "<target task>", "pad": true}]' \
  --eval-every-epochs <~E/10> --eval-num-envs 32 --eval-num-steps 1500
```

- **Non-ambient co-training runs** (`padded_cotrain`) use the same recipe without `--ambient-*`.
- **Single-hand runs** use a single-hand store, no padding flags, and `--eval-task`.
- **Re-score after training:** `rescore_selected.py --run <dir> --which best_rollout best_val last0 --envs 100 --steps 1500 --eval-seed 1234`.

### 5.3 Rewriting Branch's ambient manifest (`build_ambient_rotation_manifest.py`) to the recipe

| Field | Branch now | Change to |
| --- | --- | --- |
| `dataset` | `[target 50k, 4 × 1M]` list | `data/padded/InHand-Rotation_pad5_scarce<Hand>_K50k.zarr` (name per `padded_grid.py`) |
| `ambient-tmin` | `[0, σ, σ, σ, σ]` (target first) | the vector in the store's HANDS order (Allegro, LEAP, Shadow, Sharpa, Wuji), 0 at the target's position. Generate it with `padded_grid.target_ambient_tmin`; a wrong order silently gates the target (C 99, 118). |
| add | – | `ambient-sampler: noise-first`, `norm-mode: frozen`, `norm-artifact`, `x0-clamp: 1.0`, `val-dataset`, `val-windows: 2048`, `keep-last: 3` |
| remove | `source-norm`, `val-fraction`, `val-seed`, `val-every-epochs`, `val-max-batches`, `val-embodiment`, `wandb-*` | – |
| `eval-spec` | `[{"task": T, "embodiment": T}]` | `[{"task": T, "pad": true}]` |
| `eval-num-steps` | 800 | 1500 |
| `num-workers` | 8 | 0 |
| `num-epochs` | 50 (~700k steps) | epochs for ~784k steps (`padded_grid.TARGET_STEPS=784_000`; Bundle's fine grid was 50 epochs = 791,550 steps) |
| `save-every-epochs` | epochs+1 | epochs |
| seeds | 0–3 | Bundle used 0–2 for the main sweep and 5 seeds for the fine grid |
| σ grid | {0,1,2,3,4,5,6,8,10,12,14,16,18,20,25,100} | unchanged is fine (it covers Bundle's 09-15 peak at σ*=2–3). Bundle's main sweep was {5,10,15,20,25,30,35,50,75,100}, with σ=0 taken from the co-training runs. |

---

## 6. Things you inherit from Bundle

Matching Bundle brings these with it. For each, decide whether to copy as-is or fix while porting.

- **Ambient sampler default is `data-first`,** although F §2.3 says noise-first is "now the default" (§4).
- **`save_every_epochs` is never read by the loop;** the CLI help still describes numbered checkpoints.
- **SNR figures disagree.** `validate.py` and the logs say `std(x0)` = 0.153 / SNR=1 at t=8; `train.py`, the CLI help and `normalizer.py` say 0.179 / t=10. Under the frozen minmax the runs use, action std is 0.077 (A 2026-09-09).
- **`DenoisedValidator` says "raw radians",** but stored actions are pre-clip policy outputs (roughly ±15, A 2026-08-26), not radians.
- **`policy_best.pt` is selected by training loss.** Bundle keeps it "for continuity" only; it is not a quality signal.
- **`build_family_normalizer.py --stat` defaults to `zscore`,** but every run used minmax artifacts.
- **No 5-hand diagnostic.** `rescore_selected.py` only scores the run's own eval spec. The job-script header claims all five hands are re-scored; they aren't (R 2026-09-11).
- **The vision training code isn't in the bundle.** 92 runs in `outputs/vision/` used `obs_mode`, `max_steps`, `source_dataset`, `source_max_steps` and `crop_size`, which no bundle code defines. If needed, recover it from the cluster checkout or `cross_embodied_diffusion-history.bundle`.
- **Bundle's untracked files** (`padding.py`, `frozen_norm.py`, `validate.py`): the bundle snapshot is the only copy, so commit them in Branch.

---

## 7. Why Bundle is the way it is (log context)

Useful when reviewing the migration. All [log] unless marked.

- **Term-aligned padding.** With end-of-vector padding, obs columns mean different things per hand (Allegro `object_pos` at columns 44–46, Shadow's at 60–62). Bundle says it "would train, the loss would fall, and the rollouts would be meaningless" (J 2026-09-05; C 83). This is a design argument; neither project ran a rollout A/B of the two layouts. Column-level semantics are recovered, but joint correspondence *within* a term is not (Branch measured Allegro and LEAP joint orders differing).

- **One shared normalizer, not per hand.** A per-hand map would break the column alignment that term-aligned padding exists for.

- **Frozen normalizer.** Per-run min/max made the same Allegro column scale up to 2.42× differently across cells and drift with N (C 105; `frozen_norm.py`). Frozen was adopted for comparability. A pilot computed from `outputs/padded_frozen` vs `outputs/padded` (rotation, even mixtures, 3 seeds) came out about performance-neutral: 50k 1.02→0.99, 100k 1.36→1.38, 500k 1.65→1.53, 1M 1.43→1.62 [runs; computed during this review, not analysed in Bundle's logs]. The fit must keep 0 in range; without that, padding zeros went to −198,322 on 17 grasp columns (C 115).

- **Masked padded loss is off.** In the 09-09 A/B on the scarce hand (n=3, floor 0.113):

  | Arm | Change vs baseline |
  | --- | --- |
  | `--mask-pad-loss` | −0.177 |
  | pad-aware + clip 0.1 | −0.031 (partners +0.12) |
  | z-score (x0 4) | −0.917 (confounded by lr) |

  (A 2026-09-09.) Holding the padded channels fixed when masked is argued, not A/B'd.

- **Noise-first.** Data-first starves the low-noise steps of gradient by (N_t+N_s)/N_t: 41× on 2-hand, 79–97× on padded stores. Bundle's "ambient falsified" result was retracted (A 2026-09-01). Results with noise-first:
  - **Rotation, starved 50k target:** peak at σ*=2–3 (0.655 @0 → 1.137 @2, 0.767 @100). Sharpa is the exception (best at σ=25) (A 2026-09-15).
  - **Rotation @1M:** gating reverses sign.
  - **Grasp:** gating hurts monotonically.

  Branch's AGENTS.md "Standing findings" still carries the retracted "falsified" claim.

- **Selection.** The in-training best-of-10 overstates itself by +0.212 (rotation). After re-scoring, `best_rollout` ≈ `last0`, and `best_val` is slightly worse with the highest seed sd. Report `last0` (A 2026-09-11, 09-15).

- **Val store from 10M, deduplicated against 1M.** The 1M and 10M stores are independent collections, but 254 of 400 episode starts collide (CO 2026-09-05; C 111).

- **`num_workers=0`.** Workers recreated every epoch deadlocked (26 hangs). `persistent_workers` would change the shuffle stream (C 72).

- **CPU fallback raises.** 15 jobs once ran silently on CPU, 60× slower (F §2.9).

- **Co-training vs solo:** rotation −0.27 / −0.28 (50k / 1M); grasp +0.46 / +0.04 (A 2026-09-11).

---

## 8. Checking the migration

1. **Source parity:** after copying, `diff -r` Branch's `src/mjlab_hand` against Bundle's. The only differences should be the Branch-only features you deliberately kept (§3.5), and they should be off by default.
2. **Training equivalence:** same padded zarr, same seed, `--num-workers 0`, compile off, on the same machine type. The first ~50 per-step losses from migrated Branch and from Bundle should match (bitwise on the same hardware; to float tolerance across GPUs). [inferred procedure]
3. **Bundle's checks on Branch data:**
   - `check_padded_dataset.py --rows 1000` on each rebuilt zarr.
   - `check_frozen_norm.py` (artifact digests).
   - `check_ambient_sampler.py --dataset <scarce zarr> --tmin <vector>`: noise-first should show flat total mass per timestep.
   - `check_sampler.py` (DDIM).
4. **Eval parity:** load one Bundle checkpoint in migrated Branch and run `evaluate_diffusion_policy(pad=True, seed=1234, num_envs=100, num_steps=1500)`. It should reproduce that run's `final_eval.jsonl`, given the same checkpoint (not in the bundle; copy one from the cluster).
5. **Smoke run of the recipe** through Branch's Vista job script with the staged single

<!-- The text in IMG_0956.jpeg is cut off after "staged single". -->
