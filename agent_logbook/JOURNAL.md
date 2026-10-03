# Agent journal

Newest entries first. Link run/collection IDs from `RUNS.md` / `COLLECTIONS.md`.

---

## 2026-10-02 (19:15) — HANDOFF: co-training vs target-only (job 1043385) queued; what's next

State at handoff (the agent's idev box is being killed; nothing of this session runs any more):

**Running / queued**
- `1043385` (`ct-vs-to`, submitted ~16:30 by the user from `~/cross_embodied_diffusion` on
  `bundle-migration`): 10 array jobs x 4 nodes, PACK=1, `--time=09:00:00`, manifest
  `slurm_jobs/cotrain_vs_target_manifest.json` (gitignored; regenerate with
  `scripts/build_ambient_manifest.py --families Grasp InHand-Rotation --sigmas 0 100 --seeds 0 1
  --out slurm_jobs/cotrain_vs_target_manifest.json`). 40 runs = {Grasp, InHand-Rotation} x 5
  targets x sigma {0 = full co-training, 100 = target-only} x seeds {0, 1}, each on the target's
  `<Family>_pad5_scarce<Hand>_K50k` store (target 50k + other four hands at 1M), Bundle recipe
  (noise-first, frozen min/max, x0 1.0, val store, keep-last 3, target-only padded eval 32 x 1500).
  Array element a = one target (a 0-4 grasp Allegro..Wuji, 5-9 rotation); node i = task 4a+i
  (sigma0 s0, sigma0 s1, sigma100 s0, sigma100 s1). Outputs:
  `$SCRATCH/cross_embodied_diffusion/outputs/diffusion/ambient_ta/<Task>_sigma<s>_seed<k>/`.
  At 19:14 all 10 PENDING (Priority); queue median wait was ~10.5-11.3 h for 1-16 node jobs.
- Expected run time ~7 h alone/node (rotation measured 7.5 min/epoch uncompiled, ~50 epochs +
  10 evals). **Grasp speed is unmeasured** -- when the jobs start, check epoch time vs the 9 h
  limit (grasp obs 191 vs 91, heavier sim). A pending/running job's limit can only be lowered
  by the user (`scontrol update ... TimeLimit`), not raised.

**When 1043385 finishes**
1. Completion marker is `selection.json` per run (`ls .../ambient_ta/*/selection.json | wc -l`
   should be 40). Check logs `slurm_jobs/vista_train_manifest/logs/task_1043385_*.{out,err}`.
2. Fresh-seed re-score (user submits from a login node, from `~/cross_embodied_diffusion`):
   `sbatch -A ASC26008 --array=0-9 --export=ALL,RUNS='diffusion/ambient_ta/*' slurm_jobs/vista_eval_checkpoints.sbatch`
   -> `<run>/final_eval.jsonl` (best_rollout, best_val, last0; 100 envs x 1500 steps, seed 1234).
   Report `last0` (Bundle's choice), show all three.
3. Compare sigma0 (co-train) vs sigma100 (target-only) per target and family, mean of 2 seeds.
   What the user wants to know: does co-training with 4M source + 50k target beat 50k alone, as
   it did on the other compute? Bundle's logs (MIGRATION.md section 7): co-train minus solo =
   grasp +0.46, rotation -0.27 at 50k -- so expect a gain on grasp and possibly a loss on
   rotation. `scripts/plot_ambient_rot_sweep.py` (on `vista-ambient-rotation`) plots per-hand
   score vs sigma; it reads `posthoc_eval.json`/in-training rows and the old metric names --
   adapt it to `final_eval.jsonl` for these runs.
4. Promote runs worth keeping: `scripts/promote_outputs.sh diffusion/ambient_ta` (needs
   `selection.json`; checks /work quota).

**Ready but not launched (on hold per user: "don't focus on ambient sigma not 0/100")**
- `slurm_jobs/diag_rot_ta_manifest.json` (Allegro x sigma {0,2,100} x 3 seeds) and the 320-run
  `build_ambient_rotation_manifest.py --kind sweep`; `slurm_jobs/cotrain_solo_manifest.json`
  (60 runs, solo = single-hand 50k store with shared norm instead of sigma=100 -- superseded by
  1043385's design, kept for reference).

**Open issues**
- Normalized action std under our frozen min/max is 0.24 (rot) / 0.26 (grasp) vs Bundle's quoted
  0.077 (MIGRATION.md is itself inconsistent: 0.077/0.153/0.179). Unresolved; sigma values may
  not map 1:1 to Bundle's. See CHANGES item 63.
- Old 320-run tail-padded sweep `1038730` (`vista-ambient-rotation` code) was never re-scored;
  provisional plots + finding (co-training works only for the column-aligned Allegro/LEAP pair)
  are in that branch's JOURNAL 2026-10-01 21:10. Re-scoring needs a worktree of
  `vista-ambient-rotation` (steps in the 15:40 entry below). Probably superseded by 1043385.
- Branches diverged: `vista-ambient-rotation` (CHANGES item 62, sweep results, plot script) and
  `bundle-migration` (items 63-65). Merge vista-ambient-rotation into bundle-migration when
  convenient (conflicts only in CHANGES.md / logbooks). Both pushed; nothing merged to `main`.
- Rotation-Shadow val store has 19,541 steps (< 20k target; all 64 collected episodes kept).

**Session environment notes**
- The agent cannot `sbatch` from idev/compute nodes; hand commands to the user. GPU checks need
  a `gh` idev; `gg` idevs have no GPU.
- Monitoring pattern used: a bash loop (`squeue` state changes, error grep over the task logs,
  hourly digest of eval_metrics) run under the Monitor tool, re-armed every 30 min.

## 2026-10-02 (15:40) — Worktree consolidated; main checkout now on bundle-migration

- User asked why the migration lived in `$WORK/code/ced-migrate`: it was a git worktree of
  this repo (same history), used so the running sweep and its old-code evals were untouched.
  Consolidated on request: pending `vista-ambient-rotation` work committed there (`0687fc7`:
  plot script, sweep results, eval sbatch now imports its own checkout), main checkout switched
  to `bundle-migration`, worktree removed (its ignored manifests/logs copied over first; data
  is symlinked, untouched).
- **Re-scoring the old sweep (1038730) needs the old code**, which is no longer checked out.
  Do it from a temporary worktree:
  `git worktree add ../ced-old vista-ambient-rotation`, symlink `data/mjlab_hand_demos`,
  `outputs`, `logs`, `.venv` into it, submit `slurm_jobs/vista_eval_checkpoints.sbatch` from
  there (it puts that checkout's `src/` first), then `git worktree remove`.
- New runs (`cotrain_vs_target_manifest.json`, 40 runs) are submitted from the main checkout.

## 2026-10-01 (evening) — Migrating to Bundle's design (MIGRATION.md), from the doc only

- User pushed MIGRATION.md to `main`: make this repo ("Branch") behave like "Bundle" (another
  checkout of the project) for diffusion training/model/eval. Bundle's code is **not
  available** anywhere we can reach, so the user asked to re-implement it from the doc as
  faithfully as possible. Done on branch `bundle-migration` in a separate git worktree
  (`$WORK/code/ced-migrate`) so the running ambient sweep (`1038730`, old code, main checkout
  on `vista-ambient-rotation`) is untouched. CHANGES.md item 63 lists every change and marks
  the [reconstructed] choices and deviations (frozen normalizer and val store built from
  Vista data: 1M stores and fresh expert rollouts instead of Bundle's 10M stores).
- Why it matters now: Bundle reports, on the same rotation scarce-50k setup with noise-first
  ambient, sigma=0 0.655 -> peak 1.137 at sigma=2 -> 0.767 at sigma=100, while our tail-padded
  per-hand-normalized sweep sits near 0.1 at sigma=0. Earlier today I attributed that to
  target dilution (1.3% of samples); Bundle had the same dilution, so that explanation is
  incomplete. The leading candidates are the layout/normalization differences this migration
  removes (term-aligned obs, one shared frozen normalizer, unmasked padded loss).
- Verified on CPU (see item 63): store vs sources bitwise, frozen artifacts, padding plumbing
  and masked-loss gradients, noise-first sampler mass, val-split collision filter, manifest
  parse. Each sbatch now puts its own checkout's `src/` first on PYTHONPATH (the shared
  venv's editable install points at the main checkout) -- without that, jobs submitted from
  the worktree would silently run the old code.
- 2026-10-02: on a GPU idev (c610-072) every GPU check passed (CHANGES item 63, "GPU checks"):
  padded eval on all 5 hands, env-vs-store column alignment (with a negative control that
  fails as it should), rescore, val stores built (0 start collisions, ~20k steps/hand), and the
  recipe timing: 7.5 min/epoch uncompiled alone -> ~7 h per 50-epoch run.
- Pick up here (superseded parts done 10-02): needs a GPU node for (1) `slurm_jobs/vista_collect_val.sbatch` then
  `scripts/build_val_split.py` (val stores), (2) the 9-run diagnostic
  `slurm_jobs/diag_rot_ta_manifest.json` (Allegro x sigma {0,2,100} x 3 seeds) to compare with
  Bundle's 0.655 / 1.137 / 0.767, then `rescore_selected.py`. The other four
  `scarce<Hand>_K50k` stores are CPU builds (`padded_grid.py`). Old checkpoints (sweep
  `1038730`) must be re-scored from the `vista-ambient-rotation` branch.

## 2026-09-30 (20:00) — Sweep job shape: queue evidence, AGENTS.md scheduling procedure

- User: only 40 queued jobs allowed and some gh jobs already pending, so 40 x 2-node jobs
  won't fit; asked to confirm whether 8-node jobs schedule slower than 16-node ones.
- `scripts/queue_wait_stats.sh gh 3` (2026-09-30 20:00, gh 569/576 allocated, 0 idle): median
  submit->start of jobs started in the last 3 days -- 1 node 9.1 h (n=87), 2 nodes 2.6 h (16),
  3-4 nodes 12.6 h (13), 5-8 nodes 5.8 h (25), 9-16 nodes 13.2 h (5). Pending ages agree
  (5-8: 6.9 h, 9-16: 10.4 h). So 8-node jobs are NOT slower than 16-node ones here.
  Correction: an earlier coarse "2-4 nodes = 2.8 h" bin led me to recommend 20 x 4-node jobs;
  the 2-node jobs carried that bin, 3-4-node jobs waited 12.6 h. Retracted.
- Expected completion for 320 runs (run ~8.5 h at PACK=4, ~17 h at PACK=8): 10 x 8 nodes
  PACK=4 ~14 h; 20 x 2 nodes PACK=8 ~19-20 h; 20 x 4 nodes PACK=4 ~21 h; 5 x 16 nodes ~22 h.
  Recommended 10 x 8 nodes, pending `sbatch --test-only` confirmation from a login node.
- Wrote the procedure into AGENTS.md ("Getting Vista jobs scheduled fast") + CHANGES item 61.

## 2026-09-30 (evening) — Ambient minmax test runs done; sweep ready

- `1036217` (Allegro target, 50k + 4x1M, minmax, 1 run/node): all 4 COMPLETED, 2h20-2h24.
  In-training evals (32 envs, successes before drop, epochs 5..50):
  s0 0.00 0.00 0.09 0.03 0.06 0.12 0.12 0.19 0.22 0.41;
  s10 0.09 0.16 0.28 0.31 0.22 0.44 0.53 0.22 0.47 0.38;
  s25 0.84 0.72 0.81 0.50 0.69 0.59 0.62 0.44 0.50 0.62;
  s100 0.72 0.94 0.97 0.84 0.81 0.66 0.81 0.84 0.59 0.59.
  Pipeline works end to end. Target performance rises with sigma (more gating of the other
  hands = better Allegro), as in the Aug grasp result. Note the fixed ~700k-step budget:
  low sigma is still improving at epoch 50 (target ~1% of samples), high sigma peaks early and
  declines (target 50k repeated ~4000x) -- the per-run checkpoint selection matters, and a
  step-budget ablation may be worth adding later.
- Session moved to idev `1038503` on a `gg` (CPU-only) node, so reporting evals can't run here;
  wrote `slurm_jobs/vista_eval_checkpoints.sbatch` (CHANGES.md item 60).

## 2026-09-30 (09:30) — minmax fix verified; sweep manifest switched to minmax

- idev results (in-training evals, 32 envs, successes before drop): 1M plain 1.03 -> ~2.0-2.25;
  **1M padded minmax 1.38 -> ~1.6-2.1 (matches plain)**; 1M padded gaussian 0.00 (killed);
  50k plain 0.59-1.41; 5-hand pool minmax co-trained (Allegro 50k target) 0.00 -> 0.44, final
  0.31, still rising at epoch 50 -- below the 50k specialist, consistent with earlier
  "pooling doesn't help" findings (target is ~1.2% of samples). All finished (train_done.json).
- `build_ambient_rotation_manifest.py` pooled runs now pass `"source-norm": "minmax"`;
  `ambient_rot_manifest.json` regenerated (320 runs, all minmax).
- User submitted `1036217` (`diag_rot_ambient_minmax_manifest.json`, 4 x 1-node, PACK=1,
  ambient s=0/10/25/100 minmax) -- pending on priority.
- Post-hoc evals (`eval_checkpoints.py`, 100 envs, env seed 1000; successes before drop, drop
  rate in parentheses), `<run>/posthoc_eval.json`:

  | run | best_eval | best_val | latest |
  |---|---|---|---|
  | 1M plain | 1.74 (0.90), ep 60 | 2.10 (0.84), ep 173 | 1.91 (0.85) |
  | 1M padded minmax | 1.85 (0.89), ep 180 | 1.83 (0.88), ep 196 | 2.11 (0.76) |
  | 50k plain | 0.87 (0.98), ep 2000 | 0.66 (0.97), ep 559 | 0.92 (0.96) |
  | pool co-train minmax (Allegro 50k + 4x1M) | 0.15 (0.99), ep 35 | 0.10 (1.00), ep 43 | 0.40 (0.99) |

  Padded minmax = plain within noise at 100 episodes (fix confirmed). Selection by in-training
  eval is optimistic, as expected: the pool's best_eval scored 0.44 in training, 0.15 fresh.
  No checkpoint rule dominates in 4 runs; keep reporting all three.

## 2026-09-30 (00:00) — Diagnostic: the padded path is what zeroed pooled runs

- idev 1031788 ran diagnostic tasks 0-3 (RUNS.md diag-rot-idev). By epoch 60 (1M) / 800 (50k):
  plain specialists learn (1M 1.03 -> 2.25 successes before drop, drop rate 1.00 -> 0.78; 50k
  0.78-0.81); the identical data through `--pool-sources` scores 0.00 at every eval. So the
  09-28 scarce sweep's zeros come from the padded path itself, not from mixing hands.
- Isolated (CHANGES.md item 58): eval-side normalize/pad/unnormalize is exact; the model is
  worse. Two causes: the sampler's hard [-1, 1] clamp (fixed; not enough alone, 0.03 in 64 envs)
  and the GaussianNormalizer data scale vs a noise schedule/clamp tuned for [-1, 1] min/max data
  (first-step MSE 5x plain even unclamped-ish). Added `--source-norm minmax`, which makes a
  single-source pool numerically identical to the plain path.
- Killed the two gaussian padded runs (conclusive) and started, on the same GPU:
  `spec1M_padded_minmax` (should match plain 1M) and `pool_cotrain_minmax` (Allegro 50k + 4x1M,
  target eval/val) -- `slurm_jobs/diag_rot_idev2_manifest.json`, log
  `slurm_jobs/vista_train_manifest/logs/idev_1031788_diag2.log`. ~9.5 h; idev ends 2026-09-30 13:27.
- **Do not launch** `diag_rot_manifest.json` / `ambient_rot_manifest.json` as built: they use the
  gaussian default. If minmax checks out, rebuild them with `"source-norm": "minmax"`. Queued job
  `1035261` runs the old diag manifest -- recommended to the user to cancel it.

## 2026-09-29 (later) — Ambient rotation sweep: design, checkpoint selection, diagnostic first

- User asked for ambient diffusion on mixed-embodiment InHand-Rotation data: per target hand,
  sigma in {0,1,2,3,4,5,6,8,10,12,14,16,18,20,25,100}, 4 seeds = 320 runs, as 40 jobs x 2 nodes x
  4 runs/node. Confirmed with the user: sigma follows the code convention (other hands admitted
  at t >= sigma; 0 = full co-training, 100 = target-only); target 50k + other four 1M; run the
  diagnostic before the sweep.
- Checkpoint selection (user's choice): report **three** checkpoints per run, each re-scored
  with a fresh-seed eval after training: `policy_best_eval.pt` (best in-training closed-loop
  eval, now target hand only, 32 envs, 10 evals), `policy_best_val.pt` (best target-hand val
  loss, `--val-embodiment`), `policy_latest.pt`. The old selection averaged eval over all 5
  hands and val loss over the pooled set (~99% non-target), which selects for the wrong thing
  here. The post-hoc eval script is not written yet.
- Implemented CHANGES.md item 57 (ambient + padded, target-only val, manifests). Manifests:
  `slurm_jobs/diag_rot_manifest.json` (8), `slurm_jobs/ambient_rot_manifest.json` (320).
- Why the diagnostic: the 09-28 Vista scarce sweep scored ~0 on every hand, including the
  data-rich ones. Diagnostic = Allegro, seed 0: 50k and 1M specialists, each plain and through
  `--pool-sources` (node 0); 5-hand pool co-trained, ambient s=0/10/100 (node 1). Plain vs
  padded specialist isolates the padded/normalization/eval path; s=0 vs co-trained checks
  the ambient code path gives the same training as the DataLoader path.

## 2026-09-29 (later) — Multi-hand training without pre-built padded zarrs

- User asked whether multi-dataset policies can be trained by listing the per-hand datasets at
  training time and padding on the fly instead of building `padded/*.zarr`. Implemented as
  CHANGES.md item 56: `train-diffusion --dataset A.zarr B.zarr ...` pools in memory with the
  same code the builder now uses (`mjlab_hand.diffusion.pooling`).
- Correctness, all on Vista (login node, CPU):
  - Pre-refactor builder vs new builder vs in-memory pool on the 5 rotation 10k sets:
    bitwise-identical arrays and metadata.
  - `scripts/check_pooling.py` on all 14 existing `padded/*.zarr` (built on the old x86
    cluster): **14/14**. With the recorded stats, padding/concatenation is bitwise identical
    for every array. Refitting the stats on Vista gives means that differ by float32 rounding
    (<=2.4e-7; stds exact), so normalized values differ by <=5.7e-6. `DiffusionDataset`
    episodes, action masks, source ids, 10% val split and balanced weights are identical
    (e.g. Grasp-AllHands: 9045 train / 1005 val episodes, 4,500,570 / 501,038 windows).
  - `train-diffusion` with 0 epochs: `--dataset A..E` writes the same `source_stats.json` and
    val split as the equivalent zarr. A pooled batch through 2 DataLoader workers +
    `compute_loss(action_mask=...)` + backward: masks 22/26/28, loss and grads finite.
  - Not done: a real GPU training run on a pooled dataset.
- `--pool-sources` with one dataset trains a single hand through the padded path. That is the
  control for the ~0 scarce co-training results (see the 09-28 Vista sweep in RUNS.md): it
  separates "padded/Gaussian path broken" from "pooling hands hurts".

## 2026-09-29 — Vista storage: demos staged off `$WORK`, outputs on `$SCRATCH`

- User asked that training jobs copy `data/mjlab_hand_demos` to `$SCRATCH` (node `/tmp` for
  multi-node) and read it there, write outputs/checkpoints to
  `$SCRATCH/cross_embodied_diffusion/outputs`, and have a script to copy runs worth keeping to
  `$STOCKYARD/vista/cross_embodied_diffusion/outputs`. Motivation: the `$WORK` quota incident
  in the entry below.
- Done in CHANGES.md item 55. Only `vista_train_manifest.sbatch` changed; the other
  `train_*.sbatch` are for the original shared-node cluster (partition `allnodes`, no
  `$SCRATCH`) and were left alone.
- Stages per dataset the node's runs actually use, not all 25 GB of `mjlab_hand_demos`
  (`padded/` alone is 21 GB). Runs execute from a symlinked run root so manifests and
  `train_config.json` stay repo-relative.
- New `train_done.json` completion marker; `scripts/promote_outputs.sh` promotes only runs
  that have it (older runs need `--force`).
- Verified on a login node (fake `$SCRATCH` in a temp dir): staging in both modes, dry-run
  commands, promote script on fake runs + the real `/work` quota check, and a real 1-epoch
  CPU `train-diffusion` through the run root: the dataset loaded from the staged copy and
  `train_config.json` landed on scratch with repo-relative paths, but the CPU epoch hit my
  15-min timeout, so the `train_done.json` write itself was not exercised.
- Pick up here: first real Vista submission with this sbatch; check the `[INFO] staged ...`
  lines for staging time, then `scripts/promote_outputs.sh` the finished sweep.

## 2026-09-28 — Vista job shape: multi-node jobs, benchmark, `$WORK` quota scare

- User asked for the configuration that runs the most diffusion-BC training runs at once on
  Vista, given the 40-submitted-job cap: packed 1-node jobs or multi-node jobs. Worked in a
  4-node `gh-dev` idev allocation (`1031790`).
- Key limit (`sacctmgr show qos qgh`): 20 running / 40 submitted jobs, but **96 running nodes
  per user, 64 per job**. 1-node jobs top out at 20 nodes; multi-node jobs reach 96 (4.8x).
- Made `vista_train_manifest.sbatch` multi-node aware (CHANGES.md item 54): `-N K` fans out
  one packed runner per node via `srun`. With `-N 1` it behaves as before.
- Benchmark (ANALYSIS.md "Vista job shape"): per-node throughput flat at ~0.37 run-ep/s
  (training) from 4 to 16 runs/node, with 4 nodes loaded at once. One 4-node sbatch job ran
  32/32 runs, each node within 4% of a single node. Queue snapshot: jobs up to 16 nodes start
  about as fast as 1-node ones; 32+ node jobs can wait a day.
- Recommendation: 8-16-node jobs with 8 runs/node (~24 h per 4000-epoch-equivalent run), or
  PACK=2 on more nodes for turnaround. SU per run is unchanged, ~3 SU each. ASC26008 has
  4,123 SU, which 96 nodes spend in ~43 h.
- **Incident:** round 1's first attempt wrote checkpoints to `outputs/` on `$WORK` and pushed
  the 1 TB user quota over (runs died in `torch.save`, "unexpected pos"). I killed it, deleted
  its output and reran everything on `$SCRATCH`. Four killed processes stayed stuck in Lustre
  `cl_sync_io_wait` for the rest of the allocation (harmless). Then found that user job
  `1030560` (scarce sweep, 40 runs) would hit the quota at its next snapshot. On the user's
  instruction, deleted its `policy_epoch_0005/0010/0015.pt` (33 GB), leaving 34 GB free.
- Not verified: the sbatch through real `sbatch -N K` from a login node (sbatch is disabled
  in idev). I exercised the same path by calling the script inside the allocation.

**Pick up here:**
1. `1030560` needs ~66 GB more to reach epoch 50 and has ~34 GB: it will hit the quota around
   epoch 35-40 (~6-8 h after 13:00 on 09-28). Free more space on `$WORK` or delete the
   epoch-20/25 snapshots before then.
2. Before any big multi-node sweep, decide where checkpoints go (`$SCRATCH` + copy the
   selected ones back, or fewer saved checkpoints). ~0.8 GB/run minimum.
3. `train.py` prints without flush, so log timestamps under a pipe are approximate. Consider
   `flush=True` or `PYTHONUNBUFFERED=1` in the sbatch.

---

## 2026-09-27 (later) — Set up TACC Vista (GH200, aarch64) as a second training server

- Bulk storage on `$WORK` (`/work/09312/rudolph/vista/cross_embodied_diffusion/`), symlinked
  in as `data/mjlab_hand_demos`, `logs`, `outputs`, `.venv`. `logs` symlink added to
  `.git/info/exclude` (`.gitignore`'s `logs/` doesn't match a symlink). `$WORK` quota was at
  926 GB / 1 TB before the pull, so it is shared with other projects and tight.
- `hf_sync.py pull` (everything): 85 files, ~4 min download plus untar, 25 GB. All 10 x 1M, subsets
  10k/50k, `padded/` (AllHands + Scarce pools) and all 10 RL experts are present.
- `uv sync --frozen` installed CPU-only torch on aarch64 → fixed with the cu128 index
  (CHANGES.md item 50). Verified torch 2.10.0+cu128 + warp on the GH200.
- Smoke-trained Grasp-Allegro 50k for 4 epochs with val + env eval: works end to end. Throughput
  probe → packing runs per node (CHANGES.md item 51); new `vista_train_manifest.sbatch` verified
  by running its body inside the idev node (4 parallel runs, env eval each epoch).
- Estimates at PACK=2 (4 runs/node): 50k x 4000 epochs ≈ 18h; 1M x 200 epochs ≈ 5h solo,
  ~10h packed. Budget: project ASC26008 has ~4,350 SU (1 SU per GH node-hour); the other
  project is overdrawn.
- Verified evaluation: RL expert (`scripts/eval_expert.py`, Grasp-Allegro `model_9999.pt`) 100%
  (32 eps). 150-epoch Grasp-Allegro 50k diffusion policy (`outputs/diffusion/_vista_verify/`):
  in-training eval 78% (32 eps); standalone `eval-diffusion` 84% (32 eps, same settings),
  59% (64 eps, 2000 steps), 60% (256 eps). Eval is not bit-deterministic and 32-ep numbers are
  noisy. Rendering works headless (EGL), both `render_diffusion_rollout` and
  `scripts/render_checkpoint_video.py` (RL checkpoints only -- it dies with
  `KeyError: 'actor_state_dict'` on a diffusion policy).
- Runs-per-node sweep (table in `vista_train_manifest.sbatch` header): GPU-bound, node
  throughput plateaus ~1.9x from 6 runs. Recommended PACK=2 (4 runs) or PACK=4 (8 runs, max
  that fits 48h for 50k and RAM for the largest pool). `torch.compile` (1.34x train step, 2x
  val in the 09-13 prototype) is still not wired into `train.py` -- the next lever.
- User asked to try torch.compile: wired in as `--compile-mode` (CHANGES.md item 53).
  `reduce-overhead` (CUDA graphs) = 3.1 s/epoch vs eager 7.0 (2.3x single run); node plateau
  0.368 vs eager TF32's 0.282 run-epochs/s (1.3x). Quality unchanged (256-ep eval 56.6% vs
  53.1%). Needed `CC=gcc`, and a restore of the TF32/cudnn flags around in-training eval: env
  setup flips them via the new API and Inductor then crashes on recompile. Found in passing
  that eager runs switch FP32 -> TF32 at their first eval (unchanged). Vista sbatch now compiles by
  default; verified its full path (4 parallel runs, eval every epoch, render).
- Not submitted: this session ran inside idev, where `sbatch` is disabled. Handed submission
  commands to the user.

---

## 2026-09-27 — Git slowness fixed; demos + RL experts mirrored to Hugging Face; docs restructure

- `git status`/`add` hung for minutes: untracked `tmp/` (per-task Slurm `TMPDIR`, 26k+ files
  on NFS) was not ignored. Added `tmp/` to `.gitignore` (commit `ed33c20`).
- User wanted the main diffusion training datasets and the RL policies that collected them
  available on other servers (not the BC models). Added `scripts/hf_sync.py`; mirror is the
  private dataset repo `maxrudolph/mjlab-hand-demos`. Details in `COLLECTIONS.md`.
- `huggingface_hub` is not a project dependency; installed into `.venv` with
  `uv pip install --index-url https://pypi.org/simple huggingface_hub` (the pyproject's Aliyun
  mirror fails TLS). v2.0 has no `upload_large_folder`, so `push` is resumable per-archive.
- User will clone the repo and pull the data on another server, and wants others to be able
  to use the repo too: all data paths are now repo-relative with per-machine symlinks
  (`data/mjlab_hand_demos`, `data/hf_staging`). Rewrote the `/datastor2/...` paths in 8 sbatch
  files, the scarce manifest builder and both untracked manifests (job `97798` was running;
  the new path resolves to the same files). Documented in README "Data layout" and AGENTS.md.
- Deleted the 25 GB of staging tars after the verified upload; `data/hf_staging` symlink kept
  (now holds only `README.md`/`push.log`), `stage` recreates the tars when needed.
- Docs restructure (CHANGES.md item 49): new tool-neutral `AGENTS.md` is the single source;
  `CLAUDE.md` is just `@AGENTS.md`, the Cursor rule points to it. Fixed `CHANGES.md`/
  `ANALYSIS.md` links (they assumed the logbook files sat beside them), brought the protocol to
  cover `CHANGES.md`/`ANALYSIS.md`, backfilled `CHANGES.md` items 41-48 for 09-03 → 09-27 from
  commit diffs, replaced the stale "snapshot 2026-08-24" headers in `RUNS.md`/`COLLECTIONS.md`,
  and put the two 2026-09-26 entries below back in newest-first order.

---

## 2026-09-26 (later) — Relaunched rotation specialists as job `97798` (8 parallel)

Shortened `TMPDIR` in `train_rotation_specialist.sbatch` to `$PWD/tmp/rs_N` and set
`--array=0-9%8`. The scarce/grasp sweep's `scarce_specialist_task_N` path was exactly 107 chars
(the limit) for single-digit tasks and 108 for two-digit ones -- that sbatch will hit the same
bug for tasks >= 10 if reused. See `RUNS.md`.

**Pick up here when `97798` finishes** (expected ~20h for 1M tasks, ~26h for 50k tasks, from
2026-09-26):
1. `sacct -j 97798 -X` -- any TIMEOUT (36h limit; 50k tasks run 2x4000 epochs sequentially)?
   If a seed1 run was cut off, resubmit just that run.
2. Each `outputs/diffusion/specialist/InHand-Rotation-*_seed{0,1}/` should have
   `policy_latest.pt`, `policy_best_val.pt`, `policy_best_eval.pt` + `best_*.json` -- first real
   exercise of that code, verify it.
3. Tabulate `eval_metrics.jsonl` (avg_successes_before_drop) vs the old `*_400k` baselines
   (LEAP 2.31, Allegro 1.62), then fill in the rotation rows of the baseline inventory.
4. Remaining gaps: Grasp-Allegro 50k (both seeds), grasp seed1 runs cut short
   (Shadow 1M, Sharpa 50k, Wuji 50k), and the whole scarce co-training sweep (fix its TMPDIR
   first).

---

## 2026-09-26 — Audit: rotation specialist backfill `94864` produced nothing

Tasks 0-7 hung ~10-14h each on `OSError: AF_UNIX path too long` (per-task `TMPDIR` under the
repo path is too long for torch DataLoader's multiprocessing socket), cancelled 2026-09-23.
No rotation specialists exist at any scale; details in `RUNS.md`. Current baseline inventory
(grasp specialists 1M x5 hands x2 seeds all ~1.0 success; 50k 4/5 hands, Allegro missing;
pooled AllHands padded ~0 everywhere; rotation only `*_400k` Allegro/LEAP) summarized for the
user.

---

## 2026-09-20/22 — Audited `89262`/`90599` sweep outcome; three-way checkpoint selection; launched InHand-Rotation specialist backfill

Checked actual job/checkpoint state (`squeue`, `sacct`, `find outputs/diffusion`) against
`RUNS.md`'s stale "pending on cluster priority" note for the scarce/specialist sweep -- the
sweep was cancelled 2026-09-14, not pending. Full outcome, and what checkpoints currently exist
across every `outputs/diffusion/*` dir, recorded in `RUNS.md` (2026-09-22 edits). Headline: the
InHand-Rotation specialist configs (1M + 50k per embodiment) never ran at all -- 0/20 runs.

**Checkpoint selection was undefined before this.** `train.py` only ever wrote `policy_best.pt`
selected by lowest *training* loss (one-step noise-prediction loss on a random timestep, not a
real action-quality signal, and rarely saved due to the `latest_every_epochs` gating -- several
existing runs, e.g. `Grasp-Shadow_50k_seed0`, have no `policy_best.pt` at all) plus periodic
`policy_epoch_NNNN.pt` snapshots nothing ever selected among. User asked for three explicit,
named selections instead:

- `policy_latest.pt` -- last epoch (unchanged).
- `policy_best_val.pt` (+ `best_val.json`) -- lowest `val/action_loss`, which is
  `DiffusionPolicy.action_reconstruction_loss`: full DDIM `predict_action` rollout compared to
  the expert action in real units, *not* `compute_loss`'s one-step noise-prediction loss. Only
  populated when `--val-fraction > 0` (the specialist manifest sets 0.1, so rotation runs get
  it).
- `policy_best_eval.pt` (+ `best_eval.json`) -- highest mean env-eval headline
  (`success_rate` / `avg_successes_before_drop`) across `eval_specs`, copied from the weights
  just scored at that eval point.

Removed the old train-loss `policy_best.pt` (grepped for readers -- none in `src`/`scripts`;
only pre-existing `.err` log files reference the old save line, harmless). Change is in
`src/mjlab_hand/diffusion/train.py`; syntax-checked only (`ast.parse`), not yet exercised by a
real training run before the launch below -- first real test is job `94864`.

Also changed eval cadence in `build_scarce_specialist_manifest.py` from `epochs // 4` to
`epochs // 10` at user's request (more, cheaper eval checkpoints to select `policy_best_eval.pt`
among -- at `// 4` there were only 4 candidates per run, too few to trust over noise).

**Launched the InHand-Rotation specialist backfill** identified above: extended
`build_scarce_specialist_manifest.py` with `--family`/`--kind`/`--out` so it can emit a subset
manifest (previously always wrote the full 80-run sweep). Built
`slurm_jobs/rotation_specialist_manifest.json` (10 tasks / 20 runs, `--family InHand-Rotation
--kind specialist`), new `slurm_jobs/train_rotation_specialist.sbatch` copied from
`train_scarce_specialist.sbatch` with the same `LD_LIBRARY_PATH`/per-task
`WARP_CACHE_PATH`/per-task `TMPDIR` fixes, no node pin (matches `train_cross_embodiment.sbatch`
precedent -- pure GPU training, only a small periodic eval rollout touches `mujoco_warp`).
Submitted as **job `94864`**, `--array=0-9%10` (only 10 tasks exist, so `%10` is a no-op cap,
kept explicit per user's "only launch 10 in parallel"). 4 tasks started immediately on
`slurm-node-001`/`002`; rest pending on resources as of submission. Not yet watched to
completion -- next session should check `sacct -j 94864` and
`outputs/diffusion/specialist/InHand-Rotation-*` for the three new checkpoint files before
treating this as done, and should confirm `policy_best_val.pt`/`policy_best_eval.pt` are
actually being written (first real exercise of the new selection code).

---

## 2026-09-13 — Prototyped `torch.vmap`-over-seeds training; negative result at production batch size

User asked how to get more performance per GPU (fit more per GPU, is the pipeline compilable,
should we switch to JAX). All jobs here run on A40s (`sinfo`: every partition is
`gpu:nvidia-A40:8`), and `ConditionalUnet1D` (`down_dims=(256,512,1024)`, a few M params) looked
small enough that `batch_size=256` might leave the GPU launch/memory-bound rather than
compute-bound -- in which case training N seeds as one `torch.func.vmap`-batched step (stack
per-seed params via `stack_module_state`, `functional_call` + `vmap`, no JAX needed) should give
"extra seeds for free" versus the sweep's current sequential-per-array-task seed loop
(`run_manifest_task.py`). Prototyped in `scripts/prototype_vmap_seeds.py`.

**Mechanism validated.** Two in-process monkeypatches were needed, neither touching the real
`model.py`/`policy.py` (both files are actively imported by the running sweep, `89262`/`90599`):
`ConditionalUnet1D.forward`'s `moveaxis` has no vmap batching rule in torch 2.10 (swapped for
`transpose`, identical result); `DiffusionPolicy` has no `forward` (only `compute_loss`), and
`functional_call` always invokes `forward`, so `DiffusionPolicy.forward = DiffusionPolicy.compute_loss`
for the duration of the script. Correctness was checked on CPU (fp32, `atol=1e-5`) comparing
vmapped output per seed-slice against that seed's own un-vmapped forward on identical input --
passed (max diff 4e-6). The same comparison on GPU showed ~1e-3 max diff; that's vmap's batched
conv/GroupNorm kernels taking a different (still correct) cuDNN path than the per-sample kernels,
confirmed to be numerical-path noise rather than a logic bug by isolating the check to CPU.

**Throughput result: no benefit at `batch_size=256` (the project default), real benefit at small
batch.** Measured ms/step, vmapped vs. `n_seeds` sequential single-seed steps, same A40:

| batch_size | n_seeds=2 | 4 | 8 | 16 |
|---|---|---|---|---|
| 16  | 1.16x | 1.54x | 1.76x | 1.84x |
| 256 | 0.87x | 0.95x | 0.97x | 1.00x |

i.e. at `batch_size=16` (launch-overhead-bound regime) vmap-over-seeds is a real, growing win, up
to 1.84x at 16 seeds -- this is the mechanism actually working. At `batch_size=256`, the batch size
every current config in this repo actually trains with, there is roughly zero speedup (sometimes
slightly negative): a single seed's step at batch 256 is already compute-bound on an A40, so
running N of them as one bigger vmapped batch doesn't buy anything, and vmap's grouped-conv
batching rule uses ~30-45% more peak GPU memory than sequential for the same n_seeds (e.g. 11.9GB
vs 8.3GB at n_seeds=8) since `stack_module_state` keeps every seed's params+activations live at
once. **Conclusion: `vmap`-over-seeds is not a good fit for the current sweep as configured** --
don't wire it into `train_scarce_specialist.sbatch` or similar. It would matter if a future job
genuinely wanted many small-batch runs concurrently on one GPU.

**Follow-up same day: bf16 autocast also doesn't help, and the real headline finding is that a
single network at `batch_size=256` already saturates one A40.** Extended the same script:

- `[n_seeds=1,8]` at `batch_size=256`, `torch.autocast(dtype=torch.bfloat16)` vs fp32: **slightly
  slower**, not faster (29.72ms -> 32.27ms solo; 238.8ms -> 259.2ms at 8 sequential seeds). This
  model's compute is small conv1d/GroupNorm/Mish blocks, not large GEMMs -- there's no big matmul
  for A40 tensor cores to accelerate, GroupNorm runs in fp32 under autocast regardless, and the
  cast machinery is pure overhead here. AMP is not a free win for this architecture.
- Re-read the batch_size=256 table from the first experiment with this framing: sequential-vs-vmapped
  time scales almost exactly linearly with `n_seeds` (0.97x-1.00x "speedup", i.e. none) *for both
  methods*. That's not "vmap doesn't help" so much as **one network at batch=256 already keeps this
  A40 ~fully compute-utilized** -- there is no idle capacity left for a second network to use
  regardless of how it's scheduled (vmap, threads, or separate processes), so nothing can increase
  networks-per-GPU at the production batch size without slowing every network down proportionally.
- Quantified how bad the small-batch tradeoff actually is: vmap+bf16 at `batch_size=32`, `n_seeds=16`
  hits 2079 samples/s *combined* across all 16 seeds (~130 samples/s per seed); a single standalone
  network at `batch_size=256` alone does ~8615 samples/s. So packing seeds via small-batch vmap
  recovers *some* of the overhead small batches pay, but per-network training is still ~66x slower
  in samples/sec than just running that one network at the batch size already used in production --
  confirms it's not a viable way to finish the sweep faster, only a way to make an otherwise-forced
  small-batch regime less wasteful. `n_seeds=32` at `batch_size=32` OOM'd (~1.49GiB/seed x 32 > 46GiB),
  so ~16-24 seeds is this A40's ceiling for that regime anyway.

**Revised bottom line:** at this project's actual batch size, this A40 has no spare compute to give
away -- vmap, AMP, and (by construction) plain multi-process sharing are all dead ends for "more
networks per GPU" here. The only levers left that could still make single-network training faster
(which is the only way to get more networks *done* per GPU-hour) are `torch.compile` (kernel fusion
that reduces Python/launch overhead without changing numerics or batch size -- untested) and reducing
the model's own per-step cost (fewer channels, fewer diffusion train/inference steps) or using more
GPUs (raise the Slurm array throttle). None of this motivates a JAX rewrite; `mujoco_warp` is
Warp/CUDA, not JAX, so there's no ecosystem synergy pulling that direction either.

**Follow-up same day: `torch.compile` is a genuine win, unlike vmap/AMP.** Prototyped in
`scripts/prototype_torch_compile.py`. `DiffusionPolicy` has no `forward` (only `compute_loss` /
`predict_action`), and `torch.compile(module)` only intercepts `forward`/`__call__`, so compiling
the *bound methods* directly (`torch.compile(policy.compute_loss)`) is what actually traces them --
compiling the module itself would silently compile nothing.

Correctness: bit-exact comparison of the full `compute_loss`/`predict_action` against eager isn't
meaningful -- both draw internal `torch.randn`, and Inductor's Philox-based random codegen doesn't
consume the RNG stream identically to eager even with the same seed (documented `torch.compile`
behavior). A first attempt at seeding both sides and comparing losses showed ~5e-3 diff even on CPU,
which looked like a bug; isolating the check to the deterministic core (`noise_pred_net.forward`
called directly, no internal randomness) on CPU showed 3.8e-6 diff -- confirms the compiled graph is
correct, and the earlier "diff" was RNG-consumption mismatch, not a logic bug.

Throughput at the project's actual `batch_size=256` on the same A40:
- `compute_loss` (training step): eager 24.24ms -> compiled 18.10ms, **1.34x**.
- `predict_action` (DDIM sampling loop, batch=16 -- used by both periodic `--eval-spec` rollouts and
  the new `action_reconstruction_loss` validation metric): eager 87.70ms -> compiled 43.52ms,
  **2.02x**. Bigger win here because it's 16 sequential UNet calls; fusing each call's small ops
  (GroupNorm -> Mish -> scale/bias -> residual add) compounds over the loop. One-time compile cost
  ~32s wall -- negligible against multi-hour jobs.

**This is the one lever from this investigation that's actually worth wiring into production.**
Proposed integration point: `policy.noise_pred_net = torch.compile(policy.noise_pred_net)` right
after construction in `train.py` -- `noise_pred_net` *is* a real `nn.Module` with `forward`, so
compiling it directly (rather than the wrapper methods) means `compute_loss` and `predict_action`
both pick up the speedup for free through their existing `self.noise_pred_net(...)` calls, no
change to either method's code. Not yet wired in -- `train.py` is actively imported by the running
sweep (`89262`/`90599`); should be done as a separate, deliberate change once those jobs are clear,
not hot-patched under them. Caveat to check before wiring in: recompilation triggers on any
batch-size change (e.g. a shorter last batch if a loader isn't `drop_last`), so worth confirming the
train/val loaders always feed a constant shape before enabling this broadly.

---

## 2026-09-13 — Job `89262`: 36/40 tasks failed on node-local /tmp exhaustion; fixed + resubmitted

Checked in on the scarce-co-training/specialist sweep (job `89262`, see the previous entry) a
day after submission: only 4 of 40 tasks (`0`, `2`, `3`, `4`) were actually healthy and training;
the other 36 all failed within seconds to a minute of starting, all on `slurm-node-005`.

Root cause, from the `.err` logs: `OSError: [Errno 28] No space left on device` inside wandb's
service subprocess (`tempfile.TemporaryDirectory()` in `wandb/sdk/lib/service/service_process.py`),
which resolves to node-local `/tmp` by default. `train_scarce_specialist.sbatch` already gave
each array task its own `WARP_CACHE_PATH` (the established fix for concurrent-array-task races,
see `CLAUDE.md`), but never touched `TMPDIR` -- with several of the up-to-20 concurrent tasks
landing on the same node (`slurm-node-005`), node-local `/tmp` filled up and every `wandb.init()`
after that point died. The 4 survivors had already gotten past that step before the node's `/tmp`
ran out. This is a new, more general finding than the existing Warp-cache one: **any array job
with wandb logging AND real concurrency needs a per-task `TMPDIR` on shared storage, not just a
per-task Warp cache** -- added to `CLAUDE.md`'s required-env-vars block so it's not missed again.

**Fix:** added `export TMPDIR="$PWD/tmp/scarce_specialist_task_${SLURM_ARRAY_TASK_ID}"` (shared
storage, same pattern as `WARP_CACHE_PATH`) to `train_scarce_specialist.sbatch`. Resubmitted only
the 36 failed indices against the *same* manifest (`sbatch --array=1,5-39%16 ...`) -- job `90599`.
Confirmed via `scontrol show job` that Slurm's `SLURM_ARRAY_TASK_ID` for an explicit index list
still resolves to the real index values (not renumbered), so each resubmitted task looks up the
correct manifest entry and writes to the same output dir the original plan intended. Throttled to
`%16` rather than `%20`, since 4 tasks from job `89262` are still running toward the user's
"20 concurrent" cap -- `16 + 4 = 20` keeps the combined ceiling correct across both job IDs.
Wasted compute from the failures was minimal (all died in <1 minute); the 4 survivors are left
running untouched (task `0`, e.g., is a healthy Grasp-Allegro 50k specialist run, epoch 93/4000,
loss and val_action_loss both looking reasonable).

---

## 2026-09-12 — Scarce co-training + full specialist sweep, 80 runs / 40 Slurm tasks

User: train specialist policies (per task/embodiment, at 50k and 1M) and a new generalist
"scarce co-training" scheme -- one embodiment's data capped at 50k, the other four at full 1M,
pooled into one policy -- and test whether that scarce source should be sampled uniformly
(proportional to its row count, i.e. drowned out) or upweighted to equal representation.
Scoped via clarifying questions first (see conversation): confirmed "scarce co-training" is a
genuinely new mixed-scale-per-source scheme, not a rerun of the existing uniform-scale pooled
training from 2026-09-10/11; confirmed building the missing 50k single-embodiment datasets
rather than leaving 3 of 5 hands 1M-only; confirmed all new runs use the validation split added
earlier today. User then set final scope: 2 seeds (not 3), 2 runs packed per Slurm task, capped
at 20 concurrent.

**New code**, additive to the val-split work above:
- `DiffusionDataset.source_id_per_window` + `.source_sample_weights(mode)`: per-window source id
  (reusing `_episode_source_ids`, the same source-lookup helper the val split uses -- falls back
  to a single group for a plain non-mixed dataset) and a weight array for a
  `WeightedRandomSampler`. `"uniform"` (default, `None`) is the existing plain-shuffle behaviour,
  unchanged. `"balanced"` gives every source equal *expected* representation per epoch
  regardless of row count -- the actual mechanism that makes "scarce" sampling ratio testable;
  without it a 50k source in a ~4M pool gets ~1.2% of batches by construction.
- `TrainConfig.source_sample_mode` (`"uniform"`|`"balanced"`) + `--source-sample-mode` CLI flag,
  wired into the non-ambient DataLoader branch (mutually exclusive with ambient sampling, same as
  the existing `shuffle=True` path it replaces).

**Verified via smoke tests** before touching real data: a synthetic 20/400-episode 2-source
store confirms `"uniform"` reproduces each source's raw row-count share (~4.76% observed
~4.96%) while `"balanced"` gives ~50/50 regardless of the 20x size imbalance; a full 2-epoch CPU
`train_diffusion` run combining `source_sample_mode="balanced"` with `val_fraction=0.2`
completes cleanly (the two new features don't interact badly). Then **on real data**
(`Grasp-Scarce-Allegro.zarr`, ~4.05M steps): built train (3,645,950 windows) and val (404,146
windows) splits confirming ~10% held out per source including the scarce one (5,000 of ~50k
Allegro windows, not zero and not all of it); a `WeightedRandomSampler` batch's `action_mask`
column sums confirm every source contributes ~1/5 of the batch under `"balanced"` regardless of
its 1M vs 50k size; `compute_loss` and the new `action_reconstruction_loss` both run cleanly on
a real masked batch.

**Datasets built** -- see `COLLECTIONS.md` for the full table: 10 fresh 50k single-embodiment
sets (superseding stale/missing ones from the 2026-08-25 vintage) and 10 scarce-co-training
pools (`padded/<Family>-Scarce-<Hand>.zarr`, one per choice of scarce embodiment x task family).

**Job launched**: `scripts/build_scarce_specialist_manifest.py` generates
`slurm_jobs/scarce_specialist_manifest.json` -- 40 array-task entries, each a list of 2
`train-diffusion` arg-dicts (the config's 2 seeds). `scripts/run_manifest_task.py` (has
`--dry-run`, used to verify a few representative tasks' generated commands before submitting)
runs one task's 2 runs sequentially. `slurm_jobs/train_scarce_specialist.sbatch`, job `89262`,
`--array=0-39%20` (confirmed via `scontrol show job`: `ArrayTaskId=0-39%20
ArrayTaskThrottle=20`), no node pin (pure training + small periodic eval, per the node-011
scoping memory). Epoch counts follow this project's ~780k-gradient-step convention: 200 for 1M
specialists, 4000 for 50k specialists, 50 for the ~4M scarce pools. `val-fraction=0.1` on every
run, with `val-seed` fixed at 0 across both seeds of a config (so seed 0 and seed 1 of the same
dataset validate on the *identical* held-out trajectories -- their val losses are directly
comparable; only the training seed differs between the two runs packed in one task).

Breakdown: 20 tasks / 40 runs specialist (10 combos x {50k, 1M} x 2 seeds), 20 tasks / 40 runs
scarce co-training (2 families x 5 scarce-hand choices x {uniform, balanced} x 2 seeds).

---

## 2026-09-11 — Real validation loss: held-out trajectories, denoised-action MSE

User pointed out the existing `"train_loss"` in `eval_metrics.jsonl` isn't a validation loss at
all -- no held-out split existed, and it's a noise-prediction MSE, not a measure of actual
action-reconstruction quality. Asked for both fixed: a val split from held-out *trajectories*
(not held-out individual states), and a validation loss computed over **denoised actions**, not
predicted noise.

- `DiffusionDataset` gained `split`/`val_fraction`/`val_seed`. Split is at the **episode** level,
  not the window level -- splitting individual (obs, action) windows would let near-identical
  neighboring states from the same trajectory leak across train/val, understating how much the
  val loss actually measures generalization. `_train_val_split` partitions episodes with a seeded
  `np.random.default_rng(val_seed).permutation`, **stratified per source** (`_episode_source_ids`,
  reusing `source_real_dims()`/`source_step_bounds()` with a single-group fallback for plain
  datasets) so a cross-embodiment or onehot-mixed dataset holds out trajectories from every
  embodiment, not just whichever source happens to shuffle to the front. `val_fraction=0.0`
  (default) is a pure no-op -- every existing caller/script that doesn't pass these kwargs trains
  on 100% of episodes exactly as before; confirmed via smoke test that a plain `DiffusionDataset()`
  call still gets all episodes.
- `DiffusionPolicy.action_reconstruction_loss`: runs the *actual* DDIM reverse-diffusion sampling
  (`predict_action`, same code path used at rollout time) on held-out obs windows, then MSE against
  the ground-truth action -- both in real (unnormalized) action units. Deliberately not a
  cheaper single-step x0 estimate from one random noisy timestep (which the training loss's
  machinery would have made trivial to bolt on) -- the user asked for denoised actions, and the
  thing a BC policy is actually judged on is what it outputs after full sampling, not a partial
  one-step reconstruction.
- `TrainConfig` gained `val_fraction`/`val_seed`/`val_every_epochs`/`val_max_batches` (default 20 --
  DDIM sampling is far pricier per batch than one training step, so validation is capped to a
  small slice of an epoch's compute rather than doubling it) and CLI flags in
  `cli/train_diffusion.py`. Val loss is written to a new `val_metrics.jsonl` (epoch,
  val_action_loss) and wandb (`val/action_loss`), printed alongside the train-loss line each
  epoch. Deliberately did **not** change `policy_best.pt` checkpoint selection (still lowest
  train loss) -- the ask was for a validation signal to look at, not a change to what "best"
  means; flagged to the user as an easy follow-up if wanted.
- Normalizer fitting (`LinearNormalizer.fit(dataset.obs/.action)` in the non-padded path) now
  fits from the train-split dataset only when `val_fraction > 0`, not the full store -- otherwise
  val trajectories' own statistics would leak into the normalization the model is trained under,
  which is itself a subtle form of train/val contamination independent of the loss/split fix.

**Verified via a synthetic smoke test** (not committed -- ad hoc script, small random
single-embodiment + fake 2-source mixed `TrajectoryStore`s): train/val episode sets are disjoint
and sum to the full episode count; `val_fraction=0` reproduces the old all-episodes behavior
exactly; `split="val"` with `val_fraction=0` raises rather than silently returning nothing;
the 2-source split gives both sources nonzero held-out episodes; a real 2-epoch CPU
`train_diffusion` run with `val_fraction=0.2` writes non-NaN `val_action_loss` to
`val_metrics.jsonl` every epoch. Not yet run against a real dataset/GPU job -- next real
cross-embodiment or scaling run should pass `--val-fraction` to pick this up.

---

## 2026-09-10 — New cross-embodiment BC scheme: pad-to-max + per-source static normalization + masked loss

User: one diffusion policy per task family (Grasp, InHand-Rotation) that takes obs from an
arbitrary embodiment and outputs its action -- via padding to the family's max obs/action dim,
careful static per-dataset normalization, and correct loss masking. Confirmed scope first
(per-task-family pooling, not one policy across both tasks) via a scoping question -- the
obs/action dims differ enough between task families (grasp max 189/28 vs rotation max 89/22)
that a single global pad would waste most of rotation's capacity on grasp-only padding.

This is a deliberate reversal of `build_mixed_dataset.py`'s stance ("refuses mismatched spaces
rather than zero-padding") for a good reason: that script mixes exactly 2 same-dim embodiments
with onehot conditioning, where padding really would conflate unrelated physical quantities.
Here the pool is all 5 embodiments of one task, which necessarily have different dims, and
zero-padding to the family max plus a real/pad mask is the only way to share one network.

**New pieces** (all additive, existing single-embodiment / 2-source-mixed training paths
unaffected -- verified via a full small-scale smoke test before running on real data):

- `GaussianNormalizer` (`normalizer.py`): mean-0/var-1, `.fit()` per source dataset (never on
  the pooled mixture), `.identity()` for a no-op pass-through.
- `scripts/build_padded_dataset.py`: takes N source `.zarr` dirs (one task family, all 5
  hands), fits a `GaussianNormalizer` per source from that source's own raw data, normalizes,
  zero-pads obs/action to the family's max width (real dims front-packed, pad at the tail),
  concatenates. Writes `extra.padded=True` + per-source `{embodiment, obs_dim, action_dim,
  obs_mean/std, action_mean/std}` -- this provenance is what makes the loss mask and eval-time
  normalization reproducible from stored data alone.
- `TrajectoryStore.source_real_dims()`: reads that per-source real-dim provenance (parallel to
  the existing `source_step_bounds()` used by the onehot-mixed/ambient scheme).
- `DiffusionDataset`: when a store has `source_real_dims()`, precomputes a per-episode
  `action_mask` (episodes never span a source boundary, so it's constant within an episode) and
  includes it in every `__getitem__` batch.
- `DiffusionPolicy.compute_loss(..., action_mask=...)`: masked MSE -- `(sq_err * mask).sum() /
  mask.sum()` -- excludes padded action dims from the loss rather than training the network to
  predict zero there (which would bias the shared denoiser for no reason). `None` (default)
  reproduces the exact old unmasked behavior.
- `DiffusionPolicyConfig.normalizer_type`: `"linear"` (default, old behavior) or `"gaussian"`.
  A padded dataset uses `"gaussian"` with **identity** normalizers inside the policy --
  normalization already happened once, statically, per-source, in `build_padded_dataset.py`;
  normalizing again at the pooled level would mix embodiments' scales and defeat the purpose.
  `train_diffusion.py` detects `extra.padded` and wires this automatically, plus writes
  `source_stats.json` next to the checkpoint (self-contained -- eval doesn't need the original
  training zarr on disk).
- `CrossEmbodimentActionChunkPolicy` path in `evaluate.py` (folded into the existing
  `DiffusionActionChunkPolicy` via an `embodiment_stats` arg, mutually exclusive with `onehot`):
  at eval/rollout time, normalizes a live env's raw (real-dim) obs with that one embodiment's
  static stats, zero-pads to the policy's width, predicts, un-normalizes and slices the
  prediction back to that embodiment's real action_dim before it's sent to `env.step`. Wired
  into `train_diffusion.py`'s existing in-loop `eval_specs` (`{"task": ..., "embodiment": ...}`
  entries) so periodic training-time eval scores each embodiment separately.
- Ambient-diffusion (`--ambient-tmin`) + padded datasets together is explicitly refused
  (`sample_ambient_batch` bypasses `__getitem__` and doesn't emit `action_mask` -- combining
  them would silently leak padded dims into the loss). Not needed for this work.

**Verified correctness on a small smoke dataset** (Grasp-Allegro + Grasp-Sharpa, 8k steps
each) before running for real: per-source real dims are genuinely mean~0/std~1 and the pad
tail is exactly 0 in both obs and action; `action_mask` sums to the right per-embodiment
real `action_dim` per episode; 2 epochs of real training ran end-to-end with the masked loss
decreasing; reloaded the saved checkpoint and confirmed `normalizer_type="gaussian"` with true
identity stats, and that the eval wrapper's normalize -> pad -> predict -> un-normalize ->
slice pipeline returns an action of exactly the source embodiment's real `action_dim`.

**Real datasets built** from the existing 1M-transition collections (`COLLECTIONS.md`):
`/datastor2/mrudolph/mjlab_hand_demos/padded/Grasp-AllHands.zarr` (5,001,608 steps, obs=189,
act=28) and `.../InHand-Rotation-AllHands.zarr` (4,897,978 steps, obs=89, act=22).

**Training launched**: `slurm_jobs/train_cross_embodiment.sbatch`, job `85823`, array `0-1`,
pinned to `slurm-node-011` (per the driver lessons above -- this is real multi-hour training,
not a one-off collection). 40 epochs each (matches this project's ~780k-gradient-step
convention at this data scale), periodic per-embodiment eval every 10 epochs via 5
`{"task", "embodiment"}` eval_specs, wandb project `mjlab`. Outputs:
`outputs/diffusion/{Grasp,InHand-Rotation}-AllHands_padded/`.

## 2026-09-10 (later) — 50k-scale pooled datasets + training

User: pick N trajectories per embodiment for a ~50k-sample pool, then train those too.
Deliberately did **not** subsample the already-pooled 5M padded dataset directly -- its
episodes are 5 concatenated ~1M-step blocks in source order, so `subsample_dataset.py`'s
front-to-back greedy selector would only ever have picked from the first source (Allegro),
and it also drops the `extra.padded`/`sources` metadata `source_real_dims()`/`action_mask`
depend on. Instead: subsampled each of the 10 individual 1M per-embodiment datasets to ~10k
steps first (`subsample_dataset.py`, unchanged), then re-ran `build_padded_dataset.py` over
those 5-per-family 10k subsets -- landed at 50,060 (Grasp) and 50,326 (Rotation) steps, all 5
hands genuinely represented, fresh per-source normalizer fit on the smaller data (not reused
from the 1M fit, per the "static per that dataset" spec). Verified `action_mask` sums match
each source's real action_dim on both. See `COLLECTIONS.md` for the dataset table.

Training launched: `slurm_jobs/train_cross_embodiment_50k.sbatch`, job `86050`, array `0-1`,
also pinned to node-011. 4000 epochs (this project's existing `EPOCHS_50K` convention, same
~780k total gradient steps as every other scale), eval every 1000 epochs, same 5-embodiment
eval_specs and wandb project as the 5M runs (tagged `50k` to distinguish). Outputs:
`outputs/diffusion/{Grasp,InHand-Rotation}-AllHands_50k_padded/`.

---

## 2026-09-09 (later still) — Pinned remaining grasp RL training to node-011 only

Surveyed more nodes for the stale-driver issue (see entries above). Confirmed bad, in addition
to node-004: **node-002, node-003, node-005, node-007, node-008 (also currently drained,
`Reason=Kill task failed` since 2026-09-05), node-009** -- via direct `nvidia-smi` on running
jobs and via the "CUDA version < 12.4" warning appearing in the **original 2026-08-23 array**
(`72897`) logs too, on 5 different nodes. So this has been true since the very first Slurm
array in this project; it just didn't show up as a problem because rotation tasks (lower
per-step contact cost) only pay a mild penalty (5.7-11s/iter) while grasp tasks pay heavily
(15-34s/iter). Only `slurm-node-011` (driver 580.173.02) confirmed clean.

Node survey for 001/006/010 never completed -- the probe jobs sat PENDING (priority) for
hours and got cancelled once the decision below made them moot. **Lesson: a job's
`#SBATCH --output` path must be on storage the compute node can actually reach.** Wrote the
probes' output to this session's `/tmp/claude-*/.../scratchpad/` dir, which isn't visible from
compute nodes -- the jobs ran and exited 0, but the result files were never created. Anything
meant to be read back after an sbatch job runs needs to live under the repo
(`/scratch/cluster/.../mjlab_hand/...`) or another genuinely shared path, never a
session-local scratch dir.

User decision: **only run the remaining RL expert training on node-011**, accepting that it
may queue rather than risk more bad-driver time. Cancelled `85163` tasks 0/1/2 (Allegro/LEAP/
Shadow grasp, all on bad nodes) and resubmitted as `slurm_jobs/train_rl_experts_node011.sbatch`
(job `85382`, array `0-2`, `#SBATCH --nodelist=slurm-node-011`), resuming from their current
checkpoints (5800/5200/4900) with the iteration-count bug fixed this time -- passed the
*remaining* budget (4200/4800/5100) as `--agent.max-iterations`, not 10000, since resume adds
that value on top of the loaded checkpoint's iteration rather than treating it as an absolute
target. `85163` task 3 (Grasp-Sharpa) had, by luck, already landed on node-011 directly and is
running fast (~4.5s/iter) -- left alone rather than restarted, but it still carries the
uncorrected overshoot bug (target 12500, not 10000); not fixed since killing a
healthy fast-running task felt worse than the overshoot itself. Node-011 is often near-full
(96/96 CPUs when `85382` was submitted, including this user's own long-running `flock` jobs
`84302-84304`, 48 CPUs) so `85382` may sit pending for a while -- accepted tradeoff.

---

## 2026-09-09 (later) — Root cause of the 7-14x slowdown: stale driver on slurm-node-004

User: "we need the actual algorithms to go much faster. there's something wrong here." Dug
into why the 4 remaining tasks from array `84348` were so much slower than Grasp-Wuji/
InHand-Rotation-Sharpa (tasks 4/5, which finished at 2.5s/iter and 5.5s/iter respectively).

All 4 slow tasks landed on **`slurm-node-004`**; the 2 fast ones ran on `slurm-node-011`.
Every slow task's `.err` log carries mujoco_warp's `check_toolkit_driver()` warning ("CUDA
version < 12.4 detected ... conditional graph nodes are not available"); neither fast task's
log does. Confirmed directly: `srun --jobid=<task's real JobId> --overlap --cpu-bind=none
nvidia-smi --query-gpu=driver_version --format=csv` inside the *running* job's own allocation
(the array-task alias like `84348_0` doesn't work as a `--jobid` for `srun --overlap`; need
the real numeric JobId from `scontrol show job 84348_0 | grep JobId=`, e.g. `84368`) --
node-004 reports driver **535.216.03** (CUDA 12.2 max). Ran the same `warp.init()` /
`wp.is_conditional_graph_supported()` check interactively on node-011: driver **580.173.02**,
CUDA 13.0, conditional graph capture **True**. Also ruled out the `LD_LIBRARY_PATH=/usr/lib64`
env workaround (from `CLAUDE.md`, meant for H200 rlcompute nodes) as the cause -- reproduced
it explicitly set on node-011 and conditional-graph support stayed `True`; neither node has a
`libcuda` under `/usr/lib64` at all, so that prepend is a no-op on both. This is a genuine
per-node driver mismatch on the cluster, not anything in this repo's code or env setup.

**Fix applied:** cancelled the 4 tasks stuck on node-004 (`scancel 84348_0..3`, all were in
`CG` afterward) and resubmitted as `slurm_jobs/train_rl_experts_resume.sbatch`, job `85163`,
array `0-3`, `#SBATCH --exclude=slurm-node-004`, `--agent.resume True` (defaults for
`--agent.load-run`/`--agent.load-checkpoint` pick the alphabetically-latest run dir / model
file, unambiguous since each hand has exactly one `*_slurm2` dir) so the 2400-4900 iterations
already trained are kept rather than restarted. Also dropped the accidental
`--agent.logger tensorboard` override from the original script (see the wandb entry above) --
these resumed runs use `train`'s actual default (`wandb`) and log live. Killed the background
tensorboard->wandb sync watcher (pid 965893) since it would otherwise have synced the
now-cancelled, truncated tensorboard logs for these 4 as if they were the final result.

**Open follow-up:** only checked node-004 (bad) and node-011 (good) directly -- driver
versions on node-001/002/003/005/006/007/009/010 are unknown. If a resumed task lands on
another stale-driver node, the same "CUDA version < 12.4" line will show up in its `.err`
within the first few log lines; check with the same `srun --jobid=<real JobId> --overlap
--cpu-bind=none nvidia-smi --query-gpu=driver_version --format=csv` trick and add that node to
`--exclude` too.

---

## 2026-09-09 — Don't trust `train`'s printed "Time elapsed"/"ETA"; use checkpoint mtimes

Gave the user a "close to done" ETA for the `84348` RL array (see previous entry) sourced
from the training script's own `.out` log lines (`Time elapsed: HH:MM:SS`, `ETA: HH:MM:SS`).
User pushed back -- these jobs had been running ~25h per `sacct`, but the script's own
"Time elapsed" showed under an hour. That field is **not cumulative since job start**; it
appears to reset periodically while `sacct`'s elapsed and the `Iteration time` (instantaneous
per-step) field stay trustworthy. Its "ETA" is derived from the same broken counter, so it was
off by 10-25x (printed ~2-10h remaining vs. real ~26-76h).

**Correct way to estimate remaining time for an in-progress RL run:** take two checkpoint
files' mtimes (`stat -c '%y'`) and their iteration numbers from the filenames, compute real
iter/hour from that, then `(max_iterations - current_iter) / rate`. Cross-check against the
log's own `Iteration time` line -- it should roughly match `elapsed / iters_in_that_window`
from the mtime method; if it does, trust the mtime-derived ETA over the script's printed one.

Also surfaced: **Grasp-Sharpa (task 3) is running ~2x slower per iteration** than the other
three still-running grasp tasks (36.2s/iter vs 18-21s/iter), with its log full of
`contact match overflow: please increase Option.contact_sensor_maxmatch` warnings -- plausibly
the cause of the slowdown, not yet confirmed. At current rate it's ~76h (~3.2 days) from
iteration 10000, vs. ~26-34h for Allegro/LEAP/Shadow. Not yet restarted with a higher
`contact_sensor_maxmatch` -- open question for the user.

---

## 2026-09-07 — Launched RL expert completion pass, all 10 combos

User asked for expert policies for all 5 embodiments x both tasks. Audited actual on-disk
state first (the `RUNS.md`/`COLLECTIONS.md` 2026-08-24 snapshot doesn't match this box -- see
the 2026-09-01 reconstruction entry below; verified real checkpoints and real `data/demos/`
contents directly). Found 4 of 10 combos already `done` at `model_9999.pt` (rotation:
Allegro/LEAP/Shadow/Wuji); the other 6 (all 5 grasp hands + rotation-Sharpa) had only partial
checkpoints (iters 3800-7500) or, for Grasp-Shadow, none at all (prior Warp-cache failure).

Asked the user two scoping questions rather than assuming: (1) retrain the 4 already-done
combos too, or leave them -- chose leave them; (2) resume the 6 partial runs from checkpoint
(`--agent.resume`/`--agent.load-run` are supported by `train`) or restart fresh -- chose
restart fresh for all 6.

Wrote `slurm_jobs/train_rl_experts.sbatch` (array `0-5`, one task per missing combo,
`run-name slurm2` so new runs land in fresh dirs rather than overwriting the partial ones) and
submitted it: Slurm job `84348`. See `RUNS.md` for the per-task table and script details.
Currently pending on priority (cluster already running the ambient sweep, job `82903`, plus
unrelated `flock` jobs `84302-84304`).

---

## 2026-09-02 — First real Slurm-trained BC policies: dp-leap-rot-400k, dp-allegro-rot-400k

Jobs `80853` (LEAP) / `80854` (Allegro) -- the resubmit after fixing the
`SLURM_SUBMIT_DIR` bug (see the entry below) -- both ran to completion
overnight, 7-8h each, on separate GPU allocations (ran concurrently once
both got scheduled). Real, working BC rotation policies: LEAP reaches 2.31
goals before dropping on average, Allegro 1.62, both well above zero and in
the same ballpark as the historical ~27%-of-expert finding noted in
`ANALYSIS.md` for rotation BC. See `RUNS.md` for full numbers.

Backfill's displayed `StartTime` estimate (~1.8 days) was badly pessimistic
both times -- the first attempt (80250/80251, which failed on the
`SLURM_SUBMIT_DIR` bug) actually started ~3h after submission despite the
same estimate, and this successful pair started within a few hours too.
Don't take the scheduler's `StartTime` field as the real ETA on this
cluster; check back periodically instead of trusting it.

This also closes the loop on the ambient-sampling-bug fix from the
previous entry: these two runs used the *ordinary* (non-ambient) training
path, so they don't exercise `sample_ambient_batch` -- that fix is still
unverified by any long real run, only the short smoke test recorded there.

---

## 2026-09-01 (later) — Ambient diffusion sampling-order bug, found by the user

User: "For each training sample, you need to first sample a diffusion time
step, then sample a valid state action tuple so that you don't undersample
the early (ie low noise) tuples." The ambient mechanism reconstructed earlier
today (see below, and `CHANGES.md` items 24-26) sampled the tuple first (via
a standard shuffling `DataLoader`) and the timestep second, conditioned on
that tuple's `t_min`. Diagnosed why this is wrong: it makes the probability
of ever training below a source's `t_min` proportional to the target's share
of the *dataset*, not the schedule -- a ~40x suppression at the ambient
sweep's own N=10k/400k-source scale. Fixed by inverting the order:
`DiffusionDataset.sample_ambient_batch` now draws the timestep first, then
picks uniformly among currently-valid tuples via a sorted-index searchsorted.
`compute_loss` lost its `t_min`/reweighting path entirely -- once sampling is
correct, no downstream weighting is needed. See `CHANGES.md` for the full
writeup and the verification numbers (built a real 10k-target/400k-source
mixed dataset from data already collected today; old order put 2.5-3.4% of
the intended mass below t=50 where it should be ~50%, new order lands within
2% of ideal everywhere; a 3-epoch real training run with the fix completed
cleanly). Committed.

---

## 2026-09-01 — Reconstructed lost code from the logbook; GPU verification training

`CHANGES.md`/`ANALYSIS.md`/`COLLECTIONS.md` documented ~40 source edits and ~20
new scripts/Slurm jobs made on another box, but only the prose ever landed in
this repo (`git log` here stops right after the base diffusion pipeline,
commit `58946e8`; the commits `CHANGES.md` cites, e.g. `b3b458e`/`ad994ec`,
don't exist in this repo's history). Reconstructed the actual code from those
records: all of `src/mjlab_hand/diffusion/`'s documented fixes (DDIM sampler,
ambient-diffusion gating, checkpoint I/O, one-hot conditioning, multi-target
eval, rotation-success-filter fix), 19 new `scripts/`, 7 new Slurm job
scripts, `.gitignore`, `CLAUDE.md`. Verified what's verifiable on this box's
actual data (real regression test via `check_sampler.py` against the real
smoke checkpoint reproduced the documented before/after MSE numbers almost
exactly; real end-to-end tests of `build_mixed_dataset.py`,
`source_step_bounds`/`ambient_tmin`, `select_experts.py`, `plot_seed_curves.py`,
and the `train.py` checkpoint-cadence rewrite; unit tests of the
`plot_mixed_matrix.py`/`plot_ambient_sweep.py` correctness fixes against the
exact documented bug scenarios). Nothing was committed.

Then launched real GPU verification: **rl-leap-rot-gpu-verify** (see
`RUNS.md`), RL training on the local A40 (free, 0% util at start),
confirming the reconstructed `mjlab_hand` task registration / `train` CLI
path still works end-to-end (not something `CHANGES.md` touched, but a real
dependency of it). Healthy: reward and `episode_success` climbing from a
fresh policy within the first ~30 iterations, ~6.1s/iter steady state after
~1min of Warp/JIT warmup -- confirmed over a 6-minute sampling window (iter
17->64, reward 0.07->0.23).

**Aborted at iter ~64**, user pointed out `logs/rsl_rl/leap_inhand_rotation/2026-08-23_02-26-38_slurm`
and the Allegro equivalent already have `done` experts (`model_9999.pt`,
08-23/08-24) -- retraining was redundant. Killed both the wrapper script and
`train` process, freed the GPU, deleted the partial run dir / wandb run /
console log. The verification goal (confirm the pipeline trains) was already
met by the 64 iterations observed; no need for a trained checkpoint from
this run specifically. Lesson: check `RUNS.md` for an existing `done` expert
*before* launching a "verify it trains" run against a task that already has
one -- a shorter/synthetic smoke check would have answered the same question
without spending GPU time on a real task.

---

## 2026-08-24 — Agent logbook added

- Created `agent_logbook/` (`README.md`, `JOURNAL.md`, `RUNS.md`, `COLLECTIONS.md`) and always-on Cursor rule `.cursor/rules/agent-logbook.mdc`.
- Seeded registries from prior session artifacts (transcript [c5d65292-2612-48b0-8936-c7852d3f7ba7](c5d65292-2612-48b0-8936-c7852d3f7ba7)).

---

## 2026-08-22 → 2026-08-23 — Setup, RL training, demos, diffusion pipeline

Session: [c5d65292-2612-48b0-8936-c7852d3f7ba7](c5d65292-2612-48b0-8936-c7852d3f7ba7)

### Install & tooling

- Installed with `uv sync --group dev --default-index https://pypi.org/simple --system-certs` (Aliyun mirror TLS issues).
- Added helper scripts: `scripts/train_all.sh`, `scripts/plot_training_curves.py`, `scripts/record_trajectories.sh`, `scripts/render_checkpoint_video.py`, later `scripts/watch_eval_diffusion.py`.

### RL training

- Started **rl-allegro-grasp-abort**, user stopped; restarted as **rl-allegro-grasp** on local A40 (`logs/grasp_allegro_train.log`).
- Built Slurm array `slurm_jobs/array_20260823_022315` for 9 other hand/task combos (skip Allegro grasp).
- **rl-shadow-grasp** failed (Warp CCD kernel cache meta missing). Several grasp jobs look incomplete vs 10k iters; all in-hand rotation jobs except Sharpa reached `model_9999.pt`.
- Generated TB plots under `outputs/plots/` and rollout videos under `outputs/videos/`.

### Diffusion pipeline + data

- User asked for diffusion policies from expert checkpoints across embodiments; started with Allegro grasp.
- Smoke collect **demo-allegro-smoke** (100 eps) → smoke train **dp-allegro-smoke** (20 epochs, best loss ≈0.015, env eval 0% success).
- Full collect **demo-allegro-full** (2000 eps, ~1M steps, 1979 success) → **dp-allegro-full** (150 epochs). Watch-eval every 10 epochs still reporting 0% success through epoch 130; training loss ~0.0018.
- GitHub fork/push attempt was abandoned by user (“just forget it”); local commands were provided instead.

### Open issues / follow-ups

- Fix/retry **Grasp-Shadow** Slurm failure (Warp cache).
- Confirm whether incomplete grasp/Sharpa-rot Slurm jobs are still running or need resubmit.
- Allegro diffusion full train: finish remaining epochs; diagnose 0% rollout success despite low train loss (horizon, obs, action scaling, success filter, eval protocol).
- Collect demos + train diffusion for other hands/tasks once RL experts are solid.
