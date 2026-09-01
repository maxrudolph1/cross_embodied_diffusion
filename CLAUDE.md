# CLAUDE.md

Notes for Claude Code sessions working in this repo, covering what
`README.md` does not. See [`agent_logbook/`](agent_logbook/) for the
detailed history — this file is a summary, not a replacement.

## What this repo actually is right now

The upstream `mjlab_hand` package is an RL benchmark (grasp / in-hand
rotation, five hand embodiments: Allegro, LEAP, Shadow, Sharpa, Wuji). The
**active work is cross-embodiment diffusion-policy behavior cloning on top
of that benchmark**, not the RL benchmark itself:

1. Train RL experts per (task, embodiment) with `mjlab_hand`'s own
   training entry point (see `README.md`).
2. Collect demonstrations from an expert checkpoint:
   `collect-demos` / `src/mjlab_hand/diffusion/collect.py`.
3. Train an action-chunk diffusion policy on the demos:
   `train-diffusion` / `src/mjlab_hand/diffusion/train.py`.
4. Evaluate / render rollouts: `eval-diffusion`,
   `src/mjlab_hand/diffusion/evaluate.py`.
5. Cross-embodiment experiments (mixed-embodiment training, ambient
   diffusion, state-equivalence probes) build on steps 2-4 — see
   `agent_logbook/COLLECTIONS.md` and `ANALYSIS.md`.

## Required environment variables (every GPU job)

```bash
# rlcompute (H200) nodes ship a stale /usr/local/cuda/compat libcuda that
# ldconfig prefers over the real driver -> CUDA error 803 on every call.
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Concurrent array tasks sharing one Warp kernel cache race and fail with a
# CCD-kernel FileNotFoundError. Give each task its own.
export WARP_CACHE_PATH="$JOBDIR/warp_cache/task_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WARP_CACHE_PATH"

export MUJOCO_GL=egl   # headless rendering
```

`srun` fails with "More processors requested than permitted" when invoked
from inside an existing interactive allocation — use `sbatch`.

## Logbook protocol

Before non-trivial work, read `agent_logbook/JOURNAL.md`, `RUNS.md`,
`COLLECTIONS.md`. After: update the relevant file in the same turn. Full
rule in `.cursor/rules/agent-logbook.mdc`. Do not commit gitignored
artifacts (`logs/`, `data/`, most of `outputs/` — see `.gitignore` and
CHANGES.md item 23 for exactly what is tracked).

## Invariants worth knowing before touching this code

- **The diffusion sampler is DDIM (eta=0), not single-step ancestral
  updates.** `inference_timesteps` is a strided subsequence of the training
  schedule; a single-step DDPM update is only valid for a `t -> t-1`
  transition. See `CHANGES.md` item 1 and `scripts/check_sampler.py` for a
  standalone regression check.
- **`collect_demos` filters rotation success via
  `RotationCommand.metrics["episode_success"]`, not the `pose` command
  term** — grasp tasks have a `pose` term, rotation tasks have `rotation`
  instead, and treating their absence as "always successful" silently
  mislabels every rotation episode. See item 9.
- **Actions are stored pre-clip.** `collect_demos` records the raw policy
  output; `RslRlVecEnvWrapper` clips to `clip_actions=1.0` inside
  `env.step`. Not a bug — a BC policy is clipped identically at rollout —
  but `LinearNormalizer` fits min/max, so account for the long tail
  (measured range ~±15) before assuming the normalized range is well used.
- **Only Allegro and LEAP share observation/action dimensionality *and*
  term layout.** Every other embodiment pair needs padding, per-embodiment
  encoders, or a shared schema before it can be mixed. `build_mixed_dataset.py`
  refuses mismatched spaces rather than zero-padding.
- **Eval rows are tagged with `eval_task`, always, including solo runs.**
  Once multi-target eval (`eval_specs`) shipped, a single-target
  `--eval-task` run is internally promoted to a one-element spec list and
  still writes an `eval_task` key. A plotter that identifies "solo" rows by
  the *absence* of that key will silently blank every run trained after
  that point — see item 36.
- **`src/mjlab_hand/env_cfg.py` is dead upstream leftover.** It imports
  `mjlab_hand.anymal_c`, which does not exist in this repo, so the module
  cannot be imported. Left in place; do not extend it.

## Standing findings (see `ANALYSIS.md` for the full analysis)

- The observation identifies the embodiment perfectly (linear probe AUC
  1.000) at every diffusion timestep, including t=99 — the observation is
  never noised, so masking/invariance schemes that rely on high-noise
  unidentifiability do not apply here.
- Ambient (per-source timestep-gated) diffusion is falsified for grasp
  (decisively negative, replicated across seeds) and null for rotation.
  The one robust result is the control: an embodiment trained only on the
  coarse end of the schedule is completely non-functional (0.000), not
  merely degraded.
- Adversarial invariance training does not work on the Allegro/LEAP pair:
  a fresh probe trained after the encoder freezes stays at ≥0.977 balanced
  accuracy regardless of reversal strength, while task information drops
  substantially. Always score invariance with a fresh probe on held-out
  data, never the adversary's own training accuracy.
- Sequential transfer (pretrain on one embodiment, fine-tune on another) is
  the one cross-embodiment-sharing idea that measurement has not yet ruled
  out.
