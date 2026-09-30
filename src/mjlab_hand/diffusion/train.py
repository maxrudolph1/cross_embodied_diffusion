"""Train a diffusion policy on collected expert trajectories."""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mjlab_hand.diffusion.dataset import DiffusionDataset, TrajectoryStore
from mjlab_hand.diffusion.normalizer import GaussianNormalizer, LinearNormalizer
from mjlab_hand.diffusion.policy import DiffusionPolicy, DiffusionPolicyConfig
from mjlab_hand.diffusion.pooling import pooled_store


@dataclass
class TrainConfig:
    # One zarr, or several single-embodiment zarrs to pool on the fly into a
    # padded cross-embodiment dataset (see `pool_sources`).
    dataset: Path | list[Path]
    output_dir: Path
    obs_horizon: int = 2
    action_horizon: int = 8
    batch_size: int = 256
    num_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-6
    num_workers: int = 4
    device: str = "cuda:0"
    success_only: bool = True
    num_train_timesteps: int = 100
    num_inference_steps: int = 16
    save_every_epochs: int = 10
    # How often to write policy_latest.pt / policy_best_val.pt. A 265MB
    # torch.save to NFS costs ~0.75s, comparable to a whole epoch at small
    # data scales -- see CHANGES.md item 13. Always saved on the final epoch
    # and immediately before any eval/render regardless of this cadence.
    latest_every_epochs: int = 1
    seed: int = 0
    # Optional in-loop env evaluation (disabled unless eval_task/eval_specs is set).
    eval_task: str | None = None
    # Multi-target eval: [{"task": ..., "onehot": [..] | None}, ...] for the
    # 2-source onehot-mixed scheme, or [{"task": ..., "embodiment": "..."},
    # ...] for the N-source padded cross-embodiment scheme (embodiment must
    # match a name in the dataset's source_stats.json). Lets a
    # mixed/cross-embodiment policy be scored against each embodiment it
    # drives. `eval_task` alone is internally promoted to a single-element
    # spec (no onehot/embodiment).
    eval_specs: list[dict] | None = None
    eval_every_epochs: int = 10
    eval_num_envs: int = 32
    eval_num_steps: int = 1500
    # Optional periodic mp4 rollout rendering.
    render_every_epochs: int = 0
    render_num_steps: int = 400
    render_num_envs: int = 1
    # Ambient diffusion: one t_min per source (dataset order) for a mixed
    # dataset. Source i is admitted into the loss only at t >= ambient_tmin[i].
    ambient_tmin: list[int] | None = None
    # Held-out validation: fraction of *trajectories* (not individual states)
    # per source withheld from training and used only to compute a
    # validation loss (see DiffusionPolicy.action_reconstruction_loss). 0
    # (default) disables validation entirely -- every existing sbatch script
    # trains on 100% of its data exactly as before.
    val_fraction: float = 0.0
    val_seed: int = 0
    # Restrict the val loss (and so policy_best_val.pt) to one source of a
    # pooled dataset, by embodiment name -- e.g. the target hand of a
    # scarce/ambient run, whose val loss is otherwise ~1% of the pooled one.
    # The held-out trajectories are the same as without it. CHANGES.md item 57.
    val_embodiment: str | None = None
    val_every_epochs: int = 1
    # DDIM sampling (num_inference_steps forward passes/batch) is much
    # pricier than one training step; cap how many val batches run each
    # check so validation stays a small fraction of an epoch's compute
    # instead of doubling it. None = use the whole val set every time.
    val_max_batches: int | None = 20
    # Per-source sampling ratio for a padded/mixed dataset. "uniform"
    # (default): plain shuffling, each window equally likely (source share
    # of a batch proportional to its row count -- unchanged behaviour).
    # "balanced": WeightedRandomSampler so every source gets equal expected
    # representation per epoch regardless of size (see
    # DiffusionDataset.source_sample_weights) -- for "scarce co-training",
    # where one source has far fewer rows than the rest.
    source_sample_mode: str = "uniform"
    # torch.compile mode for the denoising UNet (None = eager, unchanged).
    # Compiled in place via nn.Module.compile(), so state_dict keys -- and
    # therefore saved checkpoints -- are identical to eager, and the
    # torch.randn/randint draws in compute_loss/predict_action stay eager.
    # Only the training process compiles; eval rollouts load a fresh eager
    # policy from disk. See CHANGES.md item 53.
    compile_mode: str | None = None
    # WandB logging. Disabled (None) by default; set wandb_project to enable.
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    # Pool `dataset` in memory with the padded scheme (mjlab_hand.diffusion.
    # pooling; CHANGES.md item 56). None = only when several datasets are
    # given; True with a single dataset trains it through the padded path
    # (per-source Gaussian stats, identity policy normalizers), e.g. to
    # compare against a plain LinearNormalizer specialist on the same data.
    pool_sources: bool | None = None
    # task_family recorded in source_stats.json; None = inferred from names.
    task_family: str | None = None
    # Per-source normalization for a pooled run: "gaussian" (mean 0 / var 1,
    # the original scheme) or "minmax" ([-1, 1], = the plain path's
    # LinearNormalizer). CHANGES.md item 58.
    source_norm: str = "gaussian"


def _dataset_paths(cfg: TrainConfig) -> list[Path]:
    ds = cfg.dataset
    return [Path(p) for p in ds] if isinstance(ds, (list, tuple)) else [Path(ds)]


def _open_store(cfg: TrainConfig) -> TrajectoryStore:
    paths = _dataset_paths(cfg)
    pool = cfg.pool_sources if cfg.pool_sources is not None else len(paths) > 1
    if not pool:
        if len(paths) > 1:
            raise ValueError("several datasets need pool_sources (the padded scheme)")
        return TrajectoryStore(paths[0], mode="r")
    return pooled_store(
        paths,
        task_family=cfg.task_family,
        success_only=cfg.success_only,
        source_norm=cfg.source_norm,
    )


def _action_range(dataset: DiffusionDataset) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim min/max of the training split's (normalized) actions."""
    low = np.full(dataset.action.shape[1], np.inf, dtype=np.float32)
    high = np.full(dataset.action.shape[1], -np.inf, dtype=np.float32)
    for start, end, _ in dataset.episodes:
        chunk = dataset.action[start:end]
        low = np.minimum(low, chunk.min(axis=0))
        high = np.maximum(high, chunk.max(axis=0))
    return low, high


def _source_index(extra: dict, embodiment: str) -> int:
    names = [s.get("embodiment", s.get("task")) for s in extra.get("sources", [])]
    if embodiment not in names:
        raise ValueError(f"val_embodiment {embodiment!r} not among dataset sources {names}")
    return names.index(embodiment)


def _dataset_repr(cfg: TrainConfig) -> str | list[str]:
    paths = _dataset_paths(cfg)
    return [str(p) for p in paths] if isinstance(cfg.dataset, (list, tuple)) else str(paths[0])


@contextmanager
def _restore_backend_flags():
    """Undo the global backend changes an env eval/render makes in-process.

    Env setup calls mjlab's configure_torch_backends(), which sets TF32 via
    the new `fp32_precision` API (and cudnn.benchmark). Once that is set, any
    read of the legacy `allow_tf32` flag raises "mix of the legacy and new
    APIs" -- and Inductor reads it when it recompiles (pad_mm), so a compiled
    run died on the first training step after its first eval. Restoring the
    flags keeps a compiled run's training precision constant. See CHANGES.md
    item 53.
    """
    m, c = torch.backends.cuda.matmul, torch.backends.cudnn
    saved = (m.fp32_precision, c.fp32_precision, c.benchmark, c.deterministic)
    try:
        yield
    finally:
        m.fp32_precision, c.fp32_precision, c.benchmark, c.deterministic = saved


def _eval_specs(cfg: TrainConfig) -> list[dict]:
    if cfg.eval_specs is not None:
        return cfg.eval_specs
    if cfg.eval_task is not None:
        return [{"task": cfg.eval_task, "onehot": None}]
    return []


def train_diffusion(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    store = _open_store(cfg)
    summary = store.summary()
    print(f"[INFO] Dataset: {summary}")
    extra = json.loads(store.root.attrs.get("extra", "{}"))
    is_padded = bool(extra.get("padded"))

    dataset = DiffusionDataset(
        store,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        success_only=cfg.success_only,
        ambient_tmin=cfg.ambient_tmin,
        split="train",
        val_fraction=cfg.val_fraction,
        val_seed=cfg.val_seed,
    )

    # Held-out trajectories for the validation loss (see TrainConfig.val_fraction).
    # Built from the *same* store/split params as `dataset` above so the two
    # partition the episode list into disjoint, complementary sets -- see
    # `_train_val_split`. Normalizer fitting below uses `dataset` (the train
    # split only), so val trajectories never leak into normalization stats.
    val_loader: DataLoader | None = None
    val_source = _source_index(extra, cfg.val_embodiment) if cfg.val_embodiment else None
    if cfg.val_fraction > 0:
        val_dataset = DiffusionDataset(
            store,
            obs_horizon=cfg.obs_horizon,
            action_horizon=cfg.action_horizon,
            success_only=cfg.success_only,
            split="val",
            val_fraction=cfg.val_fraction,
            val_seed=cfg.val_seed,
            only_source=val_source,
        )
        if cfg.val_embodiment:
            print(f"[INFO] Val loss restricted to {cfg.val_embodiment} (source {val_source})")
        print(
            f"[INFO] Val split: {len(val_dataset.episodes)} held-out trajectories, "
            f"{len(val_dataset)} windows (train: {len(dataset.episodes)} trajectories, "
            f"{len(dataset)} windows)"
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            generator=torch.Generator().manual_seed(cfg.val_seed),
        )

    # Ambient diffusion samples the diffusion timestep FIRST, then a training
    # tuple valid at that timestep (see DiffusionDataset.sample_ambient_batch)
    # -- the reverse order silently starves low-noise training whenever the
    # admitted-everywhere (target) data is a small fraction of the mixed
    # dataset. This can't be expressed as a DataLoader shuffle over fixed
    # rows, so it bypasses the DataLoader entirely; ambient_rng is exhausted
    # once per batch rather than once per dataset pass.
    # Ambient + padded is supported since CHANGES.md item 57:
    # sample_ambient_batch emits the per-episode action_mask.
    is_ambient = cfg.ambient_tmin is not None
    if is_ambient and cfg.source_sample_mode != "uniform":
        raise ValueError("source_sample_mode is not used by ambient sampling; leave it 'uniform'")
    loader: DataLoader | None = None
    ambient_rng: np.random.Generator | None = None
    num_batches_per_epoch = len(dataset) // cfg.batch_size
    if is_ambient:
        ambient_rng = np.random.default_rng(cfg.seed)
    else:
        weights = dataset.source_sample_weights(cfg.source_sample_mode)
        if weights is None:
            loader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=True,
                generator=torch.Generator().manual_seed(cfg.seed),
            )
        else:
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.from_numpy(weights),
                num_samples=len(dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(cfg.seed),
            )
            loader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                sampler=sampler,
                num_workers=cfg.num_workers,
                pin_memory=device.type == "cuda",
                drop_last=True,
            )

    if is_padded:
        # Data is already per-source normalized (mean 0, var 1, fit
        # statically per embodiment) and zero-padded by
        # build_padded_dataset.py -- normalizing again here, against the
        # *pooled* mixture, would mix scales across embodiments and defeat
        # the whole point of per-source static normalization. The policy's
        # obs/action normalizers are therefore pure identities; real
        # normalization already happened upstream, once, and is reused
        # unchanged at eval time via source_stats.json (written below).
        obs_norm: LinearNormalizer | GaussianNormalizer = GaussianNormalizer.identity(
            int(summary["obs_dim"])
        )
        act_norm: LinearNormalizer | GaussianNormalizer = GaussianNormalizer.identity(
            int(summary["action_dim"])
        )
        normalizer_type = "gaussian"
    else:
        obs_norm = LinearNormalizer.fit(dataset.obs)
        act_norm = LinearNormalizer.fit(dataset.action)
        normalizer_type = "linear"

    policy_cfg = DiffusionPolicyConfig(
        obs_dim=int(summary["obs_dim"]),
        action_dim=int(summary["action_dim"]),
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        num_train_timesteps=cfg.num_train_timesteps,
        num_inference_steps=cfg.num_inference_steps,
        normalizer_type=normalizer_type,
    )
    policy = DiffusionPolicy(policy_cfg).to(device)
    policy.set_normalizers(obs_norm, act_norm)
    if is_padded:
        # The sampler clamps its x0 estimate to this range; the +-1 default
        # only matches LinearNormalizer data (CHANGES.md item 58).
        low, high = _action_range(dataset)
        policy.set_action_clip(torch.from_numpy(low), torch.from_numpy(high))
        print(f"[INFO] action clip range: [{low.min():.2f}, {high.max():.2f}]")
    if cfg.compile_mode is not None:
        policy.noise_pred_net.compile(mode=cfg.compile_mode)
    # Eager runs keep the historical behaviour (TF32 switches on at the first
    # in-training eval and stays on); only compiled runs restore the flags.
    restore_flags = _restore_backend_flags if cfg.compile_mode is not None else nullcontext

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "train_config.json").write_text(
        json.dumps(
            {**asdict(cfg), "dataset": _dataset_repr(cfg), "output_dir": str(cfg.output_dir)},
            indent=2,
            default=str,
        )
    )
    if is_padded:
        # Self-contained copy of the per-embodiment normalization stats +
        # real dims, so eval-time code (CrossEmbodimentActionChunkPolicy)
        # doesn't need the original training dataset zarr on disk -- only
        # this policy's output_dir.
        (cfg.output_dir / "source_stats.json").write_text(
            json.dumps(
                {
                    "task_family": extra.get("task_family"),
                    "max_obs_dim": extra.get("max_obs_dim"),
                    "max_action_dim": extra.get("max_action_dim"),
                    "sources": extra.get("sources"),
                },
                indent=2,
            )
        )

    specs = _eval_specs(cfg)
    render_task = cfg.eval_task or (specs[0]["task"] if specs else None)

    wandb_run = None
    if cfg.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            tags=cfg.wandb_tags,
            config={**asdict(cfg), "dataset": _dataset_repr(cfg), "output_dir": str(cfg.output_dir)},
        )

    global_step = 0
    best_loss = float("inf")
    best_val_loss = float("inf")
    best_eval_score = float("-inf")
    latest_path = cfg.output_dir / "policy_latest.pt"
    eval_jsonl = cfg.output_dir / "eval_metrics.jsonl"

    for epoch in range(1, cfg.num_epochs + 1):
        policy.train()
        losses = []
        if is_ambient:
            assert ambient_rng is not None
            batches = (
                dataset.sample_ambient_batch(cfg.batch_size, cfg.num_train_timesteps, ambient_rng)
                for _ in range(num_batches_per_epoch)
            )
        else:
            batches = loader
        for batch in batches:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            timesteps = batch["timesteps"].to(device) if "timesteps" in batch else None
            action_mask = batch["action_mask"].to(device) if "action_mask" in batch else None
            loss = policy.compute_loss(obs, action, timesteps=timesteps, action_mask=action_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            global_step += 1

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[epoch {epoch:04d}/{cfg.num_epochs}] loss={mean_loss:.6f} steps={global_step}")
        if wandb_run is not None:
            wandb_run.log({"train/loss": mean_loss, "train/steps": global_step}, step=epoch)

        if (
            val_loader is not None
            and cfg.val_every_epochs > 0
            and epoch % cfg.val_every_epochs == 0
        ):
            policy.eval()
            val_losses = []
            for i, batch in enumerate(val_loader):
                if cfg.val_max_batches is not None and i >= cfg.val_max_batches:
                    break
                obs = batch["obs"].to(device)
                action = batch["action"].to(device)
                action_mask = batch["action_mask"].to(device) if "action_mask" in batch else None
                val_losses.append(
                    float(policy.action_reconstruction_loss(obs, action, action_mask))
                )
            policy.train()
            mean_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            print(f"[epoch {epoch:04d}/{cfg.num_epochs}] val_action_loss={mean_val_loss:.6f}")
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                policy.save(cfg.output_dir / "policy_best_val.pt")
                (cfg.output_dir / "best_val.json").write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "val_action_loss": mean_val_loss,
                            "val_embodiment": cfg.val_embodiment,
                        }
                    )
                )
            if wandb_run is not None:
                wandb_run.log({"val/action_loss": mean_val_loss}, step=epoch)
            with (cfg.output_dir / "val_metrics.jsonl").open("a") as f:
                f.write(json.dumps({"epoch": epoch, "val_action_loss": mean_val_loss}) + "\n")

        is_last = epoch == cfg.num_epochs
        due = epoch % cfg.latest_every_epochs == 0 or is_last
        if due:
            policy.save(latest_path)
        best_loss = min(best_loss, mean_loss)
        if epoch % cfg.save_every_epochs == 0:
            policy.save(cfg.output_dir / f"policy_epoch_{epoch:04d}.pt")

        if specs and cfg.eval_every_epochs > 0 and epoch % cfg.eval_every_epochs == 0:
            from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

            # evaluate_diffusion_policy loads from disk; make sure it scores
            # the current weights even if latest_every_epochs skipped this epoch.
            policy.save(latest_path)
            policy.eval()
            print(f"[INFO] Running env eval at epoch {epoch}...")
            headlines = []
            for spec in specs:
                with restore_flags():
                    metrics = evaluate_diffusion_policy(
                        task=spec["task"],
                        policy_path=latest_path,
                        num_envs=cfg.eval_num_envs,
                        num_steps=cfg.eval_num_steps,
                        device=str(device),
                        seed=cfg.seed,
                        onehot=spec.get("onehot"),
                        embodiment=spec.get("embodiment"),
                        reuse_env=True,
                    )
                row = {
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "train_steps": global_step,
                    "eval_task": spec["task"],
                    "onehot": spec.get("onehot"),
                    "metrics": metrics,
                }
                with eval_jsonl.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                headline = metrics.get(
                    "success_rate", metrics.get("avg_successes_before_drop", float("nan"))
                )
                print(f"[INFO] eval epoch={epoch} task={spec['task']} headline={headline:.3f}")
                headlines.append(headline)
                if wandb_run is not None:
                    safe_task = spec["task"].replace("/", "_")
                    wandb_run.log(
                        {f"eval/{safe_task}/{k}": v for k, v in metrics.items()}, step=epoch
                    )
            finite = [h for h in headlines if np.isfinite(h)]
            if finite and float(np.mean(finite)) > best_eval_score:
                best_eval_score = float(np.mean(finite))
                shutil.copy2(latest_path, cfg.output_dir / "policy_best_eval.pt")
                (cfg.output_dir / "best_eval.json").write_text(
                    json.dumps({"epoch": epoch, "mean_headline": best_eval_score})
                )
            policy.train()

        if (
            cfg.render_every_epochs > 0
            and epoch % cfg.render_every_epochs == 0
            and render_task is not None
        ):
            try:
                from mjlab_hand.diffusion.evaluate import render_diffusion_rollout

                policy.save(latest_path)
                with restore_flags():
                    render_diffusion_rollout(
                        task=render_task,
                        policy_path=latest_path,
                        output_dir=cfg.output_dir / "videos",
                        num_steps=cfg.render_num_steps,
                        num_envs=cfg.render_num_envs,
                        device=str(device),
                        seed=cfg.seed,
                        tag=f"epoch{epoch:04d}",
                        onehot=specs[0].get("onehot") if specs else None,
                        embodiment=specs[0].get("embodiment") if specs else None,
                    )
            except Exception as exc:  # noqa: BLE001 - rendering must never kill training
                print(f"[WARN] render failed at epoch {epoch}: {exc}")

    if specs:
        from mjlab_hand.diffusion.evaluate import close_cached_envs

        close_cached_envs()

    if wandb_run is not None:
        wandb_run.summary["best_loss"] = best_loss
        wandb_run.finish()

    # Completion marker: scripts/promote_outputs.sh copies only run dirs
    # that have it, so a run still training on $SCRATCH isn't promoted
    # half-written. See CHANGES.md item 55.
    (cfg.output_dir / "train_done.json").write_text(
        json.dumps({"num_epochs": cfg.num_epochs, "best_loss": best_loss}, indent=2)
    )
    print(f"[INFO] Training done. Best loss={best_loss:.6f}")
    print(f"[INFO] Saved {latest_path}")
    return latest_path
