"""Train a diffusion policy on collected expert trajectories.

Re-implementation of Bundle's train.py from MIGRATION.md (CHANGES.md item 63):
one store (single-hand, or a term-aligned padded multi-hand store), one shared
LinearNormalizer chosen by --norm-mode, optional ambient gating with a
data-first or noise-first sampler, held-out validation on a separate store,
and checkpoints latest / best (train loss) / last{k}_epoch{N} / best_rollout /
best_val, with selection.json written last as the completion marker.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mjlab_hand.diffusion.dataset import (
    AmbientNoiseFirstBatchSampler,
    DiffusionDataset,
    TrajectoryStore,
)
from mjlab_hand.diffusion.frozen_norm import load_artifact, select
from mjlab_hand.diffusion.normalizer import LinearNormalizer, norm_digest
from mjlab_hand.diffusion.padding import build_plan
from mjlab_hand.diffusion.policy import DiffusionPolicy, DiffusionPolicyConfig

NORM_MODES = ("shared", "pad-aware", "zscore", "frozen")
AMBIENT_SAMPLERS = ("data-first", "noise-first")


@dataclass
class TrainConfig:
    dataset: Path
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
    # Kept for CLI compatibility; the loop does not read it (as in Bundle).
    # Snapshots are the rolling last{k} below.
    save_every_epochs: int = 10
    # Cadence of policy_latest.pt and the rolling policy_last{k}_epoch{N}.pt
    # snapshots (last0 = newest). Always on the final epoch.
    latest_every_epochs: int = 1
    keep_last: int = 3
    seed: int = 0
    # In-training env eval. eval_specs: [{"task": ..., "pad": bool} or
    # {"task": ..., "onehot": [...]}]; eval_task alone = one unpadded spec.
    # A final eval always runs on the last epoch.
    eval_task: str | None = None
    eval_specs: list[dict] | None = None
    eval_every_epochs: int = 10
    eval_num_envs: int = 32
    eval_num_steps: int = 1500
    render_every_epochs: int = 0
    render_num_steps: int = 400
    render_num_envs: int = 1
    # Ambient diffusion: one t_min per source, in the store's source order
    # (HANDS order for padded stores: padded_grid.target_ambient_tmin).
    # Source i is trained only at t >= ambient_tmin[i].
    ambient_tmin: list[int] | None = None
    # "data-first": window first, then t ~ U[t_min, T) (Bundle's code default;
    # its logs call it confounded -- low-noise steps get starved).
    # "noise-first": t first, then a window admitted at t (Bundle's recipe).
    ambient_sampler: str = "data-first"
    mask_pad_loss: bool = False
    norm_mode: str = "shared"
    norm_clip_pct: float | None = None
    norm_artifact: Path | None = None
    x0_clamp: float = 1.0
    val_dataset: Path | None = None
    val_windows: int = 2048
    # Fine-tuning: start from this checkpoint's weights AND normalizers
    # (architecture must match). Fresh optimizer. CHANGES.md item 67.
    init_checkpoint: Path | None = None
    # Optional WandB logging (off unless wandb_project is set; Branch-only,
    # no effect on training).
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None


def _eval_specs(cfg: TrainConfig) -> list[dict]:
    if cfg.eval_specs is not None:
        return cfg.eval_specs
    if cfg.eval_task is not None:
        return [{"task": cfg.eval_task, "onehot": None}]
    return []


def check_config(cfg: TrainConfig) -> None:
    """Reject bad flag combinations up front."""
    if cfg.norm_mode not in NORM_MODES:
        raise ValueError(f"norm_mode must be one of {NORM_MODES}")
    if cfg.ambient_sampler not in AMBIENT_SAMPLERS:
        raise ValueError(f"ambient_sampler must be one of {AMBIENT_SAMPLERS}")
    if (cfg.norm_mode == "frozen") != (cfg.norm_artifact is not None):
        raise ValueError("--norm-mode frozen requires --norm-artifact, and an artifact needs frozen")
    if cfg.norm_mode == "zscore" and not cfg.x0_clamp > 1:
        raise ValueError("--norm-mode zscore needs --x0-clamp > 1 (z-scored data exceeds +-1)")
    if cfg.norm_clip_pct is not None and cfg.norm_mode != "pad-aware":
        raise ValueError("--norm-clip-pct requires --norm-mode pad-aware")
    if cfg.ambient_sampler == "noise-first" and cfg.ambient_tmin is None:
        raise ValueError("--ambient-sampler noise-first requires --ambient-tmin")


def _fit_normalizers(cfg: TrainConfig, dataset: DiffusionDataset, store: TrajectoryStore, extra: dict):
    padded = extra.get("pad_scheme") == "term_aligned"
    if cfg.norm_mode == "frozen":
        if not padded:
            raise ValueError("--norm-mode frozen needs a term-aligned padded store")
        art = load_artifact(cfg.norm_artifact)
        return select(art, extra["family"], dataset.obs.shape[1], dataset.action.shape[1])
    if not padded or cfg.norm_mode == "shared":
        if cfg.norm_mode == "zscore":
            return LinearNormalizer.fit_standardized(dataset.obs), LinearNormalizer.fit_standardized(dataset.action)
        # pad-aware on a single-hand store is the same as shared
        return LinearNormalizer.fit(dataset.obs), LinearNormalizer.fit(dataset.action)
    plan = build_plan(extra["family"])
    go = [(s, e, plan.obs_valid(t)) for s, e, t in store.source_step_bounds()]
    ga = [(s, e, plan.action_valid(t)) for s, e, t in store.source_step_bounds()]
    if cfg.norm_mode == "pad-aware":
        return (
            LinearNormalizer.fit_masked(dataset.obs, go, clip_pct=cfg.norm_clip_pct, include_zero=True),
            LinearNormalizer.fit_masked(dataset.action, ga, clip_pct=cfg.norm_clip_pct, include_zero=True),
        )
    return LinearNormalizer.fit_standardized(dataset.obs, go), LinearNormalizer.fit_standardized(dataset.action, ga)


def _headline(metrics: dict) -> float:
    return float(metrics.get("success_rate", metrics.get("avg_successes_before_drop", float("nan"))))


class _Snapshots:
    """Rolling policy_last{k}_epoch{N}.pt, k = 0 newest, at most keep files."""

    def __init__(self, out: Path, keep: int):
        self.out, self.keep, self.items = out, keep, []  # [(epoch, path)] newest first

    def add(self, latest: Path, epoch: int) -> None:
        for _ep, p in self.items[self.keep - 1 :]:
            p.unlink(missing_ok=True)
        kept = self.items[: self.keep - 1]
        for k in reversed(range(len(kept))):  # oldest first: shift k -> k+1
            ep, p = kept[k]
            new = self.out / f"policy_last{k + 1}_epoch{ep:04d}.pt"
            p.rename(new)
            kept[k] = (ep, new)
        first = self.out / f"policy_last0_epoch{epoch:04d}.pt"
        shutil.copy2(latest, first)
        self.items = [(epoch, first)] + kept


def train_diffusion(cfg: TrainConfig) -> Path:
    check_config(cfg)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        # A silent CPU fallback once ran 15 jobs ~60x slower (MIGRATION section 7).
        raise RuntimeError(f"device {cfg.device!r} requested but CUDA is not available")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    store = TrajectoryStore(cfg.dataset, mode="r")
    summary = store.summary()
    print(f"[INFO] Dataset: {summary}")
    extra = json.loads(store.root.attrs.get("extra", "{}"))
    padded = extra.get("pad_scheme") == "term_aligned"
    if extra.get("padded") and not padded:
        raise ValueError(f"{cfg.dataset} is an old tail-padded store; rebuild it term-aligned")

    dataset = DiffusionDataset(
        store,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        success_only=cfg.success_only,
        ambient_tmin=cfg.ambient_tmin,
        mask_pad_loss=cfg.mask_pad_loss,
    )
    # No explicit torch.Generator: the shuffle order comes from torch.manual_seed.
    if cfg.ambient_tmin is not None and cfg.ambient_sampler == "noise-first":
        sampler = AmbientNoiseFirstBatchSampler(dataset, cfg.batch_size, cfg.num_train_timesteps)
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=cfg.num_workers, pin_memory=device.type == "cuda"
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

    obs_norm, act_norm = _fit_normalizers(cfg, dataset, store, extra)
    digest = norm_digest(obs_norm, act_norm)

    policy_cfg = DiffusionPolicyConfig(
        obs_dim=int(summary["obs_dim"]),
        action_dim=int(summary["action_dim"]),
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        num_train_timesteps=cfg.num_train_timesteps,
        num_inference_steps=cfg.num_inference_steps,
        mask_pad_loss=cfg.mask_pad_loss,
        x0_clamp=cfg.x0_clamp,
    )
    policy = DiffusionPolicy(policy_cfg).to(device)
    policy.set_normalizers(obs_norm, act_norm)
    if cfg.init_checkpoint is not None:
        init = DiffusionPolicy.load(cfg.init_checkpoint, device=device)
        arch = ("obs_dim", "action_dim", "obs_horizon", "action_horizon", "num_train_timesteps",
                "num_inference_steps", "down_dims", "diffusion_step_embed_dim")
        diff = {k: (getattr(init.cfg, k), getattr(policy_cfg, k)) for k in arch
                if tuple(np.atleast_1d(getattr(init.cfg, k))) != tuple(np.atleast_1d(getattr(policy_cfg, k)))}
        if diff:
            raise ValueError(f"--init-checkpoint architecture mismatch (checkpoint, run): {diff}")
        policy.load_state_dict(init.state_dict())
        # Keep the checkpoint's scaling: the weights were trained against it.
        policy.set_normalizers(init.obs_normalizer, init.action_normalizer)
        init_digest = norm_digest(init.obs_normalizer, init.action_normalizer)
        if init_digest != digest:
            print(f"[WARN] --norm-mode gives {digest}, checkpoint has {init_digest}; using the checkpoint's")
        digest = init_digest
        print(f"[INFO] initialized from {cfg.init_checkpoint} (normalizer {digest})")
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    specs = _eval_specs(cfg)
    validator = None
    if cfg.val_dataset is not None:
        from mjlab_hand.diffusion.validate import DenoisedValidator

        if padded:
            val_tasks = [s["task"] for s in specs if s.get("pad")] or None
        else:
            val_tasks = [str(store.root.attrs.get("task"))]
        validator = DenoisedValidator(
            cfg.val_dataset,
            obs_horizon=cfg.obs_horizon,
            action_horizon=cfg.action_horizon,
            tasks=val_tasks,
            padded=padded,
            n_windows=cfg.val_windows,
        )
        print(f"[INFO] Validation: {cfg.val_windows} windows of {validator.tasks} from {cfg.val_dataset}")

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    run_meta = {**asdict(cfg), "dataset": str(cfg.dataset), "output_dir": str(out), "norm_digest": digest}
    (out / "train_config.json").write_text(json.dumps(run_meta, indent=2, default=str))
    print(f"[INFO] norm_mode={cfg.norm_mode} digest={digest}")

    wandb_run = None
    if cfg.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project, name=cfg.wandb_run_name, tags=cfg.wandb_tags,
            config=json.loads(json.dumps(run_meta, default=str)),
        )

    latest_path = out / "policy_latest.pt"
    eval_jsonl = out / "eval_metrics.jsonl"
    render_task = cfg.eval_task or (specs[0]["task"] if specs else None)
    best_loss, best_val, best_rollout = float("inf"), float("inf"), float("-inf")
    sel: dict = {"best_train_loss": None, "best_val": None, "best_rollout": None, "last": []}
    snaps = _Snapshots(out, cfg.keep_last)
    global_step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        policy.train()
        losses = []
        for batch in loader:
            loss = policy.compute_loss(
                batch["obs"].to(device),
                batch["action"].to(device),
                timesteps=batch["t"].to(device) if "t" in batch else None,
                t_min=batch["t_min"].to(device) if "t_min" in batch else None,
                action_valid=batch["action_valid"].to(device) if "action_valid" in batch else None,
            )
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
        if mean_loss < best_loss:  # selected by train loss: continuity only, not a quality signal
            best_loss = mean_loss
            policy.save(out / "policy_best.pt")
            sel["best_train_loss"] = {"epoch": epoch, "loss": mean_loss, "path": "policy_best.pt"}

        is_last = epoch == cfg.num_epochs
        if epoch % cfg.latest_every_epochs == 0 or is_last:
            policy.save(latest_path)
            snaps.add(latest_path, epoch)

        due_eval = bool(specs) and ((cfg.eval_every_epochs > 0 and epoch % cfg.eval_every_epochs == 0) or is_last)
        val = None
        if validator is not None and (due_eval or is_last):
            policy.eval()
            val = validator.evaluate(policy, device)
            print(f"[epoch {epoch:04d}/{cfg.num_epochs}] val={val['pooled']:.6f} {val['per_hand']}")
            if val["score"] < best_val:
                best_val = val["score"]
                policy.save(out / "policy_best_val.pt")
                sel["best_val"] = {"epoch": epoch, "score": best_val, "path": "policy_best_val.pt"}
            if wandb_run is not None:
                wandb_run.log({"val/pooled": val["pooled"]}, step=epoch)

        if due_eval:
            from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

            policy.save(latest_path)  # evaluate loads from disk: score the current weights
            policy.eval()
            print(f"[INFO] Running env eval at epoch {epoch}...")
            heads = []
            for spec in specs:
                metrics = evaluate_diffusion_policy(
                    task=spec["task"],
                    policy_path=latest_path,
                    num_envs=cfg.eval_num_envs,
                    num_steps=cfg.eval_num_steps,
                    device=str(device),
                    seed=cfg.seed,
                    onehot=spec.get("onehot"),
                    pad=bool(spec.get("pad", False)),
                    reuse_env=True,
                )
                row = {
                    "epoch": epoch,
                    "train_loss": mean_loss,
                    "train_steps": global_step,
                    "eval_task": spec["task"],
                    "onehot": spec.get("onehot"),
                    "pad": bool(spec.get("pad", False)),
                    "val": val,
                    "metrics": {k: v for k, v in metrics.items() if k != "per_episode"},
                }
                with eval_jsonl.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                h = _headline(metrics)
                print(f"[INFO] eval epoch={epoch} task={spec['task']} headline={h:.3f}")
                heads.append(h)
                if wandb_run is not None:
                    safe = spec["task"].replace("/", "_")
                    wandb_run.log({f"eval/{safe}/{k}": v for k, v in row["metrics"].items()}, step=epoch)
            finite = [h for h in heads if np.isfinite(h)]
            if finite and float(np.mean(finite)) > best_rollout:
                best_rollout = float(np.mean(finite))
                shutil.copy2(latest_path, out / "policy_best_rollout.pt")
                sel["best_rollout"] = {"epoch": epoch, "score": best_rollout, "path": "policy_best_rollout.pt"}

        if cfg.render_every_epochs > 0 and epoch % cfg.render_every_epochs == 0 and render_task is not None:
            try:
                from mjlab_hand.diffusion.evaluate import render_diffusion_rollout

                policy.save(latest_path)
                render_diffusion_rollout(
                    task=render_task,
                    policy_path=latest_path,
                    output_dir=out / "videos",
                    num_steps=cfg.render_num_steps,
                    num_envs=cfg.render_num_envs,
                    device=str(device),
                    seed=cfg.seed,
                    tag=f"epoch{epoch:04d}",
                    onehot=specs[0].get("onehot") if specs else None,
                    pad=bool(specs[0].get("pad", False)) if specs else False,
                )
            except Exception as exc:  # noqa: BLE001 - rendering must never kill training
                print(f"[WARN] render failed at epoch {epoch}: {exc}")

    if specs:
        from mjlab_hand.diffusion.evaluate import close_cached_envs

        close_cached_envs()
    if wandb_run is not None:
        wandb_run.summary["best_loss"] = best_loss
        wandb_run.finish()

    sel["last"] = [{"k": k, "epoch": ep, "path": p.name} for k, (ep, p) in enumerate(snaps.items)]
    sel.update({"num_epochs": cfg.num_epochs, "train_steps": global_step, "norm_digest": digest})
    # Written last: its presence means training finished (promote_outputs.sh).
    (out / "selection.json").write_text(json.dumps(sel, indent=2))
    print(f"[INFO] Training done. Best train loss={best_loss:.6f}; wrote {out / 'selection.json'}")
    return latest_path
