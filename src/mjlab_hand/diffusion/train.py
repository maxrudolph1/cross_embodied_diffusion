"""Train a diffusion policy on collected expert trajectories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mjlab_hand.diffusion.dataset import DiffusionDataset, TrajectoryStore
from mjlab_hand.diffusion.normalizer import LinearNormalizer
from mjlab_hand.diffusion.policy import DiffusionPolicy, DiffusionPolicyConfig


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
    save_every_epochs: int = 10
    # How often to write policy_latest.pt / policy_best.pt. A 265MB
    # torch.save to NFS costs ~0.75s, comparable to a whole epoch at small
    # data scales -- see CHANGES.md item 13. Always saved on the final epoch
    # and immediately before any eval/render regardless of this cadence.
    latest_every_epochs: int = 1
    seed: int = 0
    # Optional in-loop env evaluation (disabled unless eval_task/eval_specs is set).
    eval_task: str | None = None
    # Multi-target eval: [{"task": ..., "onehot": [..] | None}, ...]. Lets a
    # mixed-embodiment policy be scored against each embodiment it drives.
    # `eval_task` alone is internally promoted to a single-element spec.
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
    # WandB logging. Disabled (None) by default; set wandb_project to enable.
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None


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

    store = TrajectoryStore(cfg.dataset, mode="r")
    summary = store.summary()
    print(f"[INFO] Dataset: {summary}")

    dataset = DiffusionDataset(
        store,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        success_only=cfg.success_only,
        ambient_tmin=cfg.ambient_tmin,
    )

    # Ambient diffusion samples the diffusion timestep FIRST, then a training
    # tuple valid at that timestep (see DiffusionDataset.sample_ambient_batch)
    # -- the reverse order silently starves low-noise training whenever the
    # admitted-everywhere (target) data is a small fraction of the mixed
    # dataset. This can't be expressed as a DataLoader shuffle over fixed
    # rows, so it bypasses the DataLoader entirely; ambient_rng is exhausted
    # once per batch rather than once per dataset pass.
    is_ambient = cfg.ambient_tmin is not None
    loader: DataLoader | None = None
    ambient_rng: np.random.Generator | None = None
    num_batches_per_epoch = len(dataset) // cfg.batch_size
    if is_ambient:
        ambient_rng = np.random.default_rng(cfg.seed)
    else:
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )

    obs_norm = LinearNormalizer.fit(dataset.obs)
    act_norm = LinearNormalizer.fit(dataset.action)

    policy_cfg = DiffusionPolicyConfig(
        obs_dim=int(summary["obs_dim"]),
        action_dim=int(summary["action_dim"]),
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        num_train_timesteps=cfg.num_train_timesteps,
        num_inference_steps=cfg.num_inference_steps,
    )
    policy = DiffusionPolicy(policy_cfg).to(device)
    policy.set_normalizers(obs_norm, act_norm)

    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "train_config.json").write_text(
        json.dumps(
            {**asdict(cfg), "dataset": str(cfg.dataset), "output_dir": str(cfg.output_dir)},
            indent=2,
            default=str,
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
            config={**asdict(cfg), "dataset": str(cfg.dataset), "output_dir": str(cfg.output_dir)},
        )

    global_step = 0
    best_loss = float("inf")
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
            loss = policy.compute_loss(obs, action, timesteps=timesteps)
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

        is_last = epoch == cfg.num_epochs
        due = epoch % cfg.latest_every_epochs == 0 or is_last
        if due:
            policy.save(latest_path)
        if mean_loss < best_loss:
            best_loss = mean_loss
            if due:
                policy.save(cfg.output_dir / "policy_best.pt")
        if epoch % cfg.save_every_epochs == 0:
            policy.save(cfg.output_dir / f"policy_epoch_{epoch:04d}.pt")

        if specs and cfg.eval_every_epochs > 0 and epoch % cfg.eval_every_epochs == 0:
            from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

            # evaluate_diffusion_policy loads from disk; make sure it scores
            # the current weights even if latest_every_epochs skipped this epoch.
            policy.save(latest_path)
            policy.eval()
            print(f"[INFO] Running env eval at epoch {epoch}...")
            for spec in specs:
                metrics = evaluate_diffusion_policy(
                    task=spec["task"],
                    policy_path=latest_path,
                    num_envs=cfg.eval_num_envs,
                    num_steps=cfg.eval_num_steps,
                    device=str(device),
                    seed=cfg.seed,
                    onehot=spec.get("onehot"),
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
                if wandb_run is not None:
                    safe_task = spec["task"].replace("/", "_")
                    wandb_run.log(
                        {f"eval/{safe_task}/{k}": v for k, v in metrics.items()}, step=epoch
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
                )
            except Exception as exc:  # noqa: BLE001 - rendering must never kill training
                print(f"[WARN] render failed at epoch {epoch}: {exc}")

    if specs:
        from mjlab_hand.diffusion.evaluate import close_cached_envs

        close_cached_envs()

    if wandb_run is not None:
        wandb_run.summary["best_loss"] = best_loss
        wandb_run.finish()

    print(f"[INFO] Training done. Best loss={best_loss:.6f}")
    print(f"[INFO] Saved {latest_path}")
    return latest_path
