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
    seed: int = 0
    # Optional in-loop env evaluation (disabled unless eval_task is set).
    eval_task: str | None = None
    eval_every_epochs: int = 10
    eval_num_envs: int = 32
    eval_num_steps: int = 1500


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
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
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

    global_step = 0
    best_loss = float("inf")
    latest_path = cfg.output_dir / "policy_latest.pt"
    eval_jsonl = cfg.output_dir / "eval_metrics.jsonl"

    for epoch in range(1, cfg.num_epochs + 1):
        policy.train()
        losses = []
        for batch in loader:
            obs = batch["obs"].to(device)
            action = batch["action"].to(device)
            loss = policy.compute_loss(obs, action)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            global_step += 1

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[epoch {epoch:04d}/{cfg.num_epochs}] loss={mean_loss:.6f} steps={global_step}")
        policy.save(latest_path)
        if mean_loss < best_loss:
            best_loss = mean_loss
            policy.save(cfg.output_dir / "policy_best.pt")
        if epoch % cfg.save_every_epochs == 0:
            policy.save(cfg.output_dir / f"policy_epoch_{epoch:04d}.pt")

        if (
            cfg.eval_task
            and cfg.eval_every_epochs > 0
            and epoch % cfg.eval_every_epochs == 0
        ):
            from mjlab_hand.diffusion.evaluate import evaluate_diffusion_policy

            print(f"[INFO] Running env eval at epoch {epoch}...")
            policy.eval()
            metrics = evaluate_diffusion_policy(
                task=cfg.eval_task,
                policy_path=latest_path,
                num_envs=cfg.eval_num_envs,
                num_steps=cfg.eval_num_steps,
                device=str(device),
                seed=cfg.seed + epoch,
            )
            row = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_steps": global_step,
                "metrics": metrics,
            }
            with eval_jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"[INFO] eval epoch={epoch} "
                f"success_rate={metrics.get('success_rate', float('nan')):.3f}"
            )

    print(f"[INFO] Training done. Best loss={best_loss:.6f}")
    print(f"[INFO] Saved {latest_path}")
    return latest_path
