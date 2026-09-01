#!/usr/bin/env python3
"""Standalone A/B check of the diffusion sampler against recorded expert actions.

No env needed: loads a policy checkpoint and a demo dataset, and compares the
current DDIM sampler in `DiffusionPolicy.predict_action` against the old
(invalid) strided single-step ancestral-update sampler, plus noise/zeros
baselines, in normalized action space. Use as a regression check if rollout
success ever collapses again. See CHANGES.md item 1.

Expect DDIM MSE ~= 0.002 vs old ~= 0.53 (worse than the ~0.09 predict-zeros
baseline) on a checkpoint trained only a few epochs.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from mjlab_hand.diffusion.dataset import DiffusionDataset, TrajectoryStore
from mjlab_hand.diffusion.policy import DiffusionPolicy


@torch.no_grad()
def sample_old(policy: DiffusionPolicy, obs: torch.Tensor) -> torch.Tensor:
    """Pre-fix strided single-step ancestral update. Invalid for a strided
    inference schedule (a single-step DDPM update only holds for a t -> t-1
    transition); reproduced here only for comparison against the DDIM fix."""
    cfg = policy.cfg
    device = obs.device
    b = obs.shape[0]
    nobs = policy.obs_normalizer.normalize(obs).reshape(b, -1)
    x = torch.randn(b, cfg.action_horizon, cfg.action_dim, device=device)
    timesteps = policy.inference_timesteps.tolist()
    for i in reversed(range(len(timesteps))):
        t = torch.full((b,), int(timesteps[i]), device=device, dtype=torch.long)
        eps = policy.noise_pred_net(x, t, nobs)
        alpha = policy.alphas[t].view(-1, 1, 1)
        alpha_bar = policy.alphas_cumprod[t].view(-1, 1, 1)
        beta = policy.betas[t].view(-1, 1, 1)
        x = (1.0 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps)
        if i > 0:
            x = x + torch.sqrt(beta) * torch.randn_like(x)
    return x  # normalized action space


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-windows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = DiffusionPolicy.load(args.checkpoint, device=device)
    policy.eval()

    store = TrajectoryStore(args.dataset, mode="r")
    dataset = DiffusionDataset(
        store,
        obs_horizon=policy.cfg.obs_horizon,
        action_horizon=policy.cfg.action_horizon,
        success_only=True,
    )

    n = min(args.num_windows, len(dataset))
    idx = np.random.choice(len(dataset), size=n, replace=False)
    obs = torch.stack([dataset[int(i)]["obs"] for i in idx]).to(device)
    action = torch.stack([dataset[int(i)]["action"] for i in idx]).to(device)
    naction_true = policy.action_normalizer.normalize(action)

    with torch.no_grad():
        pred_new = policy.action_normalizer.normalize(policy.predict_action(obs))
        pred_old = sample_old(policy, obs)

    noise_baseline = torch.randn_like(naction_true)
    zeros_baseline = torch.zeros_like(naction_true)

    print(f"[INFO] {n} windows from {args.dataset}")
    print(f"DDIM (current) sampler MSE:  {mse(pred_new, naction_true):.6f}")
    print(f"Ancestral (old) sampler MSE: {mse(pred_old, naction_true):.6f}")
    print(f"noise baseline MSE:          {mse(noise_baseline, naction_true):.6f}")
    print(f"zeros baseline MSE:          {mse(zeros_baseline, naction_true):.6f}")
    print(f"true action std:             {float(naction_true.std().item()):.6f}")
    print(f"DDIM (current) pred std:     {float(pred_new.std().item()):.6f}")
    print(f"Ancestral (old) pred std:    {float(pred_old.std().item()):.6f}")


if __name__ == "__main__":
    main()
