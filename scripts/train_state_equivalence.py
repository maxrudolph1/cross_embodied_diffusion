#!/usr/bin/env python3
"""Train an encoder with task-decode + gradient-reversal-adversarial +
cross-embodiment losses, sweeping the reversal strength lambda, and ask
whether z can be embodiment-invariant *and* task-informative. Trains no
policy.

**The methodology point that decides whether the numbers mean anything:**
invariance is scored by a fresh probe trained AFTER the encoder is frozen,
on held-out episodes -- never by the adversary's own head, which reads
optimistic: the adversary's own training accuracy can fall toward chance
while a fresh probe on the same frozen encoder still separates the hands
near-perfectly.

Two further guards:

- **Shared (not per-hand) standardisation.** Per-hand standardisation would
  hand the encoder invariance for free, in a way a deployed policy (which
  sees one un-normalised observation stream) could not rely on.
- **`--disc-steps`** extra discriminator updates per encoder step, so
  gradient reversal only supplies a useful signal when the discriminator
  stays near optimal.

Writes outputs/analysis/state_equivalence_training.json and
outputs/plots/state_equivalence_tradeoff.png. See CHANGES.md item 38.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from mjlab_hand.diffusion.dataset import TrajectoryStore

FAMILIES = {
    "grasp": ("Grasp-Allegro", "Grasp-LEAP"),
    "rotation": ("InHand-Rotation-Allegro", "InHand-Rotation-LEAP"),
}


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class Encoder(nn.Module):
    def __init__(self, d_in: int, zdim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, zdim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Head(nn.Module):
    def __init__(self, zdim: int, d_out: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(zdim, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def episode_split(store: TrajectoryStore, frac_train: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episodes = store.episode_slices(success_only=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_train = int(len(episodes) * frac_train)
    train_eps, test_eps = order[:n_train], order[n_train:]

    def rows(idxs: np.ndarray) -> np.ndarray:
        if len(idxs) == 0:
            return np.array([], dtype=np.int64)
        return np.concatenate([np.arange(episodes[i][0], episodes[i][1]) for i in idxs])

    return rows(train_eps), rows(test_eps)


def load_data(path: Path, max_episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    store = TrajectoryStore(path, mode="r")
    tr_idx, te_idx = episode_split(store, 0.7, seed)
    obs = np.asarray(store.data["obs"][:], dtype=np.float64)
    cap_train = max_episodes * 500
    cap_test = max(max_episodes // 3, 1) * 500
    return obs[tr_idx][:cap_train], obs[te_idx][:cap_test]


def fresh_probe_accuracy(z_train, y_train, z_test, y_test, epochs: int = 200, lr: float = 0.05) -> float:
    """A newly-initialised MLP probe trained on the frozen encoder's
    outputs, evaluated on held-out data -- the honest invariance check."""
    model = nn.Sequential(nn.Linear(z_train.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt = torch.from_numpy(z_train).float(), torch.from_numpy(y_train).float()
    for _ in range(epochs):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(xt).squeeze(-1), yt).backward()
        opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(z_test).float()).squeeze(-1)).numpy()
    preds = (probs >= 0.5).astype(np.float64)
    accs = [float((preds[y_test == c] == c).mean()) for c in (0, 1) if (y_test == c).any()]
    return float(np.mean(accs)) if accs else float("nan")


def train_one_lambda(
    obs_a_train: np.ndarray,
    obs_b_train: np.ndarray,
    obs_a_test: np.ndarray,
    obs_b_test: np.ndarray,
    lam: float,
    zdim: int,
    epochs: int,
    disc_steps: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    d = obs_a_train.shape[1]
    encoder = Encoder(d, zdim)
    task_head = Head(zdim, d)  # task-decode: reconstruct the standardised observation
    disc = Head(zdim, 1)

    enc_opt = torch.optim.Adam(list(encoder.parameters()) + list(task_head.parameters()), lr=1e-3)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=1e-3)

    x_a = torch.from_numpy(obs_a_train).float()
    x_b = torch.from_numpy(obs_b_train).float()
    batch = min(256, len(x_a), len(x_b))

    for _ in range(epochs):
        idx_a = torch.randint(0, len(x_a), (batch,))
        idx_b = torch.randint(0, len(x_b), (batch,))
        x = torch.cat([x_a[idx_a], x_b[idx_b]], dim=0)
        y = torch.cat([torch.zeros(batch), torch.ones(batch)], dim=0)

        for _ in range(disc_steps):
            with torch.no_grad():
                z = encoder(x)
            disc_opt.zero_grad()
            F.binary_cross_entropy_with_logits(disc(z).squeeze(-1), y).backward()
            disc_opt.step()

        enc_opt.zero_grad()
        z = encoder(x)
        task_loss = F.mse_loss(task_head(z), x)
        adv_loss = F.binary_cross_entropy_with_logits(disc(grad_reverse(z, lam)).squeeze(-1), y)
        (task_loss + adv_loss).backward()
        enc_opt.step()

    # The adversary's own accuracy -- optimistic, and the trap: it can fall
    # toward chance while a fresh probe on the same frozen encoder does not.
    with torch.no_grad():
        x_final = torch.cat([x_a, x_b], dim=0)
        y_final = torch.cat([torch.zeros(len(x_a)), torch.ones(len(x_b))]).numpy()
        adv_probs = torch.sigmoid(disc(encoder(x_final)).squeeze(-1)).numpy()
    adv_train_acc = float(((adv_probs >= 0.5).astype(np.float64) == y_final).mean())

    with torch.no_grad():
        z_a_train = encoder(x_a).numpy()
        z_b_train = encoder(x_b).numpy()
        z_a_test = encoder(torch.from_numpy(obs_a_test).float()).numpy()
        z_b_test = encoder(torch.from_numpy(obs_b_test).float()).numpy()

    z_train = np.concatenate([z_a_train, z_b_train], axis=0)
    y_train = np.concatenate([np.zeros(len(z_a_train)), np.ones(len(z_b_train))])
    z_test = np.concatenate([z_a_test, z_b_test], axis=0)
    y_test = np.concatenate([np.zeros(len(z_a_test)), np.ones(len(z_b_test))])
    fresh_acc = fresh_probe_accuracy(z_train, y_train, z_test, y_test)

    x_test_all = np.concatenate([obs_a_test, obs_b_test], axis=0)
    with torch.no_grad():
        recon_test = task_head(encoder(torch.from_numpy(x_test_all).float())).numpy()
    ss_res = ((recon_test - x_test_all) ** 2).sum()
    ss_tot = ((x_test_all - x_test_all.mean(axis=0)) ** 2).sum()
    task_r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))

    return {"adv_train_acc": adv_train_acc, "fresh_probe_acc": fresh_acc, "task_r2": task_r2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos-dir", type=Path, default=Path("data/demos"))
    ap.add_argument("--pattern", default="{task}_expert.zarr")
    ap.add_argument("--size", default="400k", help="Informational tag; bake the actual scale into --pattern")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--max-episodes", type=int, default=120)
    ap.add_argument("--disc-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    ap.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for family, (task_a, task_b) in FAMILIES.items():
        path_a = args.demos_dir / args.pattern.format(task=task_a)
        path_b = args.demos_dir / args.pattern.format(task=task_b)
        if not path_a.exists() or not path_b.exists():
            print(f"[WARN] Missing dataset(s) for {family}, skipping")
            continue
        obs_a_train, obs_a_test = load_data(path_a, args.max_episodes, args.seed)
        obs_b_train, obs_b_test = load_data(path_b, args.max_episodes, args.seed + 1)
        if obs_a_train.shape[1] != obs_b_train.shape[1]:
            print(f"[WARN] Mismatched dims for {family}, skipping")
            continue

        # Shared (not per-hand) standardisation, fit on train only.
        pooled = np.concatenate([obs_a_train, obs_b_train], axis=0)
        mean, std = pooled.mean(axis=0), pooled.std(axis=0) + 1e-8

        def norm(x: np.ndarray) -> np.ndarray:
            return (x - mean) / std

        obs_a_train, obs_a_test = norm(obs_a_train), norm(obs_a_test)
        obs_b_train, obs_b_test = norm(obs_b_train), norm(obs_b_test)

        fam_results: dict[str, dict] = {}
        for lam in args.lambdas:
            r = train_one_lambda(
                obs_a_train, obs_b_train, obs_a_test, obs_b_test,
                lam, args.zdim, args.epochs, args.disc_steps, args.seed,
            )
            fam_results[str(lam)] = r
            print(
                f"[INFO] {family} lambda={lam}: adv_train_acc={r['adv_train_acc']:.4f} "
                f"fresh_probe_acc={r['fresh_probe_acc']:.4f} task_r2={r['task_r2']:.4f}"
            )
        results[family] = fam_results

    if not results:
        raise SystemExit("No matched-pair datasets found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "state_equivalence_training.json"
    out_path.write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for family, fam_results in results.items():
        lambdas = sorted(float(k) for k in fam_results)
        fresh = [fam_results[str(lam)]["fresh_probe_acc"] for lam in lambdas]
        adv = [fam_results[str(lam)]["adv_train_acc"] for lam in lambdas]
        r2 = [fam_results[str(lam)]["task_r2"] for lam in lambdas]
        axes[0].plot(lambdas, fresh, "o-", label=f"{family} fresh probe")
        axes[0].plot(lambdas, adv, "x--", alpha=0.6, label=f"{family} adversary (optimistic)")
        axes[1].plot(lambdas, r2, "o-", label=family)
    axes[0].set_xscale("symlog")
    axes[0].axhline(0.5, color="gray", linestyle=":")
    axes[0].set_xlabel("lambda (reversal strength)")
    axes[0].set_ylabel("embodiment separability accuracy")
    axes[0].set_title("Invariance: fresh probe vs adversary's own accuracy")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xscale("symlog")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("task reconstruction R^2")
    axes[1].set_title("Task information retained")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.plots_dir / "state_equivalence_tradeoff.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[INFO] Wrote {out_path}")
    print(f"[INFO] Wrote {out_png}")


if __name__ == "__main__":
    main()
