#!/usr/bin/env python3
"""Is the embodiment identifiable from a single observation (or action)?

Linear and small-MLP classifiers (hand-rolled in torch -- no sklearn, kept
out of this venv deliberately rather than adding it while other jobs were
running), held-out accuracy and AUC, plus a per-term breakdown and a
hypothetical noise-vs-separability curve using the policy's own cosine
schedule.

**Splits are by EPISODE, never by step.** Consecutive frames within an
episode are near-duplicates, so a random step-level split leaks test data
into training and drives accuracy toward 1.0 for any two datasets.

AUC is computed from the rank statistic (Mann-Whitney U), not a library
call.

Writes outputs/analysis/state_separability.json (consumed by
`probe_state_equivalence.py`'s per-term "task subspace" selection: terms
with per-term accuracy < 0.70) and outputs/plots/state_separability.png.

See CHANGES.md item 30.
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

from mjlab_hand.diffusion.dataset import TrajectoryStore
from mjlab_hand.diffusion.policy import cosine_beta_schedule

FAMILIES = {
    "grasp": ("Grasp-Allegro", "Grasp-LEAP"),
    "rotation": ("InHand-Rotation-Allegro", "InHand-Rotation-LEAP"),
}


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank-sum (Mann-Whitney U) statistic."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    sum_ranks_pos = ranks[labels == 1].sum()
    u = sum_ranks_pos - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def episode_split(store: TrajectoryStore, frac_train: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_row_idx, test_row_idx), split by whole episode."""
    episodes = store.episode_slices(success_only=False)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_train = int(len(episodes) * frac_train)
    train_eps = set(order[:n_train].tolist())
    train_idx, test_idx = [], []
    for i, (s, e, _succ) in enumerate(episodes):
        rows = np.arange(s, e)
        (train_idx if i in train_eps else test_idx).append(rows)
    return np.concatenate(train_idx), np.concatenate(test_idx)


class MLP(nn.Module):
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_and_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    kind: str,
    epochs: int = 300,
    lr: float = 0.05,
) -> tuple[float, float]:
    """Returns (held_out_accuracy, held_out_auc)."""
    d = x_train.shape[1]
    model: nn.Module = nn.Linear(d, 1) if kind == "linear" else MLP(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(x_train).float()
    yt = torch.from_numpy(y_train).float()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(xt).squeeze(-1) if kind == "linear" else model(xt)
        loss = F.binary_cross_entropy_with_logits(logits, yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        xv = torch.from_numpy(x_test).float()
        logits = model(xv).squeeze(-1) if kind == "linear" else model(xv)
        probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(np.float64)
    acc = float((preds == y_test).mean())
    auc = auc_from_scores(probs, y_test)
    return acc, auc


def load_pair(task_a: str, task_b: str, demos_dir: Path, pattern: str) -> tuple[TrajectoryStore, TrajectoryStore]:
    return (
        TrajectoryStore(demos_dir / pattern.format(task=task_a), mode="r"),
        TrajectoryStore(demos_dir / pattern.format(task=task_b), mode="r"),
    )


def stack_split(store_a, store_b, field: str, frac_train: float, seed: int, n_max: int):
    tr_a, te_a = episode_split(store_a, frac_train, seed)
    tr_b, te_b = episode_split(store_b, frac_train, seed + 1)
    rng = np.random.default_rng(seed)

    def cap(idx: np.ndarray) -> np.ndarray:
        if len(idx) > n_max:
            idx = rng.choice(idx, size=n_max, replace=False)
        return idx

    tr_a, te_a, tr_b, te_b = cap(tr_a), cap(te_a), cap(tr_b), cap(te_b)
    a_train = np.asarray(store_a.data[field][:], dtype=np.float64)[tr_a]
    a_test = np.asarray(store_a.data[field][:], dtype=np.float64)[te_a]
    b_train = np.asarray(store_b.data[field][:], dtype=np.float64)[tr_b]
    b_test = np.asarray(store_b.data[field][:], dtype=np.float64)[te_b]

    x_train = np.concatenate([a_train, b_train], axis=0)
    y_train = np.concatenate([np.zeros(len(a_train)), np.ones(len(b_train))])
    x_test = np.concatenate([a_test, b_test], axis=0)
    y_test = np.concatenate([np.zeros(len(a_test)), np.ones(len(b_test))])
    return x_train, y_train, x_test, y_test


def noised_separability(x_train, y_train, x_test, y_test, alphas_cumprod: np.ndarray, timesteps, seed):
    """Accuracy of a linear probe as a function of hypothetical noise level,
    using the policy's own cosine schedule."""
    rng = np.random.default_rng(seed)
    out = []
    for t in timesteps:
        ab = float(alphas_cumprod[t])

        def noise(x):
            return np.sqrt(ab) * x + np.sqrt(max(1.0 - ab, 0.0)) * rng.standard_normal(x.shape)

        acc, _auc = fit_and_eval(noise(x_train), y_train, noise(x_test), y_test, "linear", epochs=150)
        out.append(acc)
    return out


def per_term_separability(
    x_train, y_train, x_test, y_test, term_names: list[str], term_dims: list[int]
) -> dict[str, float]:
    out = {}
    start = 0
    for name, d in zip(term_names, term_dims, strict=True):
        end = start + d
        acc, _auc = fit_and_eval(
            x_train[:, start:end], y_train, x_test[:, start:end], y_test, "linear", epochs=150
        )
        out[name] = acc
        start = end
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos-dir", type=Path, default=Path("data/demos"))
    ap.add_argument("--pattern", default="{task}_expert.zarr")
    ap.add_argument("--schemas", type=Path, default=Path("outputs/analysis/schemas.json"))
    ap.add_argument("--frac-train", type=float, default=0.7)
    ap.add_argument("--n-max", type=int, default=20000, help="Row cap per split/embodiment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-train-timesteps", type=int, default=100)
    ap.add_argument("--noise-timesteps", type=int, nargs="+", default=[0, 25, 50, 75, 90, 99])
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    ap.add_argument("--plots-dir", type=Path, default=Path("outputs/plots"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    schemas = json.loads(args.schemas.read_text()) if args.schemas.exists() else {}
    betas = cosine_beta_schedule(args.num_train_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).numpy()

    results: dict[str, dict] = {}
    for family, (task_a, task_b) in FAMILIES.items():
        path_a = args.demos_dir / args.pattern.format(task=task_a)
        path_b = args.demos_dir / args.pattern.format(task=task_b)
        if not path_a.exists() or not path_b.exists():
            print(f"[WARN] Missing dataset(s) for {family}, skipping")
            continue
        store_a, store_b = load_pair(task_a, task_b, args.demos_dir, args.pattern)

        fam_result: dict = {}
        for field in ("obs", "action"):
            x_train, y_train, x_test, y_test = stack_split(
                store_a, store_b, field, args.frac_train, args.seed, args.n_max
            )
            lin_acc, lin_auc = fit_and_eval(x_train, y_train, x_test, y_test, "linear")
            mlp_acc, mlp_auc = fit_and_eval(x_train, y_train, x_test, y_test, "mlp")
            noise_curve = noised_separability(
                x_train, y_train, x_test, y_test, alphas_cumprod, args.noise_timesteps, args.seed
            )
            fam_result[field] = {
                "linear_acc": lin_acc,
                "linear_auc": lin_auc,
                "mlp_acc": mlp_acc,
                "mlp_auc": mlp_auc,
                "noise_timesteps": args.noise_timesteps,
                "noise_curve_acc": noise_curve,
            }
            print(f"[INFO] {family} {field}: linear_acc={lin_acc:.4f} mlp_acc={mlp_acc:.4f}")

        schema_a = schemas.get(task_a)
        if schema_a is not None and schema_a.get("obs_terms"):
            x_train, y_train, x_test, y_test = stack_split(
                store_a, store_b, "obs", args.frac_train, args.seed, args.n_max
            )
            fam_result["obs_per_term_acc"] = per_term_separability(
                x_train, y_train, x_test, y_test, schema_a["obs_terms"], schema_a["obs_term_dims"]
            )

        results[family] = fam_result

    if not results:
        raise SystemExit("No matched-pair datasets found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "state_separability.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    for i, (family, r) in enumerate(results.items()):
        ax = axes[0][i]
        for field, color in (("obs", "C0"), ("action", "C1")):
            ax.plot(r[field]["noise_timesteps"], r[field]["noise_curve_acc"], "o-", color=color, label=field)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("diffusion timestep t")
        ax.set_ylabel("linear probe accuracy")
        ax.set_ylim(0.4, 1.05)
        ax.set_title(family)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Embodiment separability vs hypothetical noise level")
    fig.tight_layout()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.plots_dir / "state_separability.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[INFO] Wrote {args.output_dir / 'state_separability.json'}")
    print(f"[INFO] Wrote {out_png}")


if __name__ == "__main__":
    main()
