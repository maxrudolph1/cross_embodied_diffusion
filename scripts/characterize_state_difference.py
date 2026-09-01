#!/usr/bin/env python3
"""Decompose *what* the Allegro/LEAP observation difference is.

Marginal overlap per dimension, a "constant-label" score (between-hand mean
gap / within-hand std), removability under per-hand
centring/standardisation/ZCA whitening, and how few dimensions suffice to
separate the hands.

Two bugs found and fixed during development, both of which would have
produced confident wrong answers (CHANGES.md item 31):

- **Fix A -- accuracy below chance.** Episode-level splits do not balance
  step counts, so a signal-free model that predicts one class scores the
  majority fraction. Reports balanced accuracy and AUC (imbalance-invariant)
  and applies class weights in the fit; `logreg()` deliberately does not
  return plain accuracy.
- **Fix B -- test statistics leaking into the alignment.** For "does
  centring remove the difference?" to mean anything, the per-hand
  mean/std used to align train and test must be fit on train only.
  `align()` takes train/test splits and uses train-only statistics.

Writes outputs/analysis/state_difference.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab_hand.diffusion.dataset import TrajectoryStore

FAMILIES = {
    "grasp": ("Grasp-Allegro", "Grasp-LEAP"),
    "rotation": ("InHand-Rotation-Allegro", "InHand-Rotation-LEAP"),
}


def episode_split(store: TrajectoryStore, frac_train: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episodes = store.episode_slices(success_only=False)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_train = int(len(episodes) * frac_train)
    train_eps = set(order[:n_train].tolist())
    train_idx, test_idx = [], []
    for i, (s, e, _s) in enumerate(episodes):
        rows = np.arange(s, e)
        (train_idx if i in train_eps else test_idx).append(rows)
    return np.concatenate(train_idx), np.concatenate(test_idx)


def balanced_accuracy(preds: np.ndarray, y: np.ndarray) -> float:
    accs = [float((preds[y == c] == c).mean()) for c in (0, 1) if (y == c).any()]
    return float(np.mean(accs)) if accs else float("nan")


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_probe(model: nn.Module, x_train, y_train, x_test, y_test, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(x_train).float()
    yt = torch.from_numpy(y_train).float()
    n1 = float((y_train == 1).sum())
    n0 = float((y_train == 0).sum())
    pos_weight = torch.tensor([n0 / max(n1, 1.0)])
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(xt).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(x_test).float()).squeeze(-1)).numpy()
    preds = (probs >= 0.5).astype(np.float64)
    return balanced_accuracy(preds, y_test), auc_from_scores(probs, y_test)


def logreg(x_train, y_train, x_test, y_test, epochs=300, lr=0.05):
    """Logistic regression with class weights, scored by balanced accuracy
    and AUC. Deliberately does not return plain accuracy -- see Fix A."""
    return _fit_probe(nn.Linear(x_train.shape[1], 1), x_train, y_train, x_test, y_test, epochs, lr)


def mlp_probe(x_train, y_train, x_test, y_test, epochs=300, lr=0.05, hidden=128):
    model = nn.Sequential(nn.Linear(x_train.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    return _fit_probe(model, x_train, y_train, x_test, y_test, epochs, lr)


def align(x_a_train, x_a_test, x_b_train, x_b_test, kind: str):
    """Per-hand alignment using TRAIN-ONLY statistics (Fix B)."""
    if kind == "raw":
        return x_a_train, x_a_test, x_b_train, x_b_test
    mean_a, mean_b = x_a_train.mean(axis=0), x_b_train.mean(axis=0)
    if kind == "centred":
        return x_a_train - mean_a, x_a_test - mean_a, x_b_train - mean_b, x_b_test - mean_b
    std_a = x_a_train.std(axis=0) + 1e-8
    std_b = x_b_train.std(axis=0) + 1e-8
    if kind == "standardised":
        return (
            (x_a_train - mean_a) / std_a,
            (x_a_test - mean_a) / std_a,
            (x_b_train - mean_b) / std_b,
            (x_b_test - mean_b) / std_b,
        )
    if kind == "whitened":

        def zca(x_train, x_test, mean):
            xc = x_train - mean
            cov = xc.T @ xc / max(len(xc) - 1, 1) + 1e-6 * np.eye(xc.shape[1])
            u, s, _ = np.linalg.svd(cov)
            w = u @ np.diag(1.0 / np.sqrt(s)) @ u.T
            return (x_train - mean) @ w, (x_test - mean) @ w

        a_train_w, a_test_w = zca(x_a_train, x_a_test, mean_a)
        b_train_w, b_test_w = zca(x_b_train, x_b_test, mean_b)
        return a_train_w, a_test_w, b_train_w, b_test_w
    raise ValueError(kind)


def constant_label_score(x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    """Between-hand mean gap / within-hand std, per dimension."""
    gap = np.abs(x_a.mean(axis=0) - x_b.mean(axis=0))
    within_std = (x_a.std(axis=0) + x_b.std(axis=0)) / 2 + 1e-8
    return gap / within_std


def marginal_overlap(x_a: np.ndarray, x_b: np.ndarray, bins: int = 50) -> np.ndarray:
    """Histogram intersection per dimension (1 = identical marginals)."""
    d = x_a.shape[1]
    out = np.zeros(d)
    for j in range(d):
        lo, hi = min(x_a[:, j].min(), x_b[:, j].min()), max(x_a[:, j].max(), x_b[:, j].max())
        if hi <= lo:
            out[j] = 1.0
            continue
        edges = np.linspace(lo, hi, bins + 1)
        ha, _ = np.histogram(x_a[:, j], bins=edges, density=True)
        hb, _ = np.histogram(x_b[:, j], bins=edges, density=True)
        out[j] = float(np.minimum(ha, hb).sum() * (edges[1] - edges[0]))
    return out


def top_k_dims_needed(x_a, x_b, target_acc: float = 0.95, max_k: int = 10, seed: int = 0):
    """How few dims (ranked by constant-label score) suffice to reach
    `target_acc` balanced accuracy."""
    score = constant_label_score(x_a, x_b)
    order = np.argsort(-score)
    x = np.concatenate([x_a, x_b], axis=0)
    y = np.concatenate([np.zeros(len(x_a)), np.ones(len(x_b))])
    idx = np.random.default_rng(seed).permutation(len(y))
    split = int(len(y) * 0.7)
    tr, te = idx[:split], idx[split:]
    acc = float("nan")
    for k in range(1, max_k + 1):
        cols = order[:k]
        acc, _auc = logreg(x[tr][:, cols], y[tr], x[te][:, cols], y[te], epochs=150)
        if acc >= target_acc:
            return k, acc
    return max_k, acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos-dir", type=Path, default=Path("data/demos"))
    ap.add_argument("--pattern", default="{task}_expert.zarr")
    ap.add_argument("--schemas", type=Path, default=Path("outputs/analysis/schemas.json"))
    ap.add_argument("--frac-train", type=float, default=0.7)
    ap.add_argument("--n-max", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    schemas = json.loads(args.schemas.read_text()) if args.schemas.exists() else {}

    results: dict[str, dict] = {}
    for family, (task_a, task_b) in FAMILIES.items():
        path_a = args.demos_dir / args.pattern.format(task=task_a)
        path_b = args.demos_dir / args.pattern.format(task=task_b)
        if not path_a.exists() or not path_b.exists():
            print(f"[WARN] Missing dataset(s) for {family}, skipping")
            continue
        store_a = TrajectoryStore(path_a, mode="r")
        store_b = TrajectoryStore(path_b, mode="r")

        tr_a, te_a = episode_split(store_a, args.frac_train, args.seed)
        tr_b, te_b = episode_split(store_b, args.frac_train, args.seed + 1)
        rng = np.random.default_rng(args.seed)

        def cap(idx: np.ndarray) -> np.ndarray:
            return rng.choice(idx, size=args.n_max, replace=False) if len(idx) > args.n_max else idx

        tr_a, te_a, tr_b, te_b = cap(tr_a), cap(te_a), cap(tr_b), cap(te_b)
        obs_a = np.asarray(store_a.data["obs"][:], dtype=np.float64)
        obs_b = np.asarray(store_b.data["obs"][:], dtype=np.float64)
        if obs_a.shape[1] != obs_b.shape[1]:
            print(f"[WARN] Mismatched obs dims for {family}, skipping")
            continue

        a_train, a_test = obs_a[tr_a], obs_a[te_a]
        b_train, b_test = obs_b[tr_b], obs_b[te_b]

        fam_result: dict = {}
        for kind in ("raw", "centred", "standardised", "whitened"):
            at, av, bt, bv = align(a_train, a_test, b_train, b_test, kind)
            x_train = np.concatenate([at, bt], axis=0)
            y_train = np.concatenate([np.zeros(len(at)), np.ones(len(bt))])
            x_test = np.concatenate([av, bv], axis=0)
            y_test = np.concatenate([np.zeros(len(av)), np.ones(len(bv))])
            lin_acc, lin_auc = logreg(x_train, y_train, x_test, y_test)
            result = {"linear_balanced_acc": lin_acc, "linear_auc": lin_auc}
            if kind in ("raw", "centred", "standardised"):
                mlp_acc, mlp_auc = mlp_probe(x_train, y_train, x_test, y_test)
                result["mlp_balanced_acc"] = mlp_acc
                result["mlp_auc"] = mlp_auc
            fam_result[kind] = result
            print(f"[INFO] {family} {kind}: linear_auc={lin_auc:.4f}")

        overlap = marginal_overlap(a_train, b_train)
        k, k_acc = top_k_dims_needed(a_train, b_train, seed=args.seed)

        fam_result["marginal_overlap_per_dim"] = overlap.tolist()
        fam_result["constant_label_score_per_dim"] = constant_label_score(a_train, b_train).tolist()
        fam_result["disjoint_support_dims"] = int((overlap < 0.01).sum())
        fam_result["top_k_dims_for_target_acc"] = {"k": k, "balanced_acc": k_acc}

        schema_a = schemas.get(task_a)
        if schema_a is not None and schema_a.get("obs_terms"):
            names, dims = schema_a["obs_terms"], schema_a["obs_term_dims"]
            start = 0
            by_term = {}
            for name, d in zip(names, dims, strict=True):
                end = start + d
                by_term[name] = float(overlap[start:end].mean())
                start = end
            fam_result["marginal_overlap_by_term"] = by_term

        results[family] = fam_result

    if not results:
        raise SystemExit("No matched-pair datasets found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "state_difference.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
