#!/usr/bin/env python3
"""Check the three preconditions for a usable Allegro/LEAP state
equivalence, before training any encoder.

- Q1 (populated equivalence classes): nearest-neighbour distance in the
  task subspace, cross-embodiment vs a within-embodiment floor.
- Q2 (joint invariance): MLP embodiment separability from the task subspace
  alone. Checked on the *concatenation* -- individually-at-chance terms can
  still be jointly identifying.
- Q3 (action correspondence): R^2 of Allegro-action -> matched-LEAP-action,
  against a task-state-only baseline. The *comparison*, not the absolute
  R^2, is the answer: does the action correspondence carry information the
  matched task state does not?

Two methodology points:

- The within-embodiment NN baseline loads **disjoint episode halves**
  (`load_half(..., half=0/1)`). Querying a set against itself returns the
  point itself or an adjacent near-identical frame, giving a meaningless
  near-zero distance.
- The task subspace is selected from measured per-term separability
  (`acc < 0.70` in `state_separability.json`), not by intuition about which
  terms "are" object state.

Writes outputs/analysis/state_equivalence.json. See CHANGES.md item 37.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from characterize_state_difference import logreg, mlp_probe

from mjlab_hand.diffusion.dataset import TrajectoryStore

FAMILIES = {
    "grasp": ("Grasp-Allegro", "Grasp-LEAP"),
    "rotation": ("InHand-Rotation-Allegro", "InHand-Rotation-LEAP"),
}
SEPARABILITY_THRESHOLD = 0.70
# Domain-informed pure-object subspace, for Q2's second row.
HAND_PICKED_OBJECT_TERMS = {
    "grasp": ["object_pos", "object_quat", "object_goal_pos"],
    "rotation": ["object_quat", "object_angvel", "goal_quat"],
}


def task_subspace(schema: dict, per_term_acc: dict[str, float], threshold: float) -> list[str]:
    return [n for n in schema["obs_terms"] if per_term_acc.get(n, 1.0) < threshold]


def term_slice_map(schema: dict) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    start = 0
    for name, d in zip(schema["obs_terms"], schema["obs_term_dims"], strict=True):
        out[name] = (start, start + d)
        start += d
    return out


def subspace_cols(schema: dict, terms: list[str]) -> np.ndarray:
    slices = term_slice_map(schema)
    cols: list[int] = []
    for t in terms:
        s, e = slices[t]
        cols.extend(range(s, e))
    return np.asarray(cols, dtype=np.int64)


def load_episodes(path: Path):
    store = TrajectoryStore(path, mode="r")
    episodes = store.episode_slices(success_only=True)
    obs = np.asarray(store.data["obs"][:], dtype=np.float64)
    action = np.asarray(store.data["action"][:], dtype=np.float64)
    return episodes, obs, action


def episode_rows(episodes, indices: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(episodes[i][0], episodes[i][1]) for i in indices])


def load_half(episodes, obs: np.ndarray, half: int, seed: int, max_episodes: int) -> np.ndarray:
    """Rows from a disjoint half of episodes -- avoids the near-zero,
    meaningless distance you get from querying a set against itself."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    mid = len(order) // 2
    chosen = order[:mid] if half == 0 else order[mid:]
    chosen = chosen[: max(max_episodes // 2, 1)]
    return obs[episode_rows(episodes, chosen)]


def nn_median_distance(query: np.ndarray, reference: np.ndarray, n_query: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if len(query) > n_query:
        query = query[rng.choice(len(query), size=n_query, replace=False)]
    q = torch.from_numpy(query).float()
    r = torch.from_numpy(reference).float()
    dists = []
    for i in range(0, q.shape[0], 512):
        dists.append(torch.cdist(q[i : i + 512], r).min(dim=1).values)
    return float(np.median(torch.cat(dists).numpy()))


def match_by_nn(query_subspace: np.ndarray, ref_subspace: np.ndarray) -> np.ndarray:
    q = torch.from_numpy(query_subspace).float()
    r = torch.from_numpy(ref_subspace).float()
    idx = []
    for i in range(0, q.shape[0], 512):
        idx.append(torch.cdist(q[i : i + 512], r).argmin(dim=1))
    return torch.cat(idx).numpy()


def fit_regression(x_train, y_train, x_test, y_test, epochs=300, lr=0.05) -> float:
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(x_train.shape[1], 128), nn.ReLU(), nn.Linear(128, y_train.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(torch.from_numpy(x_test).float()).numpy()
    ss_res = ((pred - y_test) ** 2).sum()
    ss_tot = ((y_test - y_test.mean(axis=0)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-8))


def analyze_family(
    family: str,
    task_a: str,
    task_b: str,
    demos_dir: Path,
    pattern: str,
    schemas: dict,
    separability: dict,
    threshold: float,
    max_episodes: int,
    n_match: int,
    seed: int,
) -> dict:
    path_a = demos_dir / pattern.format(task=task_a)
    path_b = demos_dir / pattern.format(task=task_b)
    schema_a = schemas[task_a]

    per_term_acc = separability.get(family, {}).get("obs_per_term_acc", {})
    subspace_terms = task_subspace(schema_a, per_term_acc, threshold)
    if not subspace_terms:
        raise RuntimeError(f"{family}: no terms below separability threshold {threshold}")
    cols = subspace_cols(schema_a, subspace_terms)

    episodes_a, obs_a, action_a = load_episodes(path_a)
    episodes_b, obs_b, _action_b = load_episodes(path_b)

    half0_a = load_half(episodes_a, obs_a, 0, seed, max_episodes)[:, cols]
    half1_a = load_half(episodes_a, obs_a, 1, seed, max_episodes)[:, cols]
    all_a_sub = obs_a[:, cols]
    all_b_sub = obs_b[:, cols]

    q1_cross = nn_median_distance(all_a_sub, all_b_sub, n_match, seed)
    q1_self = nn_median_distance(half0_a, half1_a, n_match, seed)

    y = np.concatenate([np.zeros(len(all_a_sub)), np.ones(len(all_b_sub))])
    idx = np.random.default_rng(seed).permutation(len(y))
    split = int(len(y) * 0.7)
    tr, te = idx[:split], idx[split:]

    x_sub = np.concatenate([all_a_sub, all_b_sub], axis=0)
    _lin_acc, _lin_auc = logreg(x_sub[tr], y[tr], x_sub[te], y[te])
    mlp_acc, mlp_auc = mlp_probe(x_sub[tr], y[tr], x_sub[te], y[te])

    hand_picked = [t for t in HAND_PICKED_OBJECT_TERMS[family] if t in schema_a["obs_terms"]]
    hp_cols = subspace_cols(schema_a, hand_picked) if hand_picked else cols
    x_hp = np.concatenate([obs_a[:, hp_cols], obs_b[:, hp_cols]], axis=0)
    hp_mlp_acc, _hp_mlp_auc = mlp_probe(x_hp[tr], y[tr], x_hp[te], y[te])

    x_full = np.concatenate([obs_a, obs_b], axis=0)
    full_mlp_acc, _full_mlp_auc = mlp_probe(x_full[tr], y[tr], x_full[te], y[te])

    # Q3: match by nearest neighbour in the task subspace, then compare
    # action-based vs task-state-based regression to the matched LEAP action.
    action_b_full = load_episodes(path_b)[2]
    rng = np.random.default_rng(seed)
    n = min(n_match, len(all_a_sub))
    q_idx = rng.choice(len(all_a_sub), size=n, replace=False)
    match_idx = match_by_nn(all_a_sub[q_idx], all_b_sub)

    a_action, a_state = action_a[q_idx], all_a_sub[q_idx]
    b_action_matched = action_b_full[match_idx]

    perm = rng.permutation(n)
    split_q = int(n * 0.5)
    tr_q, te_q = perm[:split_q], perm[split_q:]

    r2_action = fit_regression(a_action[tr_q], b_action_matched[tr_q], a_action[te_q], b_action_matched[te_q])
    r2_state = fit_regression(a_state[tr_q], b_action_matched[tr_q], a_state[te_q], b_action_matched[te_q])
    combined = np.concatenate([a_action, a_state], axis=1)
    r2_combined = fit_regression(
        combined[tr_q], b_action_matched[tr_q], combined[te_q], b_action_matched[te_q]
    )

    return {
        "task_subspace_terms": subspace_terms,
        "task_subspace_dim": int(len(cols)),
        "q1_nn_median_cross": q1_cross,
        "q1_nn_median_self": q1_self,
        "q1_cross_self_ratio": q1_cross / q1_self if q1_self > 0 else float("nan"),
        "q1_unrelated_scale_sqrt_d": float(np.sqrt(len(cols))),
        "q2_embodiment_sep_task_subspace_mlp_acc": mlp_acc,
        "q2_embodiment_sep_task_subspace_mlp_auc": mlp_auc,
        "q2_embodiment_sep_hand_picked_mlp_acc": hp_mlp_acc,
        "q2_embodiment_sep_full_obs_mlp_acc": full_mlp_acc,
        "q3_r2_action_to_matched_action": r2_action,
        "q3_r2_task_state_to_matched_action": r2_state,
        "q3_r2_action_plus_state_to_matched_action": r2_combined,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos-dir", type=Path, default=Path("data/demos"))
    ap.add_argument("--pattern", default="{task}_expert.zarr")
    ap.add_argument("--size", default="400k", help="Informational tag; bake the actual scale into --pattern")
    ap.add_argument("--schemas", type=Path, default=Path("outputs/analysis/schemas.json"))
    ap.add_argument("--separability", type=Path, default=Path("outputs/analysis/state_separability.json"))
    ap.add_argument("--threshold", type=float, default=SEPARABILITY_THRESHOLD)
    ap.add_argument("--max-episodes", type=int, default=120)
    ap.add_argument("--n-match", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if not args.schemas.exists():
        raise SystemExit(f"Missing {args.schemas}; run slurm_jobs/dump_schemas.sh first")
    if not args.separability.exists():
        raise SystemExit(f"Missing {args.separability}; run scripts/measure_state_separability.py first")
    schemas = json.loads(args.schemas.read_text())
    separability = json.loads(args.separability.read_text())

    results: dict[str, dict] = {}
    for family, (task_a, task_b) in FAMILIES.items():
        path_a = args.demos_dir / args.pattern.format(task=task_a)
        path_b = args.demos_dir / args.pattern.format(task=task_b)
        if not path_a.exists() or not path_b.exists():
            print(f"[WARN] Missing dataset(s) for {family}, skipping")
            continue
        print(f"[INFO] Analyzing {family}")
        results[family] = analyze_family(
            family,
            task_a,
            task_b,
            args.demos_dir,
            args.pattern,
            schemas,
            separability,
            args.threshold,
            args.max_episodes,
            args.n_match,
            args.seed,
        )

    if not results:
        raise SystemExit("No matched-pair datasets found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "state_equivalence.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
