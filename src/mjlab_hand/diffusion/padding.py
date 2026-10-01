"""Term-aligned padding for multi-hand (cross-embodiment) datasets and policies.

Every observation term (joint_pos, object_pos, keypoint_pos_rel, ...) gets
its own block in the padded vector, as wide as the widest hand's version of
that term in the family, and each hand's values are written at the start of
their block. So a padded column means the same observation term for every
hand (end-of-vector padding put e.g. Allegro's object_pos at columns 44-46 and
Shadow's at 60-62). Actions are front-packed: hand h's action is columns
[0, action_dim_h) and the rest is padding. Padding is 0 in raw units; data is
stored raw and normalized by the policy (`--norm-mode`).

Joint correspondence *within* a term is not recovered: joint_pos column 0 is
whatever joint each hand lists first (Allegro ring-finger spread, LEAP index
MCP; outputs/analysis/spaces_reference.md).

Schemas (term names and widths per task) come from, in order: $MJHAND_SCHEMAS,
<repo>/outputs/analysis/schemas.json, <repo>/configs/schemas.json (tracked).
Regenerate with slurm_jobs/dump_schemas.sh or scripts/dump_spaces.py.

Re-implementation of Bundle's padding.py from MIGRATION.md (CHANGES.md item 63).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

HANDS = ["Allegro", "LEAP", "Shadow", "Sharpa", "Wuji"]
FAMILIES = ["Grasp", "InHand-Rotation"]

_REPO = Path(__file__).resolve().parents[3]


def schemas_path() -> Path:
    env = os.environ.get("MJHAND_SCHEMAS")
    if env:
        return Path(env)
    for p in (_REPO / "outputs/analysis/schemas.json", _REPO / "configs/schemas.json"):
        if p.exists():
            return p
    raise FileNotFoundError("no schemas.json: set MJHAND_SCHEMAS or run slurm_jobs/dump_schemas.sh")


@lru_cache(maxsize=None)
def load_schemas(path: str | None = None) -> dict:
    return json.loads(Path(path or schemas_path()).read_text())


def family_of(task: str) -> str:
    for fam in FAMILIES:
        if task.startswith(fam + "-"):
            return fam
    raise ValueError(f"task {task!r} is not in a known family {FAMILIES}")


def family_tasks(family: str) -> list[str]:
    return [f"{family}-{h}" for h in HANDS]


@dataclass(frozen=True)
class PadPlan:
    family: str
    tasks: tuple[str, ...]
    terms: tuple[str, ...]
    term_widths: tuple[int, ...]  # padded width per term
    term_offsets: tuple[int, ...]  # start column per term
    obs_dim: int  # padded obs width
    action_dim: int  # padded action width
    native: dict  # task -> {"term_dims": [...], "obs_dim": int, "action_dim": int}

    def obs_index(self, task: str) -> np.ndarray:
        """Padded column for each native obs dim of `task` (scatter index)."""
        nat = self.native[task]["term_dims"]
        idx = [off + i for off, d in zip(self.term_offsets, nat, strict=True) for i in range(d)]
        return np.asarray(idx, dtype=np.int64)

    def obs_valid(self, task: str) -> np.ndarray:
        valid = np.zeros(self.obs_dim, dtype=bool)
        valid[self.obs_index(task)] = True
        return valid

    def action_valid(self, task: str) -> np.ndarray:
        valid = np.zeros(self.action_dim, dtype=bool)
        valid[: self.native[task]["action_dim"]] = True
        return valid

    def pad_obs(self, task: str, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        nat = self.native[task]["obs_dim"]
        if obs.shape[-1] != nat:
            raise ValueError(f"{task}: obs width {obs.shape[-1]} != schema width {nat}")
        out = np.zeros((*obs.shape[:-1], self.obs_dim), dtype=obs.dtype)
        out[..., self.obs_index(task)] = obs
        return out

    def pad_action(self, task: str, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action)
        nat = self.native[task]["action_dim"]
        if action.shape[-1] != nat:
            raise ValueError(f"{task}: action width {action.shape[-1]} != schema width {nat}")
        out = np.zeros((*action.shape[:-1], self.action_dim), dtype=action.dtype)
        out[..., :nat] = action
        return out

    def to_json(self) -> dict:
        return {
            "family": self.family,
            "tasks": list(self.tasks),
            "terms": list(self.terms),
            "term_widths": list(self.term_widths),
            "term_offsets": list(self.term_offsets),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }


def build_plan(tasks: list[str] | str, schemas: dict | None = None) -> PadPlan:
    """Plan over `tasks` (or a family name -> its five hands). All tasks must
    share a family and the same ordered obs-term names."""
    if isinstance(tasks, str):
        tasks = family_tasks(tasks)
    schemas = schemas or load_schemas()
    fams = {family_of(t) for t in tasks}
    if len(fams) != 1:
        raise ValueError(f"tasks span several families: {sorted(fams)}")
    missing = [t for t in tasks if t not in schemas]
    if missing:
        raise KeyError(f"no schema for {missing}")
    terms = schemas[tasks[0]]["obs_terms"]
    for t in tasks:
        if schemas[t]["obs_terms"] != terms:
            raise ValueError(f"{t}: obs terms {schemas[t]['obs_terms']} != {terms}")
    widths = [max(schemas[t]["obs_term_dims"][i] for t in tasks) for i in range(len(terms))]
    offsets = list(np.cumsum([0] + widths[:-1]))
    native = {
        t: {
            "term_dims": list(schemas[t]["obs_term_dims"]),
            "obs_dim": int(schemas[t]["obs_dim"]),
            "action_dim": int(schemas[t]["action_dim"]),
        }
        for t in tasks
    }
    return PadPlan(
        family=fams.pop(),
        tasks=tuple(tasks),
        terms=tuple(terms),
        term_widths=tuple(int(w) for w in widths),
        term_offsets=tuple(int(o) for o in offsets),
        obs_dim=int(sum(widths)),
        action_dim=int(max(native[t]["action_dim"] for t in tasks)),
        native=native,
    )
