"""Utilities for robust table pose queries in grasping tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def get_table_reference_pos_w(table: Entity) -> torch.Tensor:
    """Return a stable world-frame table reference position."""
    # Prefer named table surface geom when available.
    if (
        hasattr(table.data, "geom_pos_w")
        and table.data.geom_pos_w.shape[1] > 0
        and hasattr(table, "geom_names")
        and len(table.geom_names) > 0
    ):
        geom_idx = (
            table.geom_names.index("table_surface") if "table_surface" in table.geom_names else 0
        )
        return table.data.geom_pos_w[:, geom_idx, :]

    # For auto-wrapped fixed-base entities, body[0] is mocap anchor.
    if (
        getattr(table, "is_fixed_base", False)
        and getattr(table, "is_mocap", False)
        and hasattr(table.data, "body_link_pos_w")
        and table.data.body_link_pos_w.shape[1] > 1
    ):
        return table.data.body_link_pos_w[:, 1, :]

    # Generic body fallback.
    if hasattr(table.data, "body_link_pos_w") and table.data.body_link_pos_w.shape[1] > 0:
        return table.data.body_link_pos_w[:, 0, :]

    # Last-resort fallback.
    return table.data.root_link_pos_w


def get_table_reference_height(env: ManagerBasedRlEnv, table_name: str) -> torch.Tensor:
    """Return table reference height (z) for each environment."""
    table: Entity = env.scene[table_name]
    return get_table_reference_pos_w(table)[:, 2]
