"""Per-step diagnostic metrics for grasp tasks."""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_hand.grasp.mdp.table_utils import get_table_reference_pos_w


def object_height_beyond_table(
    env,
    *,
    object_name: str,
    table_name: str,
) -> torch.Tensor:
    """Object root height relative to the table reference height."""
    obj = env.scene[object_name]
    table = env.scene[table_name]
    table_pos = get_table_reference_pos_w(table)
    return obj.data.root_link_pos_w[:, 2] - table_pos[:, 2]


def hand_object_distance(
    env,
    *,
    object_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Distance from the configured hand body to the object root."""
    hand = env.scene[asset_cfg.name]
    obj = env.scene[object_name]
    body_ids = asset_cfg.body_ids
    body_idx = 0 if isinstance(body_ids, slice) else int(body_ids[0])
    hand_pos = hand.data.body_link_pos_w[:, body_idx]
    return torch.norm(obj.data.root_link_pos_w - hand_pos, dim=-1)


__all__ = ["hand_object_distance", "object_height_beyond_table"]
