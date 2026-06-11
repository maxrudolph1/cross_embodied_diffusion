from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import RotationCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "position_distance_reward",
    "rotation_distance_reward",
    "rotation_alignment_reward",
    "action_l2_penalty",
    "reach_goal_bonus",
    "fall_penalty",
    "time_penalty",
]

_GOAL_POS_OFFSET_FROM_HAND_BASE = (0.0, 0.0, 0.1)


def position_distance_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("hand"),
    goal_pos_offset_from_hand_base: tuple[float, float, float] = (_GOAL_POS_OFFSET_FROM_HAND_BASE),
) -> torch.Tensor:
    """Penalize object distance to goal position above hand base.

    Goal position is computed as:
      hand_base_position + goal_pos_offset_from_hand_base
    """
    hand: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice) or body_ids is None or len(body_ids) == 0:
        raise ValueError(
            f"asset_cfg must have body_names set for hand base lookup, got body_ids={body_ids}"
        )
    hand_base_pos = hand.data.body_link_pos_w[:, body_ids[0]]

    goal_pos = hand_base_pos + torch.tensor(
        goal_pos_offset_from_hand_base,
        device=obj.data.root_link_pos_w.device,
        dtype=obj.data.root_link_pos_w.dtype,
    )
    return torch.norm(obj.data.root_link_pos_w - goal_pos, dim=-1)


def rotation_distance_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
) -> torch.Tensor:
    """Backward-compatible alias for position distance reward."""
    del command_name
    return position_distance_reward(env=env, object_name=object_name)


def rotation_alignment_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    rot_eps: float = 0.1,
) -> torch.Tensor:
    """Reward orientation alignment: 1 / (|rot_dist| + eps)."""
    obj: Entity = env.scene[object_name]
    command = cast(RotationCommand, env.command_manager.get_term(command_name))
    rot_dist = quat_error_magnitude(obj.data.root_link_quat_w, command.target_quat)
    return 1.0 / (rot_dist.abs() + rot_eps)


def action_l2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """L2 penalty on actions: sum(actions^2).

    Matches TOUCHMANIP action_penalty.
    """
    return torch.sum(env.action_manager.action**2, dim=-1)


def reach_goal_bonus(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    success_tolerance: float = 0.1,
    bonus: float = 250.0,
) -> torch.Tensor:
    """Flat bonus when orientation error < threshold."""
    obj: Entity = env.scene[object_name]
    command = cast(RotationCommand, env.command_manager.get_term(command_name))
    rot_dist = quat_error_magnitude(obj.data.root_link_quat_w, command.target_quat)
    return torch.where(rot_dist.abs() <= success_tolerance, bonus, 0.0)


def fall_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    fall_dist: float = 0.24,
    penalty: float = 0.0,
) -> torch.Tensor:
    """Penalty when object drifts too far from target position."""
    obj: Entity = env.scene[object_name]
    command = cast(RotationCommand, env.command_manager.get_term(command_name))
    goal_dist = torch.norm(obj.data.root_link_pos_w - command.target_pos, dim=-1)
    return torch.where(goal_dist >= fall_dist, penalty, 0.0)


def time_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant penalty per step to discourage stalling."""
    return torch.ones(env.num_envs, device=env.device)
