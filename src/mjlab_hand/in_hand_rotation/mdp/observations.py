from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_inv, quat_mul

from .commands import RotationCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "object_position_in_base",
    "object_orientation_in_base",
    "object_linear_velocity",
    "object_angular_velocity",
    "goal_orientation",
    "goal_relative_rotation",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("hand")


def _hand_base_pos_quat(hand: Entity, asset_cfg: SceneEntityCfg):
    """Return world-frame (pos_w, quat_w) for the hand base via ``asset_cfg.body_ids[0]``.

    Intended only for relative computations (e.g. obj_pos_w - base_pos_w) where the
    per-env origin offset cancels. Do not expose pos_w directly as a policy observation.

    Raises if ``body_names`` was not configured (i.e. ``body_ids`` is a slice or empty).
    """
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice) or body_ids is None or len(body_ids) == 0:
        raise ValueError(
            f"asset_cfg must have body_names set for hand base lookup, got body_ids={body_ids}"
        )
    body_idx = body_ids[0]
    return hand.data.body_link_pos_w[:, body_idx], hand.data.body_link_quat_w[:, body_idx]


def object_position_in_base(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    hand: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    obj_pos_w = obj.data.root_link_pos_w
    base_pos_w, base_quat_w = _hand_base_pos_quat(hand, asset_cfg)
    rel_pos_w = obj_pos_w - base_pos_w
    return quat_apply(quat_inv(base_quat_w), rel_pos_w)


def object_orientation_in_base(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    hand: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    obj_quat_w = obj.data.root_link_quat_w
    _, base_quat_w = _hand_base_pos_quat(hand, asset_cfg)
    return quat_mul(quat_conjugate(base_quat_w), obj_quat_w)


def object_linear_velocity(
    env: ManagerBasedRlEnv,
    object_name: str,
    vel_scale: float = 0.2,
) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    return obj.data.root_link_vel_w[:, :3] * vel_scale


def object_angular_velocity(
    env: ManagerBasedRlEnv,
    object_name: str,
    vel_scale: float = 0.2,
) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    return obj.data.root_link_vel_w[:, 3:] * vel_scale


def goal_orientation(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = cast(RotationCommand, env.command_manager.get_term(command_name))
    return command.target_quat


def goal_relative_rotation(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
) -> torch.Tensor:
    obj: Entity = env.scene[object_name]
    command = cast(RotationCommand, env.command_manager.get_term(command_name))
    return quat_mul(obj.data.root_link_quat_w, quat_conjugate(command.target_quat))
