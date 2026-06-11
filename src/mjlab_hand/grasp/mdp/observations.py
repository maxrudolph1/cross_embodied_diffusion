from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_inv, quat_mul

from mjlab_hand.grasp.mdp.table_utils import get_table_reference_pos_w

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "hand_position",
    "hand_orientation",
    "hand_linear_velocity",
    "hand_angular_velocity",
    "object_position",
    "object_orientation",
    "object_position_in_base",
    "object_orientation_in_base",
    "object_linear_velocity",
    "object_angular_velocity",
    "keypoint_positions_in_base",
    "keypoint_orientations_in_base",
    "keypoint_linear_velocities",
    "keypoint_angular_velocities",
    "hand_to_object_distance",
    "min_keypoint_to_object_distance",
    "object_height_above_table",
    "object_goal_position",
    "object_goal_distance",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("hand")


def _get_single_body_id(asset_cfg: SceneEntityCfg) -> int:
    if not isinstance(asset_cfg.body_ids, list) or len(asset_cfg.body_ids) != 1:
        raise ValueError(
            f"Expected exactly one body in asset_cfg.body_ids, got {asset_cfg.body_ids}"
        )
    return asset_cfg.body_ids[0]


def _to_env_origin_frame(env: "ManagerBasedRlEnv", pos_w: torch.Tensor) -> torch.Tensor:
    """Convert world-frame positions to per-environment origin frame.

    In parallel simulation each environment is offset by ``env.scene.env_origins``.
    Subtracting this offset makes absolute positions consistent across envs.
    """
    return pos_w - env.scene.env_origins


def hand_position(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Hand root position relative to the environment origin."""
    hand: Entity = env.scene[asset_cfg.name]
    base_idx = _get_single_body_id(asset_cfg)
    return _to_env_origin_frame(env, hand.data.body_link_pos_w[:, base_idx])


def hand_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Hand root orientation in world frame as a quaternion."""
    hand: Entity = env.scene[asset_cfg.name]
    base_idx = _get_single_body_id(asset_cfg)
    return hand.data.body_link_quat_w[:, base_idx]


def hand_linear_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    vel_scale: float = 1.0,
) -> torch.Tensor:
    """Scaled hand root linear velocity in world frame."""
    hand: Entity = env.scene[asset_cfg.name]
    base_idx = _get_single_body_id(asset_cfg)
    return hand.data.body_link_vel_w[:, base_idx, :3] * vel_scale


def hand_angular_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    vel_scale: float = 1.0,
) -> torch.Tensor:
    """Scaled hand root angular velocity in world frame."""
    hand: Entity = env.scene[asset_cfg.name]
    base_idx = _get_single_body_id(asset_cfg)
    return hand.data.body_link_vel_w[:, base_idx, 3:] * vel_scale


def object_position(
    env: ManagerBasedRlEnv,
    object_name: str,
) -> torch.Tensor:
    """Object position relative to the environment origin."""
    obj: Entity = env.scene[object_name]
    return _to_env_origin_frame(env, obj.data.root_link_pos_w)


def object_orientation(
    env: ManagerBasedRlEnv,
    object_name: str,
) -> torch.Tensor:
    """Object orientation in world frame as a quaternion."""
    obj: Entity = env.scene[object_name]
    return obj.data.root_link_quat_w


def object_position_in_base(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Object position in the hand base frame."""
    hand: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]

    obj_pos_w = obj.data.root_link_pos_w

    base_idx = _get_single_body_id(asset_cfg)
    base_pos_w = hand.data.body_link_pos_w[:, base_idx]
    base_quat_w = hand.data.body_link_quat_w[:, base_idx]

    rel_pos_w = obj_pos_w - base_pos_w

    return quat_apply(quat_inv(base_quat_w), rel_pos_w)


def object_orientation_in_base(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Object orientation relative to the hand base frame."""
    hand: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]

    obj_quat_w = obj.data.root_link_quat_w

    base_idx = _get_single_body_id(asset_cfg)
    base_quat_w = hand.data.body_link_quat_w[:, base_idx]

    return quat_mul(quat_conjugate(base_quat_w), obj_quat_w)


def object_linear_velocity(
    env: ManagerBasedRlEnv,
    object_name: str,
    vel_scale: float = 1.0,
) -> torch.Tensor:
    """Scaled object linear velocity in world frame."""
    obj: Entity = env.scene[object_name]
    return obj.data.root_link_vel_w[:, :3] * vel_scale


def object_angular_velocity(
    env: ManagerBasedRlEnv,
    object_name: str,
    vel_scale: float = 1.0,
) -> torch.Tensor:
    """Scaled object angular velocity in world frame."""
    obj: Entity = env.scene[object_name]
    return obj.data.root_link_vel_w[:, 3:] * vel_scale


def keypoint_positions_in_base(
    env: ManagerBasedRlEnv,
    base_cfg: SceneEntityCfg,
    keypoints_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Flattened keypoint positions in the hand base frame."""
    hand: Entity = env.scene[base_cfg.name]
    base_idx = _get_single_body_id(base_cfg)
    base_pos_w = hand.data.body_link_pos_w[:, base_idx]
    base_quat_w = hand.data.body_link_quat_w[:, base_idx]

    kp_ids = keypoints_cfg.body_ids
    if not isinstance(kp_ids, list):
        raise ValueError(
            f"keypoint_positions_in_base expects keypoints_cfg.body_ids to be a list, got {kp_ids}"
        )

    kps_pos_w = hand.data.body_link_pos_w[:, kp_ids]  # (N, K, 3)
    rel_pos_w = kps_pos_w - base_pos_w.unsqueeze(1)
    rel_pos_base = quat_apply(
        quat_inv(base_quat_w).unsqueeze(1).expand(-1, len(kp_ids), -1),
        rel_pos_w,
    )
    return rel_pos_base.reshape(rel_pos_base.shape[0], -1)


def keypoint_orientations_in_base(
    env: ManagerBasedRlEnv,
    base_cfg: SceneEntityCfg,
    keypoints_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Flattened keypoint orientations relative to the hand base frame."""
    hand: Entity = env.scene[base_cfg.name]
    base_idx = _get_single_body_id(base_cfg)
    base_quat_w = hand.data.body_link_quat_w[:, base_idx]

    kp_ids = keypoints_cfg.body_ids
    if not isinstance(kp_ids, list):
        raise ValueError(
            f"keypoint_orientations_in_base expects keypoints_cfg.body_ids to be a list, "
            f"got {kp_ids}"
        )

    kps_quat_w = hand.data.body_link_quat_w[:, kp_ids]  # (N, K, 4)
    base_quat_expanded = base_quat_w.unsqueeze(1).expand(-1, len(kp_ids), -1)
    rel_quat = quat_mul(quat_conjugate(base_quat_expanded), kps_quat_w)
    return rel_quat.reshape(rel_quat.shape[0], -1)


def keypoint_linear_velocities(
    env: ManagerBasedRlEnv,
    keypoints_cfg: SceneEntityCfg,
    vel_scale: float = 1.0,
) -> torch.Tensor:
    """Flattened scaled keypoint linear velocities in world frame."""
    hand: Entity = env.scene[keypoints_cfg.name]

    kp_ids = keypoints_cfg.body_ids
    if not isinstance(kp_ids, list):
        raise ValueError(
            f"keypoint_linear_velocities expects keypoints_cfg.body_ids to be a list, got {kp_ids}"
        )

    kps_vel_w = hand.data.body_link_vel_w[:, kp_ids, :3]  # (N, K, 3)
    return (kps_vel_w * vel_scale).reshape(kps_vel_w.shape[0], -1)


def keypoint_angular_velocities(
    env: ManagerBasedRlEnv,
    keypoints_cfg: SceneEntityCfg,
    vel_scale: float = 1.0,
) -> torch.Tensor:
    """Flattened scaled keypoint angular velocities in world frame."""
    hand: Entity = env.scene[keypoints_cfg.name]

    kp_ids = keypoints_cfg.body_ids
    if not isinstance(kp_ids, list):
        raise ValueError(
            f"keypoint_angular_velocities expects keypoints_cfg.body_ids to be a list, got {kp_ids}"
        )

    kps_vel_w = hand.data.body_link_vel_w[:, kp_ids, 3:]  # (N, K, 3)
    return (kps_vel_w * vel_scale).reshape(kps_vel_w.shape[0], -1)


def hand_to_object_distance(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Distance from the palm body, rather than the mocap root, to the object."""
    hand: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]

    palm_idx = _get_single_body_id(asset_cfg)
    palm_pos = hand.data.body_link_pos_w[:, palm_idx]
    obj_pos = obj.data.root_link_pos_w

    return torch.norm(obj_pos - palm_pos, dim=-1, keepdim=True)


def min_keypoint_to_object_distance(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg,
    keypoints_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Minimum object distance over the configured contact keypoints."""
    hand: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]

    obj_pos = obj.data.root_link_pos_w  # (num_envs, 3)

    kp_ids = keypoints_cfg.body_ids
    if not isinstance(kp_ids, list):
        raise ValueError(
            f"min_keypoint_to_object_distance expects keypoints_cfg.body_ids to be a list, "
            f"got {kp_ids}"
        )

    kps_pos_w = hand.data.body_link_pos_w[:, kp_ids]  # (N, K, 3)
    dists = torch.norm(obj_pos.unsqueeze(1) - kps_pos_w, dim=-1)  # (N, K)
    min_distances, _ = dists.min(dim=-1)

    return min_distances.unsqueeze(-1)  # (num_envs, 1)


def object_height_above_table(
    env: ManagerBasedRlEnv,
    object_name: str,
    table_name: str,
) -> torch.Tensor:
    """Object height above the table reference surface."""
    obj: Entity = env.scene[object_name]
    table: Entity = env.scene[table_name]

    obj_pos = obj.data.root_link_pos_w
    table_pos = get_table_reference_pos_w(table)

    height = obj_pos[:, 2:3] - table_pos[:, 2:3]
    return height


def object_goal_position(
    env: ManagerBasedRlEnv,
    command_name: str = "pose",
) -> torch.Tensor:
    """Goal position shifted to the same origin frame as ``object_position``."""
    return _to_env_origin_frame(env, env.command_manager.get_command(command_name))


def object_goal_distance(
    env: ManagerBasedRlEnv,
    object_name: str,
    command_name: str = "pose",
) -> torch.Tensor:
    """Distance from the object to its sampled position goal."""
    obj: Entity = env.scene[object_name]
    obj_pos = obj.data.root_link_pos_w
    goal_pos = env.command_manager.get_command(command_name)

    return torch.norm(obj_pos - goal_pos, dim=-1, keepdim=True)
