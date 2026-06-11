from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_hand.grasp.mdp.table_utils import (
    get_table_reference_height,
    get_table_reference_pos_w,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "reach_reward",
    "target_tracking_reward",
    "action_l2_penalty",
    "object_distance_to_goal",
    "grasp_stability_reward",
    "EpisodicApproachReward",
    "LiftReward",
    "GoalProgressReward",
    "smoothness_reward",
]


def reach_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg,
    keypoints_cfg: SceneEntityCfg,
    reach_reward_scale: float = 1.0,
) -> torch.Tensor:
    hand: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]

    obj_pos = obj.data.root_link_pos_w  # (num_envs, 3)

    kp_ids = keypoints_cfg.body_ids
    if not isinstance(kp_ids, list):
        raise ValueError(f"reach_reward expects keypoints_cfg.body_ids to be a list, got {kp_ids}")
    kps_pos_w = hand.data.body_link_pos_w[:, kp_ids]  # (N, K, 3)
    dists = torch.norm(obj_pos.unsqueeze(1) - kps_pos_w, dim=-1)  # (N, K)
    min_distances, _ = dists.min(dim=-1)

    reward = reach_reward_scale / (min_distances + 0.05)

    return reward


def target_tracking_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    command_name: str = "pose",
    sigma: float = 0.1,
    reward_scale: float = 1.0,
    force_sensor_name: str | None = None,
    force_threshold: float | None = None,
) -> torch.Tensor:
    """Exponential goal tracking reward, optionally gated by contact force."""
    obj: Entity = env.scene[object_name]

    obj_pos = obj.data.root_link_pos_w

    goal_pos = env.command_manager.get_command(command_name)

    distance = torch.norm(obj_pos - goal_pos, dim=-1)

    reward = torch.exp(-distance / sigma) * reward_scale

    if force_sensor_name is not None and force_threshold is not None:
        fs = env.scene[force_sensor_name]
        if fs.data.force is not None:
            force = torch.nan_to_num(fs.data.force, nan=0.0, posinf=0.0, neginf=0.0)
            force_norm = torch.norm(force, dim=-1)
            if fs.data.found is not None:
                has_contact = fs.data.found > 0
                force_norm = torch.where(has_contact, force_norm, torch.zeros_like(force_norm))
            max_f = force_norm.max(dim=-1).values
            reward = reward * (max_f < force_threshold).float()

    return reward


def success_bonus(
    env: ManagerBasedRlEnv,
    object_name: str,
    table_name: str,
    success_height: float = 0.15,
    bonus: float = 100.0,
) -> torch.Tensor:
    """Bonus when the object clears ``success_height`` above the table."""
    obj: Entity = env.scene[object_name]
    table: Entity = env.scene[table_name]

    obj_pos = obj.data.root_link_pos_w
    table_pos = get_table_reference_pos_w(table)

    height_above_table = obj_pos[:, 2] - table_pos[:, 2]

    success = height_above_table > success_height

    return success.float() * bonus


def goal_proximity_bonus(
    env: ManagerBasedRlEnv,
    object_name: str,
    command_name: str = "pose",
    distance_threshold: float = 0.05,
    bonus: float = 1.0,
) -> torch.Tensor:
    """Bonus when the object is within ``distance_threshold`` of the goal."""
    obj: Entity = env.scene[object_name]
    goal_pos = env.command_manager.get_command(command_name)
    dist = torch.norm(obj.data.root_link_pos_w - goal_pos, dim=-1)
    return (dist < distance_threshold).float() * bonus


def get_table_surface_height(env: ManagerBasedRlEnv, table_name: str) -> torch.Tensor:
    """Table surface height, including static-table pose fallbacks."""
    return get_table_reference_height(env, table_name)


def action_l2_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.sum(env.action_manager.action**2, dim=-1)


def object_distance_to_goal(
    env: ManagerBasedRlEnv,
    object_name: str,
    asset_cfg: SceneEntityCfg,
    goal_pos_offset_from_palm: tuple[float, float, float] = (0.0, 0.0, 0.2),
) -> torch.Tensor:
    """Negative object distance to a goal offset from the palm body."""
    hand: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]

    if not isinstance(asset_cfg.body_ids, list) or len(asset_cfg.body_ids) != 1:
        raise ValueError(
            f"object_distance_to_goal expects asset_cfg with exactly one body name, "
            f"got body_ids={asset_cfg.body_ids}"
        )
    palm_idx = asset_cfg.body_ids[0]
    palm_pos = hand.data.body_link_pos_w[:, palm_idx]

    goal_offset = torch.tensor(
        goal_pos_offset_from_palm,
        device=env.device,
        dtype=torch.float32,
    )
    goal_pos = palm_pos + goal_offset

    obj_pos = obj.data.root_link_pos_w
    distance = torch.norm(obj_pos - goal_pos, dim=-1)

    return -distance


def grasp_stability_reward(
    env: ManagerBasedRlEnv,
    object_name: str,
    stability_reward_scale: float = 0.1,
) -> torch.Tensor:
    """Penalty on object velocity for grasp stability."""
    obj: Entity = env.scene[object_name]

    obj_vel = obj.data.root_link_vel_w

    vel_mag = torch.norm(obj_vel, dim=-1)

    reward = -vel_mag * stability_reward_scale

    return reward


class EpisodicApproachReward:
    """Per-finger closest-distance progress reward (pre-lift phase).

    Tracks the closest each keypoint has gotten to the object and rewards
    improvements. Only active before the object is lifted.

    .. note::
        Lazy initialization assumes reset events fire before the first post-reset
        reward computation.
    """

    def __init__(self, cfg, env) -> None:
        self._env = env
        self._object_name: str = cfg.params["object_name"]
        self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("hand"))
        self._keypoints_cfg: SceneEntityCfg = cfg.params["keypoints_cfg"]
        self._lift_threshold: float = cfg.params.get("lift_threshold", 0.05)

        robot: Entity = env.scene[self._asset_cfg.name]
        kp_ids = self._keypoints_cfg.body_ids
        self._keypoint_body_indices: list[int] = (
            list(range(robot.num_bodies)) if isinstance(kp_ids, slice) else [int(i) for i in kp_ids]
        )
        num_keypoints = len(self._keypoint_body_indices)

        self._closest_keypoint_dist = torch.full(
            (env.num_envs, num_keypoints),
            float("inf"),
            device=env.device,
        )
        self._object_init_z = torch.zeros(env.num_envs, device=env.device)
        self._lifted_object = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env, **params) -> torch.Tensor:
        robot: Entity = env.scene[self._asset_cfg.name]
        obj: Entity = env.scene[self._object_name]

        keypoint_pos = robot.data.body_link_pos_w[:, self._keypoint_body_indices, :]
        obj_pos = obj.data.root_link_pos_w.unsqueeze(1)
        dists = torch.norm(keypoint_pos - obj_pos, dim=-1)

        obj_z = obj.data.root_link_pos_w[:, 2]
        not_init = ~self._initialized

        if not_init.any():
            self._closest_keypoint_dist[not_init] = dists[not_init]
            self._object_init_z[not_init] = obj_z[not_init]
            self._initialized[not_init] = True

        z_lift = obj_z - self._object_init_z
        lifted = z_lift > self._lift_threshold
        self._lifted_object = lifted | self._lifted_object

        deltas = torch.clamp(self._closest_keypoint_dist - dists, 0.0, 10.0)
        reward = deltas.sum(dim=-1) * (~self._lifted_object)

        self._closest_keypoint_dist = torch.minimum(self._closest_keypoint_dist, dists)
        return reward

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._initialized.fill_(False)
            self._lifted_object.fill_(False)
            self._closest_keypoint_dist.fill_(float("inf"))
        else:
            self._initialized[env_ids] = False
            self._lifted_object[env_ids] = False
            self._closest_keypoint_dist[env_ids] = float("inf")


class LiftReward:
    """Height-based lift reward with one-time bonus (pre-lift phase)."""

    def __init__(self, cfg, env) -> None:
        self._env = env
        self._object_name: str = cfg.params["object_name"]
        self._lift_threshold: float = cfg.params.get("lift_threshold", 0.05)
        self._lift_bonus: float = cfg.params.get("lift_bonus", 10.0)
        self._lift_scale: float = cfg.params.get("lift_scale", 1.0)
        self._object_init_z = torch.zeros(env.num_envs, device=env.device)
        self._lifted_object = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env, **params) -> torch.Tensor:
        obj: Entity = env.scene[self._object_name]
        obj_z = obj.data.root_link_pos_w[:, 2]

        not_init = ~self._initialized
        if not_init.any():
            self._object_init_z[not_init] = obj_z[not_init]
            self._initialized[not_init] = True

        z_lift = obj_z - self._object_init_z
        lifted = z_lift > self._lift_threshold
        just_lifted = lifted & ~self._lifted_object
        self._lifted_object = lifted | self._lifted_object

        lifting_rew = torch.clamp(z_lift, 0.0, 0.05) * (~self._lifted_object) * self._lift_scale
        bonus_rew = self._lift_bonus * just_lifted.float()
        return lifting_rew + bonus_rew

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._initialized.fill_(False)
            self._lifted_object.fill_(False)
        else:
            self._initialized[env_ids] = False
            self._lifted_object[env_ids] = False


class GoalProgressReward:
    """Episodic progress reward toward the goal position.

    Tracks the best distance achieved to the current goal and rewards
    improvements. A bonus is awarded while the object remains within
    ``success_tolerance`` of the goal.

    Resets ``best_dist`` when the goal is resampled (detected by target pos
    change) so that progress is measured relative to the new goal.
    """

    def __init__(self, cfg, env) -> None:
        self._env = env
        self._object_name: str = cfg.params["object_name"]
        self._command_name: str = cfg.params.get("command_name", "pose")
        self._success_tolerance: float = cfg.params.get("success_tolerance", 0.03)
        self._success_bonus: float = cfg.params.get("success_bonus", 10.0)

        self._best_dist = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._last_target_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._goal_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env, **params) -> torch.Tensor:
        obj: Entity = env.scene[self._object_name]
        command = env.command_manager.get_term(self._command_name)

        obj_pos = obj.data.root_link_pos_w
        target_pos = command.goal_pos
        current_dist = torch.norm(obj_pos - target_pos, dim=-1)

        not_init = ~self._initialized
        goal_changed = self._initialized & (self._last_target_pos != target_pos).any(dim=-1)
        needs_reset = not_init | goal_changed

        if needs_reset.any():
            self._best_dist[needs_reset] = current_dist[needs_reset]
            self._last_target_pos[needs_reset] = target_pos[needs_reset].clone()
            self._initialized[needs_reset] = True
            self._goal_reached[needs_reset] = False

        progress_rew = torch.clamp(self._best_dist - current_dist, 0.0, float("inf")) * 500

        near_goal = current_dist < self._success_tolerance
        success_bonus = self._success_bonus * near_goal.float()

        self._best_dist = torch.minimum(self._best_dist, current_dist)
        return progress_rew + success_bonus

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._initialized.fill_(False)
            self._best_dist.fill_(float("inf"))
            self._last_target_pos.fill_(0.0)
            self._goal_reached.fill_(False)
        else:
            self._initialized[env_ids] = False
            self._best_dist[env_ids] = float("inf")
            self._last_target_pos[env_ids] = 0.0
            self._goal_reached[env_ids] = False


def smoothness_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    joint_names: tuple[str, ...],
    scale: float = 0.1,
) -> torch.Tensor:
    """L1 joint-velocity penalty."""
    robot: Entity = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(joint_names)
    vel = robot.data.joint_vel[:, joint_ids]
    return -scale * vel.abs().sum(dim=-1)
