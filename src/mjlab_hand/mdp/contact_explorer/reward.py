"""ContactExplorer reward manager and environment wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from mjlab_hand.mdp.contact_explorer.normalization import finite_row_mask
from mjlab_hand.mdp.contact_explorer.object_cache import get_cached_contact_db_object
from mjlab_hand.mdp.contact_explorer.state_bank import LearnedHashStateBank, infer_simhash_dim
from mjlab_hand.object.object import ContactDBObject, _fps_points

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


PRE_CONTACT_SCALE = 0.5
POST_CONTACT_SCALE = 128.0


def _sqrt_novelty(counts: torch.Tensor, novelty_decay_rate: float) -> torch.Tensor:
    return 1.0 / torch.sqrt(1.0 + float(novelty_decay_rate) * counts)


class ContactExplorerRewardManager:
    """Potential-field + contact novelty with learned-hash discretized object pose."""

    def __init__(
        self,
        obj: ContactDBObject,
        *,
        device: torch.device,
        num_envs: int,
        num_keypoints: int,
        num_key_states: int = 32,
        state_num_points: int = 16,
        cluster_k: int | None = None,
        kernel_param: float = 0.03,
        novelty_decay_rate: float = 2.0,
        hash_code_dim: int = 256,
        hash_noise_scale: float = 0.3,
        hash_lambda_binary: float = 1.0,
        hash_ae_lr: float = 3e-4,
        hash_ae_steps: int = 5,
        hash_ae_num_minibatches: int = 8,
        hash_ae_update_freq: int = 16,
        hash_seed: int = 0,
        use_goal_state_feature: bool = False,
    ) -> None:
        self.obj = obj
        self.device = device
        self.M = int(obj.canonical_pointcloud.shape[0])
        self.L = int(num_keypoints)
        self._point_to_cluster = obj.point_to_region.to(device)
        max_region = int(obj.point_to_region.max().item())
        self.K = int(cluster_k) if cluster_k is not None else max_region + 1
        self.K = max(self.K, max_region + 1)
        self.num_key_states = int(num_key_states)
        self.state_num_points = int(state_num_points)
        self.kernel_param = float(kernel_param)
        self.novelty_decay_rate = float(novelty_decay_rate)
        self.use_goal_state_feature = bool(use_goal_state_feature)

        idx, _ = _fps_points(
            obj.canonical_pointcloud.to(device), k=self.state_num_points, start_idx=0
        )
        self._state_point_indices = idx.to(torch.long)

        feature_dim = self.state_num_points * 3
        if self.use_goal_state_feature:
            feature_dim = feature_dim * 2
        buffer_size = max(int(num_envs) * int(hash_ae_update_freq), hash_ae_update_freq)
        sim_dim = infer_simhash_dim(self.num_key_states)
        self.state_bank = LearnedHashStateBank(
            num_key_states=self.num_key_states,
            feature_dim=feature_dim,
            buffer_size=buffer_size,
            num_hand_keypoints=self.L,
            num_object_bins=self.K,
            device=device,
            code_dim=hash_code_dim,
            simhash_dim=sim_dim,
            noise_scale=hash_noise_scale,
            lambda_binary=hash_lambda_binary,
            ae_lr=hash_ae_lr,
            ae_update_steps=hash_ae_steps,
            ae_update_freq=hash_ae_update_freq,
            ae_num_minibatches=hash_ae_num_minibatches,
            seed=hash_seed,
        )

        self.potential_per_kp_max: torch.Tensor | None = None
        self.contact_bonus_per_kp_max: torch.Tensor | None = None
        self.last_pre_contact_reward = torch.zeros(
            int(num_envs), device=self.device, dtype=torch.float32
        )
        self.last_post_contact_reward = torch.zeros(
            int(num_envs), device=self.device, dtype=torch.float32
        )

    def ensure_running_max_buffers(self, num_envs: int) -> None:
        n = int(num_envs)
        num_states, num_keypoints = self.num_key_states, self.L
        if self.potential_per_kp_max is None or self.potential_per_kp_max.shape != (
            n,
            num_states,
            num_keypoints,
        ):
            self.potential_per_kp_max = torch.zeros(
                (n, num_states, num_keypoints), device=self.device, dtype=torch.float32
            )
        if self.contact_bonus_per_kp_max is None or self.contact_bonus_per_kp_max.shape != (
            n,
            num_states,
            num_keypoints,
        ):
            self.contact_bonus_per_kp_max = torch.zeros(
                (n, num_states, num_keypoints), device=self.device, dtype=torch.float32
            )

    @torch.no_grad()
    def reset_running_max_buffers(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or env_ids.numel() == 0:
            return
        self.ensure_running_max_buffers(int(self.potential_per_kp_max.shape[0]))
        self.potential_per_kp_max[env_ids] = 0.0
        self.contact_bonus_per_kp_max[env_ids] = 0.0

    def build_state_features_world(self, pc_world: torch.Tensor) -> torch.Tensor:
        """pc_world: (N, M, 3)"""
        idx = self._state_point_indices
        return pc_world.index_select(1, idx).reshape(pc_world.shape[0], -1)

    @torch.no_grad()
    def step_reward(
        self,
        object_positions: torch.Tensor,
        object_orientations: torch.Tensor,
        keypoint_positions_world: torch.Tensor,
        contact_found: torch.Tensor,
        contact_pos_world: torch.Tensor,
        goal_positions: torch.Tensor | None = None,
        goal_orientations: torch.Tensor | None = None,
        force_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns per-env reward (N,)."""
        device = self.device
        n, num_keypoints, _ = keypoint_positions_world.shape
        num_points = self.M
        reward_out = torch.zeros(n, device=device, dtype=torch.float32)
        self.last_pre_contact_reward.zero_()
        self.last_post_contact_reward.zero_()
        self.ensure_running_max_buffers(n)

        valid_env = (
            finite_row_mask(object_positions)
            & finite_row_mask(object_orientations)
            & finite_row_mask(keypoint_positions_world)
        )
        if force_gate is not None:
            valid_env = valid_env & force_gate
        if not valid_env.any():
            return reward_out

        env_idx = torch.nonzero(valid_env, as_tuple=False).squeeze(1)
        object_positions = object_positions[valid_env].to(device=device, dtype=torch.float32)
        object_orientations = object_orientations[valid_env].to(device=device, dtype=torch.float32)
        keypoint_positions_world = keypoint_positions_world[valid_env].to(
            device=device, dtype=torch.float32
        )
        contact_found = contact_found[valid_env]
        contact_pos_world = contact_pos_world[valid_env] if contact_pos_world is not None else None
        n_valid = int(env_idx.shape[0])

        q_expanded = object_orientations.unsqueeze(1).expand(-1, num_points, -1)
        pc_local = self.obj.canonical_pointcloud.to(device).unsqueeze(0).expand(n_valid, -1, -1)
        pc_world = quat_apply(q_expanded, pc_local) + object_positions.unsqueeze(1)

        state_feats = self.build_state_features_world(pc_world)

        if (
            self.use_goal_state_feature
            and goal_positions is not None
            and goal_orientations is not None
        ):
            goal_positions_valid = goal_positions[valid_env].to(device=device, dtype=torch.float32)
            goal_orientations_valid = goal_orientations[valid_env].to(
                device=device, dtype=torch.float32
            )
            q_goal_expanded = goal_orientations_valid.unsqueeze(1).expand(-1, num_points, -1)
            pc_world_goal = quat_apply(q_goal_expanded, pc_local) + goal_positions_valid.unsqueeze(
                1
            )
            goal_state_feats = self.build_state_features_world(pc_world_goal)
            state_feats = torch.cat([state_feats, goal_state_feats], dim=-1)

        self.state_bank.push(state_feats)
        state_ids = self.state_bank.assign(state_feats)

        contact_found = contact_found.to(device=device, dtype=torch.float32)
        contact_mask = torch.isfinite(contact_found) & (contact_found > 0.0)

        if contact_mask.any() and contact_pos_world is not None:
            cp = contact_pos_world.to(device=device, dtype=torch.float32)
            contact_pos_valid = torch.isfinite(cp).all(dim=-1)
            contact_mask = contact_mask & contact_pos_valid
            cp = torch.nan_to_num(cp, nan=0.0, posinf=0.0, neginf=0.0)
            region_flat = self.obj.nearest_region_ids(cp, object_positions, object_orientations)
            contact_bins = region_flat.view(n_valid, num_keypoints)
        else:
            contact_bins = torch.zeros((n_valid, num_keypoints), device=device, dtype=torch.long)

        if contact_mask.any():
            self.state_bank.add_contacts(
                state_ids=state_ids,
                contact_mask=contact_mask,
                contact_bins=contact_bins,
            )

        counts_state = self.state_bank.counts
        counts_slm = counts_state.gather(
            2,
            self._point_to_cluster.view(1, 1, num_points).expand(
                self.num_key_states, num_keypoints, num_points
            ),
        )
        q_slm = _sqrt_novelty(counts_slm, self.novelty_decay_rate)

        sid = state_ids.clamp(min=0)
        q_env = q_slm[sid]

        dists = torch.norm(keypoint_positions_world.unsqueeze(2) - pc_world.unsqueeze(1), dim=-1)
        k_mat = torch.exp(-dists / self.kernel_param)

        phi = (q_env * k_mat).sum(dim=-1)

        prev_max = self.potential_per_kp_max[env_idx, sid]
        delta_kp = torch.clamp(phi - prev_max, min=0.0)
        pre_contact_reward_step = delta_kp.mean(dim=1)
        new_max = torch.maximum(prev_max, phi)
        self.potential_per_kp_max[env_idx, sid] = new_max

        contact_novelty_reward = torch.zeros(
            (n_valid, num_keypoints), dtype=phi.dtype, device=device
        )
        if contact_mask.any():
            contact_env_idx, contact_keypoint_idx = torch.nonzero(contact_mask, as_tuple=True)
            contact_bin_idx = contact_bins[contact_env_idx, contact_keypoint_idx]
            contact_state_idx = state_ids[contact_env_idx]
            point_counts = self.state_bank.counts[
                contact_state_idx, contact_keypoint_idx, contact_bin_idx
            ].to(torch.float32)
            contact_novelty_reward[contact_env_idx, contact_keypoint_idx] = _sqrt_novelty(
                point_counts, self.novelty_decay_rate
            )

        prev_bonus = self.contact_bonus_per_kp_max[env_idx, sid]
        delta_bonus = torch.clamp(contact_novelty_reward - prev_bonus, min=0.0)
        post_contact_reward_step = delta_bonus.mean(dim=1) * POST_CONTACT_SCALE
        new_bonus = torch.maximum(prev_bonus, contact_novelty_reward)
        self.contact_bonus_per_kp_max[env_idx, sid] = new_bonus

        self.last_pre_contact_reward[env_idx] = pre_contact_reward_step
        self.last_post_contact_reward[env_idx] = post_contact_reward_step
        reward_out[env_idx] = pre_contact_reward_step * PRE_CONTACT_SCALE + post_contact_reward_step
        return reward_out


def _get_or_create_manager(
    env: ManagerBasedRlEnv,
    contactdb_object_name: str,
    contactdb_scale: tuple[float, float, float],
    num_keypoints: int,
    num_key_states: int,
    use_goal_state_feature: bool = False,
    **kwargs,
) -> ContactExplorerRewardManager:
    registry = getattr(env, "_contact_explorer_managers", None)
    if registry is None:
        registry = {}
        setattr(env, "_contact_explorer_managers", registry)
    key = (contactdb_object_name, contactdb_scale, use_goal_state_feature)
    if key not in registry:
        obj = get_cached_contact_db_object(contactdb_object_name, scale=contactdb_scale)
        registry[key] = ContactExplorerRewardManager(
            obj,
            device=torch.device(env.device),
            num_envs=int(env.num_envs),
            num_keypoints=int(num_keypoints),
            num_key_states=num_key_states,
            use_goal_state_feature=use_goal_state_feature,
            **kwargs,
        )
    return registry[key]


@torch.no_grad()
def contact_explorer_reset_buffers(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    contactdb_object_name: str = "cylinder_medium",
    contactdb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_goal_state_feature: bool = False,
) -> None:
    reg = getattr(env, "_contact_explorer_managers", None)
    if reg is None:
        return
    key = (contactdb_object_name, contactdb_scale, use_goal_state_feature)
    if key not in reg:
        return
    reg[key].reset_running_max_buffers(env_ids)


def contact_explorer_reward(
    env: ManagerBasedRlEnv,
    scene_object_name: str,
    contactdb_object_name: str,
    contact_sensor_name: str,
    asset_cfg: SceneEntityCfg,
    keypoints_cfg: SceneEntityCfg,
    num_keypoints: int,
    contactdb_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_key_states: int = 32,
    reward_scale: float = 1.0,
    command_name: str | None = None,
    use_goal_state_feature: bool = False,
    force_sensor_name: str | None = None,
    force_threshold: float | None = None,
    **manager_kwargs,
) -> torch.Tensor:
    """Shaped exploration reward from ContactExplorer (potential + contact novelty)."""
    hand: Entity = env.scene[asset_cfg.name]
    obj_ent: Entity = env.scene[scene_object_name]
    sensor = env.scene[contact_sensor_name]
    data = sensor.data
    if data.found is None or data.pos is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    mgr = _get_or_create_manager(
        env,
        contactdb_object_name,
        contactdb_scale,
        num_keypoints=num_keypoints,
        num_key_states=num_key_states,
        use_goal_state_feature=use_goal_state_feature,
        **manager_kwargs,
    )

    kp_ids = keypoints_cfg.body_ids
    if not isinstance(kp_ids, list):
        raise ValueError(
            f"contact_explorer_reward expects keypoints_cfg.body_ids to be a list, got {kp_ids}"
        )
    kp = [hand.data.body_link_pos_w[:, body_id] for body_id in kp_ids]
    keypoint_positions_world = torch.stack(kp, dim=1)

    obj_pos = obj_ent.data.root_link_pos_w
    obj_quat = obj_ent.data.root_link_quat_w

    found = data.found
    pos = data.pos
    if found.shape[1] != num_keypoints:
        raise RuntimeError(
            f"ContactExplorer: contact sensor has {found.shape[1]} slots but num_keypoints={num_keypoints}. "
            "Match keypoint geom pattern to observation bodies or adjust num_keypoints."
        )

    goal_positions = None
    goal_orientations = None
    if use_goal_state_feature and command_name is not None:
        command = env.command_manager.get_term(command_name)
        goal_positions = command.goal_pos
        goal_orientations = command.goal_quat

    origins = env.scene.env_origins
    obj_pos = obj_pos - origins
    keypoint_positions_world = keypoint_positions_world - origins.unsqueeze(1)
    if pos is not None:
        pos = pos - origins.unsqueeze(1)
    if goal_positions is not None:
        goal_positions = goal_positions - origins

    force_gate = None
    if force_sensor_name is not None and force_threshold is not None:
        fs = env.scene[force_sensor_name]
        if fs.data.force is not None:
            force = torch.nan_to_num(fs.data.force, nan=0.0, posinf=0.0, neginf=0.0)
            force_norm = torch.norm(force, dim=-1)
            if fs.data.found is not None:
                has_contact = fs.data.found > 0
                force_norm = torch.where(has_contact, force_norm, torch.zeros_like(force_norm))
            max_force = force_norm.max(dim=-1).values
            force_gate = max_force < force_threshold

    reward = mgr.step_reward(
        obj_pos,
        obj_quat,
        keypoint_positions_world,
        found,
        pos,
        goal_positions=goal_positions,
        goal_orientations=goal_orientations,
        force_gate=force_gate,
    )
    return reward * float(reward_scale)
