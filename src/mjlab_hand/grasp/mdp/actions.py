from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class FloatingHandDeltaActionCfg(BaseActionCfg):
    """Configuration for floating hand delta action space with smoothing.

    Action structure: [base translation (3), base rotation (3), finger joints (N)].

    The policy outputs normalized actions in [-1, 1], which are mapped to:
    - Base translation velocity: [trans_vel_min, trans_vel_max] (m/s)
    - Base rotation velocity: [rot_vel_min, rot_vel_max] (rad/s)
    - Finger joint velocity: [joint_vel_min, joint_vel_max] (rad/s)

    Each env step integrates ``velocity * step_dt`` into position targets, then EMA.
    """

    trans_vel_min: float = -0.25
    trans_vel_max: float = 0.25

    rot_vel_min: float = -1.5
    rot_vel_max: float = 1.5

    joint_vel_min: float = -1.0
    joint_vel_max: float = 1.0

    ema_alpha: float = 0.3

    use_relative: bool = True

    clip_to_joint_limits: bool = True

    def __post_init__(self):
        self.transmission_type = TransmissionType.JOINT
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in [0, 1], got {self.ema_alpha}")

    def build(self, env: ManagerBasedRlEnv) -> FloatingHandDeltaAction:
        return FloatingHandDeltaAction(self, env)


class FloatingHandDeltaAction(BaseAction):
    """Floating hand control: velocity-scaled deltas and EMA smoothing.

    Controls base pose and finger joints by mapping actions to velocities
    (m/s, rad/s), multiplying by ``step_dt``, then integrating into targets.
    """

    cfg: FloatingHandDeltaActionCfg

    _base_pos_target: torch.Tensor
    _base_rot_target: torch.Tensor
    _joint_pos_target: torch.Tensor

    _base_pos_ema: torch.Tensor
    _base_rot_ema: torch.Tensor
    _joint_pos_ema: torch.Tensor

    _base_trans_dim: int = 3
    _base_rot_dim: int = 3
    _joint_start_idx: int = 6

    def __init__(self, cfg: FloatingHandDeltaActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        self._step_dt = float(self._env.step_dt)

        self._num_joints = self.action_dim

        self._base_pos_target = torch.zeros(self.num_envs, 3, device=self.device)
        self._base_rot_target = torch.zeros(self.num_envs, 3, device=self.device)
        self._joint_pos_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._base_pos_ema = torch.zeros(self.num_envs, 3, device=self.device)
        self._base_rot_ema = torch.zeros(self.num_envs, 3, device=self.device)
        self._joint_pos_ema = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._initialize_targets(slice(None))

    def _initialize_targets(self, env_ids: torch.Tensor | slice) -> None:
        all_joint_pos = self._entity.data.joint_pos[env_ids]
        joint_pos = all_joint_pos[:, self._target_ids].clone()

        self._base_pos_target[env_ids] = joint_pos[:, 0:3]

        self._base_rot_target[env_ids] = joint_pos[:, 3:6]

        self._joint_pos_target[env_ids] = joint_pos

        self._base_pos_ema[env_ids] = self._base_pos_target[env_ids].clone()
        self._base_rot_ema[env_ids] = self._base_rot_target[env_ids].clone()
        self._joint_pos_ema[env_ids] = self._joint_pos_target[env_ids].clone()

        self._processed_actions[env_ids] = 0.0  # Deltas start at zero

    def _quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=-1)

    def _euler_to_quat(self, euler: torch.Tensor) -> torch.Tensor:
        roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack([w, x, y, z], dim=-1)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Map actions to velocities, scale by ``step_dt``, integrate, then EMA.

        Normalized values in [-1, 1]: base (0:6) then finger joints (6:).
        """
        self._raw_actions[:] = actions

        normalized = torch.clamp(actions, -1.0, 1.0)

        dt = self._step_dt

        trans_delta = (
            self._map_range(
                normalized[:, 0:3],
                self.cfg.trans_vel_min,
                self.cfg.trans_vel_max,
            )
            * dt
        )
        rot_delta = (
            self._map_range(
                normalized[:, 3:6],
                self.cfg.rot_vel_min,
                self.cfg.rot_vel_max,
            )
            * dt
        )
        finger_delta = (
            self._map_range(
                normalized[:, 6:],
                self.cfg.joint_vel_min,
                self.cfg.joint_vel_max,
            )
            * dt
        )

        joint_delta = torch.cat([trans_delta, rot_delta, finger_delta], dim=-1)

        self._base_pos_target = self._base_pos_target + trans_delta
        self._base_rot_target = self._base_rot_target + rot_delta
        self._joint_pos_target = self._joint_pos_target + joint_delta

        if self.cfg.clip_to_joint_limits:
            joint_limits = self._entity.data.joint_pos_limits[:, self._target_ids]
            self._joint_pos_target = torch.clamp(
                self._joint_pos_target,
                joint_limits[..., 0],
                joint_limits[..., 1],
            )

            base_pos_limits = joint_limits[:, 0:3]  # (envs, 3, 2) -> (min, max)
            self._base_pos_target = torch.clamp(
                self._base_pos_target,
                base_pos_limits[..., 0],
                base_pos_limits[..., 1],
            )

            base_rot_limits = joint_limits[:, 3:6]  # (envs, 3, 2) -> (min, max)
            self._base_rot_target = torch.clamp(
                self._base_rot_target,
                base_rot_limits[..., 0],
                base_rot_limits[..., 1],
            )

        alpha = self.cfg.ema_alpha
        self._base_pos_ema = alpha * self._base_pos_target + (1 - alpha) * self._base_pos_ema
        self._base_rot_ema = alpha * self._base_rot_target + (1 - alpha) * self._base_rot_ema
        self._joint_pos_ema = alpha * self._joint_pos_target + (1 - alpha) * self._joint_pos_ema

        self._processed_actions = torch.cat([trans_delta, rot_delta, joint_delta], dim=-1)

    def _map_range(self, x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        return 0.5 * (x + 1.0) * (max_val - min_val) + min_val

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self._joint_pos_ema, joint_ids=self._target_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._initialize_targets(env_ids)


@dataclass(kw_only=True)
class TendonDeltaActionCfg(BaseActionCfg):
    """Delta tendon control with per-step integration and smoothing."""

    tendon_vel_min: float = -1.0
    tendon_vel_max: float = 1.0
    ema_alpha: float = 0.3
    clip_to_limits: bool = True

    def __post_init__(self):
        self.transmission_type = TransmissionType.TENDON
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in [0, 1], got {self.ema_alpha}")

    def build(self, env: ManagerBasedRlEnv) -> "TendonDeltaAction":
        return TendonDeltaAction(self, env)


class TendonDeltaAction(BaseAction):
    """Control tendons via integrated velocity deltas instead of absolute jumps."""

    cfg: TendonDeltaActionCfg

    def __init__(self, cfg: TendonDeltaActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        self._step_dt = float(self._env.step_dt)
        self._needs_sync = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._tendon_target = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._ema_target = torch.zeros_like(self._tendon_target)

        if isinstance(self._scale, torch.Tensor):
            self._target_min = self._offset - self._scale
            self._target_max = self._offset + self._scale
        else:
            scale = float(self._scale)
            offset = float(self._offset)
            self._target_min = offset - scale
            self._target_max = offset + scale

        self._initialize_targets(slice(None))

    def _initialize_targets(self, env_ids: torch.Tensor | slice) -> None:
        if isinstance(env_ids, slice):
            tendon_len = self._entity.data.tendon_len[env_ids, self._target_ids].clone()
        else:
            tendon_len = self._entity.data.tendon_len[
                env_ids.unsqueeze(-1), self._target_ids
            ].clone()
        self._tendon_target[env_ids] = tendon_len
        self._ema_target[env_ids] = tendon_len
        self._processed_actions[env_ids] = tendon_len

    def _map_range(self, x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        return 0.5 * (x + 1.0) * (max_val - min_val) + min_val

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        if torch.any(self._needs_sync):
            sync_env_ids = torch.nonzero(self._needs_sync, as_tuple=False).squeeze(-1)
            self._initialize_targets(sync_env_ids)
            self._needs_sync[sync_env_ids] = False

        normalized = torch.clamp(actions, -1.0, 1.0)
        tendon_delta = (
            self._map_range(
                normalized,
                self.cfg.tendon_vel_min,
                self.cfg.tendon_vel_max,
            )
            * self._step_dt
        )

        self._tendon_target = self._tendon_target + tendon_delta
        if self.cfg.clip_to_limits:
            self._tendon_target = torch.clamp(
                self._tendon_target,
                min=self._target_min,
                max=self._target_max,
            )

        alpha = self.cfg.ema_alpha
        self._ema_target = alpha * self._tendon_target + (1.0 - alpha) * self._ema_target
        self._processed_actions = self._ema_target

    def apply_actions(self) -> None:
        self._entity.set_tendon_len_target(self._processed_actions, tendon_ids=self._target_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._initialize_targets(env_ids)
        self._needs_sync[env_ids] = True


@dataclass(kw_only=True)
class FloatingHandDeltaActionWithBaseLimitsCfg(FloatingHandDeltaActionCfg):
    """Extended config with workspace limits for base position."""

    workspace_x_min: float = -0.5
    workspace_x_max: float = 0.5
    workspace_y_min: float = -0.5
    workspace_y_max: float = 0.3
    workspace_z_min: float = 0.1
    workspace_z_max: float = 0.8

    def build(self, env: ManagerBasedRlEnv) -> FloatingHandDeltaActionWithBaseLimits:
        return FloatingHandDeltaActionWithBaseLimits(self, env)


class FloatingHandDeltaActionWithBaseLimits(FloatingHandDeltaAction):
    """Floating hand delta action with workspace limits enforcement."""

    cfg: FloatingHandDeltaActionWithBaseLimitsCfg
    _initial_base_pos: torch.Tensor

    def __init__(self, cfg: FloatingHandDeltaActionWithBaseLimitsCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._initial_base_pos = self._entity.data.root_link_pos_w.clone()

    def _apply_workspace_limits(self) -> None:
        x_min = self._initial_base_pos[:, 0] + self.cfg.workspace_x_min
        x_max = self._initial_base_pos[:, 0] + self.cfg.workspace_x_max
        y_min = self._initial_base_pos[:, 1] + self.cfg.workspace_y_min
        y_max = self._initial_base_pos[:, 1] + self.cfg.workspace_y_max
        z_min = self._initial_base_pos[:, 2] + self.cfg.workspace_z_min
        z_max = self._initial_base_pos[:, 2] + self.cfg.workspace_z_max

        self._base_pos_target[:, 0] = torch.clamp(self._base_pos_target[:, 0], x_min, x_max)
        self._base_pos_target[:, 1] = torch.clamp(self._base_pos_target[:, 1], y_min, y_max)
        self._base_pos_target[:, 2] = torch.clamp(self._base_pos_target[:, 2], z_min, z_max)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._apply_workspace_limits()
