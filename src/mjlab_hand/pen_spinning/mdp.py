"""Pen spinning objectives, independent of the pen's axial roll.

Design references (independent implementation):
https://github.com/HaozhiQi/penspin (directional rotation with retention/energy costs)
https://github.com/henrycharlesworth/dexterous-gym/blob/master/dexterous_gym/envs/pen_spin.py
(horizontal pen and directed rotation). Here world Z is the spin axis and
object-local Z is the long axis. Only long-axis heading changes count as turns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from mjlab_hand.mdp.fixed_hand_reference import fixed_hand_reference_position

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def heading_delta(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Signed shortest heading change, continuous across the +/-pi boundary."""
    cross = previous[..., 0] * current[..., 1] - previous[..., 1] * current[..., 0]
    dot = (previous[..., :2] * current[..., :2]).sum(-1)
    return torch.atan2(cross, dot)


def end_over_end_rate(axis: torch.Tensor, angular_velocity: torch.Tensor) -> torch.Tensor:
    """d atan2(long_axis_y, long_axis_x)/dt; pure axial roll is exactly zero."""
    axial_speed = (axis * angular_velocity).sum(-1)
    return (angular_velocity[..., 2] - axis[..., 2] * axial_speed) / (
        1.0 - axis[..., 2].square()
    ).clamp_min(0.1)


class PenSpinCommand(CommandTerm):
    cfg: PenSpinCommandCfg

    def __init__(self, cfg: PenSpinCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        cfg.hand_asset_cfg.resolve(env.scene)
        self.hand = env.scene["hand"]
        self.object = env.scene["object"]
        self.palm_id = int(cfg.hand_asset_cfg.body_ids[0])
        self._reference_pos = fixed_hand_reference_position(env, cfg.hand_asset_cfg)
        self.offset = torch.tensor(cfg.center_offset, device=self.device)
        self.axis_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        self.previous_axis = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_axis[:, 0] = 1.0
        self.net_angle = torch.zeros(self.num_envs, device=self.device)
        self.success = torch.zeros_like(self.net_angle)
        self.best_angle = torch.zeros_like(self.net_angle)
        self.last_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.previous_valid = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.direction = torch.full_like(self.net_angle, cfg.direction)
        for name in ("signed_revolutions", "completed_revolutions", "episode_success",
                     "spin_rate", "center_error", "horizontal_error", "best_revolutions"):
            self.metrics[name] = torch.zeros_like(self.net_angle)

    @property
    def command(self) -> torch.Tensor:
        # Direction + desired rate; matching inputs across every hand.
        return torch.stack((self.direction, torch.full_like(self.direction, self.cfg.target_speed)), -1)

    @property
    def center(self) -> torch.Tensor:
        # Fixed nominal world-frame palm anchor. Live body poses are un-forwarded
        # on the first reset and can retain the preceding episode on auto-reset.
        return self._reference_pos + self.offset

    def state(self):
        axis = quat_apply(self.object.data.root_link_quat_w, self.axis_local)
        error = torch.linalg.vector_norm(self.object.data.root_link_pos_w - self.center, dim=-1)
        horizontal = axis[:, 2].abs()
        rate = end_over_end_rate(axis, self.object.data.root_link_vel_w[:, 3:]) * self.direction
        valid = (error < self.cfg.retention_radius) & (horizontal < self.cfg.max_tilt_sine)
        valid &= self.object.data.root_link_pos_w[:, 2] > self.center[:, 2] - self.cfg.drop_height
        return axis, error, horizontal, rate, valid

    def refresh_metrics(self):
        # Rewards run before command_manager.compute and resets. Refresh once per
        # physics step so terminal transitions contribute to the episode metrics.
        axis, error, horizontal, rate, valid = self.state()
        fresh = self.last_step != self._env.common_step_counter
        delta = heading_delta(self.previous_axis, axis) * self.direction
        # Invalid (near-vertical, dropped or escaped) transitions cannot earn turns.
        delta = torch.where(valid & self.previous_valid & fresh, delta, 0.0)
        self.net_angle = torch.where(fresh & ~valid, 0.0, self.net_angle + delta)
        self.best_angle = torch.maximum(self.best_angle, self.net_angle)
        self.success = torch.maximum(self.success, ((self.net_angle >= 2 * math.pi) & valid).float())
        self.previous_axis[fresh] = axis[fresh]
        self.previous_valid[fresh] = valid[fresh]
        self.last_step.fill_(self._env.common_step_counter)
        self.metrics["signed_revolutions"].copy_(self.net_angle / (2 * math.pi))
        self.metrics["completed_revolutions"].copy_((self.best_angle / (2 * math.pi)).floor())
        self.metrics["best_revolutions"].copy_(self.best_angle / (2 * math.pi))
        self.metrics["episode_success"].copy_(self.success)
        self.metrics["spin_rate"].copy_(rate)
        self.metrics["center_error"].copy_(error)
        self.metrics["horizontal_error"].copy_(horizontal)

    def _update_metrics(self):
        self.refresh_metrics()

    def _update_command(self):
        pass

    def _resample_command(self, env_ids):
        n = len(env_ids)
        # Start horizontally, with modest heading/position jitter, no injected spin.
        yaw = torch.empty(n, device=self.device).uniform_(-self.cfg.reset_heading_range, self.cfg.reset_heading_range)
        yaw += self.cfg.initial_heading
        half = yaw * 0.5
        s = math.sqrt(0.5)
        # q_z(yaw) * q_y(pi/2): pen-local Z initially points along world X.
        quat = torch.stack((s * half.cos(), -s * half.sin(), s * half.cos(), s * half.sin()), -1)
        pos = self.center[env_ids].clone()
        pos[:, 2] += self.cfg.initial_lift
        pos[:, :2] += torch.empty(n, 2, device=self.device).uniform_(-self.cfg.reset_position_jitter, self.cfg.reset_position_jitter)
        self.object.write_root_link_pose_to_sim(torch.cat((pos, quat), -1), env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        self.previous_axis[env_ids] = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
        self.net_angle[env_ids] = 0
        self.success[env_ids] = 0
        self.best_angle[env_ids] = 0
        self.previous_valid[env_ids] = True
        self.last_step[env_ids] = self._env.common_step_counter
        for value in self.metrics.values():
            value[env_ids] = 0


@dataclass(kw_only=True)
class PenSpinCommandCfg(CommandTermCfg):
    hand_asset_cfg: SceneEntityCfg
    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.12)
    direction: float = 1.0
    target_speed: float = 2.0
    retention_radius: float = 0.10
    drop_height: float = 0.08
    max_tilt_sine: float = 0.6
    reset_heading_range: float = 0.15
    initial_lift: float = 0.03
    initial_heading: float = 0.0
    reset_position_jitter: float = 0.003

    def build(self, env):
        if self.direction not in (-1.0, 1.0):
            raise ValueError("Pen spin direction must be -1 or +1")
        if self.target_speed <= 0:
            raise ValueError("Pen spin target speed must be positive")
        return PenSpinCommand(self, env)


def spin_command(env):
    return env.command_manager.get_term("pen_spin")


def target_observation(env):
    return spin_command(env).command


def pen_axis_observation(env):
    return spin_command(env).state()[0]


def directed_rate_reward(rate: torch.Tensor, target_speed: float) -> torch.Tensor:
    """Bound useful speed but retain the full cost of reversing progress."""
    return (rate / target_speed).clamp(max=1.0)


def spin_reward(env):
    cmd = spin_command(env)
    cmd.refresh_metrics()
    _, _, _, rate, valid = cmd.state()
    # Cap only desired-direction speed. Clipping reverse speed as well lets a
    # policy earn reward with slow forward / rapid reverse oscillations.
    return directed_rate_reward(rate, cmd.cfg.target_speed) * valid.float()


def retention_reward(env):
    _, error, horizontal, _, _ = spin_command(env).state()
    return torch.exp(-error.square() / 0.04**2) * torch.exp(-horizontal.square() / 0.3**2)


def tilt_cost(env):
    return spin_command(env).state()[2].square()


def linear_velocity_cost(env):
    return env.scene["object"].data.root_link_vel_w[:, :3].square().sum(-1)


def axial_velocity_cost(env):
    cmd = spin_command(env)
    axis = cmd.state()[0]
    return (axis * cmd.object.data.root_link_vel_w[:, 3:]).sum(-1).square()


def object_dropped(env):
    cmd = spin_command(env)
    cmd.refresh_metrics()
    pos = cmd.object.data.root_link_pos_w
    # Radius is looser than reward retention to permit recoveries.
    return ((pos - cmd.center).norm(dim=-1) > 0.18) | (pos[:, 2] < cmd.center[:, 2] - cmd.cfg.drop_height)
