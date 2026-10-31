"""Baoding trajectory tracking and measured (not commanded) orbit progress.

Design reference: https://github.com/amathislab/myochallenge/blob/main/src/envs/baoding.py
and docs/summary.md in that repository: labeled-ball target distances, drop
termination, random reference-phase initialization, and slow initial rotations.
This is an independent mjlab implementation for motor-actuated robot hands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_hand.mdp.fixed_hand_reference import fixed_hand_reference_position

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def orbit_positions(phase: torch.Tensor, radius: float) -> torch.Tensor:
    """Two labeled, antipodal ball centers in a horizontal orbit (N,2,3)."""
    offset = torch.stack((radius * phase.cos(), radius * phase.sin(), phase * 0), dim=-1)
    return torch.stack((offset, -offset), dim=1)


def signed_angle_delta(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Wrapped signed delta; reverse motion cancels forward motion."""
    difference = current - previous
    return torch.atan2(difference.sin(), difference.cos())


class BaodingCommand(CommandTerm):
    cfg: BaodingCommandCfg

    def __init__(self, cfg: BaodingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if cfg.period_s <= 0 or cfg.orbit_radius <= cfg.ball_radius:
            raise ValueError("Baoding requires period_s > 0 and orbit_radius > ball_radius")
        cfg.hand_asset_cfg.resolve(env.scene)
        self.hand = env.scene[cfg.hand_asset_cfg.name]
        self.palm_id = cfg.hand_asset_cfg.body_ids[0]
        self.reference_pos = fixed_hand_reference_position(env, cfg.hand_asset_cfg)
        self.balls = [env.scene[name] for name in cfg.ball_names]
        self.center_offset = torch.tensor(cfg.center_offset, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.direction = torch.ones_like(self.phase)
        self.target_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.previous_angles = torch.zeros(self.num_envs, 2, device=self.device)
        self.progress = torch.zeros_like(self.previous_angles)
        self.previous_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.progress_rate = torch.zeros_like(self.phase)
        self.error_sum = torch.zeros_like(self.phase)
        self.sample_count = torch.zeros_like(self.phase)
        self.last_metric_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        for name in (
            "revolutions",
            "best_revolutions",
            "episode_success",
            "tracking_error",
            "mean_tracking_error",
            "min_ball_height",
        ):
            self.metrics[name] = torch.zeros_like(self.phase)

    @property
    def center(self):
        # World-aligned orbit at the nominal palm, independent of stale/reset poses.
        return self.reference_pos + self.center_offset

    @property
    def positions(self):
        return torch.stack([ball.data.root_link_pos_w for ball in self.balls], dim=1)

    @property
    def command(self):
        return torch.cat(
            (
                (self.target_pos - self.center[:, None]).flatten(1),
                self.direction[:, None] * (2 * math.pi / self.cfg.period_s),
            ),
            dim=-1,
        )

    def _resample_command(self, env_ids):
        n = len(env_ids)
        if self.cfg.random_initial_phase:
            self.phase[env_ids] = torch.rand(n, device=self.device) * (2 * math.pi)
        else:
            self.phase[env_ids] = self.cfg.initial_phase
        self.direction[env_ids] = self.cfg.direction
        if self.cfg.random_direction:
            self.direction[env_ids] = torch.randint(0, 2, (n,), device=self.device) * 2 - 1
        positions = self.center[env_ids, None] + orbit_positions(
            self.phase[env_ids], self.cfg.orbit_radius
        )
        self.target_pos[env_ids] = positions
        positions = positions.clone()
        positions[..., 2] += self.cfg.initial_ball_lift
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1
        velocity = torch.zeros(n, 6, device=self.device)
        for i, ball in enumerate(self.balls):
            ball.write_root_link_pose_to_sim(
                torch.cat((positions[:, i], quat), dim=-1), env_ids=env_ids
            )
            ball.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
        offsets = positions - self.center[env_ids, None]
        self.previous_angles[env_ids] = torch.atan2(offsets[..., 1], offsets[..., 0])
        self.progress[env_ids] = 0
        self.previous_valid[env_ids] = True
        self.progress_rate[env_ids] = 0
        self.error_sum[env_ids] = 0
        self.sample_count[env_ids] = 0
        self.last_metric_step[env_ids] = self._env.common_step_counter
        self.reset_step[env_ids] = self._env.common_step_counter
        for value in self.metrics.values():
            value[env_ids] = 0

    def refresh_metrics(self):
        """Called before reset by termination/reward terms, at most once per step."""
        fresh = self.last_metric_step != self._env.common_step_counter
        if not fresh.any():
            return
        positions = self.positions
        offsets = positions - self.center[:, None]
        error = torch.linalg.vector_norm(positions - self.target_pos, dim=-1)
        radius = torch.linalg.vector_norm(offsets[..., :2], dim=-1)
        angles = torch.atan2(offsets[..., 1], offsets[..., 0])
        delta = signed_angle_delta(angles, self.previous_angles) * self.direction[:, None]
        # Require both balls near the commanded orbit and away from its singular
        # center. Breaking tracking resets progress: waving/dropping balls cannot
        # accumulate successful revolutions across disconnected trajectory pieces.
        valid = (
            (error < self.cfg.progress_tolerance).all(dim=-1)
            & (radius > self.cfg.orbit_radius * 0.5).all(dim=-1)
            & (offsets[..., 2].abs() < self.cfg.progress_tolerance).all(dim=-1)
        )
        old_progress = self.progress.amin(dim=-1)
        progress = torch.where((valid & self.previous_valid)[:, None], self.progress + delta, 0.0)
        progress_min = progress.amin(dim=-1)
        self.progress_rate[fresh] = ((progress_min - old_progress) / self._env.step_dt)[fresh]
        self.progress[fresh] = progress[fresh]
        self.previous_angles[fresh] = angles[fresh]
        self.previous_valid[fresh] = valid[fresh]
        self.metrics["revolutions"][fresh] = (progress_min / (2 * math.pi))[fresh]
        self.metrics["best_revolutions"][fresh] = torch.maximum(
            self.metrics["best_revolutions"][fresh], self.metrics["revolutions"][fresh]
        )
        success = (progress_min >= 2 * math.pi * self.cfg.success_revolutions) & valid
        self.metrics["episode_success"][fresh] = torch.maximum(
            self.metrics["episode_success"][fresh], success[fresh].float()
        )
        self.metrics["tracking_error"][fresh] = error.mean(dim=-1)[fresh]
        self.error_sum[fresh] += error.mean(dim=-1)[fresh]
        self.sample_count[fresh] += 1
        self.metrics["mean_tracking_error"][fresh] = (
            self.error_sum / self.sample_count.clamp_min(1)
        )[fresh]
        self.metrics["min_ball_height"][fresh] = offsets[..., 2].amin(dim=-1)[fresh]
        self.last_metric_step[fresh] = self._env.common_step_counter

    def _update_metrics(self):
        self.refresh_metrics()

    def _update_command(self):
        self.target_pos[:] = self.center[:, None] + orbit_positions(
            self.phase, self.cfg.orbit_radius
        )

    def compute(self, dt):
        # Advancing phase here makes the next observation contain the next target;
        # the completed transition is scored against the target the actor saw.
        self.refresh_metrics()
        advance = (self.reset_step != self._env.common_step_counter).float()
        self.phase.add_(self.direction * (2 * math.pi / self.cfg.period_s) * dt * advance)
        self.phase.remainder_(2 * math.pi)
        self._update_command()

    def _debug_vis_impl(self, visualizer):
        for index in visualizer.get_env_indices(self.num_envs):
            for ball in range(2):
                visualizer.add_sphere(
                    self.target_pos[index, ball].detach().cpu().numpy(),
                    self.cfg.ball_radius,
                    (0.1, 0.9, 0.3, 0.25),
                    label=f"baoding_target_{index}_{ball}",
                )


@dataclass(kw_only=True)
class BaodingCommandCfg(CommandTermCfg):
    hand_asset_cfg: SceneEntityCfg
    ball_names: tuple[str, str] = ("ball1", "ball2")
    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.12)
    ball_radius: float = 0.018
    orbit_radius: float = 0.028
    period_s: float = 8.0
    initial_ball_lift: float = 0.0
    initial_phase: float = 0.0
    random_initial_phase: bool = True
    direction: float = 1.0
    random_direction: bool = False
    progress_tolerance: float = 0.035
    success_revolutions: float = 1.0
    drop_distance: float = 0.16
    drop_height: float = 0.08

    def build(self, env):
        return BaodingCommand(self, env)


def ball_states(env, command_name="baoding"):
    command = env.command_manager.get_term(command_name)
    positions = (command.positions - command.center[:, None]).flatten(1)
    velocities = torch.cat([ball.data.root_link_vel_w for ball in command.balls], dim=-1)
    return torch.cat((positions, velocities * 0.2), dim=-1)


def target_observation(env, command_name="baoding"):
    command = env.command_manager.get_term(command_name)
    return torch.cat((command.command, (command.target_pos - command.positions).flatten(1)), dim=-1)


def tracking_reward(env, command_name="baoding", std=0.04):
    command = env.command_manager.get_term(command_name)
    command.refresh_metrics()
    error_sq = (command.positions - command.target_pos).square().sum(dim=-1)
    # Product-like joint score: both balls must track, not just the easier ball.
    return torch.exp(-error_sq.mean(dim=-1) / std**2)


def progress_reward(env, command_name="baoding"):
    command = env.command_manager.get_term(command_name)
    command.refresh_metrics()
    # Do not clip negative resets: clipping would allow repeated partial rotations
    # followed by tracking loss to earn unbounded forward-progress reward.
    return command.progress_rate


def balls_dropped(env, command_name="baoding"):
    command = env.command_manager.get_term(command_name)
    command.refresh_metrics()
    offsets = command.positions - command.center[:, None]
    return (offsets[..., 2] < -command.cfg.drop_height).any(dim=-1) | (
        offsets.norm(dim=-1) > command.cfg.drop_distance
    ).any(dim=-1)
