"""Grasp task evaluator: reach the first sampled target."""

from __future__ import annotations

import torch

from mjlab_hand.eval.base import TaskEvaluator, register_evaluator


@register_evaluator
class GraspEvaluator(TaskEvaluator):
    name = "grasp"

    @classmethod
    def matches(cls, env) -> bool:
        return "pose" in env.unwrapped.command_manager.active_terms

    @classmethod
    def add_cli_args(cls, parser) -> None:
        parser.add_argument(
            "--grasp-success-tolerance",
            type=float,
            default=None,
            help="Goal distance tolerance for grasp success (m). Default: 0.03",
        )
        parser.add_argument(
            "--grasp-success-steps",
            type=int,
            default=None,
            help="Consecutive steps near goal to count as grasp success. Default: 10",
        )

    def __init__(self, env, args, device: torch.device) -> None:
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.dt = env.unwrapped.step_dt
        self.success_tolerance = (
            args.grasp_success_tolerance if args.grasp_success_tolerance is not None else 0.03
        )
        self.success_steps = (
            args.grasp_success_steps if args.grasp_success_steps is not None else 10
        )

        # Per-env episode trackers
        self.first_goal = self._get_grasp_goal_pos().clone()
        self.consecutive_near = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.max_consecutive_near = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.steps_to_success = torch.full((self.num_envs,), -1, dtype=torch.int64, device=device)
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.done_ever = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        # Pre-step snapshot buffers (avoid post-reset contamination)
        self._pre_step_dist = torch.zeros(self.num_envs, device=device)
        self._pre_step_near = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        # Global accumulators
        self.completed_episodes = 0
        self.success_count = 0
        self.total_steps_to_success = 0
        self.total_final_dist = 0.0

    def _get_grasp_goal_pos(self) -> torch.Tensor:
        term = self.env.unwrapped.command_manager.get_term("pose")
        return term.goal_pos

    def _get_object_pos(self) -> torch.Tensor:
        return self.env.unwrapped.scene["object"].data.root_link_pos_w

    def snapshot(self) -> None:
        """Capture pre-step env state before the next env.step()."""
        obj_pos = self._get_object_pos()
        dist = torch.norm(obj_pos - self.first_goal, dim=-1)
        self._pre_step_dist[:] = dist
        self._pre_step_near[:] = dist < self.success_tolerance

    def bootstrap(self, env_ids: torch.Tensor) -> None:
        """Set up initial trackers after the first env.reset()."""
        self.first_goal[env_ids] = self._get_grasp_goal_pos()[env_ids]
        self.max_consecutive_near[env_ids] = 0
        self.consecutive_near[env_ids] = 0
        self.steps_to_success[env_ids] = -1
        self.episode_step[env_ids] = 0
        self.done_ever[env_ids] = False

    def on_reset(self, done_mask: torch.Tensor) -> None:
        if not done_mask.any():
            return
        # Only flush envs that haven't been flushed before
        newly_done = done_mask & ~self.done_ever
        if newly_done.any():
            self._flush_done(newly_done)
            self.done_ever |= newly_done

        # Still reset trackers for all done envs so they don't affect future steps
        self.max_consecutive_near[done_mask] = 0
        self.consecutive_near[done_mask] = 0
        self.steps_to_success[done_mask] = -1
        self.episode_step[done_mask] = 0
        # Capture new first goals for envs that auto-reset
        self.first_goal[done_mask] = self._get_grasp_goal_pos()[done_mask]

    def on_step(self, obs, rewards, dones, extras) -> None:
        # Use pre-step snapshot to avoid post-reset contamination.
        # Note: this is the state at the *start* of the step (end of prev step),
        # so for envs that terminate now it is the pre-terminal state (1-step lag).
        active = ~self.done_ever
        if not active.any():
            return

        near = self._pre_step_near & active

        # Update consecutive counters only for active envs
        self.consecutive_near = (self.consecutive_near + 1) * near.long()
        self.consecutive_near = torch.where(
            active, self.consecutive_near, torch.zeros_like(self.consecutive_near)
        )
        self.max_consecutive_near = torch.maximum(self.max_consecutive_near, self.consecutive_near)
        self.max_consecutive_near = torch.where(
            active,
            self.max_consecutive_near,
            torch.zeros_like(self.max_consecutive_near),
        )

        # Record first time success
        just_succeeded = (
            active & (self.steps_to_success < 0) & (self.consecutive_near >= self.success_steps)
        )
        self.steps_to_success = torch.where(
            just_succeeded, self.episode_step + 1, self.steps_to_success
        )

        self.episode_step += 1
        self.episode_step = torch.where(
            active, self.episode_step, torch.zeros_like(self.episode_step)
        )

    def _flush_done(self, done_mask: torch.Tensor) -> None:
        if not done_mask.any():
            return
        done_idx = done_mask.nonzero(as_tuple=False).squeeze(-1)
        n_done = done_idx.numel()

        succeeded = (self.max_consecutive_near[done_idx] >= self.success_steps).long()
        n_succ = int(succeeded.sum().item())
        # steps_to_success >= 0 already implies success, so no extra & needed
        valid_steps = self.steps_to_success[done_idx] >= 0
        if valid_steps.any():
            self.total_steps_to_success += int(
                self.steps_to_success[done_idx][valid_steps].sum().item()
            )
        self.total_final_dist += float(self._pre_step_dist[done_idx].sum().item())
        self.completed_episodes += n_done
        self.success_count += n_succ

    def finalize(self) -> dict[str, float]:
        # Flush envs that never terminated during the eval window
        never_done = ~self.done_ever
        if never_done.any():
            nd_idx = never_done.nonzero(as_tuple=False).squeeze(-1)
            n_nd = nd_idx.numel()
            succeeded = (self.max_consecutive_near[nd_idx] >= self.success_steps).long()
            n_succ = int(succeeded.sum().item())
            valid_steps = self.steps_to_success[nd_idx] >= 0
            if valid_steps.any():
                self.total_steps_to_success += int(
                    self.steps_to_success[nd_idx][valid_steps].sum().item()
                )
            self.total_final_dist += float(self._pre_step_dist[nd_idx].sum().item())
            self.completed_episodes += n_nd
            self.success_count += n_succ

        success_rate = self.success_count / max(self.completed_episodes, 1)
        avg_time_to_success = (self.total_steps_to_success / max(self.success_count, 1)) * self.dt
        avg_final_dist = self.total_final_dist / max(self.completed_episodes, 1)

        return {
            "completed_episodes": float(self.completed_episodes),
            "success_rate": success_rate,
            "avg_time_to_success_s": avg_time_to_success,
            "avg_final_dist_to_first_goal_m": avg_final_dist,
        }

    def report(self, metrics: dict[str, float]) -> None:
        print("\n========== Grasp Evaluation Results ==========")
        print(f"Completed episodes:        {int(metrics['completed_episodes'])}")
        print(f"Success rate:              {metrics['success_rate']:.2%}")
        print(f"Avg time to success:       {metrics['avg_time_to_success_s']:.3f} s")
        print(f"Avg final dist to goal:    {metrics['avg_final_dist_to_first_goal_m']:.4f} m")
        print("==============================================\n")
