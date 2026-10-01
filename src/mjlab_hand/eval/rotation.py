"""Rotation task evaluator: count distinct targets reached before object drops.

Each env's first episode is scored. Besides avg_successes_before_drop it
reports (CHANGES.md item 63, re-implementing Bundle's rotation.py):
  success_rate_any          fraction of episodes with >= 1 target reached
  avg_targets_offered       targets presented while the episode ran (1 + resamples)
  per_target_success_rate   reached / offered, over episodes that ended
                            within the eval window (dropped or timed out)
  per_target_success_rate_incl_truncated   same over all episodes, including
                            ones still running when the window ended
  per_episode               [{successes, targets_offered, survival_s, ended}]
"""

from __future__ import annotations

import torch

from mjlab_hand.eval.base import TaskEvaluator, register_evaluator


@register_evaluator
class RotationEvaluator(TaskEvaluator):
    name = "rotation"

    @classmethod
    def matches(cls, env) -> bool:
        return "rotation" in env.unwrapped.command_manager.active_terms

    @classmethod
    def add_cli_args(cls, parser) -> None:
        parser.add_argument(
            "--rotation-success-tolerance",
            type=float,
            default=None,
            help="Rotation distance tolerance for rotation success (rad). Default: 0.1",
        )
        parser.add_argument(
            "--rotation-success-steps",
            type=int,
            default=None,
            help="Consecutive steps within tolerance to count a target as reached. Default: 10",
        )

    def __init__(self, env, args, device: torch.device) -> None:
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.dt = env.unwrapped.step_dt
        self.success_tolerance = (
            args.rotation_success_tolerance if args.rotation_success_tolerance is not None else 0.1
        )
        self.success_steps = (
            args.rotation_success_steps if args.rotation_success_steps is not None else 10
        )

        # Per-env episode trackers
        self.success_count = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.consecutive_success = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.target_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.last_target_quat = torch.zeros(self.num_envs, 4, device=device)
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.survival_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.done_ever = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.targets_offered = torch.ones(self.num_envs, dtype=torch.int64, device=device)
        self.per_episode: list[dict] = []

        # Pre-step snapshot
        self._pre_step_at_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._pre_step_rot_dist = torch.zeros(self.num_envs, device=device)

        # Global accumulators
        self.completed_episodes = 0
        self.total_success_count = 0
        self.total_survival_steps = 0
        self.total_rot_dist = 0.0
        self.total_steps = 0

    def _get_rotation_rot_dist(self) -> torch.Tensor:
        term = self.env.unwrapped.command_manager.get_term("rotation")
        return term.metrics["rot_dist"]

    def _get_target_quat(self) -> torch.Tensor:
        term = self.env.unwrapped.command_manager.get_term("rotation")
        return term.target_quat

    def snapshot(self) -> None:
        """Capture pre-step env state before the next env.step()."""
        rot_dist = self._get_rotation_rot_dist()
        target_quat = self._get_target_quat()

        # Detect target resampling: reset streaks so each new target can be counted
        target_changed = (self.last_target_quat != target_quat).any(dim=-1)
        self.targets_offered += (target_changed & ~self.done_ever).long()
        if target_changed.any():
            self.consecutive_success[target_changed] = 0
            self.target_reached[target_changed] = False
            self.last_target_quat = target_quat.clone()

        self._pre_step_rot_dist[:] = rot_dist
        self._pre_step_at_goal[:] = rot_dist < self.success_tolerance

    def bootstrap(self, env_ids: torch.Tensor) -> None:
        """Set up initial trackers after the first env.reset()."""
        self.success_count[env_ids] = 0
        self.consecutive_success[env_ids] = 0
        self.target_reached[env_ids] = False
        self.last_target_quat[env_ids] = self._get_target_quat()[env_ids]
        self.episode_step[env_ids] = 0
        self.survival_steps[env_ids] = 0
        self.done_ever[env_ids] = False
        self.targets_offered[env_ids] = 1

    def on_reset(self, done_mask: torch.Tensor) -> None:
        if not done_mask.any():
            return
        newly_done = done_mask & ~self.done_ever
        if newly_done.any():
            self._flush_done(newly_done)
            self.done_ever |= newly_done

        # Still reset per-step trackers for done envs so they don't affect future steps
        self.consecutive_success[done_mask] = 0
        self.target_reached[done_mask] = False
        self.episode_step[done_mask] = 0

    def on_step(self, obs, rewards, dones, extras) -> None:
        active = ~self.done_ever
        if not active.any():
            return

        at_goal = self._pre_step_at_goal & active

        # Update consecutive success counter for current target
        self.consecutive_success = (self.consecutive_success + 1) * at_goal.long()
        self.consecutive_success = torch.where(
            active, self.consecutive_success, torch.zeros_like(self.consecutive_success)
        )

        # Count a new target success only once per target, after N consecutive steps
        just_reached = (
            active & ~self.target_reached & (self.consecutive_success >= self.success_steps)
        )
        self.success_count += just_reached.long()
        self.target_reached |= just_reached

        self.episode_step += 1
        self.episode_step = torch.where(
            active, self.episode_step, torch.zeros_like(self.episode_step)
        )
        self.survival_steps += active.long()

        self.total_rot_dist += (self._pre_step_rot_dist * active).sum().item()
        self.total_steps += int(active.sum().item())

    def _flush_done(self, done_mask: torch.Tensor) -> None:
        if not done_mask.any():
            return
        done_idx = done_mask.nonzero(as_tuple=False).squeeze(-1)
        n_done = done_idx.numel()
        self.total_success_count += int(self.success_count[done_idx].sum().item())
        self.total_survival_steps += int(self.survival_steps[done_idx].sum().item())
        self.completed_episodes += n_done
        self._record(done_idx, ended=True)

    def _record(self, idx: torch.Tensor, ended: bool) -> None:
        for i in idx.tolist():
            self.per_episode.append({
                "successes": int(self.success_count[i]),
                "targets_offered": int(self.targets_offered[i]),
                "survival_s": float(self.survival_steps[i]) * self.dt,
                "ended": ended,
            })

    def finalize(self) -> dict[str, float]:
        # Flush envs that never terminated during the eval window
        never_done = ~self.done_ever
        if never_done.any():
            nd_idx = never_done.nonzero(as_tuple=False).squeeze(-1)
            n_nd = nd_idx.numel()
            self.total_success_count += int(self.success_count[nd_idx].sum().item())
            self.total_survival_steps += int(self.survival_steps[nd_idx].sum().item())
            self.completed_episodes += n_nd
            self._record(nd_idx, ended=False)

        avg_successes = self.total_success_count / max(self.completed_episodes, 1)
        avg_survival_time = (self.total_survival_steps / max(self.completed_episodes, 1)) * self.dt
        avg_rot_dist = self.total_rot_dist / max(self.total_steps, 1)
        drop_rate = self.done_ever.float().mean().item()

        return {
            "completed_episodes": float(self.completed_episodes),
            "avg_successes_before_drop": avg_successes,
            "avg_survival_time_s": avg_survival_time,
            "drop_rate": drop_rate,
            "avg_rot_dist": avg_rot_dist,
            **self._episode_metrics(),
        }

    def _episode_metrics(self) -> dict:
        eps = self.per_episode
        if not eps:
            return {}
        ended = [e for e in eps if e["ended"]]

        def rate(rows):
            offered = sum(e["targets_offered"] for e in rows)
            return sum(e["successes"] for e in rows) / offered if offered else float("nan")

        return {
            "success_rate_any": sum(e["successes"] > 0 for e in eps) / len(eps),
            "avg_targets_offered": sum(e["targets_offered"] for e in eps) / len(eps),
            "per_target_success_rate": rate(ended),
            "per_target_success_rate_incl_truncated": rate(eps),
            "per_episode": eps,
        }

    def report(self, metrics: dict[str, float]) -> None:
        print("\n========== Rotation Evaluation Results ==========")
        print(f"Completed envs:              {int(metrics['completed_episodes'])}")
        print(f"Avg successes before drop:   {metrics['avg_successes_before_drop']:.2f}")
        print(f"Avg survival time:           {metrics['avg_survival_time_s']:.3f} s")
        print(f"Drop rate:                   {metrics['drop_rate']:.2%}")
        print(f"Avg rotation distance:       {metrics['avg_rot_dist']:.4f} rad")
        if "success_rate_any" in metrics:
            print(f"Success rate (any target):   {metrics['success_rate_any']:.2%}")
            print(f"Per-target success rate:     {metrics['per_target_success_rate']:.3f}")
            print(f"Avg targets offered:         {metrics['avg_targets_offered']:.2f}")
        print("=================================================\n")
