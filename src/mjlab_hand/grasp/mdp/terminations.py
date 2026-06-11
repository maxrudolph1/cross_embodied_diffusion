from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity

from mjlab_hand.grasp.mdp.table_utils import get_table_reference_pos_w

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "object_fallen",
    "success_termination",
    "GoalReachedTermination",
]


def object_fallen(
    env: ManagerBasedRlEnv,
    object_name: str,
    table_name: str,
    fall_height: float = -0.2,
) -> torch.Tensor:
    """Terminate when object has fallen off the table."""
    obj: Entity = env.scene[object_name]
    table: Entity = env.scene[table_name]

    obj_pos = obj.data.root_link_pos_w
    table_pos = get_table_reference_pos_w(table)

    height_above_table = obj_pos[:, 2] - table_pos[:, 2]

    return height_above_table < fall_height


def success_termination(
    env: ManagerBasedRlEnv,
    object_name: str,
    table_name: str,
    success_height: float = 0.2,
    success_steps: int = 10,
) -> torch.Tensor:
    """Terminate when the object clears ``success_height`` above the table."""
    obj: Entity = env.scene[object_name]
    table: Entity = env.scene[table_name]

    obj_pos = obj.data.root_link_pos_w
    table_pos = get_table_reference_pos_w(table)

    height_above_table = obj_pos[:, 2] - table_pos[:, 2]

    lifted = height_above_table > success_height

    return lifted


class GoalReachedTermination:
    """Terminate when the object satisfies the sampled goal for several steps."""

    def __init__(self, cfg, env) -> None:
        self._object_name: str = cfg.params["object_name"]
        self._command_name: str = cfg.params.get("command_name", "pose")
        self._success_tolerance: float = cfg.params.get("success_tolerance", 0.05)
        self._success_steps: int = cfg.params.get("success_steps", 1)
        self._near_goal_steps = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
        obj: Entity = env.scene[self._object_name]
        command = env.command_manager.get_term(self._command_name)

        obj_pos = obj.data.root_link_pos_w
        goal_pos = command.goal_pos
        distance = torch.norm(obj_pos - goal_pos, dim=-1)
        near_goal = torch.isfinite(distance) & (distance < self._success_tolerance)
        self._near_goal_steps = (self._near_goal_steps + near_goal.int()) * near_goal.int()
        return self._near_goal_steps >= self._success_steps

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._near_goal_steps.fill_(0)
        else:
            self._near_goal_steps[env_ids] = 0


def object_out_of_bounds(
    env: ManagerBasedRlEnv,
    object_name: str,
    table_name: str,
    bounds: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Terminate when object has moved too far from the table."""
    obj: Entity = env.scene[object_name]
    table: Entity = env.scene[table_name]

    obj_pos = obj.data.root_link_pos_w
    table_pos = get_table_reference_pos_w(table)

    delta = torch.abs(obj_pos - table_pos)

    out_of_bounds = (
        (delta[:, 0] > bounds[0]) | (delta[:, 1] > bounds[1]) | (delta[:, 2] > bounds[2])
    )

    return out_of_bounds
