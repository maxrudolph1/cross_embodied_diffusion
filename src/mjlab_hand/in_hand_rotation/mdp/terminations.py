from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity

from .commands import RotationCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["object_dropped"]


def object_dropped(
    env: ManagerBasedRlEnv,
    command_name: str,
    object_name: str,
    fall_dist: float = 0.24,
) -> torch.Tensor:
    """Terminate when object drifts too far from target position.

    Matches TOUCHMANIP: goal_dist >= fall_dist triggers reset.
    """
    obj: Entity = env.scene[object_name]
    command = cast(RotationCommand, env.command_manager.get_term(command_name))
    goal_dist = torch.norm(obj.data.root_link_pos_w - command.target_pos, dim=-1)
    return goal_dist >= fall_dist
