"""HandSpec descriptor shared across grasp and in-hand rotation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class HandSpec:
    """Single source of truth for hand identity data.

    Both ``grasp`` and ``in_hand_rotation`` tasks consume this descriptor
    via ``make_grasp_env_cfg`` and ``make_rotation_env_cfg``.
    """

    name: str

    palm_body: str

    keypoint_bodies: tuple[str, ...]

    keypoint_geom_pattern: str

    num_keypoints: int

    get_fixed_cfg: Callable[[], EntityCfg]

    get_floating_cfg: Callable[..., EntityCfg]
    """Factory returning a floating-base ``EntityCfg`` for this hand.

    Accepts optional ``pos`` and ``rot`` kwargs so callers can override the
    floating-base initial pose without mutating shared state. See
    :meth:`make_floating_entity_cfg` for the canonical call site.
    """

    action_limit_scale: dict[str, float] | None = None
    action_limit_offset: dict[str, float] | None = None

    floating_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.8)
    floating_init_rot: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)

    has_tendon_actuation: bool = False
    extra_actions_factory: Callable[..., dict] | None = None

    def make_floating_entity_cfg(self) -> EntityCfg:
        """Build a floating-base EntityCfg with this spec's grasp-task pose.

        Threads ``floating_init_pos`` and ``floating_init_rot`` into the
        underlying factory, so the returned cfg owns a fresh ``init_state``.
        Use this from grasp env wiring instead of mutating the cfg afterwards.
        """
        return self.get_floating_cfg(
            pos=self.floating_init_pos,
            rot=self.floating_init_rot,
        )
