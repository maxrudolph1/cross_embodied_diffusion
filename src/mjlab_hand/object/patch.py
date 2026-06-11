"""Shared utilities for patching ContactDB objects into env configs."""

from __future__ import annotations

from mjlab.entity import EntityCfg

from mjlab_hand.object.object import make_contactdb_spec_fn


def patch_object(
    cfg, object_name: str, scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> None:
    """Patch object-related fields on an env cfg.

    Updates:
      - cfg.scene.entities["object"]
      - Any reward/event/command terms that carry ``contactdb_object_name`` /
        ``contactdb_scale`` attributes.
    """
    if "object" not in cfg.scene.entities:
        return

    cfg.scene.entities["object"] = EntityCfg(
        spec_fn=make_contactdb_spec_fn(object_name, scale=scale),
    )

    for term in cfg.rewards.values():
        params = getattr(term, "params", None)
        if params is None:
            continue
        if "contactdb_object_name" in params:
            params["contactdb_object_name"] = object_name
        if "contactdb_scale" in params:
            params["contactdb_scale"] = scale

    for term in cfg.events.values():
        params = getattr(term, "params", None)
        if params is None:
            continue
        if "contactdb_object_name" in params:
            params["contactdb_object_name"] = object_name
        if "contactdb_scale" in params:
            params["contactdb_scale"] = scale

    for term in cfg.commands.values():
        if hasattr(term, "contactdb_object_name"):
            term.contactdb_object_name = object_name
        if hasattr(term, "contactdb_scale"):
            term.contactdb_scale = scale


def parse_scale(scale_str: str | None) -> tuple[float, float, float]:
    """Parse a comma-separated scale string into a 3-tuple."""
    if scale_str is None:
        return (1.0, 1.0, 1.0)
    parts = [float(v.strip()) for v in scale_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Scale must have exactly 3 comma-separated values, got: {scale_str}")
    return tuple(parts)
