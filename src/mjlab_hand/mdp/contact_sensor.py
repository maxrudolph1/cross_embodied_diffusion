"""Shared contact sensor configurations for grasp and in-hand manipulation tasks."""

from mjlab.sensor import ContactMatch, ContactSensorCfg


def make_keypoint_object_contact_sensor_cfg(
    primary_geom_pattern: str = r".*",
) -> ContactSensorCfg:
    """Contact sensors on keypoint collision geoms vs object subtree."""
    return ContactSensorCfg(
        name="keypoint_contacts",
        primary=ContactMatch(
            mode="geom",
            pattern=primary_geom_pattern,
            entity="hand",
        ),
        secondary=ContactMatch(
            mode="subtree",
            pattern="object",
            entity="object",
        ),
        fields=("found", "force", "pos", "normal"),
        reduce="netforce",
        num_slots=1,
        global_frame=False,
    )


def make_hand_force_sensor_cfg(name: str = "hand_force") -> ContactSensorCfg:
    """Contact sensor monitoring net contact force on all hand geoms vs any partner."""
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(
            mode="body",
            pattern=".*",
            entity="hand",
        ),
        secondary=None,
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        global_frame=True,
    )
