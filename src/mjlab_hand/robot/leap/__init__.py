"""LEAP hand package."""

from mjlab_hand.robot.hand_spec import HandSpec
from mjlab_hand.robot.leap.leap_constants import (
    LEAP_ACTION_LIMIT_OFFSET,
    LEAP_ACTION_LIMIT_SCALE,
    LEAP_KEYPOINT_GEOMS,
    get_leap_floating_hand_cfg,
    get_leap_hand_cfg,
)

LEAP_HAND = HandSpec(
    name="leap",
    palm_body="palm",
    keypoint_bodies=("if_ds", "mf_ds", "rf_ds", "th_ds"),
    keypoint_geom_pattern=LEAP_KEYPOINT_GEOMS,
    num_keypoints=4,
    get_fixed_cfg=get_leap_hand_cfg,
    get_floating_cfg=get_leap_floating_hand_cfg,
    action_limit_scale=LEAP_ACTION_LIMIT_SCALE,
    action_limit_offset=LEAP_ACTION_LIMIT_OFFSET,
)

__all__ = ["LEAP_HAND"]
