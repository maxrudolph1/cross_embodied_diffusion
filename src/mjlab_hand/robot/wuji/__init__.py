"""Wuji hand package."""

from mjlab_hand.robot.hand_spec import HandSpec
from mjlab_hand.robot.wuji.wuji_constants import (
    WUJI_ACTION_LIMIT_OFFSET,
    WUJI_ACTION_LIMIT_SCALE,
    WUJI_KEYPOINT_GEOMS,
    get_wuji_floating_hand_cfg,
    get_wuji_hand_cfg,
)

WUJI_HAND = HandSpec(
    name="wuji",
    palm_body="right_palm_link",
    keypoint_bodies=(
        "right_finger1_link4",
        "right_finger2_link4",
        "right_finger3_link4",
        "right_finger4_link4",
        "right_finger5_link4",
    ),
    keypoint_geom_pattern=WUJI_KEYPOINT_GEOMS,
    num_keypoints=5,
    get_fixed_cfg=get_wuji_hand_cfg,
    get_floating_cfg=get_wuji_floating_hand_cfg,
    action_limit_scale=WUJI_ACTION_LIMIT_SCALE,
    action_limit_offset=WUJI_ACTION_LIMIT_OFFSET,
    floating_init_pos=(-0.3, 0.0, 0.8),
    floating_init_rot=(0.71, 0.0, 0.71, 0.0),
)

__all__ = ["WUJI_HAND"]
