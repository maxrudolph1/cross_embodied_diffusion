"""Sharpa hand package."""

from mjlab_hand.robot.hand_spec import HandSpec
from mjlab_hand.robot.sharpa_wave.sharpa_constants import (
    SHARPA_KEYPOINT_GEOMS,
    get_sharpa_floating_hand_cfg,
    get_sharpa_hand_cfg,
)

SHARPA_HAND = HandSpec(
    name="sharpa",
    palm_body="left_hand_C_MC",
    keypoint_bodies=(
        "left_thumb_DP",
        "left_index_DP",
        "left_middle_DP",
        "left_ring_DP",
        "left_pinky_DP",
    ),
    keypoint_geom_pattern=SHARPA_KEYPOINT_GEOMS,
    num_keypoints=5,
    get_fixed_cfg=get_sharpa_hand_cfg,
    get_floating_cfg=get_sharpa_floating_hand_cfg,
    floating_init_pos=(-0.2, 0.0, 0.8),
    floating_init_rot=(0.71, 0.0, 0.71, 0.0),
)

__all__ = ["SHARPA_HAND"]
