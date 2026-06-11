"""Shadow hand package."""

from mjlab_hand.robot.hand_spec import HandSpec
from mjlab_hand.robot.shadow_hand.shadow_constants import (
    SHADOW_KEYPOINT_GEOMS,
    get_shadow_actions,
    get_shadow_floating_hand_cfg,
    get_shadow_hand_cfg,
)

SHADOW_HAND = HandSpec(
    name="shadow",
    palm_body="lh_palm",
    keypoint_bodies=(
        "lh_ffknuckle",
        "lh_ffproximal",
        "lh_ffmiddle",
        "lh_ffdistal",
        "lh_mfknuckle",
        "lh_mfproximal",
        "lh_mfmiddle",
        "lh_mfdistal",
        "lh_rfknuckle",
        "lh_rfproximal",
        "lh_rfmiddle",
        "lh_rfdistal",
        "lh_lfmetacarpal",
        "lh_lfknuckle",
        "lh_lfproximal",
        "lh_lfmiddle",
        "lh_lfdistal",
        "lh_thbase",
        "lh_thproximal",
        "lh_thhub",
        "lh_thmiddle",
        "lh_thdistal",
    ),
    keypoint_geom_pattern=SHADOW_KEYPOINT_GEOMS,
    num_keypoints=22,
    get_fixed_cfg=get_shadow_hand_cfg,
    get_floating_cfg=get_shadow_floating_hand_cfg,
    has_tendon_actuation=True,
    extra_actions_factory=get_shadow_actions,
    floating_init_pos=(-0.4, 0.0, 0.9),
)

__all__ = ["SHADOW_HAND"]
