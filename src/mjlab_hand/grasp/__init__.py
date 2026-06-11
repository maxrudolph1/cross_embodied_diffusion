"""Grasping task with floating hand.

This module provides a grasping task where a floating hand
(Allegro, LEAP, Shadow, or Sharpa) must grasp a cylinder_medium from a table.

Example usage:
    from mjlab_hand.grasp.config.allegro.env_cfgs import allegro_grasp_env_cfg

    cfg = allegro_grasp_env_cfg()
    env = ManagerBasedRLEnv(cfg=cfg)
"""

from mjlab_hand.grasp.grasp_env_cfg import make_grasp_env_cfg

__all__ = ["make_grasp_env_cfg"]
