"""Fixed-hand pen spinning environments; primitive asset needs no external download."""
from __future__ import annotations

import math
from importlib import import_module

import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as common_mdp

from . import mdp

HANDS = ("Allegro", "LEAP", "Shadow", "Sharpa", "Wuji")


def make_pen_spec() -> mujoco.MjSpec:
    # 16 cm end-to-end, 14 mm diameter, 20 g. The colored end markings make
    # end-over-end rotation distinguishable from long-axis rolling in videos.
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="pen")
    body.add_freejoint(name="free")
    body.add_geom(name="pen_collision", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  size=(0.007, 0.073, 0.0), mass=0.020,
                  rgba=(0.85, 0.85, 0.85, 1.0), friction=(1.0, 0.005, 0.0001), condim=4)
    for name, z, color in (("red_end", 0.068, (0.9, 0.1, 0.1, 1)),
                           ("blue_end", -0.068, (0.1, 0.3, 0.9, 1))):
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=(0.0072, 0.005, 0.0), pos=(0, 0, z),
                      rgba=color, contype=0, conaffinity=0, mass=0, group=2)
    return spec


def make_pen_spinning_env_cfg(hand: str = "Allegro", play: bool = False):
    matches = {h.lower(): h for h in HANDS}
    if hand.lower() not in matches:
        raise ValueError(f"Unknown hand {hand!r}; expected one of {HANDS}")
    hand = matches[hand.lower()]
    module = import_module(f"mjlab_hand.in_hand_rotation.config.{hand.lower()}.env_cfgs")
    cfg = getattr(module, f"{hand.lower()}_rotation_env_cfg")()
    rotation = cfg.commands["rotation"]
    if hand == "Shadow":
        # The rotation benchmark home pose is palm-down; spinning needs support.
        cfg.scene.entities["hand"].init_state.rot = (0.0, 0.0, 1.0, 0.0)
    cfg.scene.entities["object"] = EntityCfg(spec_fn=make_pen_spec)
    cfg.scene.sensors = ()
    cfg.commands = {"pen_spin": mdp.PenSpinCommandCfg(
        hand_asset_cfg=rotation.hand_asset_cfg,
        center_offset={
            "Allegro": (-0.04, 0.0, 0.10),
            "LEAP": (0.015, 0.0, 0.09),
            "Shadow": (-0.04, -0.025, 0.06),
            "Sharpa": (-0.07, -0.025, 0.06),
            "Wuji": (-0.07, 0.025, 0.06),
        }[hand],
        initial_lift=0.03 if hand in ("Allegro", "LEAP") else 0.02,
        initial_heading=math.pi / 2 if hand in ("Sharpa", "Wuji") else 0.0,
        reset_heading_range=0.025 if hand == "Sharpa" else 0.15,
        reset_position_jitter=0.001 if hand == "Sharpa" else 0.003,
        resampling_time_range=(1e9, 1e9),
    )}
    # ContactDB's surface-exploration objective is not defined for this primitive.
    cfg.events.pop("contact_explorer_reset", None)
    cfg.events["reset_hand_joints"].params["position_range"] = (-0.025, 0.025)
    for group in cfg.observations.values():
        group.terms.pop("goal_quat", None)
        group.terms.pop("goal_rel_rot", None)
        group.terms["pen_axis"] = ObservationTermCfg(func=mdp.pen_axis_observation)
        group.terms["spin_target"] = ObservationTermCfg(func=mdp.target_observation)
    cfg.rewards = {
        "spin": RewardTermCfg(func=mdp.spin_reward, weight=4.0),
        "retain": RewardTermCfg(func=mdp.retention_reward, weight=0.5),
        "tilt": RewardTermCfg(func=mdp.tilt_cost, weight=-0.5),
        "linear_velocity": RewardTermCfg(func=mdp.linear_velocity_cost, weight=-0.1),
        "axial_velocity": RewardTermCfg(func=mdp.axial_velocity_cost, weight=-0.002),
        "action_rate": RewardTermCfg(func=common_mdp.action_rate_l2, weight=-0.002),
        "joint_speed": RewardTermCfg(func=common_mdp.joint_vel_l2, weight=-0.0001,
                                      params={"asset_cfg": SceneEntityCfg("hand")}),
    }
    cfg.terminations["object_dropped"] = TerminationTermCfg(func=mdp.object_dropped)
    # Thin, fast objects need finer contact integration than the rotation baseline.
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.episode_length_s = 20.0
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def pen_spinning_ppo_runner_cfg(hand: str = "Allegro"):
    module = import_module(f"mjlab_hand.in_hand_rotation.config.{hand.lower()}.rl_cfg")
    cfg = getattr(module, f"{hand.lower()}_rotation_ppo_runner_cfg")()
    cfg.experiment_name = f"{hand.lower()}_pen_spinning"
    cfg.num_steps_per_env = 32
    cfg.algorithm.entropy_coef = 0.005
    return cfg
