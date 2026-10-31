"""Baoding robot-hand task factories; no external object meshes required."""

from copy import deepcopy
from dataclasses import dataclass

import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp

from mjlab_hand.in_hand_rotation.config.leap.rl_cfg import leap_rotation_ppo_runner_cfg
from mjlab_hand.in_hand_rotation.mdp.actions import JointPositionEMAActionCfg
from mjlab_hand.in_hand_rotation.rotation_env_cfg import make_rotation_env_cfg
from mjlab_hand.robot.allegro import ALLEGRO_HAND
from mjlab_hand.robot.leap import LEAP_HAND
from mjlab_hand.robot.shadow_hand import SHADOW_HAND
from mjlab_hand.robot.sharpa_wave import SHARPA_HAND
from mjlab_hand.robot.wuji import WUJI_HAND

from . import mdp as baoding_mdp

_HANDS = {
    hand.name: hand for hand in (ALLEGRO_HAND, LEAP_HAND, SHADOW_HAND, SHARPA_HAND, WUJI_HAND)
}
# World-aligned orbit centers above the palm/finger workspace, tuned against
# native MuJoCo home-state geometry; learned feasibility still needs training.
_CENTERS = {
    "allegro": (-0.025, 0.0, 0.050),
    "leap": (-0.010, 0.01, 0.050),
    "shadow": (-0.045, 0.0, 0.040),
    "sharpa": (-0.060, 0.0, 0.045),
    "wuji": (-0.060, 0.0, 0.045),
}

_INITIAL_LIFT = {"allegro": 0.04, "leap": 0.04, "shadow": 0.065, "sharpa": 0.055, "wuji": 0.055}


@dataclass(frozen=True)
class BallSpec:
    radius: float = 0.018
    mass: float = 0.025
    color: tuple[float, float, float, float] = (0.95, 0.6, 0.1, 1.0)

    def __call__(self):
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="ball")
        body.add_freejoint(name="free")
        body.add_geom(
            name="ball_collision",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(self.radius, 0, 0),
            mass=self.mass,
            rgba=self.color,
            friction=(1.0, 0.005, 0.0001),
            condim=4,
        )
        return spec


def make_baoding_env_cfg(hand: str = "leap", play: bool = False):
    """Build one of five motor-actuated Baoding environments.

    Action dimensions and mappings match that hand's rotation expert. The state
    includes both balls' positions/velocities and phase-conditioned goals. A
    successful episode requires one measured full orbit with both balls tracking.
    """
    hand = hand.lower()
    if hand not in _HANDS:
        raise ValueError(f"Unknown hand {hand!r}; choose {tuple(_HANDS)}")
    spec = _HANDS[hand]
    cfg = make_rotation_env_cfg(spec, contactdb_object=None)
    if hand == "shadow":
        # Rotate the inherited palm-down pose 180 degrees around world X.
        cfg.scene.entities["hand"].init_state.rot = (0.0, 0.0, 1.0, 0.0)
    if hand in ("sharpa", "wuji"):
        # Horizontal palm for two-ball support; inherited rotation poses tilt 60°.
        cfg.scene.entities["hand"].init_state.rot = (0.70710678, 0.0, -0.70710678, 0.0)
    if spec.extra_actions_factory is not None:
        cfg.actions = deepcopy(spec.extra_actions_factory(use_tendon_control=True))
        action = cfg.actions["joint_pos"]
        cfg.actions["joint_pos"] = JointPositionEMAActionCfg(
            entity_name=action.entity_name,
            actuator_names=action.actuator_names,
            clip=action.clip,
            preserve_order=action.preserve_order,
            scale=action.scale,
            offset=action.offset,
            use_default_offset=action.use_default_offset,
            ema_alpha=0.8,
        )
        if "tendon_pos" in cfg.actions:
            cfg.actions["tendon_pos"].ema_alpha = 0.8
    cfg.scene.entities["ball1"] = EntityCfg(spec_fn=BallSpec())
    cfg.scene.entities["ball2"] = EntityCfg(spec_fn=BallSpec(color=(0.15, 0.45, 0.95, 1.0)))
    cfg.commands = {
        "baoding": baoding_mdp.BaodingCommandCfg(
            hand_asset_cfg=SceneEntityCfg("hand", body_names=[spec.palm_body]),
            center_offset=_CENTERS[hand],
            initial_ball_lift=_INITIAL_LIFT[hand],
            resampling_time_range=(1e9, 1e9),
            debug_vis=True,
            random_initial_phase=not play,
        )
    }
    terms = {
        name: deepcopy(cfg.observations["actor"].terms[name])
        for name in ("joint_pos", "joint_vel", "actions")
    }
    terms["ball_states"] = ObservationTermCfg(func=baoding_mdp.ball_states)
    terms["ball_targets"] = ObservationTermCfg(func=baoding_mdp.target_observation)
    cfg.observations = {
        "actor": ObservationGroupCfg(terms, enable_corruption=not play, nan_policy="error"),
        "critic": ObservationGroupCfg(deepcopy(terms), enable_corruption=False, nan_policy="error"),
    }
    # Positive tracking reward avoids the early-drop optimum of distance-only
    # negative rewards. Progress is a signed potential difference, so holding
    # still cannot earn it and back-and-forth motion cancels.
    cfg.rewards = {
        "tracking": RewardTermCfg(func=baoding_mdp.tracking_reward, weight=3.0),
        "orbit_progress": RewardTermCfg(func=baoding_mdp.progress_reward, weight=0.25),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.002),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "nan_term": TerminationTermCfg(func=mdp.nan_detection),
        "balls_dropped": TerminationTermCfg(func=baoding_mdp.balls_dropped),
    }
    cfg.events["reset_hand_joints"].params["position_range"] = (-0.02, 0.02)
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4
    cfg.episode_length_s = 24.0
    cfg.sim.nconmax = 160
    cfg.sim.njmax = 1000
    cfg.viewer.distance = 0.5
    cfg.viewer.elevation = -35.0
    return cfg


def baoding_ppo_runner_cfg(hand: str = "leap"):
    cfg = leap_rotation_ppo_runner_cfg()
    cfg.experiment_name = f"{hand.lower()}_baoding"
    cfg.num_steps_per_env = 32
    cfg.max_iterations = 10000
    cfg.algorithm.learning_rate = 3e-4
    return cfg
