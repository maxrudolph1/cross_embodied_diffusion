"""CPU regressions for physical spin accounting; no rollout/GPU required."""
import math

import torch

from mjlab_hand.pen_spinning.mdp import end_over_end_rate, heading_delta


def test_axial_roll_does_not_count_as_spin():
    axis = torch.tensor([[0.8, 0.0, 0.6]])
    assert torch.allclose(end_over_end_rate(axis, axis * 9), torch.zeros(1), atol=1e-5)


def test_horizontal_end_over_end_is_directed():
    axis = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    vel = torch.tensor([[0, 0, 2.0], [0, 0, -2.0]])
    assert torch.allclose(end_over_end_rate(axis, vel), torch.tensor([2.0, -2.0]))


def test_heading_wrap_and_oscillation():
    phases = torch.tensor([math.pi - 0.1, -math.pi + 0.1, math.pi - 0.1])
    axes = torch.stack((phases.cos(), phases.sin(), torch.zeros_like(phases)), -1)
    delta = heading_delta(axes[:-1], axes[1:])
    assert torch.allclose(delta, torch.tensor([0.2, -0.2]), atol=1e-6)
    assert abs(delta.sum().item()) < 1e-6


def test_one_geometric_revolution():
    phases = torch.linspace(0, 2 * math.pi, 101)
    axes = torch.stack((phases.cos(), phases.sin(), torch.zeros_like(phases)), -1)
    assert abs(heading_delta(axes[:-1], axes[1:]).sum().item() - 2 * math.pi) < 1e-5


def test_pen_asset_is_free_and_has_finite_inertia():
    from mjlab_hand.pen_spinning.env_cfg import make_pen_spec
    model = make_pen_spec().compile()
    assert model.nq == 7 and model.nv == 6
    assert abs(model.body_mass[1] - 0.020) < 1e-9
    assert (model.body_inertia[1] > 0).all()


def test_all_hands_have_pen_specific_goals():
    from mjlab_hand.pen_spinning.env_cfg import HANDS, make_pen_spinning_env_cfg
    for hand in HANDS:
        cfg = make_pen_spinning_env_cfg(hand)
        assert list(cfg.commands) == ["pen_spin"]
        assert "contact_explorer_reset" not in cfg.events
        assert "goal_quat" not in cfg.observations["actor"].terms
        assert "spin_target" in cfg.observations["actor"].terms
        assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02


def test_initial_pen_does_not_interpenetrate_hand():
    import mujoco
    import numpy as np
    from mjlab.scene import Scene

    from mjlab_hand.pen_spinning.env_cfg import HANDS, make_pen_spinning_env_cfg

    for hand in HANDS:
        cfg = make_pen_spinning_env_cfg(hand)
        model = Scene(cfg.scene, device="cpu").compile()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        command = cfg.commands["pen_spin"]
        palm_id = model.body("hand/" + command.hand_asset_cfg.body_names[0]).id
        center = data.xpos[palm_id] + np.array(command.center_offset)
        adr = model.jnt_qposadr[model.joint("object/free").id]
        obj_id = model.body("object/pen").id
        for yaw in (-command.reset_heading_range, 0, command.reset_heading_range):
            j = command.reset_position_jitter
            for x, y in ((-j, -j), (-j, j), (j, -j), (j, j)):
                data.qpos[adr:adr+3] = center + np.array([x, y, command.initial_lift])
                h, s = (yaw + command.initial_heading) / 2, math.sqrt(.5)
                data.qpos[adr+3:adr+7] = [s*np.cos(h), -s*np.sin(h), s*np.cos(h), s*np.sin(h)]
                mujoco.mj_forward(model, data)
                for contact in data.contact:
                    if obj_id in (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]):
                        assert contact.dist > -0.0005, (hand, yaw, x, y, contact.dist)


def test_episode_turns_survive_drop_without_counting_invalid_resets():
    from types import SimpleNamespace

    from mjlab_hand.pen_spinning.mdp import PenSpinCommand

    metric_names = ("signed_revolutions", "completed_revolutions", "episode_success",
                    "spin_rate", "center_error", "horizontal_error", "best_revolutions")
    state = [torch.tensor([[1., 0., 0.]]), torch.zeros(1), torch.zeros(1),
             torch.ones(1), torch.ones(1, dtype=torch.bool)]
    cmd = SimpleNamespace(
        _env=SimpleNamespace(common_step_counter=0), state=lambda: state,
        previous_axis=state[0].clone(), net_angle=torch.zeros(1), best_angle=torch.zeros(1),
        success=torch.zeros(1), direction=torch.ones(1), last_step=torch.zeros(1, dtype=torch.long),
        previous_valid=torch.ones(1, dtype=torch.bool),
        metrics={key: torch.zeros(1) for key in metric_names},
    )
    for step, angle in enumerate(torch.linspace(0, 2 * math.pi + .01, 100), 1):
        cmd._env.common_step_counter = step
        state[0] = torch.tensor([[angle.cos(), angle.sin(), 0.]])
        PenSpinCommand.refresh_metrics(cmd)
        first = cmd.net_angle.clone()
        PenSpinCommand.refresh_metrics(cmd)
        assert torch.equal(first, cmd.net_angle), "double metric refresh counted a step twice"
    assert cmd.metrics["completed_revolutions"].item() == 1
    state[4] = torch.zeros(1, dtype=torch.bool)
    cmd._env.common_step_counter += 1
    PenSpinCommand.refresh_metrics(cmd)
    assert cmd.metrics["signed_revolutions"].item() == 0
    assert cmd.metrics["completed_revolutions"].item() == 1
    assert cmd.metrics["episode_success"].item() == 1


def test_slow_forward_fast_reverse_does_not_harvest_reward():
    from mjlab_hand.pen_spinning.mdp import directed_rate_reward

    rate = torch.tensor([2., -20.])
    duration = torch.tensor([1., .1])
    # Both trajectories rotate two radians, with opposite signs.
    assert (directed_rate_reward(rate, 2.) * duration).sum().item() <= 0
    assert (directed_rate_reward(-rate, 2.) * duration).sum().item() <= 0


def test_first_and_partial_reset_use_nominal_anchor_not_live_palm(monkeypatch):
    """Command-level regression: live xposes can be zero or from the previous episode."""
    from types import SimpleNamespace

    from mjlab_hand.pen_spinning import mdp

    reference = torch.tensor([[0., 0., .5], [2., 3., .5]])
    monkeypatch.setattr(mdp, "fixed_hand_reference_position", lambda env, asset: reference.clone())
    pose = torch.zeros(2, 7)
    velocity = torch.ones(2, 6)
    hand = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=torch.zeros(2, 1, 3)))
    obj = SimpleNamespace(
        write_root_link_pose_to_sim=lambda value, env_ids: pose.__setitem__(env_ids, value),
        write_root_link_velocity_to_sim=lambda value, env_ids: velocity.__setitem__(env_ids, value),
    )
    asset = SimpleNamespace(resolve=lambda scene: None, body_ids=[0])
    cfg = mdp.PenSpinCommandCfg(
        hand_asset_cfg=asset, center_offset=(-.04, 0., .10),
        resampling_time_range=(1e9, 1e9), reset_heading_range=0., reset_position_jitter=0.,
    )
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"hand": hand, "object": obj},
                          common_step_counter=0)
    command = cfg.build(env)
    command.reset(torch.tensor([0, 1]))
    expected = reference + torch.tensor([-.04, 0., .13])
    assert torch.allclose(pose[:, :3], expected)
    assert torch.equal(velocity, torch.zeros_like(velocity))
    first_pose = pose.clone()
    hand.data.body_link_pos_w.fill_(123.)  # stale terminal pose of an earlier episode
    env.common_step_counter = 200
    command.reset(torch.tensor([1]))
    assert torch.equal(pose, first_pose), "auto-reset moved the spawn or another environment"
    assert torch.allclose(command.center, reference + torch.tensor([-.04, 0., .10]))
