"""CPU regressions for Baoding progress, reset isolation and primitive assets.

Run: .venv/bin/python scripts/check_baoding.py
Uses synthetic object trajectories, not a trained policy or simulator rollout.
"""

import re
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene

from mjlab_hand.baoding.env_cfg import BallSpec, make_baoding_env_cfg
from mjlab_hand.baoding.mdp import BaodingCommand, BaodingCommandCfg, orbit_positions


class Ball:
    def __init__(self, n):
        self.data = SimpleNamespace(
            root_link_pos_w=torch.zeros(n, 3), root_link_vel_w=torch.zeros(n, 6)
        )

    def write_root_link_pose_to_sim(self, pose, env_ids):
        self.data.root_link_pos_w[env_ids] = pose[:, :3]

    def write_root_link_velocity_to_sim(self, velocity, env_ids):
        self.data.root_link_vel_w[env_ids] = velocity


def make_command():
    hand_cfg = SimpleNamespace(name="hand", body_ids=[0], resolve=lambda scene: None)
    cfg = BaodingCommandCfg(
        hand_asset_cfg=hand_cfg,
        resampling_time_range=(1e9, 1e9),
        random_initial_phase=False,
        center_offset=(0, 0, 0),
    )
    scene = {
        "hand": SimpleNamespace(data=SimpleNamespace(body_link_pos_w=torch.zeros(2, 1, 3))),
        "ball1": Ball(2),
        "ball2": Ball(2),
    }
    env = SimpleNamespace(
        num_envs=2, device="cpu", scene=scene, common_step_counter=0, step_dt=0.02
    )
    with patch(
        "mjlab_hand.baoding.mdp.fixed_hand_reference_position", return_value=torch.zeros(2, 3)
    ):
        command = BaodingCommand(cfg, env)
    command.reset(torch.arange(2))
    return env, command


def check_reset_geometry(cfg):
    """No ball/hand penetration across 48 orbit phases and reset-joint jitter."""
    model = Scene(cfg.scene, device="cpu").spec.compile()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1234)
    command = cfg.commands["baoding"]
    palm_id = model.body("hand/" + command.hand_asset_cfg.body_names[0]).id
    for phase in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        data.qpos[:] = model.qpos0
        for pattern, value in cfg.scene.entities["hand"].init_state.joint_pos.items():
            for joint_id in range(model.njnt):
                name = model.joint(joint_id).name.removeprefix("hand/")
                if re.fullmatch(pattern, name):
                    data.qpos[model.jnt_qposadr[joint_id]] = value + rng.uniform(-0.02, 0.02)
        mujoco.mj_forward(model, data)
        center = data.xpos[palm_id].copy() + command.center_offset
        for index, name in enumerate(command.ball_names):
            angle = phase + index * np.pi
            offset = command.orbit_radius * np.array([np.cos(angle), np.sin(angle), 0])
            qadr = model.joint(name + "/free").qposadr[0]
            data.qpos[qadr : qadr + 3] = center + offset + [0, 0, command.initial_ball_lift]
            data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            if any(model.geom(int(g)).name.startswith("ball") for g in contact.geom):
                assert contact.dist >= -1e-5, (
                    command.hand_asset_cfg.body_names,
                    phase,
                    contact.dist,
                )


def main():
    for hand in ("allegro", "leap", "shadow", "sharpa", "wuji"):
        cfg = make_baoding_env_cfg(hand)
        assert set(cfg.scene.entities) == {"hand", "ball1", "ball2"}
        assert "rotation" not in cfg.commands
        check_reset_geometry(cfg)
    model = BallSpec()().compile()
    assert model.nq == 7 and model.nv == 6

    env, command = make_command()
    # One tracked revolution vs a stationary hand holding both balls.
    for _ in range(410):
        env.common_step_counter += 1
        target = command.target_pos.clone()
        for ball_i, ball in enumerate(command.balls):
            ball.data.root_link_pos_w[0] = target[0, ball_i]
        command.refresh_metrics()
        progress = command.progress.clone()
        command.refresh_metrics()
        torch.testing.assert_close(command.progress, progress)  # no double-count
        command.compute(env.step_dt)
    assert command.metrics["episode_success"][0] == 1
    assert command.metrics["episode_success"][1] == 0
    assert command.metrics["revolutions"][0] > 1
    assert command.metrics["revolutions"][1] == 0
    other_progress = command.progress[0].clone()
    other_phase = command.phase[0].clone()
    command.reset(torch.tensor([1]))
    torch.testing.assert_close(command.progress[0], other_progress)
    torch.testing.assert_close(command.phase[0], other_phase)
    assert command.progress[1].abs().sum() == 0
    reset_phase = command.phase[1].clone()
    command.compute(env.step_dt)
    torch.testing.assert_close(command.phase[1], reset_phase)  # auto-reset must match fresh reset
    # Drop: measured progress resets, and its reward does not become positive.
    env.common_step_counter += 1
    command.balls[0].data.root_link_pos_w[0, 2] = -1
    command.refresh_metrics()
    assert command.metrics["revolutions"][0] == 0
    assert command.metrics["best_revolutions"][0] > 1
    assert command.progress_rate[0] < 0

    # Back-and-forth must cancel rather than accumulate positive-only motion.
    env, command = make_command()
    for phase in torch.cat((torch.linspace(0, 2, 50), torch.linspace(2, 0, 50))):
        env.common_step_counter += 1
        targets = orbit_positions(phase.expand(2), command.cfg.orbit_radius)
        command.target_pos[:] = targets
        for i, ball in enumerate(command.balls):
            ball.data.root_link_pos_w[:] = targets[:, i]
        command.refresh_metrics()
    assert command.metrics["revolutions"].abs().max() < 1e-5
    assert command.metrics["episode_success"].sum() == 0
    print(
        "Baoding: five factories, assets, full turn, static hold, oscillation, drop and reset isolation OK"
    )


if __name__ == "__main__":
    main()
