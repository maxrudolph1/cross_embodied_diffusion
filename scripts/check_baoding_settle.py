"""Native CPU reset/settle diagnostic for all Baoding hands.

Run: .venv/bin/python scripts/check_baoding_settle.py

Applies zero normalized actions for 2s from phases 0 and pi/2, reporting
ball/hand contact and first drop. This is a geometry/control diagnostic, NOT
an RL rollout or expert-performance evaluation: it uses nominal friction,
no observation noise, and native MuJoCo rather than Warp. LEAP software PD is
reproduced without command delay (constant zero action). No GPUs are used.
"""

import re

import mujoco
import numpy as np
from mjlab.scene import Scene

from mjlab_hand.baoding.env_cfg import _HANDS, make_baoding_env_cfg
from mjlab_hand.robot.leap.leap_constants import LEAP_ACTUATORS


def lookup(mapping, name, default=0.0):
    if isinstance(mapping, (int, float)):
        return mapping
    return next((v for k, v in mapping.items() if re.fullmatch(k, name)), default)


def run(
    hand,
    phase=0.0,
    seconds=2.0,
    hold_home=False,
    center_override=None,
    rot_override=None,
    lift_override=None,
):
    c = make_baoding_env_cfg(hand)
    if rot_override is not None:
        c.scene.entities["hand"].init_state.rot = rot_override
    if lift_override is not None:
        c.commands["baoding"].initial_ball_lift = lift_override
    m = Scene(c.scene, device="cpu").compile()
    d = mujoco.MjData(m)
    m.opt.timestep = c.sim.mujoco.timestep
    m.opt.iterations = 20
    m.opt.ls_iterations = 20
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    for i in range(m.njnt):
        name = m.joint(i).name.removeprefix("hand/")
        if m.joint(i).name.startswith("hand/"):
            d.qpos[m.jnt_qposadr[i]] = lookup(c.scene.entities["hand"].init_state.joint_pos, name)
    mujoco.mj_forward(m, d)
    palm_id = m.body("hand/" + _HANDS[hand].palm_body).id
    center_offset = np.array(
        center_override if center_override is not None else c.commands["baoding"].center_offset
    )
    center = d.xpos[palm_id].copy() + center_offset
    for i, name in enumerate(("ball1", "ball2")):
        adr = m.joint(name + "/free").qposadr[0]
        angle = phase + i * np.pi
        d.qpos[adr : adr + 3] = (
            center
            + c.commands["baoding"].orbit_radius * np.array([np.cos(angle), np.sin(angle), 0])
            + [0, 0, c.commands["baoding"].initial_ball_lift]
        )
        d.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
    targets = np.zeros(m.nu)
    for i in range(m.nu):
        trn = m.actuator_trntype[i]
        target_id = m.actuator_trnid[i, 0]
        if trn == mujoco.mjtTrn.mjTRN_JOINT:
            name = m.joint(target_id).name.removeprefix("hand/")
            a = c.actions["joint_pos"]
            targets[i] = (
                lookup(c.scene.entities["hand"].init_state.joint_pos, name)
                if a.use_default_offset or hold_home
                else lookup(a.offset, name)
            )
        elif trn == mujoco.mjtTrn.mjTRN_TENDON:
            name = m.tendon(target_id).name.removeprefix("hand/")
            targets[i] = lookup(c.actions["tendon_pos"].offset, name)
        else:
            raise ValueError(trn)
    contacted = np.zeros(2, bool)
    drop = None
    last = np.zeros((2, 3))
    for step in range(int(seconds / m.opt.timestep)):
        d.ctrl[:] = targets
        if hand == "leap":
            for i in range(m.nu):
                ji = m.actuator_trnid[i, 0]
                name = m.joint(ji).name.removeprefix("hand/")
                ac = next(x for x in LEAP_ACTUATORS if name in x.target_names_expr)
                d.ctrl[i] = np.clip(
                    ac.stiffness * (targets[i] - d.qpos[m.jnt_qposadr[ji]])
                    - ac.damping * d.qvel[m.jnt_dofadr[ji]],
                    -ac.effort_limit,
                    ac.effort_limit,
                )
        mujoco.mj_step(m, d)
        for ct in d.contact:
            names = [m.geom(int(g)).name for g in ct.geom]
            for i in range(2):
                if any(x.startswith(f"ball{i + 1}/") for x in names) and any(
                    x.startswith("hand/") for x in names
                ):
                    contacted[i] = True
        last = np.stack([d.xpos[m.body(f"ball{i + 1}/ball").id] - center for i in range(2)])
        if np.any(last[:, 2] < -c.commands["baoding"].drop_height) or np.any(
            np.linalg.norm(last, axis=1) > c.commands["baoding"].drop_distance
        ):
            drop = round(d.time, 5)
            break
    return {
        "hand": hand,
        "phase": round(phase, 2),
        "home": hold_home,
        "contacted": contacted.tolist(),
        "drop": drop,
        "relative_end": last.round(3).tolist(),
    }


if __name__ == "__main__":
    for hand in _HANDS:
        for phase in (0, np.pi / 2):
            print(run(hand, phase), flush=True)
