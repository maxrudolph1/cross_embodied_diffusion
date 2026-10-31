"""CPU regression: task anchors ignore stale live poses and offset worlds once."""

from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene

from mjlab_hand.baoding.env_cfg import make_baoding_env_cfg
from mjlab_hand.mdp.fixed_hand_reference import fixed_hand_reference_position


class FakeScene(dict):
    env_origins = torch.tensor([[2.0, 3.0, 0.0], [-4.0, 1.0, 0.0]])


def main():
    for name in ("allegro", "leap", "shadow", "sharpa", "wuji"):
        cfg = make_baoding_env_cfg(name)
        model = Scene(cfg.scene, device="cpu").compile()
        reference = mujoco.MjData(model)
        # The scene keyframe independently encodes all entities' authored home
        # states. Compare it to the helper's selected-hand-only reconstruction.
        mujoco.mj_resetDataKeyframe(model, reference, model.key("init_state").id)
        mujoco.mj_forward(model, reference)
        palm = model.body("hand/" + cfg.commands["baoding"].hand_asset_cfg.body_names[0]).id
        qadr = np.array(
            [
                model.jnt_qposadr[i]
                for i in range(model.njnt)
                if model.joint(i).name.startswith("hand/")
            ]
        )
        mocap_id = next(
            model.body_mocapid[i]
            for i in range(model.nbody)
            if model.body(i).name.startswith("hand/") and model.body_mocapid[i] >= 0
        )
        root = cfg.scene.entities["hand"].init_state
        state = SimpleNamespace(
            default_joint_pos=torch.tensor(reference.qpos[qadr][None]),
            default_root_state=torch.tensor([root.pos + root.rot]),
            body_link_pos_w=torch.zeros(2, 1, 3),
        )
        hand = SimpleNamespace(
            is_fixed_base=True,
            data=state,
            indexing=SimpleNamespace(
                joint_q_adr=torch.tensor(qadr),
                mocap_id=int(mocap_id),
                body_ids=torch.tensor([palm]),
            ),
        )
        scene = FakeScene(hand=hand)
        env = SimpleNamespace(scene=scene, sim=SimpleNamespace(mj_model=model), device="cpu")
        asset = SimpleNamespace(name="hand", body_ids=[0], resolve=lambda scene: None)
        actual = fixed_hand_reference_position(env, asset)
        expected = torch.tensor(reference.xpos[palm], dtype=torch.float32) + scene.env_origins
        torch.testing.assert_close(actual, expected)
        # Mimic both the all-zero first reset and a previous episode with wildly
        # different wrist poses: neither is allowed to move the task anchor.
        state.body_link_pos_w[:] = 123.0
        torch.testing.assert_close(fixed_hand_reference_position(env, asset), actual)
        assert (actual[:, 2] > 0.3).all(), name
    print("Fixed-hand anchors: five home keyframes, nonzero origins and stale-pose immunity OK")


if __name__ == "__main__":
    main()
