"""Nominal world anchors for fixed-hand manipulation tasks.

Command reset runs before mjlab's forward pass, so live body poses can still
contain zeros on the first reset or the previous episode's wrist pose later.
Compute the reference once from the compiled model and configured home state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


def fixed_hand_reference_position(
    env: ManagerBasedRlEnv, hand_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return nominal selected-body position per world, with origins added once.

    The caller must use a fixed hand and deterministic root reset (no randomized
    root pose). Wrist/finger motion does not move this task anchor. This performs
    one small native CPU forward pass at command construction; it neither steps
    nor mutates the live simulator. The returned tensor is safe before first reset.
    """
    hand_asset_cfg.resolve(env.scene)
    hand = env.scene[hand_asset_cfg.name]
    if not hand.is_fixed_base:
        raise ValueError("A fixed-hand task reference requires a fixed-base hand")
    body_ids = hand_asset_cfg.body_ids
    if isinstance(body_ids, slice) or len(body_ids) != 1:
        raise ValueError("Select exactly one hand reference body")
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    joint_addresses = hand.indexing.joint_q_adr.detach().cpu().numpy()
    data.qpos[joint_addresses] = hand.data.default_joint_pos[0].detach().cpu().numpy()
    mocap_id = hand.indexing.mocap_id
    if mocap_id is not None:
        root = hand.data.default_root_state[0].detach().cpu().numpy()
        data.mocap_pos[mocap_id] = root[:3]
        data.mocap_quat[mocap_id] = root[3:7]
    mujoco.mj_forward(model, data)
    body_id = int(hand.indexing.body_ids[body_ids[0]].item())
    nominal = torch.as_tensor(data.xpos[body_id].copy(), dtype=torch.float32, device=env.device)
    return env.scene.env_origins + nominal
