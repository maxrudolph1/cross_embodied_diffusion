"""Compatibility exports for ContactExplorer reward components."""

from mjlab_hand.mdp.contact_explorer import (
    POST_CONTACT_SCALE,
    PRE_CONTACT_SCALE,
    ContactExplorerRewardManager,
    EmpiricalNormalization,
    LearnedHashStateBank,
    contact_explorer_reset_buffers,
    contact_explorer_reward,
    finite_row_mask,
    get_cached_contact_db_object,
    infer_simhash_dim,
    state_id_entropy,
)

_finite_row_mask = finite_row_mask

__all__ = [
    "ContactExplorerRewardManager",
    "EmpiricalNormalization",
    "LearnedHashStateBank",
    "POST_CONTACT_SCALE",
    "PRE_CONTACT_SCALE",
    "_finite_row_mask",
    "contact_explorer_reward",
    "contact_explorer_reset_buffers",
    "finite_row_mask",
    "get_cached_contact_db_object",
    "infer_simhash_dim",
    "state_id_entropy",
]
