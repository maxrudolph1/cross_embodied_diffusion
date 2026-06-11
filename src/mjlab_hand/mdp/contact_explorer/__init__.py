"""ContactExplorer reward components."""

from mjlab_hand.mdp.contact_explorer.normalization import EmpiricalNormalization, finite_row_mask
from mjlab_hand.mdp.contact_explorer.object_cache import get_cached_contact_db_object
from mjlab_hand.mdp.contact_explorer.reward import (
    POST_CONTACT_SCALE,
    PRE_CONTACT_SCALE,
    ContactExplorerRewardManager,
    contact_explorer_reset_buffers,
    contact_explorer_reward,
)
from mjlab_hand.mdp.contact_explorer.state_bank import (
    LearnedHashStateBank,
    infer_simhash_dim,
    state_id_entropy,
)

__all__ = [
    "ContactExplorerRewardManager",
    "EmpiricalNormalization",
    "LearnedHashStateBank",
    "POST_CONTACT_SCALE",
    "PRE_CONTACT_SCALE",
    "contact_explorer_reset_buffers",
    "contact_explorer_reward",
    "finite_row_mask",
    "get_cached_contact_db_object",
    "infer_simhash_dim",
    "state_id_entropy",
]
