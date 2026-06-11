"""Task registration entry point for mjlab_hand."""

from importlib import import_module

_TASK_MODULES = (
    "mjlab_hand.grasp.config.allegro",
    "mjlab_hand.grasp.config.leap",
    "mjlab_hand.grasp.config.shadow",
    "mjlab_hand.grasp.config.sharpa",
    "mjlab_hand.grasp.config.wuji",
    "mjlab_hand.in_hand_rotation.config.allegro",
    "mjlab_hand.in_hand_rotation.config.leap",
    "mjlab_hand.in_hand_rotation.config.shadow",
    "mjlab_hand.in_hand_rotation.config.sharpa",
    "mjlab_hand.in_hand_rotation.config.wuji",
)


def register_tasks() -> None:
    """Import task modules so their mjlab registry side effects run."""
    for module_name in _TASK_MODULES:
        import_module(module_name)


__all__ = ["register_tasks"]
