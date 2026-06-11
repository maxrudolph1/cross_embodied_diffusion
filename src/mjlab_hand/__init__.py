"""mjlab_hand package entry point."""

from mjlab_hand.registration import register_tasks

register_tasks()

__all__ = ["register_tasks"]
