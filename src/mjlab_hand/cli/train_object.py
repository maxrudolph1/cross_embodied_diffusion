#!/usr/bin/env python
"""Wrapper around mjlab's train script with --object and --scale CLI args."""

import argparse
import sys

from mjlab_hand.object.patch import parse_scale, patch_object


def main() -> None:
    # Pre-parse --object and --scale before passing everything else to mjlab train.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="ContactDB object name (e.g. mug, bowl, cube_medium, light_bulb)",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Object scale as comma-separated values (e.g. 1.0,1.0,1.0 or 0.8,0.8,0.8)",
    )
    known, remaining = parser.parse_known_args()

    # Determine task ID the same way mjlab does (first positional arg).
    task_id = None
    for arg in remaining:
        if not arg.startswith("-"):
            task_id = arg
            break

    if known.object is not None and task_id is not None:
        from mjlab.tasks.registry import _REGISTRY

        import mjlab_hand  # noqa: F401

        scale = parse_scale(known.scale)

        if task_id in _REGISTRY:
            patch_object(_REGISTRY[task_id].env_cfg, known.object, scale)
            patch_object(_REGISTRY[task_id].play_env_cfg, known.object, scale)

    # Reconstruct sys.argv so mjlab train sees only the args it understands.
    sys.argv = ["train"] + remaining

    from mjlab.scripts.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
