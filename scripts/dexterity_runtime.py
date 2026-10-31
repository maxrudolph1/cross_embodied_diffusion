"""Shared task identity checks for expert checkpoints, without loading saved code."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict
from pathlib import Path


def task_signature(task: str, training_env_cfg) -> str:
    """Hash the canonical training task, ignoring runtime count/seed/camera options.

    Include task source because primitive-object factories capture dimensions in
    closures that a YAML dump cannot represent. This intentionally also rejects
    implementation edits with unchanged observation dimensions.
    """
    import yaml

    data = asdict(training_env_cfg)
    data.pop("seed", None)
    data.pop("viewer", None)
    data["scene"].pop("num_envs", None)
    payload = yaml.dump(data, sort_keys=True).encode()
    digest = hashlib.sha256(task.encode() + b"\n" + payload)
    family = "baoding" if task.startswith("Baoding-") else "pen_spinning"
    root = Path(__file__).resolve().parents[1] / "src" / "mjlab_hand" / family
    for source in sorted(root.rglob("*.py")):
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    shared_reference = root.parent / "mdp" / "fixed_hand_reference.py"
    digest.update(b"mdp/fixed_hand_reference.py")
    digest.update(shared_reference.read_bytes())
    return digest.hexdigest()


def check_checkpoint_task(checkpoint: Path, task: str, signature: str) -> None:
    metadata = checkpoint.parent / "training.json"
    if not metadata.is_file():
        warnings.warn(f"No training.json beside {checkpoint}; cannot verify task compatibility", stacklevel=2)
        return
    saved = json.loads(metadata.read_text())
    if saved.get("task") != task:
        raise ValueError(f"Checkpoint task {saved.get('task')!r} does not match requested {task!r}")
    if "task_signature" not in saved:
        warnings.warn("Legacy training metadata has no task signature; compatibility is unverified", stacklevel=2)
        return
    if saved["task_signature"] != signature:
        raise ValueError(
            "Task settings or task source changed since this checkpoint was trained. "
            "Use the matching code/configuration before evaluation or resume; "
            "loading weights alone would silently change the experiment."
        )
