#!/usr/bin/env bash
# Build each task env once and dump observation/action term names and dims
# to outputs/analysis/schemas.json. The zarr stores hold flat vectors with
# no column names, so without this the analysis scripts cannot say which
# slice of e.g. a 115-d observation is "object position".
#
# Run directly (no sbatch needed, it's a handful of CPU env builds):
#   bash slurm_jobs/dump_schemas.sh
#
# See CHANGES.md item 21.
set -euo pipefail
cd "$(dirname "$0")/.."

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$PWD/warp_cache/dump_schemas}"
mkdir -p "$WARP_CACHE_PATH"

.venv/bin/python - <<'PYEOF'
import json
from pathlib import Path

import mjlab_hand  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

TASKS = [
    "Grasp-Allegro", "Grasp-LEAP", "Grasp-Shadow", "Grasp-Sharpa", "Grasp-Wuji",
    "InHand-Rotation-Allegro", "InHand-Rotation-LEAP", "InHand-Rotation-Shadow",
    "InHand-Rotation-Sharpa", "InHand-Rotation-Wuji",
]

configure_torch_backends()


def obs_term_dims(mgr, group: str = "policy") -> dict[str, int]:
    # Probe attribute names since they differ across mjlab versions.
    for attr in ("group_obs_term_dim", "group_obs_term_dims"):
        d = getattr(mgr, attr, None)
        if d is not None and group in d:
            names = mgr.active_terms[group]
            dims = d[group]
            return {
                name: int(sum(shape) if isinstance(shape, (tuple, list)) else shape)
                for name, shape in zip(names, dims, strict=True)
            }
    raise AttributeError(f"no group_obs_term_dim(s) attribute found on {type(mgr)}")


schemas: dict[str, dict] = {}
for task in TASKS:
    print(f"[INFO] {task}")
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    try:
        obs_terms = obs_term_dims(env.observation_manager)
        action_names = env.action_manager.active_terms
        action_dims = env.action_manager.action_term_dim
        schemas[task] = {
            "obs_terms": list(obs_terms.keys()),
            "obs_term_dims": list(obs_terms.values()),
            "obs_dim": int(sum(obs_terms.values())),
            "action_terms": list(action_names),
            "action_term_dims": list(action_dims),
            "action_dim": int(env.action_manager.total_action_dim),
            "command_terms": list(env.command_manager.active_terms),
        }
    finally:
        env.close()

out = Path("outputs/analysis/schemas.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(schemas, indent=2))
print(f"[INFO] Wrote {out}")
PYEOF
