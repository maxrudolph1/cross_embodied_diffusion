from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import shadow_grasp_env_cfg
from .rl_cfg import shadow_grasp_ppo_runner_cfg

register_mjlab_task(
    task_id="Grasp-Shadow",
    env_cfg=shadow_grasp_env_cfg(),
    play_env_cfg=shadow_grasp_env_cfg(play=True),
    rl_cfg=shadow_grasp_ppo_runner_cfg(),
    runner_cls=None,
)

__all__ = ["shadow_grasp_env_cfg", "shadow_grasp_ppo_runner_cfg"]
