from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import allegro_grasp_env_cfg
from .rl_cfg import allegro_grasp_ppo_runner_cfg

register_mjlab_task(
    task_id="Grasp-Allegro",
    env_cfg=allegro_grasp_env_cfg(),
    play_env_cfg=allegro_grasp_env_cfg(play=True),
    rl_cfg=allegro_grasp_ppo_runner_cfg(),
    runner_cls=None,
)

__all__ = ["allegro_grasp_env_cfg", "allegro_grasp_ppo_runner_cfg"]
