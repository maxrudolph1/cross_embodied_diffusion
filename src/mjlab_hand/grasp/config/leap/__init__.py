from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import leap_grasp_env_cfg
from .rl_cfg import leap_grasp_ppo_runner_cfg

register_mjlab_task(
    task_id="Grasp-LEAP",
    env_cfg=leap_grasp_env_cfg(),
    play_env_cfg=leap_grasp_env_cfg(play=True),
    rl_cfg=leap_grasp_ppo_runner_cfg(),
    runner_cls=None,
)

__all__ = ["leap_grasp_env_cfg", "leap_grasp_ppo_runner_cfg"]
