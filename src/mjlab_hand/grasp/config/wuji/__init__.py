from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import wuji_grasp_env_cfg
from .rl_cfg import wuji_grasp_ppo_runner_cfg

register_mjlab_task(
    task_id="Grasp-Wuji",
    env_cfg=wuji_grasp_env_cfg(),
    play_env_cfg=wuji_grasp_env_cfg(play=True),
    rl_cfg=wuji_grasp_ppo_runner_cfg(),
    runner_cls=None,
)

__all__ = ["wuji_grasp_env_cfg", "wuji_grasp_ppo_runner_cfg"]
