from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import sharpa_grasp_env_cfg
from .rl_cfg import sharpa_grasp_ppo_runner_cfg

register_mjlab_task(
    task_id="Grasp-Sharpa",
    env_cfg=sharpa_grasp_env_cfg(),
    play_env_cfg=sharpa_grasp_env_cfg(play=True),
    rl_cfg=sharpa_grasp_ppo_runner_cfg(),
    runner_cls=None,
)

__all__ = ["sharpa_grasp_env_cfg", "sharpa_grasp_ppo_runner_cfg"]
