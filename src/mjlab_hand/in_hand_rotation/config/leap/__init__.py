from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.in_hand_rotation.runner import InHandRotationOnPolicyRunner

from .env_cfgs import leap_rotation_env_cfg
from .rl_cfg import leap_rotation_ppo_runner_cfg

register_mjlab_task(
    task_id="InHand-Rotation-LEAP",
    env_cfg=leap_rotation_env_cfg(),
    play_env_cfg=leap_rotation_env_cfg(play=True),
    rl_cfg=leap_rotation_ppo_runner_cfg(),
    runner_cls=InHandRotationOnPolicyRunner,
)
