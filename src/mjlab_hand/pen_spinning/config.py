"""Register Pen-Spinning tasks without modifying the original rotation tasks."""
from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.in_hand_rotation.runner import InHandRotationOnPolicyRunner

from .env_cfg import HANDS, make_pen_spinning_env_cfg, pen_spinning_ppo_runner_cfg

for hand in HANDS:
    register_mjlab_task(
        task_id=f"Pen-Spinning-{hand}",
        env_cfg=make_pen_spinning_env_cfg(hand),
        play_env_cfg=make_pen_spinning_env_cfg(hand, play=True),
        rl_cfg=pen_spinning_ppo_runner_cfg(hand),
        runner_cls=InHandRotationOnPolicyRunner,
    )
