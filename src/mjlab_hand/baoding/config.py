"""Register motor-actuated Baoding tasks for the five existing hands."""

from mjlab.tasks.registry import register_mjlab_task

from mjlab_hand.in_hand_rotation.runner import InHandRotationOnPolicyRunner

from .env_cfg import baoding_ppo_runner_cfg, make_baoding_env_cfg

for hand in ("Allegro", "LEAP", "Shadow", "Sharpa", "Wuji"):
    register_mjlab_task(
        task_id=f"Baoding-{hand}",
        env_cfg=make_baoding_env_cfg(hand),
        play_env_cfg=make_baoding_env_cfg(hand, play=True),
        rl_cfg=baoding_ppo_runner_cfg(hand),
        runner_cls=InHandRotationOnPolicyRunner,
    )
