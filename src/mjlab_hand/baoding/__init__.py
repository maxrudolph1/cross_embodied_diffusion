"""Two-ball, goal-conditioned Baoding manipulation."""

from .env_cfg import baoding_ppo_runner_cfg, make_baoding_env_cfg

__all__ = ["make_baoding_env_cfg", "baoding_ppo_runner_cfg"]
