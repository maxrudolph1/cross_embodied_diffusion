"""Diffusion policy: predict action chunks conditioned on obs history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from mjlab_hand.diffusion.model import ConditionalUnet1D
from mjlab_hand.diffusion.normalizer import LinearNormalizer


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.999)


@dataclass
class DiffusionPolicyConfig:
    obs_dim: int
    action_dim: int
    obs_horizon: int = 2
    action_horizon: int = 8
    num_train_timesteps: int = 100
    num_inference_steps: int = 16
    down_dims: tuple[int, ...] = (256, 512, 1024)
    diffusion_step_embed_dim: int = 128


class DiffusionPolicy(torch.nn.Module):
    def __init__(self, cfg: DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_normalizer = LinearNormalizer(
            low=torch.zeros(cfg.obs_dim), high=torch.ones(cfg.obs_dim)
        )
        self.action_normalizer = LinearNormalizer(
            low=torch.zeros(cfg.action_dim), high=torch.ones(cfg.action_dim)
        )
        self.noise_pred_net = ConditionalUnet1D(
            action_dim=cfg.action_dim,
            global_cond_dim=cfg.obs_dim * cfg.obs_horizon,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=cfg.down_dims,
        )

        betas = cosine_beta_schedule(cfg.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Inference schedule (uniform subset of train timesteps).
        steps = torch.linspace(
            0, cfg.num_train_timesteps - 1, cfg.num_inference_steps
        ).long()
        self.register_buffer("inference_timesteps", steps)

    def set_normalizers(self, obs_norm: LinearNormalizer, act_norm: LinearNormalizer) -> None:
        self.obs_normalizer = obs_norm
        self.action_normalizer = act_norm

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def compute_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            obs: (B, To, Do)
            action: (B, Ta, Da)
            timesteps: (B,) optional pre-sampled diffusion timesteps (long).
                For ambient-diffusion training, the timestep must be sampled
                *before* the training tuple -- see
                `DiffusionDataset.sample_ambient_batch` -- so that low-noise
                timesteps aren't undersampled just because few tuples are
                valid there (sampling the tuple first and then a timestep
                conditioned on it dilutes low-t training frequency by
                whatever fraction of the dataset is admitted there, which is
                exactly wrong). When None, timesteps are sampled uniformly
                over the full schedule (original, ungated behaviour, exact).
        """
        b = obs.shape[0]
        device = obs.device
        T = self.cfg.num_train_timesteps
        nobs = self.obs_normalizer.normalize(obs).reshape(b, -1)
        naction = self.action_normalizer.normalize(action)

        noise = torch.randn_like(naction)
        if timesteps is None:
            timesteps = torch.randint(0, T, (b,), device=device, dtype=torch.long)
        else:
            timesteps = timesteps.to(device=device, dtype=torch.long)

        noisy = (
            self._extract(self.sqrt_alphas_cumprod, timesteps, naction.shape) * naction
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, naction.shape)
            * noise
        )
        pred = self.noise_pred_net(noisy, timesteps, nobs)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, To, Do) or (B, Do) — last obs_horizon frames preferred.
        Returns:
            action chunk (B, Ta, Da) in env action scale.
        """
        if obs.ndim == 2:
            obs = obs[:, None, :].expand(-1, self.cfg.obs_horizon, -1)
        elif obs.shape[1] != self.cfg.obs_horizon:
            # Take last To frames or pad.
            if obs.shape[1] > self.cfg.obs_horizon:
                obs = obs[:, -self.cfg.obs_horizon :]
            else:
                pad = obs[:, :1].expand(-1, self.cfg.obs_horizon - obs.shape[1], -1)
                obs = torch.cat([pad, obs], dim=1)

        b = obs.shape[0]
        device = obs.device
        nobs = self.obs_normalizer.normalize(obs).reshape(b, -1)

        x = torch.randn(
            b, self.cfg.action_horizon, self.cfg.action_dim, device=device
        )
        timesteps = self.inference_timesteps.tolist()
        # Reverse diffusion from high noise -> low noise via DDIM (eta=0).
        # `inference_timesteps` is a strided subsequence of the training
        # schedule, so a single-step ancestral update is invalid here (it
        # only holds for a t -> t-1 transition); DDIM uses cumulative alphas
        # at the two *inference* timesteps and is valid for arbitrary
        # strides. See CHANGES.md item 1.
        for i in reversed(range(len(timesteps))):
            t_cur = int(timesteps[i])
            t = torch.full((b,), t_cur, device=device, dtype=torch.long)
            eps = self.noise_pred_net(x, t, nobs)
            alpha_bar_t = self.alphas_cumprod[t_cur]
            if i > 0:
                alpha_bar_prev = self.alphas_cumprod[int(timesteps[i - 1])]
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
            x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0 = x0.clamp(-1.0, 1.0)
            x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps

        return self.action_normalizer.unnormalize(x)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "model": self.state_dict(),
                "obs_normalizer": self.obs_normalizer.state_dict(),
                "action_normalizer": self.action_normalizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "DiffusionPolicy":
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = DiffusionPolicyConfig(**payload["cfg"])
        policy = cls(cfg)
        policy.load_state_dict(payload["model"], strict=False)
        policy.obs_normalizer.load_state_dict(payload["obs_normalizer"])
        policy.action_normalizer.load_state_dict(payload["action_normalizer"])
        return policy.to(device)
