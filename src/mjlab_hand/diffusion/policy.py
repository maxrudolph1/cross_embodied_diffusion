"""Diffusion policy: predict action chunks conditioned on obs history."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from mjlab_hand.diffusion.model import ConditionalUnet1D
from mjlab_hand.diffusion.normalizer import GaussianNormalizer, LinearNormalizer


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
    # "linear" (default): fit a LinearNormalizer from the training dataset as
    # before. "gaussian": obs/action normalizers are GaussianNormalizer
    # identities (mean=0, std=1) -- for the cross-embodiment padded scheme,
    # where each source's data is already normalized (mean 0, var 1, fit
    # per-source, statically) and zero-padded by `build_padded_dataset.py`
    # before it ever reaches this class; normalizing again here (e.g. against
    # the *pooled* mixture, or per-source stats a batch-mixed forward pass
    # has no way to apply per-row) would be wrong. See `train_diffusion.py`.
    normalizer_type: str = "linear"


# Clamp for gaussian-normalized checkpoints saved before CHANGES.md item 58.
LEGACY_GAUSSIAN_CLIP = 5.0


class DiffusionPolicy(torch.nn.Module):
    def __init__(self, cfg: DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.normalizer_type == "gaussian":
            self.obs_normalizer: LinearNormalizer | GaussianNormalizer = GaussianNormalizer.identity(
                cfg.obs_dim
            )
            self.action_normalizer: LinearNormalizer | GaussianNormalizer = (
                GaussianNormalizer.identity(cfg.action_dim)
            )
        else:
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

        # Per-dim bounds the sampler clamps its x0 estimate to, in normalized
        # action space (CHANGES.md item 58). +-1 is exactly the training data's
        # range under LinearNormalizer. A GaussianNormalizer (padded scheme)
        # puts ~16% of values outside +-1 (up to ~13), so train.py sets these
        # to the training data's per-dim min/max for those runs; clamping
        # them to +-1 truncated every executed action to mean +- 1 std.
        self.register_buffer("action_clip_low", -torch.ones(cfg.action_dim))
        self.register_buffer("action_clip_high", torch.ones(cfg.action_dim))

    def set_normalizers(
        self,
        obs_norm: LinearNormalizer | GaussianNormalizer,
        act_norm: LinearNormalizer | GaussianNormalizer,
    ) -> None:
        self.obs_normalizer = obs_norm
        self.action_normalizer = act_norm

    def set_action_clip(self, low: torch.Tensor, high: torch.Tensor) -> None:
        self.action_clip_low.copy_(torch.as_tensor(low, dtype=torch.float32))
        self.action_clip_high.copy_(torch.as_tensor(high, dtype=torch.float32))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def compute_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
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
            action_mask: (B, Da) optional 0/1 mask, 1 where `action`'s last
                dim is a real (non-padded) value for that row's source
                embodiment. Cross-embodiment padded datasets carry a
                different real action_dim per source (see
                `TrajectoryStore.source_real_dims`); the zero-padded tail has
                no ground truth to regress and must not contribute to the
                loss, so it's excluded here rather than trained toward zero
                (which would bias the shared denoising net for no reason).
                Broadcasts over the action horizon. When None, every element
                contributes (plain, unmasked MSE -- unchanged behaviour for
                every existing single-embodiment or onehot-mixed dataset).
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
        if action_mask is None:
            return F.mse_loss(pred, noise)
        mask = action_mask.to(device=device, dtype=pred.dtype)[:, None, :].expand_as(pred)
        sq_err = (pred - noise) ** 2
        return (sq_err * mask).sum() / mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def action_reconstruction_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Validation metric: MSE between the fully denoised predicted
        action and the ground-truth action, both in real (unnormalized)
        action units.

        Unlike `compute_loss` (the training objective), which only checks
        one-step noise prediction at a random timestep and never produces an
        actual action, this runs the same DDIM reverse-diffusion sampling
        used at inference (`predict_action`) end-to-end and compares its
        output directly to the expert action -- the quantity a BC policy is
        actually judged on. `action` must be in the same units `dataset`
        yields (i.e. not pre-normalized by the caller); `predict_action`
        already unnormalizes its output to match. `action_mask` excludes
        zero-padded cross-embodiment dims, same convention as `compute_loss`.
        """
        pred = self.predict_action(obs)
        sq_err = (pred - action) ** 2
        if action_mask is None:
            return sq_err.mean()
        mask = action_mask.to(device=pred.device, dtype=pred.dtype)[:, None, :].expand_as(pred)
        return (sq_err * mask).sum() / mask.sum().clamp_min(1.0)

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
            x0 = torch.maximum(torch.minimum(x0, self.action_clip_high), self.action_clip_low)
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
        if cfg.normalizer_type == "gaussian" and "action_clip_low" not in payload["model"]:
            # Saved before CHANGES.md item 58: no stored range. +-LEGACY_CLIP
            # covers >99.6% of normalized training actions.
            print(
                f"[WARN] {path}: gaussian-normalized checkpoint without a stored action "
                f"clip range; clamping sampled actions to +-{LEGACY_GAUSSIAN_CLIP}"
            )
            policy.set_action_clip(
                -LEGACY_GAUSSIAN_CLIP * torch.ones(cfg.action_dim),
                LEGACY_GAUSSIAN_CLIP * torch.ones(cfg.action_dim),
            )
        policy.obs_normalizer.load_state_dict(payload["obs_normalizer"])
        policy.action_normalizer.load_state_dict(payload["action_normalizer"])
        return policy.to(device)
