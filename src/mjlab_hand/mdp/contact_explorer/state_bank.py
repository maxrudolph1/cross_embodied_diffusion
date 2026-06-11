"""Learned hash state bank for ContactExplorer contact novelty."""

from __future__ import annotations

import math

import torch
from torch import nn

from mjlab_hand.mdp.contact_explorer.normalization import EmpiricalNormalization, finite_row_mask


@torch.no_grad()
def _bits_to_u64(bits01: torch.Tensor) -> torch.Tensor:
    bits = bits01.to(torch.long)
    k = bits.shape[1]
    shifts = torch.arange(k, device=bits.device, dtype=torch.long)
    return (bits << shifts).sum(dim=1)


def state_id_entropy(state_ids: torch.Tensor, s: int) -> float:
    counts = torch.bincount(state_ids, minlength=s).float()
    probs = counts / counts.sum().clamp_min(1.0)
    ent = -(probs * (probs + 1e-12).log()).sum()
    return (ent / math.log(max(s, 2))).item()


class _HashAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, code_dim: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.code = nn.Linear(hidden_dim, code_dim)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor, noise_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        b = torch.sigmoid(self.code(h))
        if noise_scale > 0:
            n = (torch.rand_like(b) * 2.0 - 1.0) * float(noise_scale)
            b_noisy = (b + n).clamp(0.0, 1.0)
        else:
            b_noisy = b
        x_hat = self.dec(b_noisy)
        return x_hat, b


def infer_simhash_dim(num_key_states: int) -> int:
    s = int(num_key_states)
    if s < 2 or (s & (s - 1)) != 0:
        raise ValueError(f"num_key_states must be a power of two >= 2, got {s}")
    return s.bit_length() - 1


class LearnedHashStateBank:
    def __init__(
        self,
        *,
        num_key_states: int,
        feature_dim: int,
        buffer_size: int,
        num_hand_keypoints: int,
        num_object_bins: int,
        device: torch.device,
        code_dim: int = 256,
        simhash_dim: int | None = None,
        hidden_dim: int = 512,
        noise_scale: float = 0.3,
        lambda_binary: float = 10.0,
        ae_lr: float = 1e-3,
        ae_update_steps: int = 10,
        ae_update_freq: int = 16,
        ae_num_minibatches: int = 8,
        seed: int = 0,
    ) -> None:
        self.S = int(num_key_states)
        self.F = int(feature_dim)
        self.B = int(buffer_size)
        self.L = int(num_hand_keypoints)
        self.K = int(num_object_bins)
        self.device = device

        self.code_dim = int(code_dim)
        self.simhash_dim = (
            int(simhash_dim) if simhash_dim is not None else infer_simhash_dim(self.S)
        )
        if not (1 <= self.simhash_dim <= 64):
            raise ValueError("simhash_dim must be in [1, 64] for u64 packing")

        self.noise_scale = float(noise_scale)
        self.lambda_binary = float(lambda_binary)
        self.ae_update_steps = int(ae_update_steps)
        self.ae_update_freq = int(ae_update_freq)
        self.ae_num_minibatches = int(ae_num_minibatches)

        with torch.inference_mode(False):
            self.counts = torch.zeros(
                (self.S, self.L, self.K), device=self.device, dtype=torch.float32
            )

            self.ae = _HashAE(self.F, int(hidden_dim), self.code_dim).to(self.device)
            self.opt = torch.optim.Adam(self.ae.parameters(), lr=float(ae_lr))

            g = torch.Generator(device=self.device)
            g.manual_seed(int(seed))
            self.A = torch.randn(
                (self.simhash_dim, self.code_dim),
                generator=g,
                device=self.device,
                dtype=torch.float32,
            )

            self.buffer = torch.zeros((self.B, self.F), device=self.device, dtype=torch.float32)
            self.buf_n = torch.zeros((), device=self.device, dtype=torch.long)

            self.normalizer = EmpiricalNormalization(self.F).to(self.device)

            self.last_recon = torch.tensor(0.0, device=self.device)
            self.last_reg = torch.tensor(0.0, device=self.device)
            self.last_entropy = torch.tensor(0.0, device=self.device)

        self.update_enabled = True

    def reset(self, *, reset_counters: bool = False) -> None:
        self.buf_n.zero_()
        if reset_counters:
            self.counts.zero_()

    @torch.no_grad()
    def push(self, feats: torch.Tensor) -> None:
        if not self.update_enabled:
            return
        feats = feats.to(self.device, dtype=torch.float32)
        feats = feats[finite_row_mask(feats)]
        if feats.shape[0] == 0:
            return
        self.normalizer.update(feats)

        start = int(self.buf_n.item())
        end = min(self.B, start + int(feats.shape[0]))
        n_take = end - start
        if n_take > 0:
            self.buffer[start:end] = feats[:n_take]
            self.buf_n.fill_(end)

        if int(self.buf_n.item()) >= self.B:
            with torch.inference_mode(False):
                self._update_autoencoder(self.buffer[: self.B].clone())
            self.buf_n.zero_()

    @torch.no_grad()
    def _state_id_from_feats(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.to(self.device, dtype=torch.float32)
        self.ae.eval()
        _, b = self.ae(x, noise_scale=0.0)
        bits = (b >= 0.5).to(torch.long)
        v = (bits * 2 - 1).to(torch.float32)
        proj = v @ self.A.t()
        sim_bits = (proj >= 0).to(torch.long)
        code_u64 = _bits_to_u64(sim_bits)
        self.last_entropy = torch.tensor(
            state_id_entropy(code_u64.to(torch.long), self.S), device=self.device
        )
        return code_u64.to(torch.long)

    @torch.no_grad()
    def assign(self, feats: torch.Tensor) -> torch.Tensor:
        feats = feats.to(self.device, dtype=torch.float32)
        state_ids = torch.zeros(feats.shape[0], device=self.device, dtype=torch.long)
        valid_rows = finite_row_mask(feats)
        if not valid_rows.any():
            return state_ids
        feats_n = self.normalizer(feats[valid_rows])
        state_ids[valid_rows] = self._state_id_from_feats(feats_n)
        return state_ids

    def _update_autoencoder(self, data: torch.Tensor) -> None:
        self.ae.train()
        data = data[finite_row_mask(data)]
        n = int(data.shape[0])
        if n <= 0:
            return
        x_normalized = self.normalizer(data)
        x_normalized = x_normalized[finite_row_mask(x_normalized)]
        n = int(x_normalized.shape[0])
        if n <= 0:
            return

        num_minibatches = max(1, int(self.ae_num_minibatches))
        chunks = torch.chunk(torch.randperm(n, device=self.device), num_minibatches)
        chunks = [chunk for chunk in chunks if chunk.numel() > 0]
        if not chunks:
            return

        steps = max(1, int(self.ae_update_steps)) * len(chunks)
        recon = reg = torch.tensor(0.0, device=self.device)
        for step in range(steps):
            idx = chunks[step % len(chunks)]
            x = x_normalized.index_select(0, idx).clone()
            x_hat, b = self.ae(x, noise_scale=self.noise_scale)
            recon = torch.mean((x_hat - x) ** 2)
            reg = torch.minimum((1.0 - b) ** 2, b**2).mean()
            loss = recon + self.lambda_binary * reg

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

        self.last_recon = recon.detach()
        self.last_reg = reg.detach()

    @torch.no_grad()
    def add_contacts(
        self,
        *,
        state_ids: torch.Tensor,
        contact_mask: torch.Tensor,
        contact_bins: torch.Tensor,
    ) -> None:
        state_ids = state_ids.to(self.device, dtype=torch.long).clamp(0, self.S - 1)
        contact_mask = contact_mask.to(self.device, dtype=torch.bool)
        contact_bins = contact_bins.to(self.device, dtype=torch.long).clamp(0, self.K - 1)

        if not contact_mask.any():
            return

        env_idx, keypoint_idx = torch.nonzero(contact_mask, as_tuple=True)
        state_idx = state_ids[env_idx]
        bin_idx = contact_bins[env_idx, keypoint_idx]

        lin = ((state_idx * self.L + keypoint_idx) * self.K + bin_idx).to(torch.long)
        binc = torch.bincount(lin, minlength=self.S * self.L * self.K).view(self.S, self.L, self.K)
        self.counts.add_(binc.to(self.counts.dtype))

    def get_metrics(self) -> dict[str, float]:
        return {
            "hash_recon_loss": float(self.last_recon),
            "hash_binary_reg": float(self.last_reg),
            "stateid_entropy": float(self.last_entropy),
        }
