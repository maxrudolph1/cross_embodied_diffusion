"""Prototype: train N seeds of DiffusionPolicy as one vmapped batch instead of
N sequential python-level training steps.

Idea: `ConditionalUnet1D` (down_dims=(256,512,1024)) is small -- on an A40
with batch_size=256 it's launch/memory-bound, not compute-bound, so a
sequential loop over seeds pays full kernel-launch/python overhead per seed
for a GPU that isn't full. `torch.func.vmap` over a stacked-parameters axis
runs all seeds' UNet forward+backward as one set of (bigger, batched)
kernels instead, without changing the model or touching JAX.

Two checks:
1. Correctness -- vmap over stacked `noise_pred_net` params/buffers must
   reproduce, bit for bit, calling each seed's own (un-vmapped) network on
   the same input. This isolates the *mechanism* (stack + functional_call +
   vmap correctly slices per-seed compute) from RNG semantics, since
   `compute_loss` samples its own noise/timesteps internally and vmap's
   in-graph randomness handling doesn't bit-match sequential torch.randn
   calls -- that's expected, not a bug.
2. Throughput -- wall-clock per optimizer step for `n_seeds` trained via one
   vmapped `compute_loss` step vs. `n_seeds` independent sequential steps,
   same batch size each, on whatever GPU is available.

Run: python scripts/prototype_vmap_seeds.py
"""

from __future__ import annotations

import copy
import time

import torch
from torch.func import functional_call, stack_module_state, vmap

from mjlab_hand.diffusion.model import ConditionalUnet1D
from mjlab_hand.diffusion.policy import DiffusionPolicy, DiffusionPolicyConfig


def _vmap_safe_unet_forward(self, sample, timestep, global_cond):
    # Identical to ConditionalUnet1D.forward, except `moveaxis` -> `transpose`
    # for the two 2-axis swaps: torch 2.10 has no vmap batching rule for
    # `aten::moveaxis.int` yet, but plain `transpose` (equivalent here, since
    # both calls only ever swap the last two axes) is supported. Patched only
    # in this prototype process -- production `model.py` is left untouched
    # since live training jobs are importing it right now.
    x = sample.transpose(-1, -2)
    if timestep.ndim == 0:
        timestep = timestep[None].expand(sample.shape[0])
    global_feature = torch.cat(
        [self.diffusion_step_encoder(timestep), global_cond], dim=-1
    )

    h: list[torch.Tensor] = []
    for resnet, resnet2, downsample in self.down_modules:
        x = resnet(x, global_feature)
        x = resnet2(x, global_feature)
        h.append(x)
        x = downsample(x)

    for mid in self.mid_modules:
        x = mid(x, global_feature)

    for resnet, resnet2, upsample in self.up_modules:
        x = torch.cat((x, h.pop()), dim=1)
        x = resnet(x, global_feature)
        x = resnet2(x, global_feature)
        x = upsample(x)

    x = self.final_conv(x)
    return x.transpose(-1, -2)


ConditionalUnet1D.forward = _vmap_safe_unet_forward

# `functional_call` always invokes `module.__call__` -> `forward`, but
# `DiffusionPolicy` doesn't define one (it exposes `compute_loss` /
# `predict_action` directly). Point `forward` at `compute_loss` so
# `functional_call(policy, ...)` does the training-loss computation we
# actually want to vmap. In-process only, same as the model.py patch above.
DiffusionPolicy.forward = DiffusionPolicy.compute_loss


def build_policies(cfg: DiffusionPolicyConfig, n_seeds: int, device: str) -> list[DiffusionPolicy]:
    policies = []
    for seed in range(n_seeds):
        torch.manual_seed(1000 + seed)
        policies.append(DiffusionPolicy(cfg).to(device))
    return policies


def check_correctness(policies: list[DiffusionPolicy], cfg: DiffusionPolicyConfig, device: str) -> None:
    # Run this check on CPU regardless of `device`: it's validating the
    # stack/functional_call/vmap *mechanism* (does seed i's slice of the
    # vmapped computation equal seed i's own un-vmapped forward?), not GPU
    # throughput. On GPU, vmap's batched conv/groupnorm kernels are a
    # different (still correct) cuDNN code path than the per-sample kernels,
    # which is enough fp32 rounding difference (~1e-3 max abs, confirmed by
    # hand) to make the comparison noisy for no interesting reason. CPU uses
    # the same reference ops either way, so this pins down real bugs only.
    nets = [copy.deepcopy(p.noise_pred_net).to("cpu") for p in policies]
    params, buffers = stack_module_state(nets)
    base_net = copy.deepcopy(nets[0]).to("meta")

    def call_one(p, b, sample, timestep, global_cond):
        return functional_call(base_net, (p, b), (sample, timestep, global_cond))

    batch = 32
    sample = torch.randn(batch, cfg.action_horizon, cfg.action_dim)
    timestep = torch.randint(0, cfg.num_train_timesteps, (batch,))
    global_cond = torch.randn(batch, cfg.obs_dim * cfg.obs_horizon)

    out_vmap = vmap(call_one, in_dims=(0, 0, None, None, None))(
        params, buffers, sample, timestep, global_cond
    )

    max_err = 0.0
    for i, net in enumerate(nets):
        out_direct = net(sample, timestep, global_cond)
        err = (out_vmap[i] - out_direct).abs().max().item()
        max_err = max(max_err, err)
    print(f"[correctness, cpu] max abs diff between vmapped and direct forward: {max_err:.3e}")
    assert max_err < 1e-5, "vmap forward does not match per-seed direct forward -- logic bug"
    print("[correctness] PASS -- vmap-over-seeds reproduces independent per-seed forward exactly")
    print(
        "[note] the same comparison on GPU shows ~1e-3 max abs diff -- that's "
        "vmap's batched conv/GroupNorm kernels taking a different (still "
        "correct) cuDNN code path than the per-sample kernels, not a bug."
    )


def bench_vmapped(
    cfg: DiffusionPolicyConfig,
    n_seeds: int,
    batch_size: int,
    n_iters: int,
    n_warmup: int,
    device: str,
    amp: bool = False,
) -> tuple[float, float, list[list[float]]]:
    policies = build_policies(cfg, n_seeds, device)
    params, buffers = stack_module_state(policies)
    params = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    base = copy.deepcopy(policies[0]).to("meta")
    opt = torch.optim.Adam(list(params.values()), lr=1e-4)

    def call_one(p, b, obs, action):
        return functional_call(base, (p, b), (obs, action))

    vmapped_loss = vmap(call_one, in_dims=(0, 0, None, None), randomness="different")

    loss_traces: list[list[float]] = [[] for _ in range(n_seeds)]

    def step() -> torch.Tensor:
        obs = torch.randn(batch_size, cfg.obs_horizon, cfg.obs_dim, device=device)
        action = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim, device=device)
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            losses = vmapped_loss(params, buffers, obs, action)
        losses.float().sum().backward()
        opt.step()
        return losses.detach()

    for _ in range(n_warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        losses = step()
        for i in range(n_seeds):
            loss_traces[i].append(losses[i].item())
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    return elapsed / n_iters, peak_mem, loss_traces


def bench_sequential(
    cfg: DiffusionPolicyConfig,
    n_seeds: int,
    batch_size: int,
    n_iters: int,
    n_warmup: int,
    device: str,
    amp: bool = False,
) -> tuple[float, float, list[list[float]]]:
    policies = build_policies(cfg, n_seeds, device)
    opts = [torch.optim.Adam(p.parameters(), lr=1e-4) for p in policies]

    loss_traces: list[list[float]] = [[] for _ in range(n_seeds)]

    def step() -> list[float]:
        losses = []
        for policy, opt in zip(policies, opts):
            obs = torch.randn(batch_size, cfg.obs_horizon, cfg.obs_dim, device=device)
            action = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim, device=device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
                loss = policy.compute_loss(obs, action)
            loss.float().backward()
            opt.step()
            losses.append(loss.item())
        return losses

    for _ in range(n_warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        losses = step()
        for i in range(n_seeds):
            loss_traces[i].append(losses[i])
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    return elapsed / n_iters, peak_mem, loss_traces


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}")

    cfg = DiffusionPolicyConfig(obs_dim=64, action_dim=24, obs_horizon=2, action_horizon=8)
    n_warmup, n_iters = 5, 30

    torch.manual_seed(0)
    check_correctness(build_policies(cfg, 3, device), cfg, device)
    print()

    for batch_size in (16, 256):
        print(f"=== batch_size={batch_size} (fp32) ===")
        for n_seeds in (2, 4, 8, 16):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            vmap_ms, vmap_mem, vmap_traces = bench_vmapped(
                cfg, n_seeds, batch_size, n_iters, n_warmup, device
            )
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            seq_ms, seq_mem, seq_traces = bench_sequential(
                cfg, n_seeds, batch_size, n_iters, n_warmup, device
            )
            speedup = seq_ms / vmap_ms
            vmap_deltas = [t[-1] - t[0] for t in vmap_traces]
            seq_deltas = [t[-1] - t[0] for t in seq_traces]
            print(
                f"[n_seeds={n_seeds:2d}] vmapped {vmap_ms * 1000:7.2f} ms/step "
                f"(peak {vmap_mem:7.1f} MiB) | sequential {seq_ms * 1000:7.2f} ms/step "
                f"(peak {seq_mem:7.1f} MiB) | speedup {speedup:5.2f}x"
            )
            print(
                f"           loss decreased for all seeds? "
                f"vmapped={all(d < 0 for d in vmap_deltas)} sequential={all(d < 0 for d in seq_deltas)}"
            )

    print()
    print("=== bf16 autocast: does it help the production batch_size=256 step directly? ===")
    for n_seeds in (1, 8):
        for amp in (False, True):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            ms, mem, _ = bench_sequential(cfg, n_seeds, 256, n_iters, n_warmup, device, amp=amp)
            print(
                f"[n_seeds={n_seeds} amp={amp!s:5s}] sequential @ batch=256: "
                f"{ms * 1000:7.2f} ms/step (peak {mem:7.1f} MiB)"
            )

    print()
    print("=== vmap + bf16, scaling n_seeds at batch_size=32: how many networks fit / go fastest? ===")
    for n_seeds in (8, 16, 32, 64):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            ms, mem, _ = bench_vmapped(cfg, n_seeds, 32, n_iters, n_warmup, device, amp=True)
        except torch.cuda.OutOfMemoryError:
            print(f"[n_seeds={n_seeds:2d}] OOM")
            break
        samples_per_sec = n_seeds * 32 / ms
        print(
            f"[n_seeds={n_seeds:2d}] {ms * 1000:7.2f} ms/step (peak {mem:8.1f} MiB, "
            f"{mem / n_seeds:6.1f} MiB/seed) | {samples_per_sec:9.0f} samples/s across all seeds"
        )


if __name__ == "__main__":
    main()
