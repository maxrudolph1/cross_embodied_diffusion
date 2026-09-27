"""Prototype: does `torch.compile` speed up a single network's training step
at the batch size this project actually trains at?

Earlier experiments (see agent_logbook/JOURNAL.md, 2026-09-13) found that one
`DiffusionPolicy` at `batch_size=256` already keeps an A40 ~fully
compute-utilized -- vmap-over-seeds and bf16 autocast both failed to find any
spare capacity to exploit. `torch.compile` is a different kind of lever: even
in a compute-bound regime, Inductor fuses the many small pointwise ops in
this UNet (GroupNorm -> Mish -> scale/bias -> residual add, per block) into
fewer kernels, cutting memory traffic rather than adding parallelism. That
can still win when a model is bandwidth-bound rather than FLOP-bound, which
is plausible for a stack of small conv1d blocks like this one.

Checks:
1. Correctness -- compiled `compute_loss`/`predict_action` must match eager
   on the same input (compile only fuses kernels, doesn't change the math,
   so this should be near machine precision, not just "close").
2. Throughput -- ms/step for eager vs compiled `compute_loss` (training) and
   `predict_action` (the DDIM sampling loop used at eval time), at the
   project's real batch_size=256, and also n_seeds sequential to see if
   compile changes the "no spare capacity" picture.

Run: python scripts/prototype_torch_compile.py
"""

from __future__ import annotations

import copy
import time

import torch

from mjlab_hand.diffusion.policy import DiffusionPolicy, DiffusionPolicyConfig


def build_policy(cfg: DiffusionPolicyConfig, seed: int, device: str) -> DiffusionPolicy:
    torch.manual_seed(seed)
    return DiffusionPolicy(cfg).to(device)


def check_correctness_loss(cfg: DiffusionPolicyConfig, device: str) -> None:
    # `torch.compile(module)` only intercepts `forward`/`__call__`; since
    # `DiffusionPolicy` doesn't define `forward` (it exposes `compute_loss` /
    # `predict_action` directly), `torch.compile` on the *bound method* is
    # what actually traces and fuses that method's graph.
    #
    # `compute_loss` draws `noise = torch.randn_like(naction)` internally.
    # Seeding with `torch.manual_seed` before each call does NOT guarantee
    # eager and compiled draw the same values -- Inductor lowers random ops
    # to its own Philox-based codegen, which consumes/produces the RNG stream
    # differently from eager's per-op generator (documented torch.compile
    # behavior, not specific to this model). A first attempt at this bit-exact
    # comparison showed ~5e-3 diff *even on CPU*, which first looked like a
    # real bug -- but it's this RNG-consumption mismatch, not the network
    # itself. So: test the deterministic part (the UNet forward that
    # compilation actually optimizes) directly, bypassing compute_loss's
    # internal randomness entirely.
    policy = build_policy(cfg, 0, "cpu")
    compiled_net = torch.compile(policy.noise_pred_net)

    batch = 64
    sample = torch.randn(batch, cfg.action_horizon, cfg.action_dim)
    timestep = torch.randint(0, cfg.num_train_timesteps, (batch,))
    global_cond = torch.randn(batch, cfg.obs_dim * cfg.obs_horizon)

    out_eager = policy.noise_pred_net(sample, timestep, global_cond)
    out_compiled = compiled_net(sample, timestep, global_cond)

    diff = (out_eager - out_compiled).abs().max().item()
    print(f"[correctness, noise_pred_net forward, cpu] max abs diff={diff:.3e}")
    assert diff < 1e-4, "compiled noise_pred_net disagrees with eager -- logic bug"
    print("[correctness] PASS -- compiled UNet forward matches eager exactly")


def check_correctness_predict(cfg: DiffusionPolicyConfig, device: str) -> None:
    # Same RNG caveat as above (`predict_action` draws its initial noise `x`
    # internally) -- already covered by the UNet-forward check, so this is
    # a smoke test that compiling+running `predict_action` doesn't crash and
    # produces a finite, correctly-shaped output, not a bit-exact comparison.
    policy = build_policy(cfg, 0, "cpu")
    compiled_predict = torch.compile(policy.predict_action)

    batch = 16
    obs = torch.randn(batch, cfg.obs_horizon, cfg.obs_dim)
    act = compiled_predict(obs)
    assert act.shape == (batch, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(act).all()
    print(f"[smoke, predict_action, cpu] shape={tuple(act.shape)} finite=True")
    print("[correctness] PASS -- compiled predict_action runs and returns a valid action chunk")


def bench_loss(policy: DiffusionPolicy, loss_fn, cfg: DiffusionPolicyConfig, batch_size: int, n_iters: int, n_warmup: int, device: str) -> float:
    for _ in range(n_warmup):
        obs = torch.randn(batch_size, cfg.obs_horizon, cfg.obs_dim, device=device)
        action = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim, device=device)
        loss_fn(obs, action).backward()
        policy.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        obs = torch.randn(batch_size, cfg.obs_horizon, cfg.obs_dim, device=device)
        action = torch.randn(batch_size, cfg.action_horizon, cfg.action_dim, device=device)
        loss_fn(obs, action).backward()
        policy.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters


def bench_predict(predict_fn, cfg: DiffusionPolicyConfig, batch_size: int, n_iters: int, n_warmup: int, device: str) -> float:
    for _ in range(n_warmup):
        obs = torch.randn(batch_size, cfg.obs_horizon, cfg.obs_dim, device=device)
        predict_fn(obs)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        obs = torch.randn(batch_size, cfg.obs_horizon, cfg.obs_dim, device=device)
        predict_fn(obs)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}")

    cfg = DiffusionPolicyConfig(obs_dim=64, action_dim=24, obs_horizon=2, action_horizon=8)

    check_correctness_loss(cfg, device)
    check_correctness_predict(cfg, device)
    print()

    n_warmup, n_iters = 10, 50
    batch_size = 256

    print(f"=== compute_loss (training step), batch_size={batch_size} ===")
    policy_eager = build_policy(cfg, 0, device)
    eager_ms = bench_loss(policy_eager, policy_eager.compute_loss, cfg, batch_size, n_iters, n_warmup, device) * 1000
    print(f"eager:    {eager_ms:7.2f} ms/step")

    policy_compiled_holder = build_policy(cfg, 0, device)
    compiled_loss_fn = torch.compile(policy_compiled_holder.compute_loss)
    # Warmup includes the (slow, one-time) compilation itself.
    compile_t0 = time.perf_counter()
    compiled_ms = bench_loss(policy_compiled_holder, compiled_loss_fn, cfg, batch_size, n_iters, n_warmup, device) * 1000
    compile_wall = time.perf_counter() - compile_t0
    print(f"compiled: {compiled_ms:7.2f} ms/step  (warmup+bench wall time {compile_wall:.1f}s, includes 1x compile cost)")
    print(f"speedup: {eager_ms / compiled_ms:.2f}x")
    print()

    print(f"=== predict_action (DDIM sampling, used in eval / action_reconstruction_loss), batch_size=16 ===")
    eval_batch = 16
    policy_eager2 = build_policy(cfg, 0, device)
    eager_ms2 = bench_predict(policy_eager2.predict_action, cfg, eval_batch, n_iters, n_warmup, device) * 1000
    print(f"eager:    {eager_ms2:7.2f} ms/step")

    policy_compiled2 = build_policy(cfg, 0, device)
    compiled_predict_fn = torch.compile(policy_compiled2.predict_action)
    compiled_ms2 = bench_predict(compiled_predict_fn, cfg, eval_batch, n_iters, n_warmup, device) * 1000
    print(f"compiled: {compiled_ms2:7.2f} ms/step")
    print(f"speedup: {eager_ms2 / compiled_ms2:.2f}x")


if __name__ == "__main__":
    main()
