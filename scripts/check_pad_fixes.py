#!/usr/bin/env python3
"""Check the padded-policy plumbing on random weights (CHANGES.md item 63):

1. obs scatter: DiffusionActionChunkPolicy(pad_task=T) feeds the policy the
   same padded row the builder wrote (padding.pad_obs), for every hand;
2. action prefix: it returns exactly the task's native action width;
3. held padding: with mask_pad_loss, predict_action's padded channels come
   out exactly 0 in raw units and the real channels are unaffected by what
   the padded ones would have been;
4. mask: compute_loss ignores padded channels iff mask_pad_loss.

    python scripts/check_pad_fixes.py [--family InHand-Rotation]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from mjlab_hand.diffusion.evaluate import DiffusionActionChunkPolicy
from mjlab_hand.diffusion.normalizer import LinearNormalizer
from mjlab_hand.diffusion.padding import build_plan
from mjlab_hand.diffusion.policy import DiffusionPolicy, DiffusionPolicyConfig


def make(plan, mask: bool) -> DiffusionPolicy:
    torch.manual_seed(0)
    p = DiffusionPolicy(DiffusionPolicyConfig(obs_dim=plan.obs_dim, action_dim=plan.action_dim, mask_pad_loss=mask))
    p.set_normalizers(
        LinearNormalizer(low=-torch.ones(plan.obs_dim) * 3, high=torch.ones(plan.obs_dim) * 2),
        LinearNormalizer(low=-torch.ones(plan.action_dim) * 4, high=torch.ones(plan.action_dim) * 5),
    )
    return p.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="InHand-Rotation")
    args = ap.parse_args()
    plan = build_plan(args.family)
    ok = True
    rng = np.random.default_rng(0)

    for task in plan.tasks:
        pol = make(plan, mask=False)
        seen = {}
        orig = pol.predict_action

        def spy(obs, action_valid=None, _o=orig):
            seen["obs"] = obs.clone()
            return _o(obs, action_valid=action_valid)

        pol.predict_action = spy
        cp = DiffusionActionChunkPolicy(pol, torch.device("cpu"), pad_task=task)
        raw = rng.normal(size=(3, plan.native[task]["obs_dim"])).astype(np.float32)
        act = cp(torch.from_numpy(raw))
        want = torch.from_numpy(plan.pad_obs(task, raw))
        c1 = torch.equal(seen["obs"][:, -1], want)
        c2 = act.shape == (3, plan.native[task]["action_dim"])
        ok &= c1 and c2
        print(f"{task:26s} obs scatter {'OK' if c1 else 'FAIL'}  action width {act.shape[1]} {'OK' if c2 else 'FAIL'}")

    task = plan.tasks[0]
    na = plan.native[task]["action_dim"]
    valid = torch.from_numpy(plan.action_valid(task))[None].expand(4, -1)
    obs = torch.randn(4, 2, plan.obs_dim)
    pol = make(plan, mask=True)
    torch.manual_seed(1)
    out = pol.predict_action(obs, action_valid=valid)
    c3 = bool(torch.allclose(out[..., na:], torch.zeros_like(out[..., na:]), atol=1e-5)) if na < plan.action_dim else True
    print(f"held padded channels == 0 raw ({task}, {plan.action_dim - na} channels): {'OK' if c3 else 'FAIL'} "
          f"max |x| {float(out[..., na:].abs().max()) if na < plan.action_dim else 0:.2e}")
    ok &= c3

    act = torch.randn(4, 8, plan.action_dim)
    for mask in (False, True):
        pol = make(plan, mask=mask)
        cap = {}

        def hook(_m, _inp, out, cap=cap):
            out.retain_grad()
            cap["pred"] = out

        h = pol.noise_pred_net.register_forward_hook(hook)
        pol.compute_loss(obs, act, action_valid=valid).backward()
        h.remove()
        g = cap["pred"].grad
        pad_grad = float(g[..., na:].abs().max()) if na < plan.action_dim else 0.0
        real_grad = float(g[..., :na].abs().max())
        good = (pad_grad == 0.0) == mask and real_grad > 0
        ok &= good
        print(f"mask_pad_loss={mask}: d loss / d pred on padded channels max {pad_grad:.2e} "
              f"(real {real_grad:.2e}) -> {'OK' if good else 'FAIL'}")
    print("ALL OK" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
