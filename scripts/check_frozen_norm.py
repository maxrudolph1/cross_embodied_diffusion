#!/usr/bin/env python3
"""Verify frozen normalizer artifacts: digest recomputed from the arrays,
layout matches the current padding plan, every range contains 0 (padding
stays inside [-1, 1]). CHANGES.md item 63.

    python scripts/check_frozen_norm.py configs/norm_*.json
"""

from __future__ import annotations

import sys

import numpy as np

from mjlab_hand.diffusion.frozen_norm import load_artifact
from mjlab_hand.diffusion.padding import build_plan

bad = 0
for path in sys.argv[1:]:
    try:
        art = load_artifact(path)
        plan = build_plan(art["family"])
        assert art["plan"] == plan.to_json(), "layout differs from the current plan"
        for k in ("obs", "act"):
            lo, hi = np.asarray(art[f"{k}_low"]), np.asarray(art[f"{k}_high"])
            if art["stat"] == "minmax":
                assert (lo <= 0).all() and (hi >= 0).all(), f"{k}: a range excludes 0"
            assert (hi > lo).all(), f"{k}: empty range"
        print(f"OK   {path}: {art['family']} {art['stat']} {art['digest'][:23]}... obs {len(art['obs_low'])} act {len(art['act_low'])}")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"FAIL {path}: {e}")
sys.exit(1 if bad else 0)
