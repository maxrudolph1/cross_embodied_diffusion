#!/usr/bin/env python3
"""Build the frozen per-family normalizer artifact (--norm-mode frozen;
mjlab_hand.diffusion.frozen_norm). CHANGES.md item 63.

Default sources: the five 1M single-hand stores of the family (Bundle used
10M stores, which are not on Vista). Default --stat minmax: what every Bundle
run used (Bundle's own script defaulted to zscore).

    python scripts/build_family_normalizer.py --family InHand-Rotation
      -> outputs/analysis/norm_rotation_minmax.json (+ tracked copy in configs/)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mjlab_hand.diffusion.frozen_norm import build_artifact
from mjlab_hand.diffusion.padding import FAMILIES, HANDS

SHORT = {"Grasp": "grasp", "InHand-Rotation": "rotation"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=FAMILIES, required=True)
    ap.add_argument("--stat", choices=["minmax", "zscore"], default="minmax")
    ap.add_argument("--sources", type=Path, nargs="*")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    srcs = args.sources or [Path(f"data/mjlab_hand_demos/{args.family}-{h}_expert_1M.zarr") for h in HANDS]
    out = args.out or Path(f"outputs/analysis/norm_{SHORT[args.family]}_{args.stat}.json")
    art = build_artifact(args.family, srcs, args.stat)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=1))
    tracked = Path("configs") / out.name
    shutil.copyfile(out, tracked)
    print(f"[INFO] {args.family} {args.stat} digest {art['digest']} -> {out} (+ {tracked})")


if __name__ == "__main__":
    main()
