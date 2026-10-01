#!/usr/bin/env python3
"""Names, sources and sizes of every term-aligned padded store, per-source
ambient t_min vectors, and epoch counts for the ~784k-step budget
(CHANGES.md item 63; re-implementation of Bundle's padded_grid.py).

Configs per family (sources always concatenated in HANDS order):
  scarce<Hand>_K50k / _K10k   target hand at 50k / 10k, the other four at 1M
  all_1M / all_50k / all_10k   every hand at that size

Paths follow this repo's data layout (deviation from Bundle's flat
data/demos/ and data/padded/): single-hand stores under data/mjlab_hand_demos
(1M at the top, subsets_<size>/), padded stores in
data/mjlab_hand_demos/padded_ta/<Family>_pad5_<config>.zarr, so the Vista
sbatch stages them like any other demo store.

  python scripts/padded_grid.py --list
  python scripts/padded_grid.py --family InHand-Rotation --index 0 --build
  python scripts/padded_grid.py --family InHand-Rotation --config scarceLEAP_K50k --build
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab_hand.diffusion.padding import FAMILIES, HANDS  # noqa: E402

DEMOS = Path("data/mjlab_hand_demos")
PADDED = DEMOS / "padded_ta"
TARGET_STEPS = 784_000
BATCH = 256
SIZES = ("1M", "50k", "10k")


def demo_path(family: str, hand: str, size: str) -> Path:
    name = f"{family}-{hand}_expert_{size}.zarr"
    return DEMOS / name if size == "1M" else DEMOS / f"subsets_{size}" / name


def configs(family: str) -> list[str]:
    del family
    return [f"scarce{h}_K{k}" for k in ("50k", "10k") for h in HANDS] + [f"all_{s}" for s in SIZES]


def sources(family: str, config: str) -> list[Path]:
    if config.startswith("scarce"):
        hand, k = config[len("scarce"):].split("_K")
        if hand not in HANDS:
            raise ValueError(f"unknown hand in {config!r}")
        return [demo_path(family, h, k if h == hand else "1M") for h in HANDS]
    if config.startswith("all_"):
        return [demo_path(family, h, config[len("all_"):]) for h in HANDS]
    raise ValueError(f"unknown config {config!r}")


def store_path(family: str, config: str) -> Path:
    return PADDED / f"{family}_pad5_{config}.zarr"


def target_ambient_tmin(target_hand: str, sigma: int) -> list[int]:
    """--ambient-tmin for a pad5 store: one entry per source in HANDS order,
    0 for the target, sigma for the others. A wrong order silently gates the
    target instead (Bundle C 99, 118)."""
    if target_hand not in HANDS:
        raise ValueError(target_hand)
    return [0 if h == target_hand else int(sigma) for h in HANDS]


def epochs_for(n_windows: int, target_steps: int = TARGET_STEPS) -> int:
    """Epochs so that epochs * (n_windows // 256) is ~target_steps."""
    return max(1, round(target_steps / max(1, n_windows // BATCH)))


def n_windows(path: Path, success_only: bool = True) -> int:
    from mjlab_hand.diffusion.dataset import TrajectoryStore

    st = TrajectoryStore(path, mode="r")
    return sum(e - s for s, e, _ in st.episode_slices(success_only=success_only))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--index", type=int)
    ap.add_argument("--config")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    fams = [args.family] if args.family else FAMILIES
    if args.list:
        for fam in fams:
            for i, c in enumerate(configs(fam)):
                p = store_path(fam, c)
                info = f"{n_windows(p):,} windows, {epochs_for(n_windows(p))} epochs" if p.exists() else "not built"
                print(f"{fam} [{i}] {c}: {p} ({info})")
        return
    if not args.family or (args.index is None and not args.config):
        raise SystemExit("need --family and --index or --config (or --list)")
    config = args.config or configs(args.family)[args.index]
    out = store_path(args.family, config)
    srcs = sources(args.family, config)
    print(f"[INFO] {args.family} {config} -> {out}")
    if args.build:
        from build_padded_dataset import build

        out.parent.mkdir(parents=True, exist_ok=True)
        build(srcs, out, overwrite=args.overwrite)
        w = n_windows(out)
        print(f"[INFO] {w:,} windows -> {epochs_for(w)} epochs for ~{TARGET_STEPS:,} steps")


if __name__ == "__main__":
    main()
