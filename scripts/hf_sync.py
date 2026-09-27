#!/usr/bin/env python3
"""Mirror the diffusion training demos and their source RL experts to a
private Hugging Face dataset repo, and pull them back on another server.

What goes up:

  * Every `.zarr` under `--demo-root` (the 1M expert sets, `subsets_10k/`,
    `subsets_50k/`, `padded/`), each packed into one uncompressed
    `demos/<relpath>.tar`. Zarr chunks are already compressed; tarring just
    turns ~60k small files into ~45 archives, which the Hub handles far
    better (per-repo / per-folder file-count limits, per-file overhead).
  * The RL expert that produced each 1M set -- read from that zarr's own
    `checkpoint` attribute, not guessed -- stored at its original
    `logs/rsl_rl/...` path together with the run's `params/` (env/agent
    yaml) and `git/` diff. Keeping the path means the zarr's `checkpoint`
    attribute still resolves after a pull.

Not included: BC/diffusion checkpoints, intermediate RL checkpoints, and the
older ad-hoc sets in the repo's own `data/`.

Usage (defaults are the repo-local `data/mjlab_hand_demos` and
`data/hf_staging`; symlink them to bulk storage first):

    python scripts/hf_sync.py stage   # tar zarrs into --stage-dir, write README
    python scripts/hf_sync.py push    # upload staged files missing on the Hub
    python scripts/hf_sync.py pull [--include 'demos/subsets_50k/*' ...]

`push` is resumable: one commit per archive, skipping anything already on
the Hub at the same size. `pull` skips datasets/checkpoints already present
locally. Needs `huggingface_hub` (not a project dependency; install with
`uv pip install --index-url https://pypi.org/simple huggingface_hub`) and
`hf auth login`.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

REPO_ID = "maxrudolph/mjlab-hand-demos"
REPO_TYPE = "dataset"
REPO_ROOT = Path(__file__).resolve().parent.parent
# Repo-local paths; on each machine these are symlinks to wherever the bulk
# storage actually is (see README "Data layout").
DEFAULT_DEMO_ROOT = REPO_ROOT / "data" / "mjlab_hand_demos"
DEFAULT_STAGE_DIR = REPO_ROOT / "data" / "hf_staging"


def find_zarrs(demo_root: Path) -> list[Path]:
    """Top-level zarr stores under demo_root, as paths relative to it."""
    out = []
    for p in sorted(demo_root.rglob("*.zarr")):
        rel = p.relative_to(demo_root)
        if not any(part.endswith(".zarr") for part in rel.parts[:-1]):
            out.append(rel)
    return out


def expert_checkpoints(demo_root: Path) -> dict[str, str]:
    """{dataset name: RL checkpoint path relative to the repo root}, from the
    `checkpoint` attribute each collected (non-derived) zarr records."""
    out = {}
    for rel in find_zarrs(demo_root):
        meta = demo_root / rel / "zarr.json"
        if not meta.exists():
            continue
        ckpt = json.loads(meta.read_text()).get("attributes", {}).get("checkpoint")
        if ckpt and rel.parent == Path("."):
            out[rel.name] = ckpt
    return out


def rl_files(ckpt: str) -> list[Path]:
    """The checkpoint plus its run's params/ and git/ files, repo-relative."""
    ckpt_path = REPO_ROOT / ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"expert checkpoint missing: {ckpt_path}")
    run_dir = ckpt_path.parent
    files = [ckpt_path]
    for sub in ("params", "git"):
        if (run_dir / sub).is_dir():
            files += sorted(f for f in (run_dir / sub).rglob("*") if f.is_file())
    return [f.relative_to(REPO_ROOT) for f in files]


def write_readme(stage_dir: Path, zarrs: list[Path], experts: dict[str, str]):
    lines = [
        "---",
        "license: other",
        "tags: [robotics, dexterous-manipulation, behavior-cloning, diffusion-policy]",
        "---",
        "",
        "# mjlab_hand diffusion training demos",
        "",
        "Expert demonstrations for cross-embodiment diffusion-policy BC on the",
        "`mjlab_hand` grasp / in-hand-rotation benchmark, plus the RL experts that",
        "collected them. Managed by `scripts/hf_sync.py` in the project repo.",
        "",
        "## Layout",
        "",
        "- `demos/<relpath>.zarr.tar` -- one uncompressed tar per zarr store;",
        "  extracts to `data/mjlab_hand_demos/` in the project repo (`hf_sync.py",
        "  pull` does this; that path is usually a symlink to bulk storage).",
        "- `logs/rsl_rl/<experiment>/<run>/` -- source RL expert checkpoint with",
        "  its `params/` (env/agent yaml) and `git/` diff, at the path recorded",
        "  in each dataset's `checkpoint` attribute (relative to the repo root).",
        "",
        "## Datasets",
        "",
    ]
    lines += [f"- `{rel}`" for rel in zarrs]
    lines += ["", "## Provenance (1M sets)", "", "| dataset | RL expert checkpoint |", "|---|---|"]
    lines += [f"| `{name}` | `{ckpt}` |" for name, ckpt in sorted(experts.items())]
    lines += [
        "",
        "`subsets_*` are subsamples of the 1M sets; `padded/` are pooled,",
        "per-source-normalized, zero-padded mixes built by",
        "`scripts/build_padded_dataset.py`.",
        "",
    ]
    (stage_dir / "README.md").write_text("\n".join(lines))


def cmd_stage(args):
    demo_root, stage_dir = args.demo_root, args.stage_dir
    zarrs = find_zarrs(demo_root)
    experts = expert_checkpoints(demo_root)
    for ckpt in experts.values():
        rl_files(ckpt)  # fail early if a source checkpoint is gone
    for rel in zarrs:
        tar_path = stage_dir / "demos" / f"{rel}.tar"
        if tar_path.exists() and not args.force:
            print(f"[stage] exists  {tar_path.relative_to(stage_dir)}")
            continue
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tar_path.with_suffix(".tar.partial")
        subprocess.run(
            ["tar", "-cf", str(tmp), "-C", str(demo_root / rel.parent), rel.name],
            check=True,
        )
        tmp.rename(tar_path)
        print(f"[stage] wrote   {tar_path.relative_to(stage_dir)}")
    write_readme(stage_dir, zarrs, experts)
    print(f"[stage] {len(zarrs)} datasets, {len(experts)} RL experts")


def remote_sizes(api: HfApi) -> dict[str, int]:
    return {
        f.path: f.size
        for f in api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True)
        if hasattr(f, "size")
    }


def cmd_push(args):
    api = HfApi()
    api.create_repo(REPO_ID, repo_type=REPO_TYPE, private=True, exist_ok=True)
    remote = remote_sizes(api)

    def missing(local: Path, path_in_repo: str) -> bool:
        return remote.get(path_in_repo) != local.stat().st_size

    # Small files (README, RL experts) in a single commit.
    small = [(args.stage_dir / "README.md", "README.md")]
    for ckpt in sorted(set(expert_checkpoints(args.demo_root).values())):
        small += [(REPO_ROOT / f, f.as_posix()) for f in rl_files(ckpt)]
    ops = [
        CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(local))
        for local, p in small
        if missing(local, p)
    ]
    if ops:
        api.create_commit(
            REPO_ID,
            repo_type=REPO_TYPE,
            operations=ops,
            commit_message=f"Add README and {len(ops)} RL expert files",
        )
        print(f"[push] committed {len(ops)} small files")

    # Archives one commit each, so an interrupted push resumes where it left off.
    tars = sorted((args.stage_dir / "demos").rglob("*.tar"))
    for i, tar in enumerate(tars, 1):
        path_in_repo = tar.relative_to(args.stage_dir).as_posix()
        if not missing(tar, path_in_repo):
            print(f"[push] {i}/{len(tars)} on hub  {path_in_repo}")
            continue
        print(
            f"[push] {i}/{len(tars)} upload   {path_in_repo} ({tar.stat().st_size / 1e9:.2f} GB)",
            flush=True,
        )
        api.upload_file(
            path_or_fileobj=str(tar),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Add {path_in_repo}",
        )


def cmd_pull(args):
    api = HfApi()
    wanted = [
        p
        for p in remote_sizes(api)
        if not args.include or any(fnmatch.fnmatch(p, g) for g in args.include)
    ]

    def target(p: str) -> Path:
        if p.startswith("demos/"):
            return args.demo_root / p[len("demos/") :].removesuffix(".tar")
        return args.rl_root / p

    todo = [p for p in wanted if p != "README.md" and not target(p).exists()]
    print(f"[pull] {len(wanted)} matched, {len(todo)} not present locally")
    if not todo:
        return
    download_dir = args.demo_root / ".hf_download"
    snapshot_download(REPO_ID, repo_type=REPO_TYPE, local_dir=download_dir, allow_patterns=todo)
    for p in todo:
        src, dst = download_dir / p, target(p)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if p.startswith("demos/"):
            subprocess.run(["tar", "-xf", str(src), "-C", str(dst.parent)], check=True)
            src.unlink()
        else:
            shutil.move(src, dst)
        print(f"[pull] {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["stage", "push", "pull"])
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument(
        "--rl-root", type=Path, default=REPO_ROOT, help="pull: where logs/rsl_rl/... is restored"
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="pull: glob(s) over repo paths, e.g. 'demos/*_1M.zarr.tar'",
    )
    parser.add_argument(
        "--force", action="store_true", help="stage: re-tar even if the archive exists"
    )
    args = parser.parse_args()
    {"stage": cmd_stage, "push": cmd_push, "pull": cmd_pull}[args.command](args)


if __name__ == "__main__":
    main()
