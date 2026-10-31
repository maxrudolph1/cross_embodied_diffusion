#!/usr/bin/env python3
"""CPU-only checks of local trainer argument handling and checkpoint retention."""

import json
import tempfile
import unittest
from pathlib import Path

from dexterity_runtime import check_checkpoint_task
from train_dexterity_expert import parse_args, prune_checkpoints, repo_path, warp_cache_path


class TrainerTest(unittest.TestCase):
    def test_pilot_defaults(self):
        args = parse_args(["--task", "Baoding-Allegro", "--output-dir", "outputs/test_new_pilot"])
        self.assertEqual(args.iterations, 100)
        self.assertEqual(args.num_envs, 1024)
        self.assertEqual(args.device, "cuda:0")

    def test_only_prune_numbered_checkpoints_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("model_9.pt", "model_10.pt", "model_100.pt", "model_best.pt", "training.json")
            for name in names:
                (root / name).touch()
            prune_checkpoints(root, 2)
            self.assertEqual({path.name for path in root.iterdir()}, set(names) - {"model_9.pt"})

    def test_paths_stay_repo_relative(self):
        import argparse
        for path in ("/tmp/output", "outputs/../bad"):
            with self.assertRaises(argparse.ArgumentTypeError):
                repo_path(path)

    def test_launcher_cache_reused_across_stage_output_directories(self):
        environment = {"DEXTERITY_CACHE_ROOT": "/tmp/test_task_cache"}
        self.assertEqual(warp_cache_path(Path("outputs/stage_500"), environment),
                         warp_cache_path(Path("outputs/stage_1000"), environment))
        self.assertEqual(warp_cache_path(Path("outputs/stage_500"), {}),
                         Path("outputs/stage_500/warp_cache"))

    def test_checkpoint_mismatch_fails_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training.json").write_text(json.dumps({
                "task": "Baoding-Allegro", "task_signature": "original"
            }))
            checkpoint = root / "model_99.pt"
            check_checkpoint_task(checkpoint, "Baoding-Allegro", "original")
            with self.assertRaisesRegex(ValueError, "changed"):
                check_checkpoint_task(checkpoint, "Baoding-Allegro", "different")
            with self.assertRaisesRegex(ValueError, "does not match"):
                check_checkpoint_task(checkpoint, "Pen-Spinning-Allegro", "original")


if __name__ == "__main__":
    unittest.main()
