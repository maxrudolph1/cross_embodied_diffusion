#!/usr/bin/env python3
"""CPU checks for checkpoint recovery after a bootstrap time limit."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from run_dexterity_batch_node import resume_checkpoint


class CheckpointRecoveryTest(unittest.TestCase):
    def test_latest_truncated_archive_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "stage_500"
            stage.mkdir()
            previous = stage / "model_250.pt"
            latest = stage / "model_499.pt"
            with zipfile.ZipFile(previous, "w") as archive:
                archive.writestr("model_250/data.pkl", b"checkpoint fixture")
            latest.write_bytes(previous.read_bytes()[:-12])
            self.assertEqual(resume_checkpoint(Path(directory)), previous)

    def test_crc_corruption_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "stage_500"
            stage.mkdir()
            previous = stage / "model_250.pt"
            latest = stage / "model_499.pt"
            with zipfile.ZipFile(previous, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("model_250/data.pkl", b"checkpoint fixture")
            latest.write_bytes(previous.read_bytes().replace(b"checkpoint fixture", b"checkpoint corrupt"))
            self.assertEqual(resume_checkpoint(Path(directory)), previous)

    def test_no_intact_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "stage_500"
            stage.mkdir()
            (stage / "model_499.pt").write_bytes(b"interrupted torch.save")
            with self.assertRaisesRegex(FileNotFoundError, "No intact bootstrap checkpoint"):
                resume_checkpoint(Path(directory))


if __name__ == "__main__":
    unittest.main()
