#!/usr/bin/env python3
"""CPU-only staged-workflow regression checks with fake train/eval subprocesses."""

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_dexterity_local as workflow


class LocalWorkflowTest(unittest.TestCase):
    def test_cumulative_stages_resume_and_evaluate(self):
        commands = []

        def fake_execute(command, log_path, environment):
            commands.append(command)
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True)
            if "train_dexterity_expert.py" in command[1]:
                count = int(output.name.split("_")[-1])
                (output / "training.json").write_text(json.dumps({
                    "status": "complete", "final_iteration": count - 1,
                }))
                (output / f"model_{count - 1}.pt").touch()
            else:
                (output / "policy_seed1234.json").write_text(json.dumps({"summary": {"success_rate": 0}}))
                (output / "policy_seed1234_env0.mp4").write_bytes(b"fake video")

        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            Path("src/mjlab_hand").mkdir(parents=True)
            Path(".venv/bin").mkdir(parents=True)
            Path(".venv/bin/python").touch()
            with patch("sys.argv", ["run_dexterity_local.py", "--task", "Baoding-Allegro",
                                    "--output-dir", "outputs/run", "--stages", "2", "5"]), \
                    patch.object(workflow, "source_hashes", return_value={}), \
                    patch.object(workflow, "execute", side_effect=fake_execute), \
                    contextlib.redirect_stdout(io.StringIO()):
                workflow.main()
            self.assertEqual(len(commands), 4)
            self.assertEqual(commands[0][commands[0].index("--iterations") + 1], "2")
            self.assertEqual(commands[2][commands[2].index("--iterations") + 1], "3")
            self.assertEqual(commands[2][commands[2].index("--resume") + 1], "outputs/run/stage_2/model_1.pt")
            saved = json.loads(Path("outputs/run/workflow.json").read_text())
            self.assertEqual(saved["status"], "complete")
            self.assertTrue(all(stage["status"] == "complete" for stage in saved["stages"]))

    def test_failed_subprocess_stops_and_records_phase(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            Path("src/mjlab_hand").mkdir(parents=True)
            Path(".venv/bin").mkdir(parents=True)
            Path(".venv/bin/python").touch()
            with patch("sys.argv", ["run_dexterity_local.py", "--task", "Baoding-Allegro",
                                    "--output-dir", "outputs/run"]), \
                    patch.object(workflow, "source_hashes", return_value={}), \
                    patch.object(workflow, "execute", side_effect=subprocess.CalledProcessError(1, "fake")) as execute:
                with self.assertRaises(subprocess.CalledProcessError):
                    workflow.main()
                self.assertEqual(execute.call_count, 1)
            saved = json.loads(Path("outputs/run/workflow.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["stages"][0]["failed_phase"], "training")
            self.assertEqual(saved["stages"][1]["status"], "pending")

    def test_invalid_stage_order_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            workflow.parse_args(["--task", "Baoding-Allegro", "--output-dir", "outputs/new",
                                 "--stages", "1000", "1000"])


if __name__ == "__main__":
    unittest.main()
