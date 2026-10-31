"""CPU-only checks for waiting/continuation decisions; never starts GPU work."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import continue_dexterity_after_stage as continuation


class ContinuationTest(unittest.TestCase):
    def prepare(self, status="complete"):
        previous = Path("outputs/previous")
        stage = previous / "stage_1000"
        stage.mkdir(parents=True)
        (stage / "training.json").write_text(json.dumps({
            "task": "Baoding-Allegro", "status": status, "final_iteration": 999,
        }))
        (stage / "model_999.pt").touch()
        (previous / "workflow.json").write_text(json.dumps({
            "status": "failed", "error": "Source changed during training",
            "arguments": {"seed": 0, "num_envs": 1024, "eval_seed": 1234, "eval_envs": 100},
            "stages": [{"cumulative_iterations": 1000, "status": "failed"},
                       {"cumulative_iterations": 3000, "status": "pending"}],
        }))
        return previous

    def test_failed_training_never_launches_eval_or_resume(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            previous = self.prepare("failed")
            with patch("sys.argv", ["continue.py", "--previous-root", str(previous),
                                    "--output-dir", "outputs/continued"]), \
                    patch.object(continuation, "execute") as execute:
                with self.assertRaisesRegex(RuntimeError, "did not complete"):
                    continuation.main()
                execute.assert_not_called()

    def test_completed_source_guard_stage_is_evaluated_then_resumed(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            previous = self.prepare()
            commands = []
            def fake_execute(command, environment, log):
                commands.append(command)
                self.assertEqual(environment["PYTHONUNBUFFERED"], "1")
                if "scripts/eval_dexterity_expert.py" in command:
                    output = Path(command[command.index("--output-dir") + 1])
                    output.mkdir()
                    (output / "policy_seed1234.json").write_text(json.dumps({
                        "task": "Baoding-Allegro", "checkpoint": str(previous / "stage_1000/model_999.pt"),
                        "seed": 1234, "num_envs": 100, "summary": {"episodes": 100, "censored_rate": 0},
                    }))
                    (output / "policy_seed1234_env0.mp4").write_bytes(b"video")
            with patch("sys.argv", ["continue.py", "--previous-root", str(previous),
                                    "--output-dir", "outputs/continued"]), \
                    patch.object(continuation, "execute", side_effect=fake_execute):
                continuation.main()
            self.assertEqual(len(commands), 2)
            self.assertIn("scripts/run_dexterity_local.py", commands[1])
            self.assertEqual(commands[1][commands[1].index("--resume") + 1],
                             str(previous / "stage_1000/model_999.pt"))
            self.assertEqual(commands[1][commands[1].index("--stages") + 1:commands[1].index("--num-envs")],
                             ["3000", "10000"])
            self.assertFalse(Path("outputs/continued").exists())  # driver owns its fresh output
            metadata = json.loads(Path("outputs/continued.continuation.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertTrue(metadata["supersedes_previous_workflow"])


if __name__ == "__main__":
    unittest.main()
