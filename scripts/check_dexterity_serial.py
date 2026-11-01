"""Exercise round scheduling and checkpoint resume without GPU processes."""
import json
import sys
import zipfile
from unittest.mock import patch

from run_dexterity_serial import main


def test_rounds_resume_and_skip_existing_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    waiting = tmp_path / "initial"
    waiting.mkdir()
    (waiting / "workflow.json").write_text(json.dumps({"status": "complete", "task": "Baoding-LEAP"}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [
        {"task": name, "seed": 0, "num_envs": 8}
        for name in ["Baoding-LEAP", "Baoding-Shadow", "Pen-Spinning-Wuji"]],
        "eval_seed": 1234, "eval_envs": 2}))
    commands = []

    class Child:
        pid = 123

        def __init__(self, command, **kwargs):
            commands.append(command)
            stage = int(command[command.index("--stages") + 1])
            from pathlib import Path
            output = Path(command[command.index("--output-dir") + 1]) / f"stage_{stage}"
            output.mkdir(parents=True)
            with zipfile.ZipFile(output / f"model_{stage - 1}.pt", "w") as archive:
                archive.writestr("data.pkl", "fake checkpoint")

        def wait(self):
            return 0

    argv = ["serial", "--manifest", str(manifest), "--output-dir", "outputs/serial",
            "--wait-root", str(waiting), "--deadline", "2099-01-01T00:00:00+00:00"]
    with patch.object(sys, "argv", argv), patch("run_dexterity_serial.subprocess.Popen", Child):
        main()
    assert [command[command.index("--stages") + 1] for command in commands] == [
        "1000", "1000", "3000", "3000", "10000", "10000"]
    assert all("Baoding-LEAP" not in command for command in commands)
    assert all("--resume" not in command for command in commands[:2])
    assert all("--resume" in command for command in commands[2:])
    state = json.loads((tmp_path / "outputs/serial/serial.json").read_text())
    assert state["status"] == "finished"
    assert all(row["status"] == "complete" for row in state["jobs"])
