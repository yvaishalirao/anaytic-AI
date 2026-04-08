import json
import subprocess
import sys
from pathlib import Path

import pytest


RUNNER_PATH = Path(__file__).resolve().parents[1] / "src" / "agent" / "subprocess_runner.py"


def _run_runner(code: str, df_payload: dict, outputs_dir: Path):
    payload = {
        "code": code,
        "df_payload": df_payload,
        "session_id": "test-session",
        "outputs_dir": str(outputs_dir),
    }
    proc = subprocess.run(
        [sys.executable, str(RUNNER_PATH)],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parents[1]),
        check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        raise AssertionError(
            f"Runner did not return valid JSON. stdout={stdout!r}, stderr={proc.stderr.decode('utf-8', errors='replace')!r}"
        )
    return proc, result


def test_simple_exec(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n2\n")

    proc, result = _run_runner("print(2)", df_payload, tmp_path / "outputs")

    assert proc.returncode == 0
    assert result["status"] == "success"
    assert result["output"] == "2"
    assert result["error"] is None
    assert result["charts"] == []


def test_exec_error(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n")

    proc, result = _run_runner("raise ValueError('boom')", df_payload, tmp_path / "outputs")

    assert proc.returncode == 0
    assert result["status"] == "error"
    assert result["error"] is not None
    assert "ValueError" in result["error"]
    assert "Traceback" in result["error"]


def test_chart_capture(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n")

    code = "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.savefig(outputs_dir + '/chart.png')"
    proc, result = _run_runner(code, df_payload, tmp_path / "outputs")

    assert proc.returncode == 0
    assert result["status"] == "success"
    assert result["error"] is None
    assert "chart.png" in result["charts"]
    assert (tmp_path / "outputs" / "chart.png").exists()


def test_no_escape_write(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n")

    code = "with open('escape.txt', 'w') as f:\n    f.write('escape')"
    proc, result = _run_runner(code, df_payload, tmp_path / "outputs")

    assert proc.returncode == 0
    assert result["status"] == "success"
    assert result["error"] is None
    assert "outside outputs_dir" in result["output"].lower() or "removed files" in result["output"].lower()
    assert not (Path(__file__).resolve().parents[1] / "escape.txt").exists()


def test_exit_code_on_error(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n")

    proc, result = _run_runner("raise RuntimeError('boom')", df_payload, tmp_path / "outputs")

    assert proc.returncode == 0
    assert result["status"] == "error"
    assert result["error"] is not None
