import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.executor import execute_code


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


def test_execute_code_interface_success(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n2\n")

    result = execute_code(
        code="print(2)",
        df_payload=df_payload,
        session_id="test-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )

    assert result["status"] == "success"
    assert result["output"] == "2"
    assert result["error"] is None
    assert result["charts"] == []


def test_execute_code_timeout(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n")

    result = execute_code(
        code="import time\ntime.sleep(2)",
        df_payload=df_payload,
        session_id="test-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=1,
    )

    assert result["status"] == "timeout"
    assert result["error"] == "execution exceeded time limit"
    assert result["output"] == ""
    assert result["charts"] == []


def test_timeout_enforcement(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n")

    start = time.monotonic()
    result = execute_code(
        code="while True:\n    pass",
        df_payload=df_payload,
        session_id="test-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=1,
    )
    elapsed = time.monotonic() - start

    assert result["status"] == "timeout"
    assert elapsed < 1.5


def test_never_raises(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n")

    try:
        result = execute_code(
            code="raise ValueError('boom')",
            df_payload=df_payload,
            session_id="test-session",
            outputs_dir=str(tmp_path / "outputs"),
            timeout=10,
        )
    except Exception as exc:
        pytest.fail(f"execute_code raised an exception: {exc}")

    assert result["status"] == "error"
    assert result["error"] is not None


def test_valid_code_success(tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n2\n")

    result = execute_code(
        code="print('ok')",
        df_payload=df_payload,
        session_id="test-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )

    assert result["status"] == "success"
    assert result["output"] == "ok"
    assert result["error"] is None


def test_empty_stdout(monkeypatch, tmp_path):
    df_payload = {"mode": "path", "path": str(tmp_path / "data.csv")}
    (tmp_path / "data.csv").write_text("col\n1\n")

    fake_proc = SimpleNamespace(stdout="", stderr="", returncode=1)

    def fake_run(*args, **kwargs):
        return fake_proc

    monkeypatch.setattr("agent.executor.subprocess.run", fake_run)

    result = execute_code(
        code="print(2)",
        df_payload=df_payload,
        session_id="test-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )

    assert result["status"] == "timeout"
    assert result["output"] == ""
    assert result["error"] == "subprocess produced no output"
    assert result["charts"] == []
