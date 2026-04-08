import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.executor import execute_code, run_analysis_step


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


def test_histogram_generation(sales_csv, tmp_path):
    df_payload = {"mode": "path", "path": str(sales_csv)}
    result = execute_code(
        code="plt.hist(df['sales'])\nplt.savefig(outputs_dir + '/hist.png')",
        df_payload=df_payload,
        session_id="hist-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )

    assert result["status"] == "success"
    assert "hist.png" in result["charts"]
    assert (tmp_path / "outputs" / "hist-session" / "hist.png").exists()


def test_correlation_matrix(sales_csv, tmp_path):
    df_payload = {"mode": "path", "path": str(sales_csv)}
    result = execute_code(
        code="print(df.corr(numeric_only=True).to_string())",
        df_payload=df_payload,
        session_id="corr-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )

    assert result["status"] == "success"
    assert "sales" in result["output"]
    assert "units" in result["output"]
    assert "returned" in result["output"]


def test_bad_import(sales_csv, tmp_path):
    df_payload = {"mode": "path", "path": str(sales_csv)}
    result = execute_code(
        code="import nonexistent_package_xyz",
        df_payload=df_payload,
        session_id="bad-import-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )

    assert result["status"] == "error"
    assert "ModuleNotFoundError" in result["error"]


def test_divide_by_zero(sales_csv, tmp_path):
    df_payload = {"mode": "path", "path": str(sales_csv)}
    result = execute_code(
        code="x = 1/0",
        df_payload=df_payload,
        session_id="divide-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )

    assert result["status"] == "error"
    assert "ZeroDivisionError" in result["error"]


def test_infinite_loop_killed(sales_csv, tmp_path):
    df_payload = {"mode": "path", "path": str(sales_csv)}
    start = time.monotonic()
    result = execute_code(
        code="while True:\n    pass",
        df_payload=df_payload,
        session_id="infinite-session",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=3,
    )
    elapsed = time.monotonic() - start

    assert result["status"] == "timeout"
    assert elapsed < 4.0


def test_session_isolation_creates_separate_outputs(tmp_path):
    df_path = tmp_path / "data.csv"
    df_path.write_text("col\n1\n2\n")
    df_payload = {"mode": "path", "path": str(df_path)}

    result_a = execute_code(
        code="import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.savefig(outputs_dir + '/histogram.png')",
        df_payload=df_payload,
        session_id="session_a",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )
    result_b = execute_code(
        code="import matplotlib.pyplot as plt\nplt.plot([4, 5, 6])\nplt.savefig(outputs_dir + '/histogram.png')",
        df_payload=df_payload,
        session_id="session_b",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )

    assert result_a["status"] == "success"
    assert result_b["status"] == "success"
    assert "histogram.png" in result_a["charts"]
    assert "histogram.png" in result_b["charts"]
    assert (tmp_path / "outputs" / "session_a" / "histogram.png").exists()
    assert (tmp_path / "outputs" / "session_b" / "histogram.png").exists()


def test_outputs_dir_prefix(tmp_path, monkeypatch):
    df_path = tmp_path / "data.csv"
    df_path.write_text("col\n1\n")
    df_payload = {"mode": "path", "path": str(df_path)}
    captured = {}

    def fake_run(args, input, capture_output, text, timeout):
        payload = json.loads(input)
        captured["payload"] = payload
        fake_proc = SimpleNamespace(stdout=json.dumps({
            "status": "success",
            "output": "ok",
            "error": None,
            "charts": [],
        }), stderr="", returncode=0)
        return fake_proc

    monkeypatch.setattr("agent.executor.subprocess.run", fake_run)

    result = execute_code(
        code="print('ok')",
        df_payload=df_payload,
        session_id="session_x",
        outputs_dir=str(tmp_path / "outputs"),
        timeout=10,
    )

    assert result["status"] == "success"
    assert captured["payload"]["outputs_dir"].startswith(str(tmp_path / "outputs" / "session_x"))


def test_dotdot_session_id(tmp_path):
    with pytest.raises(ValueError):
        execute_code(
            code="print('ok')",
            df_payload={"mode": "path", "path": str(tmp_path / "data.csv")},
            session_id="..",
            outputs_dir=str(tmp_path / "outputs"),
            timeout=10,
        )


def test_slash_session_id(tmp_path):
    with pytest.raises(ValueError):
        execute_code(
            code="print('ok')",
            df_payload={"mode": "path", "path": str(tmp_path / "data.csv")},
            session_id="session/evil",
            outputs_dir=str(tmp_path / "outputs"),
            timeout=10,
        )


# Mock classes for testing run_analysis_step
class MockSessionMemory:
    def __init__(self):
        self.completed = []
        self.failed = []
        self.timeout = []
        self.results = []

    def record_completed(self, analysis_type, output=None, chart_path=None):
        self.completed.append(analysis_type)
        self.results.append({"analysis_type": analysis_type, "status": "COMPLETED"})

    def record_failed(self, analysis_type, error=None):
        self.failed.append(analysis_type)
        self.results.append({"analysis_type": analysis_type, "status": "FAILED"})

    def record_timeout(self, analysis_type):
        self.timeout.append(analysis_type)
        self.results.append({"analysis_type": analysis_type, "status": "TIMEOUT"})

    def is_done(self, analysis_type):
        return any(r["analysis_type"] == analysis_type and r["status"] == "COMPLETED" for r in self.results)


class MockReasoningLogger:
    def __init__(self):
        self.entries = []
        self.seq = 0

    def log_action(self, content):
        import time
        self.seq += 1
        self.entries.append({
            "type": "ACTION",
            "content": content,
            "seq": self.seq,
            "timestamp": time.time()
        })

    def log_observe(self, content):
        import time
        self.seq += 1
        self.entries.append({
            "type": "OBSERVE", 
            "content": content,
            "seq": self.seq,
            "timestamp": time.time()
        })


def test_run_analysis_step_timeout_calls_record_timeout(sales_csv, tmp_path, monkeypatch):
    df_payload = {"mode": "path", "path": str(sales_csv)}
    memory = MockSessionMemory()
    log = MockReasoningLogger()

    # Mock execute_code to return timeout
    def mock_execute_code(code, df_payload, session_id, outputs_dir, timeout):
        return {"status": "timeout", "output": "", "error": "timeout", "charts": []}

    monkeypatch.setattr("agent.executor.execute_code", mock_execute_code)

    result = run_analysis_step(
        code="while True: pass",
        analysis_type="infinite_loop",
        df_payload=df_payload,
        session_id="timeout-session",
        memory=memory,
        log=log,
        timeout=1,
    )

    assert result["status"] == "timeout"
    assert memory.timeout == ["infinite_loop"]
    assert memory.failed == []
    assert not memory.is_done("infinite_loop")
    assert len(log.entries) == 2
    assert log.entries[0]["type"] == "ACTION"
    assert log.entries[1]["type"] == "OBSERVE"


def test_run_analysis_step_error_calls_record_failed(sales_csv, tmp_path, monkeypatch):
    df_payload = {"mode": "path", "path": str(sales_csv)}
    memory = MockSessionMemory()
    log = MockReasoningLogger()

    # Mock execute_code to return error
    def mock_execute_code(code, df_payload, session_id, outputs_dir, timeout):
        return {"status": "error", "output": "", "error": "ZeroDivisionError", "charts": []}

    monkeypatch.setattr("agent.executor.execute_code", mock_execute_code)

    result = run_analysis_step(
        code="1/0",
        analysis_type="division_by_zero",
        df_payload=df_payload,
        session_id="error-session",
        memory=memory,
        log=log,
        timeout=10,
    )

    assert result["status"] == "error"
    assert memory.failed == ["division_by_zero"]
    assert memory.timeout == []
    assert not memory.is_done("division_by_zero")
    assert len(log.entries) == 2
    assert log.entries[0]["type"] == "ACTION"
    assert log.entries[1]["type"] == "OBSERVE"


def test_run_analysis_step_action_log_before_execute(sales_csv, tmp_path, monkeypatch):
    df_payload = {"mode": "path", "path": str(sales_csv)}
    memory = MockSessionMemory()
    log = MockReasoningLogger()

    executed = []

    def mock_execute_code(code, df_payload, session_id, outputs_dir, timeout):
        executed.append(True)
        return {"status": "success", "output": "ok", "error": None, "charts": []}

    monkeypatch.setattr("agent.executor.execute_code", mock_execute_code)

    result = run_analysis_step(
        code="print('test')",
        analysis_type="test_analysis",
        df_payload=df_payload,
        session_id="test-session",
        memory=memory,
        log=log,
        timeout=10,
    )

    assert executed == [True]
    assert len(log.entries) == 2
    assert log.entries[0]["type"] == "ACTION"
    assert log.entries[1]["type"] == "OBSERVE"
    # ACTION logged before execution
    assert log.entries[0]["seq"] == 1
    assert log.entries[1]["seq"] == 2


def test_run_analysis_step_observe_log_after_execute(sales_csv, tmp_path, monkeypatch):
    df_payload = {"mode": "path", "path": str(sales_csv)}
    memory = MockSessionMemory()
    log = MockReasoningLogger()

    def mock_execute_code(code, df_payload, session_id, outputs_dir, timeout):
        return {"status": "success", "output": "computed result", "error": None, "charts": ["chart.png"]}

    monkeypatch.setattr("agent.executor.execute_code", mock_execute_code)

    result = run_analysis_step(
        code="compute something",
        analysis_type="computation",
        df_payload=df_payload,
        session_id="observe-session",
        memory=memory,
        log=log,
        timeout=10,
    )

    assert result["status"] == "success"
    assert memory.completed == ["computation"]
    assert memory.is_done("computation")
    assert len(log.entries) == 2
    assert log.entries[0]["type"] == "ACTION"
    assert log.entries[1]["type"] == "OBSERVE"
    assert "computed result" in log.entries[1]["content"]
    assert "chart.png" in log.entries[1]["content"]


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


def test_timeout_memory_record(sales_csv, tmp_path, monkeypatch):
    """4.5 TC-1: Timeout calls record_timeout on memory, is_done returns False, TIMEOUT in results"""
    df_payload = {"mode": "path", "path": str(sales_csv)}
    memory = MockSessionMemory()
    log = MockReasoningLogger()

    def mock_execute_code(code, df_payload, session_id, outputs_dir, timeout):
        return {"status": "timeout", "output": "", "error": "execution exceeded time limit", "charts": []}

    monkeypatch.setattr("agent.executor.execute_code", mock_execute_code)

    result = run_analysis_step(
        code="while True: pass",
        analysis_type="infinite_loop",
        df_payload=df_payload,
        session_id="timeout-session",
        memory=memory,
        log=log,
        timeout=1,
    )

    # Verify timeout result
    assert result["status"] == "timeout"
    assert result["error"] == "execution exceeded time limit"
    
    # Verify memory recording
    assert memory.timeout == ["infinite_loop"]
    assert not memory.is_done("infinite_loop")
    
    # Verify TIMEOUT in results
    timeout_results = [r for r in memory.results if r["status"] == "TIMEOUT"]
    assert len(timeout_results) == 1
    assert timeout_results[0]["analysis_type"] == "infinite_loop"


def test_error_memory_record(sales_csv, tmp_path, monkeypatch):
    """4.5 TC-2: Error calls record_failed on memory, is_done returns False, FAILED in results"""
    df_payload = {"mode": "path", "path": str(sales_csv)}
    memory = MockSessionMemory()
    log = MockReasoningLogger()

    def mock_execute_code(code, df_payload, session_id, outputs_dir, timeout):
        return {"status": "error", "output": "", "error": "ZeroDivisionError: division by zero", "charts": []}

    monkeypatch.setattr("agent.executor.execute_code", mock_execute_code)

    result = run_analysis_step(
        code="1/0",
        analysis_type="division_by_zero",
        df_payload=df_payload,
        session_id="error-session",
        memory=memory,
        log=log,
        timeout=10,
    )

    # Verify error result
    assert result["status"] == "error"
    assert "ZeroDivisionError" in result["error"]
    
    # Verify memory recording
    assert memory.failed == ["division_by_zero"]
    assert not memory.is_done("division_by_zero")
    
    # Verify FAILED in results
    failed_results = [r for r in memory.results if r["status"] == "FAILED"]
    assert len(failed_results) == 1
    assert failed_results[0]["analysis_type"] == "division_by_zero"


def test_action_logged_before_exec(sales_csv, tmp_path, monkeypatch):
    """4.5 TC-3: ACTION log written before subprocess spawn, log entry timestamp precedes execution"""
    df_payload = {"mode": "path", "path": str(sales_csv)}
    memory = MockSessionMemory()
    log = MockReasoningLogger()

    execution_timestamp = None

    def mock_execute_code(code, df_payload, session_id, outputs_dir, timeout):
        nonlocal execution_timestamp
        import time
        time.sleep(0.001)  # Small delay to ensure different timestamps
        execution_timestamp = time.time()
        return {"status": "success", "output": "ok", "error": None, "charts": []}

    monkeypatch.setattr("agent.executor.execute_code", mock_execute_code)

    result = run_analysis_step(
        code="print('test')",
        analysis_type="test_analysis",
        df_payload=df_payload,
        session_id="test-session",
        memory=memory,
        log=log,
        timeout=10,
    )

    # Verify ACTION log exists and precedes execution
    action_entries = [e for e in log.entries if e["type"] == "ACTION"]
    assert len(action_entries) == 1
    action_entry = action_entries[0]
    
    # ACTION timestamp should be before execution timestamp
    assert action_entry["timestamp"] < execution_timestamp


def test_observe_after_action(sales_csv, tmp_path, monkeypatch):
    """4.5 TC-4: OBSERVE log written after execution, OBSERVE entry seq > ACTION entry seq"""
    df_payload = {"mode": "path", "path": str(sales_csv)}
    memory = MockSessionMemory()
    log = MockReasoningLogger()

    def mock_execute_code(code, df_payload, session_id, outputs_dir, timeout):
        return {"status": "success", "output": "result", "error": None, "charts": []}

    monkeypatch.setattr("agent.executor.execute_code", mock_execute_code)

    result = run_analysis_step(
        code="print('test')",
        analysis_type="test_analysis",
        df_payload=df_payload,
        session_id="test-session",
        memory=memory,
        log=log,
        timeout=10,
    )

    # Verify both ACTION and OBSERVE entries exist
    action_entries = [e for e in log.entries if e["type"] == "ACTION"]
    observe_entries = [e for e in log.entries if e["type"] == "OBSERVE"]
    
    assert len(action_entries) == 1
    assert len(observe_entries) == 1
    
    # OBSERVE sequence number should be greater than ACTION
    assert observe_entries[0]["seq"] > action_entries[0]["seq"]
