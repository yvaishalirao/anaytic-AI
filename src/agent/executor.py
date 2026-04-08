import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.loop import ReasoningLogger
    from agent.memory import SessionMemory

RUNNER_PATH = Path(__file__).parent / "subprocess_runner.py"
assert RUNNER_PATH.exists(), f"subprocess runner not found: {RUNNER_PATH}"


def get_session_outputs_dir(session_id: str, base_dir: str = "outputs") -> Path:
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        raise ValueError("Invalid session_id")

    base_path = Path(base_dir)
    session_path = base_path / session_id
    session_path.mkdir(parents=True, exist_ok=True)

    try:
        if session_path.resolve().relative_to(base_path.resolve()):
            pass
    except Exception:
        raise ValueError("Invalid session_id")

    return session_path


def execute_code(
    code: str,
    df_payload: dict,
    session_id: str,
    outputs_dir: str,
    timeout: int = 60,
) -> dict:
    """Execute code in a child subprocess and return structured results.

    Args:
        code: Python code to execute.
        df_payload: DataFrame transfer payload.
        session_id: Session identifier.
        outputs_dir: Absolute path to outputs directory.
        timeout: Subprocess timeout in seconds.

    Returns:
        Dict with keys status, output, error, charts.
    """
    outputs_path = get_session_outputs_dir(session_id, outputs_dir)
    payload = {
        "code": code,
        "df_payload": df_payload,
        "session_id": session_id,
        "outputs_dir": str(outputs_path),
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(RUNNER_PATH)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "output": "",
            "error": "execution exceeded time limit",
            "charts": [],
        }

    stdout = proc.stdout or ""
    if not stdout.strip():
        return {
            "status": "timeout",
            "output": "",
            "error": "subprocess produced no output",
            "charts": [],
        }

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "status": "timeout",
            "output": "",
            "error": "subprocess produced no output",
            "charts": [],
        }

    if not isinstance(result, dict):
        return {
            "status": "timeout",
            "output": "",
            "error": "subprocess produced no output",
            "charts": [],
        }

    return result


def run_analysis_step(
    code: str,
    analysis_type: str,
    df_payload: dict,
    session_id: str,
    memory: "SessionMemory",
    log: "ReasoningLogger",
    timeout: int = 60
) -> dict:
    """
    Wraps execute_code() and writes results to memory and reasoning log.
    - Writes ACTION log entry BEFORE calling execute_code (I-20)
    - Calls execute_code()
    - On status=success: calls memory.record_completed()
    - On status=timeout: calls memory.record_timeout()
    - On status=error: calls memory.record_failed()
    - Writes OBSERVE log entry with the result summary
    - Returns the execute_code result dict
    """
    # Write ACTION log entry BEFORE calling execute_code
    log.log_action(f"Executing {analysis_type} analysis: {code[:100]}{'...' if len(code) > 100 else ''}")

    # Call execute_code
    result = execute_code(code, df_payload, session_id, "outputs", timeout)

    # Record to memory based on status
    output_text = result.get("output") or ""
    chart_paths = result.get("charts", [])
    chart_path = chart_paths[0] if chart_paths else None

    # Resolve basename to full path so os.path.exists() works at report time
    if chart_path is not None:
        chart_path = str(Path("outputs") / session_id / chart_path)

    if result["status"] == "success":
        memory.record_completed(analysis_type, output_text, chart_path)
    elif result["status"] == "timeout":
        memory.record_timeout(analysis_type)
    elif result["status"] == "error":
        memory.record_failed(analysis_type, result.get("error") or "")

    # Write OBSERVE log entry with result summary
    summary = f"Result: {result['status']}"
    if result["output"]:
        summary += f", output: {result['output'][:100]}{'...' if len(result['output']) > 100 else ''}"
    if result["error"]:
        summary += f", error: {result['error'][:100]}{'...' if len(result['error']) > 100 else ''}"
    if result["charts"]:
        summary += f", charts: {result['charts']}"
    log.log_observe(summary)

    return result
