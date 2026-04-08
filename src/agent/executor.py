import json
import subprocess
import sys
from pathlib import Path

RUNNER_PATH = Path(__file__).parent / "subprocess_runner.py"
assert RUNNER_PATH.exists(), f"subprocess runner not found: {RUNNER_PATH}"


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
    payload = {
        "code": code,
        "df_payload": df_payload,
        "session_id": session_id,
        "outputs_dir": outputs_dir,
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
