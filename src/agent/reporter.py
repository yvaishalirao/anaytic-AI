"""Report generator — stub with correct signature for loop.py import.

Full implementation delivered in Session 6 (tasks 6.1–6.4).
This stub satisfies the import and provides the FileExistsError guard (I-19)
so the integration test for Session 5 can verify the file is created.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.loop import ReasoningLogger
    from agent.memory import SessionMemory


def generate_report(
    session_id: str,
    memory: "SessionMemory",
    profile: dict,
    conn,
    logger: "ReasoningLogger",
    api_key: str,
    outputs_base: str = "outputs",
) -> Path:
    """Generate and write the session report exactly once (I-19).

    This stub writes a minimal four-section Markdown file so that:
    - The Session 5 integration test can assert the file exists.
    - I-19 (write exactly once) is enforced via FileExistsError.

    Session 6 replaces the body with LLM-generated insights.
    """
    report_path = Path(outputs_base) / session_id / "report.md"

    # I-19: raise before any write if report already exists
    if report_path.exists():
        raise FileExistsError(f"Report already exists: {report_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    completed = memory.get_completed()
    completed_text = (
        "\n".join(f"- {a}" for a in completed) if completed else "_No analyses completed._"
    )

    content = f"""## Dataset Summary

Dataset: {profile.get('file_name', 'unknown')}
Rows: {profile.get('row_count', 0)} | Columns: {profile.get('col_count', 0)}

## Key Trends

{completed_text}

## Anomalies

_Anomaly detection results from completed analyses above._

## Recommendations

_Recommendations will be generated in the full report (Session 6)._
"""

    report_path.write_text(content, encoding="utf-8")

    logger.log("OBSERVE", f"Report written to {report_path}")

    return report_path
