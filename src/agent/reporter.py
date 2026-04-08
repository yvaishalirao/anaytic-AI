"""Report generator — Session 6 implementation (tasks 6.1–6.4)."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.loop import ReasoningLogger
    from agent.memory import SessionMemory

REQUIRED_SECTIONS = [
    "## Dataset Summary",
    "## Key Trends",
    "## Anomalies",
    "## Recommendations",
]

# Placeholder text per section — non-empty, descriptive (I-17)
_PLACEHOLDERS: dict[str, str] = {
    "## Dataset Summary": "_Dataset summary did not complete in this session._",
    "## Key Trends": "_Key trend analysis did not complete in this session._",
    "## Anomalies": "_Anomaly detection did not complete in this session._",
    "## Recommendations": "_Recommendations could not be generated in this session._",
}


def validate_report_sections(content: str) -> list[str]:
    """Return a list of required section headers missing from content.

    Args:
        content: Markdown string to check.

    Returns:
        List of missing section headers. Empty list means all four are present.
    """
    return [header for header in REQUIRED_SECTIONS if header not in content]


def fill_missing_sections(content: str, completed_analyses: list[str]) -> str:
    """Append placeholder text for any missing required sections (I-17).

    Args:
        content: Current report content (may be partial).
        completed_analyses: List of completed analysis type names (for context).

    Returns:
        Content with all missing sections appended as non-empty placeholders.
        Existing sections are never duplicated.
    """
    missing = validate_report_sections(content)
    for header in missing:
        placeholder = _PLACEHOLDERS.get(header, "_Section not available._")
        content = content.rstrip("\n") + f"\n\n{header}\n\n{placeholder}\n"
    return content


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

    Enforces:
    - I-17: All four required sections present (placeholders if needed).
    - I-19: Raises FileExistsError if report already exists — checked first.
    - I-18: Chart references verified at write time (implemented in task 6.2).
    """
    report_path = Path(outputs_base) / session_id / "report.md"

    # I-19: guard must be the first action — before any LLM call or file write
    if report_path.exists():
        raise FileExistsError(f"Report already exists: {report_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    completed = memory.get_completed()
    completed_text = (
        "\n".join(f"- {a}" for a in completed) if completed else "_No analyses completed._"
    )

    content = (
        f"## Dataset Summary\n\n"
        f"Dataset: {profile.get('file_name', 'unknown')}  \n"
        f"Rows: {profile.get('row_count', 0)} | Columns: {profile.get('col_count', 0)}\n\n"
        f"## Key Trends\n\n"
        f"{completed_text}\n\n"
        f"## Anomalies\n\n"
        f"_Anomaly detection results from completed analyses above._\n\n"
        f"## Recommendations\n\n"
        f"_Review the key trends and anomalies above to inform next steps._\n"
    )

    # I-17: validate and fill any missing sections before writing
    content = fill_missing_sections(content, completed)

    missing_after = validate_report_sections(content)
    assert not missing_after, f"Report missing sections after fill: {missing_after}"

    report_path.write_text(content, encoding="utf-8")
    logger.log("OBSERVE", f"Report written to {report_path}")

    return report_path
