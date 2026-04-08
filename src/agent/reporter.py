"""Report generator — Session 6 implementation (tasks 6.1–6.4)."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import openai

if TYPE_CHECKING:
    from agent.loop import ReasoningLogger
    from agent.memory import SessionMemory

_DEFAULT_BASE_URL = "https://api.x.ai/v1"
_DEFAULT_MODEL = "grok-3-mini"


def _make_client(api_key: str) -> openai.OpenAI:
    base_url = os.getenv("LLM_API_BASE_URL", _DEFAULT_BASE_URL)
    return openai.OpenAI(api_key=api_key, base_url=base_url)

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


def validate_chart_paths(
    chart_paths: list[str],
    session_outputs_dir: str,
) -> tuple[list[str], list[str]]:
    """Split chart paths into existing and missing at report-write time (I-18).

    Args:
        chart_paths: Absolute or relative paths to chart PNG files.
        session_outputs_dir: The session output directory (unused for the check
            itself, but kept in the signature for callers that need it for context).

    Returns:
        (valid_paths, missing_paths) — paths verified with os.path.exists().
    """
    valid_paths: list[str] = []
    missing_paths: list[str] = []
    for path in chart_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            missing_paths.append(path)
    return valid_paths, missing_paths


def build_chart_section(chart_paths: list[str], session_outputs_dir: str) -> str:
    """Build a Markdown string embedding charts and noting any that are missing (I-18).

    Valid charts are embedded as ``![chart](path)``.
    Missing charts produce an italic note — never a broken image reference.

    Args:
        chart_paths: Paths to chart files (produced during analysis).
        session_outputs_dir: Session output directory path.

    Returns:
        Markdown string. Empty string if chart_paths is empty.
    """
    if not chart_paths:
        return ""

    valid_paths, missing_paths = validate_chart_paths(chart_paths, session_outputs_dir)

    lines: list[str] = []

    for path in valid_paths:
        lines.append(f"![chart]({path})")

    for path in missing_paths:
        basename = os.path.basename(path)
        lines.append(
            f"_Chart not available: {basename} (generation did not complete)_"
        )

    return "\n\n".join(lines)


def generate_insights_for_analysis(
    analysis_type: str,
    output: str,
    api_key: str,
) -> str:
    """Call the LLM to generate 2-3 sentence insight for one analysis result.

    Args:
        analysis_type: Short name of the analysis (e.g. "histogram_sales").
        output: Text output produced by the analysis code.
        api_key: Grok (xAI) API key.

    Returns:
        Insight string from the LLM. Falls back to a plain summary on error.
    """
    model = os.getenv("PLANNER_MODEL", _DEFAULT_MODEL)
    prompt = (
        f"Analysis type: {analysis_type}\n\n"
        f"Analysis output:\n{output[:1000]}\n\n"
        "Write 2-3 sentences explaining what this analysis shows and why it matters "
        "for understanding the dataset. Be concise and specific."
    )
    client = _make_client(api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"_Insight generation failed: {exc}_"


def generate_report(
    session_id: str,
    memory: "SessionMemory",
    profile: dict,
    conn,
    logger: "ReasoningLogger",
    api_key: str,
    outputs_base: str = "outputs",
) -> Path:
    """Generate and write the final report exactly once per session (I-17–I-19).

    Enforces:
    - I-19: FileExistsError guard is the very first action — before any LLM call.
    - I-17: All four required sections present; missing ones get placeholders.
    - I-18: Chart paths verified with os.path.exists() at write time.
    """
    report_path = Path(outputs_base) / session_id / "report.md"

    # I-19: raise before any LLM call or file write
    if report_path.exists():
        raise FileExistsError(f"Report already exists: {report_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    session_outputs_dir = str(report_path.parent)
    all_results = memory.get_all_attempted()
    completed_results = [r for r in all_results if r["status"] == "COMPLETED"]

    # --- Dataset Summary ---
    summary_body = (
        f"**File:** {profile.get('file_name', 'unknown')}  \n"
        f"**Rows:** {profile.get('row_count', 0)} | "
        f"**Columns:** {profile.get('col_count', 0)}\n"
    )
    quality = profile.get("quality_issues", [])
    if quality:
        summary_body += "\n**Quality issues:**\n" + "\n".join(f"- {q}" for q in quality)

    # --- Key Trends (LLM insights per completed analysis) ---
    trends_parts: list[str] = []
    for result in completed_results:
        a_type = result["analysis_type"]
        output = result.get("output") or ""
        insight = generate_insights_for_analysis(a_type, output, api_key)
        trends_parts.append(f"### {a_type}\n\n{insight}")
    trends_body = "\n\n".join(trends_parts) if trends_parts else "_No analyses completed._"

    # --- Charts (validated at write time, I-18) ---
    chart_paths = [r["chart_path"] for r in completed_results if r.get("chart_path")]
    chart_md = build_chart_section(chart_paths, session_outputs_dir)

    # --- Anomalies ---
    anomaly_body = (
        "_Review the analysis outputs above for outliers and unexpected patterns._"
    )

    # --- Recommendations ---
    recs_body = (
        "_Based on the key trends identified, consider further investigation of "
        "the patterns noted above._"
    )

    content = (
        f"## Dataset Summary\n\n{summary_body}\n\n"
        f"## Key Trends\n\n{trends_body}\n"
    )
    if chart_md:
        content += f"\n{chart_md}\n"
    content += (
        f"\n## Anomalies\n\n{anomaly_body}\n\n"
        f"## Recommendations\n\n{recs_body}\n"
    )

    # I-17: validate and fill any missing sections before writing
    content = fill_missing_sections(content, [r["analysis_type"] for r in completed_results])
    missing_after = validate_report_sections(content)
    assert not missing_after, f"Report missing sections after fill: {missing_after}"

    report_path.write_text(content, encoding="utf-8")
    logger.log("OBSERVE", f"Report written to {report_path}")

    return report_path


def validate_session_output(
    session_id: str,
    outputs_base: str = "outputs",
    conn=None,
) -> dict:
    """Structural validator. Run after every session (I-24).

    Args:
        session_id: UUID of the session to validate.
        outputs_base: Base directory for session outputs.
        conn: Optional SQLite connection. When provided, the reasoning_log is
              used as a fallback to confirm the agent ran when no charts exist.

    Returns:
        {"valid": bool, "errors": list[str]}
    """
    session_dir = Path(outputs_base) / session_id
    errors: list[str] = []

    # 1. Report exists
    report_path = session_dir / "report.md"
    if not report_path.exists():
        errors.append(f"Report missing: {report_path}")
        # Can't do further content checks — return early
        return {"valid": False, "errors": errors}

    content = report_path.read_text(encoding="utf-8")

    # 2. All four required section headers are present
    missing_headers = validate_report_sections(content)
    for h in missing_headers:
        errors.append(f"Report missing required section: {h}")

    # 3. No section body is empty (header followed immediately by the next header or EOF)
    for i, header in enumerate(REQUIRED_SECTIONS):
        if header not in content:
            continue  # already flagged in check 2
        start = content.index(header) + len(header)
        # Find the start of the next section header (or end of string)
        next_start = len(content)
        for other in REQUIRED_SECTIONS:
            if other == header:
                continue
            idx = content.find(other, start)
            if idx != -1 and idx < next_start:
                next_start = idx
        body = content[start:next_start].strip()
        if not body:
            errors.append(f"Section body is empty: {header}")

    # 4. At least one chart exists in session_dir, or the reasoning_log has >= 1 entry
    png_files = list(session_dir.glob("*.png")) if session_dir.exists() else []
    has_charts = len(png_files) > 0
    has_log_entries = False
    if not has_charts and conn is not None:
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM reasoning_log WHERE session_id=?",
                (session_id,),
            ).fetchone()
            has_log_entries = (row[0] if row else 0) > 0
        except Exception:
            pass
    if not has_charts and not has_log_entries:
        errors.append(
            "No chart files found in session directory and no reasoning log entries "
            "(analysis may not have run)"
        )

    # 5. All ![...](path) image references in the report resolve to existing files
    import re
    image_refs = re.findall(r"!\[.*?\]\((.*?)\)", content)
    for ref in image_refs:
        if not os.path.exists(ref):
            errors.append(f"Broken image reference in report: {ref}")

    return {"valid": len(errors) == 0, "errors": errors}
