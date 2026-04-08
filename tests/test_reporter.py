"""Tests for Session 6: Report Generator (tasks 6.1–6.4)."""

import os
import re
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.db import enqueue_job, init_db, new_session_id
from agent.loop import ReasoningLogger
from agent.memory import SessionMemory
from agent.reporter import (
    REQUIRED_SECTIONS,
    build_chart_section,
    fill_missing_sections,
    generate_insights_for_analysis,
    generate_report,
    validate_chart_paths,
    validate_report_sections,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_content() -> str:
    """Minimal content that contains all four required section headers."""
    return (
        "## Dataset Summary\n\nSome summary.\n\n"
        "## Key Trends\n\nSome trends.\n\n"
        "## Anomalies\n\nSome anomalies.\n\n"
        "## Recommendations\n\nSome recommendations.\n"
    )


# ---------------------------------------------------------------------------
# 6.1 — validate_report_sections
# ---------------------------------------------------------------------------

def test_validate_all_present():
    """All four sections present → empty list returned (6.1 TC-1)."""
    result = validate_report_sections(_full_content())
    assert result == []


def test_validate_missing_section():
    """Missing '## Anomalies' → list contains that header (6.1 TC-2)."""
    content = (
        "## Dataset Summary\n\nSummary.\n\n"
        "## Key Trends\n\nTrends.\n\n"
        "## Recommendations\n\nRecs.\n"
    )
    result = validate_report_sections(content)
    assert result == ["## Anomalies"]


def test_validate_missing_multiple_sections():
    """Multiple missing sections are all reported."""
    content = "## Dataset Summary\n\nSummary only.\n"
    result = validate_report_sections(content)
    assert "## Key Trends" in result
    assert "## Anomalies" in result
    assert "## Recommendations" in result
    assert "## Dataset Summary" not in result


def test_validate_empty_content():
    """Empty string is missing all four sections."""
    result = validate_report_sections("")
    assert set(result) == set(REQUIRED_SECTIONS)


def test_validate_order_preserved():
    """Missing sections are returned in REQUIRED_SECTIONS order."""
    content = "## Key Trends\n\nTrends.\n\n## Recommendations\n\nRecs.\n"
    result = validate_report_sections(content)
    # Missing: Dataset Summary, Anomalies — in that relative order
    assert result.index("## Dataset Summary") < result.index("## Anomalies")


# ---------------------------------------------------------------------------
# 6.1 — fill_missing_sections
# ---------------------------------------------------------------------------

def test_placeholder_nonempty():
    """fill_missing_sections adds a non-empty placeholder for each missing section (6.1 TC-3)."""
    content = "## Dataset Summary\n\nSummary.\n"
    filled = fill_missing_sections(content, completed_analyses=[])

    # All four headers must now be present
    assert validate_report_sections(filled) == []

    # Each injected placeholder must be non-empty (not just the header line)
    for header in ["## Key Trends", "## Anomalies", "## Recommendations"]:
        idx = filled.index(header)
        # Text after the header (skip header line + blank line)
        after_header = filled[idx + len(header):].lstrip("\n")
        assert after_header.strip(), f"Placeholder for {header!r} is empty"


def test_no_duplicate_sections():
    """fill_missing_sections does not duplicate already-present sections (6.1 TC-4)."""
    content = _full_content()
    filled = fill_missing_sections(content, completed_analyses=[])

    for header in REQUIRED_SECTIONS:
        assert filled.count(header) == 1, (
            f"{header!r} appears more than once after fill_missing_sections"
        )


def test_fill_preserves_existing_content():
    """Existing section bodies are not altered by fill_missing_sections."""
    original_body = "Existing trend analysis here."
    content = (
        "## Dataset Summary\n\nSummary.\n\n"
        f"## Key Trends\n\n{original_body}\n\n"
        "## Anomalies\n\nAnomalies.\n\n"
        "## Recommendations\n\nRecs.\n"
    )
    filled = fill_missing_sections(content, completed_analyses=[])
    assert original_body in filled


def test_fill_all_missing_from_empty():
    """fill_missing_sections on empty string produces all four sections."""
    filled = fill_missing_sections("", completed_analyses=["analysis_a"])
    assert validate_report_sections(filled) == []
    for header in REQUIRED_SECTIONS:
        assert header in filled


# ---------------------------------------------------------------------------
# 6.2 — validate_chart_paths
# ---------------------------------------------------------------------------

def test_validate_chart_paths(tmp_path):
    """validate_chart_paths splits correctly into valid and missing (6.2 TC-3)."""
    existing = tmp_path / "hist.png"
    existing.write_bytes(b"\x89PNG")
    missing = str(tmp_path / "nonexistent.png")

    valid, absent = validate_chart_paths(
        [str(existing), missing], session_outputs_dir=str(tmp_path)
    )

    assert str(existing) in valid
    assert missing in absent
    assert missing not in valid
    assert str(existing) not in absent


def test_validate_chart_paths_all_valid(tmp_path):
    """All paths existing → missing list is empty."""
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p1.write_bytes(b"PNG")
    p2.write_bytes(b"PNG")

    valid, missing = validate_chart_paths(
        [str(p1), str(p2)], session_outputs_dir=str(tmp_path)
    )
    assert len(valid) == 2
    assert missing == []


def test_validate_chart_paths_all_missing(tmp_path):
    """No paths existing → valid list is empty."""
    valid, missing = validate_chart_paths(
        [str(tmp_path / "x.png"), str(tmp_path / "y.png")],
        session_outputs_dir=str(tmp_path),
    )
    assert valid == []
    assert len(missing) == 2


# ---------------------------------------------------------------------------
# 6.2 — build_chart_section
# ---------------------------------------------------------------------------

def test_chart_valid_embed(tmp_path):
    """Existing chart path appears as a Markdown image reference (6.2 TC-1)."""
    chart = tmp_path / "histogram.png"
    chart.write_bytes(b"PNG")

    section = build_chart_section([str(chart)], session_outputs_dir=str(tmp_path))

    assert f"![chart]({str(chart).replace(chr(92), '/')})" in section


def test_chart_missing_no_broken_ref(tmp_path):
    """Missing chart produces an italic note — no ![...] reference (6.2 TC-2)."""
    missing = str(tmp_path / "gone.png")

    section = build_chart_section([missing], session_outputs_dir=str(tmp_path))

    # Must NOT contain any Markdown image syntax
    assert "![" not in section
    # Must contain a plain-text note mentioning the filename
    assert "gone.png" in section
    assert "not available" in section.lower()


def test_chart_no_broken_refs_for_nonexistent(tmp_path):
    """No ![...](path) reference points to a non-existent file (6.2 TC-3 extended, I-18)."""
    existing = tmp_path / "real.png"
    existing.write_bytes(b"PNG")
    missing = str(tmp_path / "fake.png")

    section = build_chart_section(
        [str(existing), missing], session_outputs_dir=str(tmp_path)
    )

    # Extract all image paths from Markdown image syntax ![...](path)
    image_refs = re.findall(r"!\[.*?\]\((.*?)\)", section)
    for ref in image_refs:
        import os
        assert os.path.exists(ref), (
            f"Broken image reference in chart section: {ref!r}"
        )


def test_build_chart_section_empty_input(tmp_path):
    """Empty chart list returns an empty string."""
    section = build_chart_section([], session_outputs_dir=str(tmp_path))
    assert section == ""


def test_build_chart_section_mixed(tmp_path):
    """Mixed valid/missing: image tag for valid, note for missing."""
    good = tmp_path / "good.png"
    good.write_bytes(b"PNG")
    bad = str(tmp_path / "bad.png")

    section = build_chart_section([str(good), bad], session_outputs_dir=str(tmp_path))

    assert f"![chart]({str(good).replace(chr(92), '/')})" in section
    assert "bad.png" in section
    assert "not available" in section.lower()
    # The missing path must NOT appear inside an image tag
    assert f"![chart]({bad})" not in section


# ---------------------------------------------------------------------------
# 6.3 helpers — minimal DB + memory + logger for generate_report tests
# ---------------------------------------------------------------------------

@pytest.fixture
def report_env(tmp_path):
    """Fresh DB, session, memory, and ReasoningLogger for generate_report tests."""
    db_path = str(tmp_path / "test.db")
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row
    session_id = new_session_id(conn)
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {})
    conn.execute(
        "UPDATE jobs SET status='PROCESSING' WHERE id=?", (job_id,)
    )
    memory = SessionMemory(conn, session_id)
    memory.set_job_id(job_id)
    logger = ReasoningLogger(conn, session_id, job_id)
    profile = {
        "file_name": "sales.csv",
        "row_count": 50,
        "col_count": 4,
        "columns": [],
        "quality_issues": [],
    }
    return {
        "conn": conn,
        "session_id": session_id,
        "job_id": job_id,
        "memory": memory,
        "logger": logger,
        "profile": profile,
        "outputs_base": str(tmp_path / "outputs"),
    }


# ---------------------------------------------------------------------------
# 6.3 — generate_insights_for_analysis
# ---------------------------------------------------------------------------

@patch("agent.reporter._make_client")
def test_generate_insights_returns_string(mock_client):
    """generate_insights_for_analysis returns the LLM response string."""
    mock_client.return_value.chat.completions.create.return_value.choices[
        0
    ].message.content = "Sales show an upward trend throughout 2023."

    result = generate_insights_for_analysis("histogram_sales", "count 50", api_key="k")

    assert isinstance(result, str)
    assert len(result) > 0


@patch("agent.reporter._make_client")
def test_generate_insights_fallback_on_error(mock_client):
    """generate_insights_for_analysis returns a fallback string on LLM error."""
    mock_client.return_value.chat.completions.create.side_effect = RuntimeError("oops")

    result = generate_insights_for_analysis("bad_analysis", "output", api_key="k")

    assert isinstance(result, str)
    assert len(result) > 0  # never raises, never returns empty


# ---------------------------------------------------------------------------
# 6.3 — generate_report
# ---------------------------------------------------------------------------

@patch("agent.reporter.generate_insights_for_analysis")
def test_report_written(mock_insights, report_env):
    """generate_report writes a file at outputs/{session_id}/report.md (6.3 TC-1)."""
    mock_insights.return_value = "Insight text."
    env = report_env

    path = generate_report(
        env["session_id"], env["memory"], env["profile"],
        env["conn"], env["logger"], api_key="test",
        outputs_base=env["outputs_base"],
    )

    assert Path(path).exists()
    assert path == Path(env["outputs_base"]) / env["session_id"] / "report.md"


@patch("agent.reporter.generate_insights_for_analysis")
def test_report_no_overwrite(mock_insights, report_env):
    """Calling generate_report twice raises FileExistsError (6.3 TC-2, I-19)."""
    mock_insights.return_value = "Insight."
    env = report_env

    generate_report(
        env["session_id"], env["memory"], env["profile"],
        env["conn"], env["logger"], api_key="test",
        outputs_base=env["outputs_base"],
    )

    with pytest.raises(FileExistsError):
        generate_report(
            env["session_id"], env["memory"], env["profile"],
            env["conn"], env["logger"], api_key="test",
            outputs_base=env["outputs_base"],
        )


@patch("agent.reporter.generate_insights_for_analysis")
def test_report_all_sections(mock_insights, report_env):
    """Written report contains all 4 required section headers (6.3 TC-3, I-17)."""
    mock_insights.return_value = "Insight text."
    env = report_env

    path = generate_report(
        env["session_id"], env["memory"], env["profile"],
        env["conn"], env["logger"], api_key="test",
        outputs_base=env["outputs_base"],
    )

    content = Path(path).read_text(encoding="utf-8")
    missing = validate_report_sections(content)
    assert missing == [], f"Report is missing sections: {missing}"


@patch("agent.reporter.generate_insights_for_analysis")
def test_no_broken_refs(mock_insights, report_env, tmp_path):
    """Every ![...](path) in the written report resolves to an existing file (6.3 TC-4, I-18)."""
    mock_insights.return_value = "Insight."
    env = report_env

    # Pre-create a real chart file and record it in memory
    chart_dir = Path(env["outputs_base"]) / env["session_id"]
    chart_dir.mkdir(parents=True, exist_ok=True)
    real_chart = chart_dir / "hist.png"
    real_chart.write_bytes(b"PNG")

    from agent.db import write_result
    write_result(
        env["conn"], env["session_id"], env["job_id"],
        "histogram_sales", "COMPLETED", "some output", str(real_chart),
    )

    path = generate_report(
        env["session_id"], env["memory"], env["profile"],
        env["conn"], env["logger"], api_key="test",
        outputs_base=env["outputs_base"],
    )

    content = Path(path).read_text(encoding="utf-8")
    image_refs = re.findall(r"!\[.*?\]\((.*?)\)", content)
    for ref in image_refs:
        assert os.path.exists(ref), f"Broken image reference in report: {ref!r}"
