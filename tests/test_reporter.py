"""Tests for Session 6: Report Generator (tasks 6.1–6.4)."""

import pytest

from agent.reporter import (
    REQUIRED_SECTIONS,
    fill_missing_sections,
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
