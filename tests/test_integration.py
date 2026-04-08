"""Integration tests — Session 5, tasks 5.5 TC-4 and 5.6."""

import csv
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.db import (
    enqueue_job,
    get_conn,
    get_session_log,
    get_session_results,
    init_db,
    new_session_id,
)
from agent.memory import SessionMemory
from agent.profiler import get_df_transfer_payload, profile_csv
from agent.reporter import validate_session_output


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn(tmp_path):
    db_path = str(tmp_path / "integration.db")
    conn = init_db(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture
def sales_csv(tmp_path):
    path = tmp_path / "sales.csv"
    regions = ["North", "South", "East", "West"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "region", "sales", "units"])
        for i in range(40):
            writer.writerow([
                f"2023-{(i % 12) + 1:02d}-01",
                regions[i % 4],
                round(1000 + i * 10.0, 2),
                20 + (i % 5),
            ])
    return path


def _make_analysis_job(conn, session_id: str, csv_path: str) -> dict:
    """Enqueue an ANALYSIS job and return its row as a dict."""
    job_id = enqueue_job(
        conn, session_id, "ANALYSIS",
        {"csv_path": str(csv_path), "file_name": Path(csv_path).name},
    )
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row)


# ---------------------------------------------------------------------------
# 5.5 TC-4 — Service processes a queued job end-to-end (mocked LLM)
# ---------------------------------------------------------------------------

@patch("agent.loop.call_planner_llm")
@patch("agent.loop.generate_report")
def test_service_processes_job(mock_report, mock_llm, db_conn, sales_csv):
    """Agent service claims a PENDING job and transitions it to DONE (5.5 TC-4)."""
    from agent.agent_service import dispatch_job

    mock_llm.return_value = {"analysis_type": "DONE", "rationale": "done", "code": ""}
    mock_report.return_value = None

    session_id = new_session_id(db_conn)
    job = _make_analysis_job(db_conn, session_id, str(sales_csv))

    # Claim the job (simulating what claim_next_job does)
    db_conn.execute(
        "UPDATE jobs SET status='PROCESSING', claimed_at=unixepoch() WHERE id=?",
        (job["id"],),
    )
    job["status"] = "PROCESSING"

    dispatch_job(job, db_conn, api_key="test-key")

    # Job must now be DONE
    row = db_conn.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()
    assert row["status"] == "DONE", f"Expected DONE, got {row['status']}"


# ---------------------------------------------------------------------------
# 5.6 — Full end-to-end session with mocked LLM
# ---------------------------------------------------------------------------

@patch("agent.loop.call_planner_llm")
def test_full_session_mocked_llm(mock_llm, db_conn, sales_csv, monkeypatch, tmp_path):
    """Full PLAN→ACTION→OBSERVE loop with two real analysis steps then DONE (5.6).

    Assertions:
      1. reasoning_log has >= 6 entries
      2. reasoning_log entries are in monotonically increasing seq order
      3. memory.get_completed() contains both analysis types
      4. memory.is_done("histogram_sales") is True
      5. job status is DONE in the DB
      6. No analysis appears twice in results
      7. Report file exists at outputs/{session_id}/report.md
    """
    # Change CWD to tmp_path so outputs/ is created there (auto-restored by monkeypatch)
    monkeypatch.chdir(tmp_path)

    import pandas as pd

    df = pd.read_csv(str(sales_csv))
    profile = profile_csv(str(sales_csv))
    df_payload = get_df_transfer_payload(df, str(sales_csv))

    session_id = new_session_id(db_conn)
    job = _make_analysis_job(db_conn, session_id, str(sales_csv))
    db_conn.execute(
        "UPDATE jobs SET status='PROCESSING', claimed_at=unixepoch() WHERE id=?",
        (job["id"],),
    )
    job["status"] = "PROCESSING"

    # LLM sequence: two real analyses then DONE
    mock_llm.side_effect = iter([
        {
            "analysis_type": "histogram_sales",
            "rationale": "understand sales distribution",
            "code": "print(df['sales'].describe())",
        },
        {
            "analysis_type": "count_by_region",
            "rationale": "understand regional breakdown",
            "code": "print(df.groupby('region').size())",
        },
        {
            "analysis_type": "DONE",
            "rationale": "all key analyses complete",
            "code": "",
        },
    ])

    from agent.loop import run_session
    run_session(job, db_conn, profile, df_payload, api_key="test")

    # --- 1. reasoning_log has >= 6 entries ---
    log_entries = get_session_log(db_conn, session_id)
    assert len(log_entries) >= 6, (
        f"Expected >= 6 log entries, got {len(log_entries)}: "
        + str([(e["step_type"], e["seq"]) for e in log_entries])
    )

    # --- 2. seq values are monotonically increasing ---
    seqs = [e["seq"] for e in log_entries]
    assert seqs == sorted(seqs), f"seq not monotonically increasing: {seqs}"
    assert len(seqs) == len(set(seqs)), f"duplicate seq values: {seqs}"

    # --- 3. both analyses recorded as COMPLETED ---
    memory = SessionMemory(db_conn, session_id)
    completed = memory.get_completed()
    assert "histogram_sales" in completed, f"histogram_sales missing from: {completed}"
    assert "count_by_region" in completed, f"count_by_region missing from: {completed}"

    # --- 4. is_done returns True for histogram_sales ---
    assert memory.is_done("histogram_sales") is True

    # --- 5. job status is DONE ---
    row = db_conn.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()
    assert row["status"] == "DONE", f"Expected DONE, got {row['status']}"

    # --- 6. no analysis type appears twice in results ---
    results = get_session_results(db_conn, session_id)
    analysis_types = [r["analysis_type"] for r in results]
    assert len(analysis_types) == len(set(analysis_types)), (
        f"Duplicate analysis types in results: {analysis_types}"
    )

    # --- 7. report file exists ---
    report_path = tmp_path / "outputs" / session_id / "report.md"
    assert report_path.exists(), f"Report not found at {report_path}"

    # --- 8. structural validator passes (6.4 TC-1, I-24) ---
    result = validate_session_output(
        session_id,
        outputs_base=str(tmp_path / "outputs"),
        conn=db_conn,
    )
    assert result["valid"] is True, f"validate_session_output failed: {result['errors']}"


# ---------------------------------------------------------------------------
# 6.4 — validate_session_output unit tests
# ---------------------------------------------------------------------------

def _poll_status(conn, session_id: str) -> str:
    """Replicate the polling query from app.py without Streamlit."""
    import sqlite3
    conn.row_factory = sqlite3.Row
    job_row = conn.execute(
        "SELECT status FROM jobs WHERE session_id=?", (session_id,)
    ).fetchone()
    return job_row["status"] if job_row else "PENDING"


def test_polling_no_agent(db_conn, sales_csv):
    """Polling with no agent running returns PENDING without error (7.2 TC-1, I-22)."""
    session_id = new_session_id(db_conn)
    enqueue_job(
        db_conn,
        session_id,
        "ANALYSIS",
        {"csv_path": str(sales_csv), "file_name": "sales.csv"},
    )

    # Replicate app.py polling logic — no agent service, job stays PENDING
    status = _poll_status(db_conn, session_id)
    assert status == "PENDING", f"Expected PENDING, got {status}"

    # get_session_log must also not raise and return empty list
    log = get_session_log(db_conn, session_id)
    assert log == []


def test_log_render_types(db_conn):
    """PLAN, ACTION, OBSERVE log entries are all handled by the icon mapping (7.2 TC-2)."""
    from agent.db import write_log_entry

    session_id = new_session_id(db_conn)
    job_id = enqueue_job(db_conn, session_id, "ANALYSIS", {})

    for seq, step_type in enumerate(["PLAN", "ACTION", "OBSERVE"], start=1):
        write_log_entry(db_conn, session_id, job_id, step_type, f"content {seq}", seq)

    log = get_session_log(db_conn, session_id)
    step_types = [e["step_type"] for e in log]
    assert "PLAN" in step_types
    assert "ACTION" in step_types
    assert "OBSERVE" in step_types


def test_polling_stops_on_done(db_conn, sales_csv):
    """Polling returns DONE once the job is complete (7.2 TC-3)."""
    session_id = new_session_id(db_conn)
    job_id = enqueue_job(
        db_conn,
        session_id,
        "ANALYSIS",
        {"csv_path": str(sales_csv), "file_name": "sales.csv"},
    )

    # Transition job to DONE (simulating agent completing)
    db_conn.execute("UPDATE jobs SET status='PROCESSING' WHERE id=?", (job_id,))
    db_conn.execute("UPDATE jobs SET status='DONE' WHERE id=?", (job_id,))

    status = _poll_status(db_conn, session_id)
    assert status == "DONE"


def test_upload_creates_job(db_conn, sales_csv):
    """Uploading a CSV enqueues a PENDING ANALYSIS job in the DB (7.1 TC-3)."""
    from agent.db import new_session_id, enqueue_job

    # Simulate what app.py does on file upload
    session_id = new_session_id(db_conn)
    csv_path = str(sales_csv)

    job_id = enqueue_job(
        db_conn,
        session_id,
        "ANALYSIS",
        {"csv_path": csv_path, "file_name": "sales.csv"},
    )

    # Verify job is visible with PENDING status
    row = db_conn.execute(
        "SELECT status, job_type, session_id FROM jobs WHERE id=?", (job_id,)
    ).fetchone()

    assert row is not None
    assert row["status"] == "PENDING"
    assert row["job_type"] == "ANALYSIS"
    assert row["session_id"] == session_id


def test_validator_catches_missing_section(tmp_path):
    """Partial report without Anomalies section → valid=False, errors mention Anomalies (6.4 TC-2)."""
    session_id = "test-session-validator-001"
    session_dir = tmp_path / "outputs" / session_id
    session_dir.mkdir(parents=True)

    # Write a report that is missing ## Anomalies
    (session_dir / "report.md").write_text(
        "## Dataset Summary\n\nSome summary.\n\n"
        "## Key Trends\n\nSome trends.\n\n"
        "## Recommendations\n\nSome recs.\n",
        encoding="utf-8",
    )
    # Put a dummy chart so check 4 passes
    (session_dir / "hist.png").write_bytes(b"PNG")

    result = validate_session_output(session_id, outputs_base=str(tmp_path / "outputs"))

    assert result["valid"] is False
    assert any("Anomalies" in e for e in result["errors"]), (
        f"Expected an error mentioning 'Anomalies', got: {result['errors']}"
    )


def test_validator_catches_broken_ref(tmp_path):
    """Report with a broken ![...](path) reference → valid=False (6.4 TC-3)."""
    session_id = "test-session-validator-002"
    session_dir = tmp_path / "outputs" / session_id
    session_dir.mkdir(parents=True)

    broken_path = str(session_dir / "missing_chart.png")  # file does NOT exist
    (session_dir / "report.md").write_text(
        f"## Dataset Summary\n\nSummary.\n\n"
        f"## Key Trends\n\nTrends.\n\n"
        f"## Anomalies\n\nAnomalies.\n\n"
        f"## Recommendations\n\nRecs.\n\n"
        f"![chart]({broken_path})\n",
        encoding="utf-8",
    )
    (session_dir / "real.png").write_bytes(b"PNG")  # chart count satisfied

    result = validate_session_output(session_id, outputs_base=str(tmp_path / "outputs"))

    assert result["valid"] is False
    assert any("Broken image" in e or "broken" in e.lower() for e in result["errors"]), (
        f"Expected broken-ref error, got: {result['errors']}"
    )


def test_validator_catches_empty_body(tmp_path):
    """Report with an empty section body → valid=False (6.4 TC-4)."""
    session_id = "test-session-validator-003"
    session_dir = tmp_path / "outputs" / session_id
    session_dir.mkdir(parents=True)

    # Anomalies section exists but has no body text
    (session_dir / "report.md").write_text(
        "## Dataset Summary\n\nSummary.\n\n"
        "## Key Trends\n\nTrends.\n\n"
        "## Anomalies\n\n"          # header only — body is empty
        "## Recommendations\n\nRecs.\n",
        encoding="utf-8",
    )
    (session_dir / "hist.png").write_bytes(b"PNG")

    result = validate_session_output(session_id, outputs_base=str(tmp_path / "outputs"))

    assert result["valid"] is False
    assert any("empty" in e.lower() for e in result["errors"]), (
        f"Expected empty-body error, got: {result['errors']}"
    )


def test_validator_passes_complete_session(tmp_path):
    """A fully valid report + chart → valid=True, errors=[] (6.4 TC-1 standalone)."""
    session_id = "test-session-validator-004"
    session_dir = tmp_path / "outputs" / session_id
    session_dir.mkdir(parents=True)

    chart = session_dir / "hist.png"
    chart.write_bytes(b"PNG")

    (session_dir / "report.md").write_text(
        f"## Dataset Summary\n\nSummary text.\n\n"
        f"## Key Trends\n\nTrend text.\n\n"
        f"## Anomalies\n\nAnomaly text.\n\n"
        f"## Recommendations\n\nRecommendations text.\n\n"
        f"![chart]({chart})\n",
        encoding="utf-8",
    )

    result = validate_session_output(session_id, outputs_base=str(tmp_path / "outputs"))

    assert result["valid"] is True
    assert result["errors"] == []
