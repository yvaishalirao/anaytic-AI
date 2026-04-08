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
