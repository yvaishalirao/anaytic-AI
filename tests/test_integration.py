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
