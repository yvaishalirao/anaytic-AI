"""Integration tests — Session 5, tasks 5.5 TC-4 and 5.6."""

import csv
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.db import (
    enqueue_job,
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
    from agent.db import enqueue_job, new_session_id

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


def _run_display_logic(conn, session_id: str, outputs_base: str) -> dict:
    """Replicate app.py display logic without Streamlit.

    Returns a dict recording which paths were checked with os.path.exists()
    and whether each existed, mirroring what app.py would render.
    """
    results = get_session_results(conn, session_id)
    chart_paths = [r["chart_path"] for r in results if r.get("chart_path")]
    report_path = os.path.join(outputs_base, session_id, "report.md")

    checked = {}
    for path in chart_paths:
        checked[path] = os.path.exists(path)
    checked[report_path] = os.path.exists(report_path)

    return {
        "chart_paths": chart_paths,
        "report_path": report_path,
        "checked": checked,
    }


def test_chart_display_missing(db_conn, tmp_path):
    """Display logic skips charts whose files are gone — no exception raised (7.3 TC-1)."""
    from agent.db import write_result

    session_id = new_session_id(db_conn)
    job_id = enqueue_job(db_conn, session_id, "ANALYSIS", {})

    # Record a chart path that does NOT exist on disk
    phantom_chart = str(tmp_path / "outputs" / session_id / "ghost.png")
    write_result(
        db_conn, session_id, job_id, "histogram", "COMPLETED", "output", phantom_chart
    )

    info = _run_display_logic(db_conn, session_id, str(tmp_path / "outputs"))

    # os.path.exists must have been called for the chart path
    assert phantom_chart in info["checked"]
    # The file does not exist — display logic must not raise
    assert info["checked"][phantom_chart] is False


def test_report_display(db_conn, tmp_path, monkeypatch):
    """Report Markdown is readable when report.md exists (7.3 TC-2)."""
    monkeypatch.chdir(tmp_path)

    session_id = new_session_id(db_conn)
    enqueue_job(db_conn, session_id, "ANALYSIS", {})

    # Create the report file
    report_dir = tmp_path / "outputs" / session_id
    report_dir.mkdir(parents=True)
    report_file = report_dir / "report.md"
    report_file.write_text(
        "## Dataset Summary\n\nSome content.\n\n"
        "## Key Trends\n\nTrends.\n\n"
        "## Anomalies\n\nAnomalies.\n\n"
        "## Recommendations\n\nRecs.\n",
        encoding="utf-8",
    )

    info = _run_display_logic(db_conn, session_id, str(tmp_path / "outputs"))

    # os.path.exists must have been called for the report
    assert info["report_path"] in info["checked"]
    # And the file must exist
    assert info["checked"][info["report_path"]] is True
    # Content is readable
    content = Path(info["report_path"]).read_text(encoding="utf-8")
    assert "Dataset Summary" in content


def test_failed_status_display(db_conn):
    """A FAILED job can be detected by polling (7.3 TC-3)."""
    session_id = new_session_id(db_conn)
    job_id = enqueue_job(db_conn, session_id, "ANALYSIS", {})
    db_conn.execute("UPDATE jobs SET status='FAILED' WHERE id=?", (job_id,))

    status = _poll_status(db_conn, session_id)
    assert status == "FAILED"


# ---------------------------------------------------------------------------
# 7.4 — Follow-up Q&A (plan test IDs)
# ---------------------------------------------------------------------------

def test_followup_dispatch(db_conn):
    """Enqueuing a follow-up question creates a FOLLOWUP job in the jobs table (7.4 TC-1)."""
    session_id = new_session_id(db_conn)
    # Simulate what app.py does when the user submits a follow-up question
    job_id = enqueue_job(
        db_conn,
        session_id,
        "FOLLOWUP",
        {"question": "What are the trends?", "session_id": session_id},
    )

    row = db_conn.execute(
        "SELECT job_type, status, session_id FROM jobs WHERE id=?", (job_id,)
    ).fetchone()

    assert row is not None
    assert row["job_type"] == "FOLLOWUP"
    assert row["status"] == "PENDING"
    assert row["session_id"] == session_id


@patch("agent.agent_service._make_llm_client")
def test_followup_answered(mock_client, db_conn):
    """FOLLOWUP job processed by agent service writes a result with followup: prefix (7.4 TC-2)."""
    from agent.agent_service import dispatch_job
    from agent.db import write_result

    mock_client.return_value.chat.completions.create.return_value.choices[
        0
    ].message.content = "Sales are increasing steadily."

    session_id = new_session_id(db_conn)
    analysis_job_id = enqueue_job(db_conn, session_id, "ANALYSIS", {})
    write_result(
        db_conn, session_id, analysis_job_id,
        "histogram_sales", "COMPLETED", "mean=1200", None,
    )

    job_id = enqueue_job(
        db_conn, session_id, "FOLLOWUP",
        {"question": "Any trends?", "session_id": session_id},
    )
    db_conn.execute(
        "UPDATE jobs SET status='PROCESSING', claimed_at=unixepoch() WHERE id=?",
        (job_id,),
    )
    job = dict(db_conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    dispatch_job(job, db_conn, api_key="test-key")

    results = get_session_results(db_conn, session_id)
    followup_results = [r for r in results if r["analysis_type"].startswith("followup:")]
    assert len(followup_results) >= 1, (
        f"Expected at least one followup: result, got: {[r['analysis_type'] for r in results]}"
    )
    assert followup_results[0]["status"] == "COMPLETED"
    assert followup_results[0]["output"] == "Sales are increasing steadily."


@patch("agent.agent_service._make_llm_client")
def test_followup_no_code_exec(mock_client, db_conn):
    """execute_code is never called during a FOLLOWUP dispatch (7.4 TC-3)."""
    from agent.agent_service import dispatch_job

    mock_client.return_value.chat.completions.create.return_value.choices[
        0
    ].message.content = "Answer without code."

    session_id = new_session_id(db_conn)
    job_id = enqueue_job(
        db_conn, session_id, "FOLLOWUP",
        {"question": "Is there a trend?", "session_id": session_id},
    )
    db_conn.execute(
        "UPDATE jobs SET status='PROCESSING', claimed_at=unixepoch() WHERE id=?",
        (job_id,),
    )
    job = dict(db_conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    with patch("agent.executor.execute_code") as mock_exec:
        dispatch_job(job, db_conn, api_key="test-key")

    mock_exec.assert_not_called()


# ---------------------------------------------------------------------------
# 7.4 — Follow-up Q&A (additional coverage)
# ---------------------------------------------------------------------------

@patch("agent.agent_service._make_llm_client")
def test_followup_job_dispatched_and_answered(mock_client, db_conn):
    """FOLLOWUP job dispatched by service writes answer with followup: prefix (7.4 TC-1)."""
    from agent.agent_service import dispatch_job
    from agent.db import write_result

    mock_client.return_value.chat.completions.create.return_value.choices[
        0
    ].message.content = "The sales trend is upward."

    session_id = new_session_id(db_conn)
    # Seed a completed analysis so the context is non-empty
    analysis_job_id = enqueue_job(db_conn, session_id, "ANALYSIS", {})
    write_result(
        db_conn, session_id, analysis_job_id,
        "histogram_sales", "COMPLETED", "mean=1200, std=150", None,
    )
    db_conn.execute(
        "UPDATE jobs SET status='DONE' WHERE id=?", (analysis_job_id,)
    )

    # Enqueue a FOLLOWUP job
    followup_job_id = enqueue_job(
        db_conn, session_id, "FOLLOWUP",
        {"question": "What is the sales trend?", "session_id": session_id},
    )
    db_conn.execute(
        "UPDATE jobs SET status='PROCESSING', claimed_at=unixepoch() WHERE id=?",
        (followup_job_id,),
    )
    followup_job = dict(db_conn.execute(
        "SELECT * FROM jobs WHERE id=?", (followup_job_id,)
    ).fetchone())

    dispatch_job(followup_job, db_conn, api_key="test-key")

    # Job must be DONE
    row = db_conn.execute(
        "SELECT status FROM jobs WHERE id=?", (followup_job_id,)
    ).fetchone()
    assert row["status"] == "DONE"

    # A result with analysis_type starting with "followup:" must exist
    results = get_session_results(db_conn, session_id)
    followup_results = [r for r in results if r["analysis_type"].startswith("followup:")]
    assert len(followup_results) == 1, (
        f"Expected 1 followup result, got: {[r['analysis_type'] for r in followup_results]}"
    )
    assert "What is the sales trend?" in followup_results[0]["analysis_type"]
    assert followup_results[0]["output"] == "The sales trend is upward."


@patch("agent.agent_service._make_llm_client")
def test_followup_answer_readable_in_ui(mock_client, db_conn):
    """Prior Q&A answers written by agent are readable by the UI display logic (7.4 TC-2)."""
    from agent.db import write_result

    mock_client.return_value.chat.completions.create.return_value.choices[
        0
    ].message.content = "Sales peaked in Q3."

    session_id = new_session_id(db_conn)
    job_id = enqueue_job(db_conn, session_id, "ANALYSIS", {})

    # Simulate agent writing a followup answer
    write_result(
        db_conn, session_id, job_id,
        "followup:How did sales perform?", "COMPLETED", "Sales peaked in Q3.", None,
    )

    # UI display logic: filter results where analysis_type starts with "followup:"
    results = get_session_results(db_conn, session_id)
    followup_results = [r for r in results if r["analysis_type"].startswith("followup:")]

    assert len(followup_results) == 1
    question = followup_results[0]["analysis_type"].replace("followup:", "", 1)
    assert question == "How did sales perform?"
    assert followup_results[0]["output"] == "Sales peaked in Q3."


@patch("agent.agent_service._make_llm_client")
def test_followup_no_findings_context(mock_client, db_conn):
    """FOLLOWUP job with no prior analyses falls back gracefully (7.4 TC-3)."""
    from agent.agent_service import dispatch_job

    mock_client.return_value.chat.completions.create.return_value.choices[
        0
    ].message.content = "No data to answer from."

    session_id = new_session_id(db_conn)
    job_id = enqueue_job(
        db_conn, session_id, "FOLLOWUP",
        {"question": "Any trends?", "session_id": session_id},
    )
    db_conn.execute(
        "UPDATE jobs SET status='PROCESSING', claimed_at=unixepoch() WHERE id=?",
        (job_id,),
    )
    job = dict(db_conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    # Must not raise even with empty session findings
    dispatch_job(job, db_conn, api_key="test-key")

    row = db_conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "DONE"

    results = get_session_results(db_conn, session_id)
    followup_results = [r for r in results if r["analysis_type"].startswith("followup:")]
    assert len(followup_results) == 1
    assert followup_results[0]["output"] == "No data to answer from."


@patch("agent.agent_service._make_llm_client")
def test_followup_llm_error_still_writes_result(mock_client, db_conn):
    """LLM failure in FOLLOWUP still writes a result entry and marks job DONE (7.4 TC-4)."""
    from agent.agent_service import dispatch_job

    mock_client.return_value.chat.completions.create.side_effect = RuntimeError("API down")

    session_id = new_session_id(db_conn)
    job_id = enqueue_job(
        db_conn, session_id, "FOLLOWUP",
        {"question": "What happened?", "session_id": session_id},
    )
    db_conn.execute(
        "UPDATE jobs SET status='PROCESSING', claimed_at=unixepoch() WHERE id=?",
        (job_id,),
    )
    job = dict(db_conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    dispatch_job(job, db_conn, api_key="test-key")

    row = db_conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "DONE"

    results = get_session_results(db_conn, session_id)
    followup_results = [r for r in results if r["analysis_type"].startswith("followup:")]
    assert len(followup_results) == 1
    # Output must be non-empty (error message, not silence)
    assert followup_results[0]["output"]
    assert "Error" in followup_results[0]["output"] or "error" in followup_results[0]["output"]


def test_display_checks_both_chart_and_report(db_conn, tmp_path):
    """os.path.exists is called for both the chart path and the report path (7.3 spec)."""
    from agent.db import write_result

    session_id = new_session_id(db_conn)
    job_id = enqueue_job(db_conn, session_id, "ANALYSIS", {})

    chart_path = str(tmp_path / "outputs" / session_id / "hist.png")
    write_result(
        db_conn, session_id, job_id, "histogram", "COMPLETED", "output", chart_path
    )
    report_path = str(tmp_path / "outputs" / session_id / "report.md")

    # Spy on os.path.exists to record every call
    original_exists = os.path.exists
    checked_paths: list[str] = []

    def spy_exists(p: str) -> bool:
        checked_paths.append(str(p))
        return original_exists(p)

    with patch("os.path.exists", side_effect=spy_exists):
        _run_display_logic(db_conn, session_id, str(tmp_path / "outputs"))

    assert chart_path in checked_paths, (
        f"os.path.exists not called for chart: {chart_path}"
    )
    assert report_path in checked_paths, (
        f"os.path.exists not called for report: {report_path}"
    )


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
