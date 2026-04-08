import sqlite3
import uuid

import pytest

from agent.db import (
    claim_next_job,
    complete_job,
    detect_stalled_jobs,
    enqueue_job,
    fail_job,
    get_session_log,
    get_session_results,
    init_db,
    new_session_id,
    write_log_entry,
    write_result,
)
from agent.memory import SessionMemory


def test_wal_mode(tmp_path):
    db_path = tmp_path / "agent.db"
    conn = init_db(str(db_path))
    result = conn.execute("PRAGMA journal_mode;").fetchone()
    assert result is not None
    assert result[0].lower() == "wal"


def test_job_session_not_null(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO jobs(id, session_id, job_type) VALUES (?, ?, ?)",
            ("job-1", None, "ANALYSIS"),
        )


def test_result_session_not_null(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO results(session_id, job_id, analysis_type, status) VALUES (?, ?, ?, ?)",
            (None, "job-1", "summary", "COMPLETED"),
        )


def test_job_status_enum(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO jobs(id, session_id, job_type, status) VALUES (?, ?, ?, ?)",
            ("job-2", "test-session", "ANALYSIS", "INVALID"),
        )


def test_log_step_type_enum(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO reasoning_log(session_id, job_id, step_type, content, seq) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test-session", "job-1", "OTHER", "test content", 1),
        )


def test_session_id_uniqueness(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    ids = set()
    for _ in range(10_000):
        session_id = new_session_id(conn)
        assert len(session_id) == 36
        assert session_id.count("-") == 4
        ids.add(session_id)
    assert len(ids) == 10_000


def test_session_id_format(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)
    assert len(session_id) == 36
    assert session_id.count("-") == 4


def test_session_collision_raises(tmp_path, monkeypatch):
    conn = init_db(str(tmp_path / "agent.db"))
    collision_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO jobs(id, session_id, job_type) VALUES (?, ?, ?)",
        ("job-collision", collision_id, "ANALYSIS"),
    )

    class DummyUUID:
        def __str__(self):
            return collision_id

    monkeypatch.setattr("uuid.uuid4", lambda: DummyUUID())
    with pytest.raises(RuntimeError):
        new_session_id(conn)


def test_claim_empty_queue(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    job = claim_next_job(conn)
    assert job is None


def test_concurrent_claim(tmp_path):
    # Test atomicity by simulating concurrent access with separate connections
    db_path = str(tmp_path / "agent.db")

    # Initialize DB with first connection
    conn1 = init_db(db_path)
    session_id = new_session_id(conn1)

    # Enqueue a job with first connection
    job_id = enqueue_job(conn1, session_id, "ANALYSIS", {"test": "data"})

    # Create second connection and try to claim
    conn2 = init_db(db_path)  # This creates a new connection to the same DB

    # First claim should succeed
    job1 = claim_next_job(conn1)
    assert job1 is not None
    assert job1["id"] == job_id

    # Second claim should return None (job already claimed)
    job2 = claim_next_job(conn2)
    assert job2 is None


def test_job_lifecycle(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)

    # Enqueue
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {"test": "data"})

    # Claim
    job = claim_next_job(conn)
    assert job is not None
    assert job["id"] == job_id
    assert job["status"] == "PROCESSING"

    # Complete
    complete_job(conn, job_id)

    # Check final status
    row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row[0] == "DONE"


def test_stall_detection(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)

    # Enqueue and claim a job
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {"test": "data"})
    job = claim_next_job(conn)
    assert job is not None

    # Manually set claimed_at to old time (simulate stall)
    conn.execute(
        "UPDATE jobs SET claimed_at = unixepoch() - 400 WHERE id=?",
        (job_id,),
    )

    # Detect stalled jobs
    count = detect_stalled_jobs(conn, stall_threshold_s=300)
    assert count == 1

    # Check job is now FAILED
    row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row[0] == "FAILED"


def test_fail_job_reason(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)

    # Enqueue a job
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {"original": "data"})

    # Fail it with reason
    fail_job(conn, job_id, "test failure reason")

    # Check status and payload
    row = conn.execute("SELECT status, payload FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row[0] == "FAILED"

    import json
    payload = json.loads(row[1])
    assert payload["failure_reason"] == "test failure reason"
    assert payload["original"] == "data"


def test_log_no_update(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {})

    # Insert a log entry
    write_log_entry(conn, session_id, job_id, "PLAN", "test content", 1)

    # Try to update - should raise IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE reasoning_log SET content='modified' WHERE id=1")


def test_log_no_delete(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {})

    # Insert a log entry
    write_log_entry(conn, session_id, job_id, "PLAN", "test content", 1)

    # Try to delete - should raise IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM reasoning_log WHERE id=1")


def test_log_ordering(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {})

    # Insert log entries out of order
    write_log_entry(conn, session_id, job_id, "PLAN", "step 3", 3)
    write_log_entry(conn, session_id, job_id, "ACTION", "step 1", 1)
    write_log_entry(conn, session_id, job_id, "OBSERVE", "step 2", 2)

    # Get log should return in seq order
    log_entries = get_session_log(conn, session_id)
    assert len(log_entries) == 3
    assert log_entries[0]["seq"] == 1
    assert log_entries[1]["seq"] == 2
    assert log_entries[2]["seq"] == 3
    assert log_entries[0]["content"] == "step 1"
    assert log_entries[1]["content"] == "step 2"
    assert log_entries[2]["content"] == "step 3"


def test_session_isolation_results(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))

    # Create two sessions
    session_a = new_session_id(conn)
    session_b = new_session_id(conn)

    job_a = enqueue_job(conn, session_a, "ANALYSIS", {})
    enqueue_job(conn, session_b, "ANALYSIS", {})

    # Write results for session A
    write_result(conn, session_a, job_a, "summary", "COMPLETED", "output A", "chart.png")

    # Session B should see no results
    results_b = get_session_results(conn, session_b)
    assert len(results_b) == 0

    # Session A should see its result
    results_a = get_session_results(conn, session_a)
    assert len(results_a) == 1
    assert results_a[0]["session_id"] == session_a
    assert results_a[0]["output"] == "output A"


def test_memory_failed_not_done(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {})

    memory = SessionMemory(conn, session_id)
    memory.set_job_id(job_id)

    # Record a failed analysis
    memory.record_failed("summary", "error message")

    # Should not be considered done
    assert not memory.is_done("summary")


def test_memory_timeout_not_done(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {})

    memory = SessionMemory(conn, session_id)
    memory.set_job_id(job_id)

    # Record a timed-out analysis
    memory.record_timeout("summary")

    # Should not be considered done
    assert not memory.is_done("summary")


def test_memory_completed_done(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {})

    memory = SessionMemory(conn, session_id)
    memory.set_job_id(job_id)

    # Record a completed analysis
    memory.record_completed("summary", "analysis output", "chart.png")

    # Should be considered done
    assert memory.is_done("summary")


def test_memory_all_attempted(tmp_path):
    conn = init_db(str(tmp_path / "agent.db"))
    session_id = new_session_id(conn)
    job_id = enqueue_job(conn, session_id, "ANALYSIS", {})

    memory = SessionMemory(conn, session_id)
    memory.set_job_id(job_id)

    # Record different types of results
    memory.record_failed("summary", "error")
    memory.record_timeout("trends")
    memory.record_completed("anomalies", "output", None)

    # Get all attempted
    attempted = memory.get_all_attempted()
    assert len(attempted) == 3

    # Check statuses
    statuses = {result["status"] for result in attempted}
    assert "FAILED" in statuses
    assert "TIMEOUT" in statuses
    assert "COMPLETED" in statuses

    # Check analysis types
    types = {result["analysis_type"] for result in attempted}
    assert "summary" in types
    assert "trends" in types
    assert "anomalies" in types

