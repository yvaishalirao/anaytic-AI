import sqlite3
import uuid

import pytest

from agent.db import init_db, new_session_id, enqueue_job, claim_next_job, complete_job, fail_job, detect_stalled_jobs


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
            "INSERT INTO reasoning_log(session_id, job_id, step_type, content, seq) VALUES (?, ?, ?, ?, ?)",
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

