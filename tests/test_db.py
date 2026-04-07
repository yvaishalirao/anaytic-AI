import sqlite3
import uuid

import pytest

from agent.db import init_db, new_session_id


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

