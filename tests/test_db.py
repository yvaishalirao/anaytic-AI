import sqlite3

import pytest

from agent.db import init_db


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

