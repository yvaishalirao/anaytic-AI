import sqlite3


def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def new_session_id(conn: sqlite3.Connection) -> str:
    import uuid

    session_id = str(uuid.uuid4())
    row = conn.execute(
        "SELECT 1 FROM jobs WHERE session_id=? LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is not None:
        raise RuntimeError(
            f"Session ID collision (should never happen): {session_id}"
        )
    return session_id


def init_db(db_path: str) -> sqlite3.Connection:
    conn = get_conn(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                job_type TEXT NOT NULL CHECK(job_type IN ('ANALYSIS','FOLLOWUP')),
                status TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK(status IN ('PENDING','PROCESSING','DONE','FAILED')),
                payload TEXT,
                claimed_at REAL,
                created_at REAL NOT NULL DEFAULT (unixepoch())
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('COMPLETED','FAILED','TIMEOUT')),
                output TEXT,
                chart_path TEXT,
                created_at REAL NOT NULL DEFAULT (unixepoch())
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reasoning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                step_type TEXT NOT NULL CHECK(step_type IN ('PLAN','ACTION','OBSERVE')),
                content TEXT NOT NULL,
                seq INTEGER NOT NULL,
                created_at REAL NOT NULL DEFAULT (unixepoch())
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_results_session ON results(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_log_session ON reasoning_log(session_id)"
        )
    return conn


def enqueue_job(conn: sqlite3.Connection, session_id: str, job_type: str, payload: dict) -> str:
    import uuid
    import json

    job_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO jobs(id, session_id, job_type, payload) VALUES (?, ?, ?, ?)",
        (job_id, session_id, job_type, json.dumps(payload)),
    )
    return job_id


def claim_next_job(conn: sqlite3.Connection) -> dict | None:
    import json

    with conn:
        cursor = conn.execute(
            """
            UPDATE jobs SET status='PROCESSING', claimed_at=unixepoch()
            WHERE id=(
                SELECT id FROM jobs WHERE status='PENDING'
                ORDER BY created_at LIMIT 1
            ) AND status='PENDING'
            """,
        )
        if cursor.rowcount != 1:
            return None

        # Get the claimed job
        row = conn.execute(
            "SELECT id, session_id, job_type, status, payload, claimed_at, created_at FROM jobs WHERE status='PROCESSING' ORDER BY claimed_at DESC LIMIT 1"
        ).fetchone()

        if row is None:
            return None

        return {
            "id": row[0],
            "session_id": row[1],
            "job_type": row[2],
            "status": row[3],
            "payload": json.loads(row[4]) if row[4] else None,
            "claimed_at": row[5],
            "created_at": row[6],
        }


def complete_job(conn: sqlite3.Connection, job_id: str):
    cursor = conn.execute(
        "UPDATE jobs SET status='DONE' WHERE id=?",
        (job_id,),
    )
    assert cursor.rowcount == 1


def fail_job(conn: sqlite3.Connection, job_id: str, reason: str):
    import json

    # Get current payload
    row = conn.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row and row[0]:
        current_payload = json.loads(row[0])
    else:
        current_payload = {}

    # Add failure reason
    current_payload["failure_reason"] = reason

    conn.execute(
        "UPDATE jobs SET status='FAILED', payload=? WHERE id=?",
        (json.dumps(current_payload), job_id),
    )


def detect_stalled_jobs(conn: sqlite3.Connection, stall_threshold_s: int = 300) -> int:
    stalled_jobs = conn.execute(
        "SELECT id FROM jobs WHERE status='PROCESSING' AND claimed_at < (unixepoch() - ?)",
        (stall_threshold_s,),
    ).fetchall()

    count = 0
    for (job_id,) in stalled_jobs:
        fail_job(conn, job_id, "stalled")
        count += 1

    return count
