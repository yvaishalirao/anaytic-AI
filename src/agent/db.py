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
