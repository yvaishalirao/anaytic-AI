import sqlite3

from .db import write_result, get_session_results


class SessionMemory:
    def __init__(self, conn: sqlite3.Connection, session_id: str):
        self.conn = conn
        self.session_id = session_id
        self._job_id = None  # Will be set when we have a job

    def set_job_id(self, job_id: str):
        """Set the current job ID for this session."""
        self._job_id = job_id

    def record_completed(self, analysis_type: str, output: str, chart_path: str | None):
        """Record a completed analysis."""
        if self._job_id is None:
            raise ValueError("Job ID not set. Call set_job_id() first.")
        write_result(
            self.conn,
            self.session_id,
            self._job_id,
            analysis_type,
            "COMPLETED",
            output,
            chart_path,
        )

    def record_failed(self, analysis_type: str, error: str):
        """Record a failed analysis."""
        if self._job_id is None:
            raise ValueError("Job ID not set. Call set_job_id() first.")
        write_result(
            self.conn,
            self.session_id,
            self._job_id,
            analysis_type,
            "FAILED",
            error,
            None,
        )

    def record_timeout(self, analysis_type: str):
        """Record a timed-out analysis."""
        if self._job_id is None:
            raise ValueError("Job ID not set. Call set_job_id() first.")
        write_result(
            self.conn,
            self.session_id,
            self._job_id,
            analysis_type,
            "TIMEOUT",
            None,
            None,
        )

    def get_completed(self) -> list[str]:
        """Get list of completed analysis types."""
        results = get_session_results(self.conn, self.session_id)
        return [
            result["analysis_type"]
            for result in results
            if result["status"] == "COMPLETED"
        ]

    def get_all_attempted(self) -> list[dict]:
        """Get all attempted analyses (any status)."""
        return get_session_results(self.conn, self.session_id)

    def is_done(self, analysis_type: str) -> bool:
        """Check if analysis_type has been completed successfully."""
        results = get_session_results(self.conn, self.session_id)
        for result in results:
            if result["analysis_type"] == analysis_type and result["status"] == "COMPLETED":
                return True
        return False
