import sqlite3

from agent.db import write_log_entry

_VALID_STEP_TYPES = frozenset({"PLAN", "ACTION", "OBSERVE"})


class ReasoningLogger:
    """Append-only wrapper around write_log_entry with a monotonic seq counter.

    Each instance owns its own counter — two loggers for different sessions
    never share state. step_type is validated before any DB write so a bad
    call raises ValueError without leaving a partial record (I-21).
    """

    def __init__(self, conn: sqlite3.Connection, session_id: str, job_id: str) -> None:
        self._conn = conn
        self._session_id = session_id
        self._job_id = job_id
        self._seq = 0

    def log(self, step_type: str, content: str) -> int:
        """Write one log entry and return its seq number.

        Args:
            step_type: Must be one of PLAN, ACTION, OBSERVE.
            content: Human-readable description of the step.

        Returns:
            The seq number assigned to this entry.

        Raises:
            ValueError: If step_type is not in the allowed set (raised before DB write).
        """
        if step_type not in _VALID_STEP_TYPES:
            raise ValueError(
                f"step_type must be one of {sorted(_VALID_STEP_TYPES)}, got {step_type!r}"
            )
        self._seq += 1
        write_log_entry(
            self._conn,
            self._session_id,
            self._job_id,
            step_type,
            content,
            self._seq,
        )
        return self._seq

    def next_seq(self) -> int:
        """Return the seq number that would be assigned to the next log() call."""
        return self._seq + 1

    # ------------------------------------------------------------------
    # Convenience methods used by executor.run_analysis_step()
    # ------------------------------------------------------------------

    def log_plan(self, content: str) -> int:
        return self.log("PLAN", content)

    def log_action(self, content: str) -> int:
        return self.log("ACTION", content)

    def log_observe(self, content: str) -> int:
        return self.log("OBSERVE", content)
