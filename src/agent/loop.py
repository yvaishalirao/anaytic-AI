import sqlite3

from agent.db import complete_job, write_log_entry
from agent.executor import run_analysis_step
from agent.memory import SessionMemory
from agent.planner import build_planner_prompt, call_planner_llm, maybe_summarise
from agent.reporter import generate_report

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


def run_session(
    job: dict,
    conn: sqlite3.Connection,
    profile: dict,
    df_payload: dict,
    api_key: str,
    max_iterations: int = 10,
    executor_timeout: int = 60,
) -> None:
    """Main agent loop. Called by the agent service for ANALYSIS jobs.

    PLAN → ACTION → OBSERVE per analysis step, with:
    - I-13: memory reloaded from DB at the top of every iteration
    - I-14: history compressed every MAX_HISTORY_STEPS steps
    - I-15: completed analyses skipped via memory.is_done()
    - I-20: PLAN entry logged before LLM call; ACTION entry logged inside
            run_analysis_step() before subprocess launch
    """
    session_id = job["session_id"]
    job_id = job["id"]

    memory = SessionMemory(conn, session_id)
    memory.set_job_id(job_id)

    logger = ReasoningLogger(conn, session_id, job_id)
    message_history: list[dict] = []
    iteration = 0

    while iteration < max_iterations:
        # 1. Load memory state from the live DB store (I-13, I-15)
        completed_analyses = memory.get_completed()
        all_attempted = memory.get_all_attempted()

        # 2. Maybe compress message history (I-14)
        message_history = maybe_summarise(message_history, iteration, api_key)

        # 3. Build planner prompt (metadata only, I-12)
        prior_observations = [
            e["content"]
            for e in _get_observe_entries(conn, session_id)
        ]
        prompt = build_planner_prompt(
            profile, completed_analyses, all_attempted, prior_observations
        )

        # 4. Log PLAN entry BEFORE calling the LLM (I-20 intent: pre-hoc logging)
        logger.log("PLAN", f"Planning step {iteration + 1}: calling planner")

        # 5. Call the LLM planner
        plan = call_planner_llm(prompt, message_history, api_key)

        analysis_type = plan.get("analysis_type", "DONE")
        rationale = plan.get("rationale", "")
        code = plan.get("code", "")

        # 6. Stop if planner signals done
        if analysis_type == "DONE":
            logger.log("OBSERVE", f"Planner signalled DONE: {rationale}")
            break

        # 7. Skip if already completed (I-15) — check live store, not local cache
        if memory.is_done(analysis_type):
            logger.log("OBSERVE", f"skipping — already done: {analysis_type}")
            iteration += 1
            continue

        # 8–9. ACTION is logged inside run_analysis_step() before subprocess (I-20)
        result = run_analysis_step(
            code=code,
            analysis_type=analysis_type,
            df_payload=df_payload,
            session_id=session_id,
            memory=memory,
            log=logger,
            timeout=executor_timeout,
        )

        # 10. Append planner exchange to history for context
        obs_summary = (
            f"analysis_type={analysis_type} status={result['status']} "
            f"output={result.get('output', '')[:200]}"
        )
        message_history.append({"role": "assistant", "content": str(plan)})
        message_history.append({"role": "user", "content": f"OBSERVATION: {obs_summary}"})

        # 12. Advance iteration
        iteration += 1

    # After loop: generate report and mark job done
    generate_report(session_id, memory, profile, conn, logger, api_key)
    complete_job(conn, job_id)


def _get_observe_entries(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Return OBSERVE log entries for this session, ordered by seq."""
    from agent.db import get_session_log
    return [e for e in get_session_log(conn, session_id) if e["step_type"] == "OBSERVE"]
