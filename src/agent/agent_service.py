"""
Agent service: polls SQLite job queue and processes jobs.
Run independently of the Streamlit UI.

Start with:
    python -m agent.agent_service

or via start.sh alongside the Streamlit UI process.
"""

import json
import os
import time

import openai

from agent.db import (
    complete_job,
    detect_stalled_jobs,
    get_conn,
    get_session_results,
    init_db,
    claim_next_job,
    write_result,
)
from agent.loop import run_session
from agent.profiler import load_csv, profile_csv, get_df_transfer_payload

# I-23: streamlit must never be imported here — verified by test_service_no_streamlit_import
# I-22: this service runs without the UI process being alive

POLL_INTERVAL = 2  # seconds


def _make_llm_client(api_key: str) -> openai.OpenAI:
    """OpenAI-compatible client pointed at the Grok (xAI) endpoint."""
    base_url = os.getenv("LLM_API_BASE_URL", "https://api.x.ai/v1")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def dispatch_job(job: dict, conn, api_key: str) -> None:
    """Route a claimed job to the correct handler.

    Supported job types:
    - ANALYSIS: profile the CSV and run the full agent reasoning loop.
    - FOLLOWUP: answer a follow-up question using prior session findings.
    """
    job_type = job.get("job_type")
    if job_type == "ANALYSIS":
        _handle_analysis_job(job, conn, api_key)
    elif job_type == "FOLLOWUP":
        _handle_followup_job(job, conn, api_key)
    else:
        from agent.db import fail_job
        fail_job(conn, job["id"], f"unknown job_type: {job_type}")


def _handle_analysis_job(job: dict, conn, api_key: str) -> None:
    """Profile the uploaded CSV and run the full agent reasoning loop."""
    payload = job.get("payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)

    csv_path = payload.get("csv_path", "")

    try:
        df = load_csv(csv_path)
        profile = profile_csv(csv_path)
        df_payload = get_df_transfer_payload(df, csv_path)
    except Exception as exc:
        from agent.db import fail_job
        fail_job(conn, job["id"], f"profiling failed: {exc}")
        return

    max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    executor_timeout = int(os.getenv("EXECUTOR_TIMEOUT", "60"))

    try:
        run_session(
            job=job,
            conn=conn,
            profile=profile,
            df_payload=df_payload,
            api_key=api_key,
            max_iterations=max_iterations,
            executor_timeout=executor_timeout,
        )
    except Exception as exc:
        from agent.db import fail_job
        fail_job(conn, job["id"], f"run_session error: {exc}")


def _handle_followup_job(job: dict, conn, api_key: str) -> None:
    """Answer a follow-up question using prior session findings (no code execution).

    Loads COMPLETED results from the session, builds a context prompt, and
    writes the LLM answer as a new result entry. Never calls execute_code().
    """
    session_id = job["session_id"]
    payload = job.get("payload") or {}
    if isinstance(payload, str):
        payload = json.loads(payload)

    question = payload.get("question", "")

    # Load completed findings for context
    results = get_session_results(conn, session_id)
    findings_text = "\n".join(
        f"- {r['analysis_type']}: {(r['output'] or '')[:300]}"
        for r in results
        if r["status"] == "COMPLETED"
    )
    if not findings_text:
        findings_text = "(No completed analyses found for this session.)"

    prompt = (
        f"You are a data analyst. Prior findings from this session:\n"
        f"{findings_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer in 2-4 sentences based only on the findings above."
    )

    model = os.getenv("PLANNER_MODEL", "grok-3-mini")
    client = _make_llm_client(api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        answer = f"[Error generating answer: {exc}]"

    # Write answer — analysis_type prefixed with "followup:" (I-07: agent writes results)
    write_result(
        conn,
        session_id,
        job["id"],
        f"followup:{question[:100]}",
        "COMPLETED",
        output=answer,
    )
    complete_job(conn, job["id"])


def run_service(db_path: str, api_key: str) -> None:
    """Poll the SQLite job queue and dispatch jobs until interrupted (I-22, I-23)."""
    conn = get_conn(db_path)
    init_db(db_path)
    print(f"Agent service started. Polling every {POLL_INTERVAL}s...")

    while True:
        try:
            detect_stalled_jobs(conn)
            job = claim_next_job(conn)
            if job is None:
                time.sleep(POLL_INTERVAL)
                continue
            dispatch_job(job, conn, api_key)
        except KeyboardInterrupt:
            print("Shutting down.")
            break
        except Exception as exc:
            print(f"Error processing job: {exc}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    # NOTE: OPENAI_API_KEY holds the Grok (xAI) API key in this project.
    run_service(
        os.getenv("DB_PATH", "agent_state.db"),
        os.getenv("OPENAI_API_KEY", ""),
    )
