"""Streamlit UI for the Autonomous Data Analyst Agent.

Process boundary (I-07, I-23):
  - This file ONLY reads from / writes to the SQLite jobs table via enqueue_job().
  - Results and reasoning log are read-only from the UI side.
  - The agent service process is never imported here.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent.db import (
    enqueue_job,
    get_conn,
    get_session_log,
    get_session_results,
    init_db,
    new_session_id,
)

# UI WRITE BOUNDARY: only enqueue_job() is permitted
# write_result() and write_log_entry() must never be called from this file.

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "agent_state.db")
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))

st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("Autonomous Data Analyst Agent")

# ---------------------------------------------------------------------------
# Initialise DB (idempotent — safe to call on every page load)
# ---------------------------------------------------------------------------
init_db(DB_PATH)

# ---------------------------------------------------------------------------
# File upload section
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None and "session_id" not in st.session_state:
    # Generate session_id first so we can create an isolated upload directory
    conn = get_conn(DB_PATH)
    session_id = new_session_id(conn)

    # Save CSV to uploads/{session_id}/ so the agent service can access it by path
    upload_dir = UPLOADS_DIR / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    csv_path = upload_dir / uploaded_file.name
    csv_path.write_bytes(uploaded_file.getvalue())

    # Enqueue ANALYSIS job — the only write the UI makes (I-07)
    enqueue_job(
        conn,
        session_id,
        "ANALYSIS",
        {"csv_path": str(csv_path), "file_name": uploaded_file.name},
    )

    # Persist session state across reruns
    st.session_state["session_id"] = session_id
    st.session_state["file_name"] = uploaded_file.name

    st.success(f"Analysis queued. Session: {session_id[:8]}...")
    conn.close()

# ---------------------------------------------------------------------------
# Polling + display section (rendered whenever a session is active)
# ---------------------------------------------------------------------------
if "session_id" in st.session_state:
    session_id = st.session_state["session_id"]
    conn = get_conn(DB_PATH)
    conn.row_factory = __import__("sqlite3").Row

    # Poll current job status
    job_row = conn.execute(
        "SELECT status FROM jobs WHERE session_id=?", (session_id,)
    ).fetchone()
    status = job_row["status"] if job_row else "PENDING"

    st.caption(f"Session `{session_id[:8]}...` — status: **{status}**")

    # --- Reasoning Log ---
    st.subheader("Reasoning Log")
    log_entries = get_session_log(conn, session_id)
    if not log_entries:
        st.info("Waiting for agent to start...")
    else:
        for entry in log_entries:
            icon = {"PLAN": "🗺️", "ACTION": "⚙️", "OBSERVE": "🔍"}.get(
                entry["step_type"], "•"
            )
            st.write(f"{icon} **{entry['step_type']}** — {entry['content']}")

    # --- Charts and Report (only when done) ---
    if status == "DONE":
        st.subheader("Generated Charts")
        results = get_session_results(conn, session_id)
        chart_paths = [r["chart_path"] for r in results if r.get("chart_path")]
        if chart_paths:
            cols = st.columns(min(len(chart_paths), 3))
            for i, path in enumerate(chart_paths):
                if os.path.exists(path):
                    cols[i % 3].image(path)
        else:
            st.info("No charts were generated.")

        st.subheader("Final Report")
        report_path = f"outputs/{session_id}/report.md"
        if os.path.exists(report_path):
            st.markdown(Path(report_path).read_text(encoding="utf-8"))
        else:
            st.warning("Report not yet available.")

        # --- Follow-up Q&A ---
        st.subheader("Ask a Follow-up Question")
        followup_q = st.text_input("Your question:", key="followup_input")
        if st.button("Ask") and followup_q.strip():
            # UI WRITE BOUNDARY: only enqueue_job() is permitted
            enqueue_job(
                conn,
                session_id,
                "FOLLOWUP",
                {"question": followup_q, "session_id": session_id},
            )
            st.info("Question queued. Refresh to see the answer.")

        # Display prior Q&A answers written by the agent service
        followup_results = [
            r for r in results if r["analysis_type"].startswith("followup:")
        ]
        for r in followup_results:
            question = r["analysis_type"].replace("followup:", "", 1)
            st.markdown(f"**Q:** {question}")
            st.markdown(f"**A:** {r['output']}")
            st.divider()

    elif status == "FAILED":
        st.error("Analysis failed. Check the reasoning log for details.")

    # Auto-rerun while still processing
    if status not in ("DONE", "FAILED"):
        import time
        time.sleep(2)
        st.rerun()

    conn.close()
