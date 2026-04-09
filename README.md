# Data Analyst Agent

An autonomous local data analysis agent. Upload a CSV file, and the agent plans and executes a series of analyses, generates labelled charts, and writes a structured four-section report — without any hardcoded analysis steps.

> **All analysis runs locally on uploaded data — no data leaves your machine.**

---

## Architecture

The system runs as **two independent OS processes** that communicate exclusively through a SQLite database (WAL mode). No sockets, no shared memory, no direct function calls across the process boundary.

```
┌─────────────────────────┐      SQLite (WAL)          ┌──────────────────────────┐
│     Streamlit UI        │ ◄─── jobs / results ────►  |     Agent Service        │
│   src/ui/app.py         │      reasoning_log         │  src/agent/agent_service │
│                         │                            │                          │
│  • File upload          │                            │  • Claims PENDING jobs   │
│  • Job enqueue          │                            │  • Profiles CSV          │
│  • Polling / log display│                            │  • Runs reasoning loop   │
│  • Chart / report render│                            │  • Generates report      │
│  • Follow-up Q&A        │                            │  • Writes results        │
└─────────────────────────┘                            └──────────────────────────┘
                                                                   │
                                                      ┌────────────┴─────────────┐
                                                      │   Subprocess Sandbox     │
                                                      │  subprocess_runner.py    │
                                                      │                          │
                                                      │  • LLM-generated code    │
                                                      │    runs in child process │
                                                      │  • Output isolated to    │
                                                      │    outputs/{session_id}/ │
                                                      │  • Killed on timeout     │
                                                      └──────────────────────────┘
```

**Reasoning loop** (PLAN → ACTION → OBSERVE) runs inside the Agent Service:

1. **PLAN** — LLM chooses the next analysis given the dataset profile and memory of completed work.
2. **ACTION** — Generated Python code is sent to the subprocess sandbox.
3. **OBSERVE** — Results and chart paths are written back; loop repeats until LLM returns `DONE`.

After the loop, the Agent Service calls the report generator, which assembles a four-section Markdown report and embeds verified chart references.

---

## Prerequisites

- Python 3.10 or later
- A Grok (xAI) API key — stored as `OPENAI_API_KEY` in `.env` (the OpenAI-compatible SDK is used with the xAI base URL)

> **Report generation requires an API key.** The LLM is called for analysis planning, history summarisation, section insights, and follow-up Q&A.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Configure the API key

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and set:

```
OPENAI_API_KEY=your-grok-api-key-here
```

### 3. Start both processes

**Option A — single script (recommended):**

```bash
# macOS / Linux
bash start.sh

# Windows PowerShell
.\venv\Scripts\python.exe -m agent.agent_service &
streamlit run src/ui/app.py
```

**Option B — two terminals:**

Terminal 1 (Agent Service):
```bash
python -m agent.agent_service
```

Terminal 2 (Streamlit UI):
```bash
streamlit run src/ui/app.py
```

### 4. Use the app

1. Open `http://localhost:8501`
2. Upload a CSV file
3. Watch the reasoning log populate (PLAN → ACTION → OBSERVE)
4. Charts and the four-section report appear when the session completes
5. Ask follow-up questions in the Q&A panel

---

## Running Tests

```bash
# Windows
powershell -File check.ps1

# macOS / Linux
bash check.sh
```

---

## Project Layout

```
src/
  agent/
    agent_service.py   # Service entry point, job dispatcher
    loop.py            # Reasoning loop, ReasoningLogger
    planner.py         # LLM planner, history summarisation
    executor.py        # run_analysis_step(), memory integration
    executor_core.py   # execute_code(), subprocess launcher
    subprocess_runner.py  # Sandboxed code execution (child process)
    reporter.py        # Report generation, section validation
    memory.py          # SessionMemory facade (SQLite-backed)
    profiler.py        # CSV profiling, DataFrame transfer
    db.py              # Schema, job queue, results store
  ui/
    app.py             # Streamlit UI (read-only on results)
tests/
  test_db.py           # Schema, job queue, memory, triggers
  test_executor.py     # Subprocess sandbox, timeout, isolation
  test_loop.py         # Planner, reasoning loop, import boundaries
  test_profiler.py     # CSV profiling, DataFrame serialisation
  test_reporter.py     # Report generation and validation
  test_integration.py  # End-to-end session and UI integration
outputs/               # Created at runtime — one subdirectory per session_id
INVARIANT_COVERAGE.md  # All 24 hard invariants mapped to their tests
SMOKE_TEST_CHECKLIST.md  # Manual end-to-end verification steps
```

---

## Invariant Coverage

All 24 hard invariants from the system specification are verified by automated tests.
See [INVARIANT_COVERAGE.md](INVARIANT_COVERAGE.md) for the full mapping.

---

## Notes

- **All analysis runs locally on uploaded data — no data leaves your machine.** The only external call is to the LLM API (Grok/xAI) for reasoning and report generation.
- The uploaded CSV is never modified or imputed — quality issues are detected and reported only.
- Each upload creates an isolated session (UUID4). Sessions do not share state or output directories.
- The UI remains responsive while the agent runs — polling is non-blocking and the agent service can be restarted independently.
