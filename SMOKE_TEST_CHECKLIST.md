# Smoke Test Checklist

Manual end-to-end verification steps after every major change.

---

## Setup

1. Copy the example env file and fill in your API key:
   ```
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY=<your-grok-or-openai-key>
   ```

2. Start both processes:
   ```
   bash start.sh
   ```

---

## Steps

3. Open `http://localhost:8501` in a browser.

4. Click **Browse files** and upload `fixtures/sample_sales.csv`.

5. Confirm an **"Analysis queued"** (or equivalent) message appears and the session ID is shown.

6. Watch the **Reasoning Log** panel update automatically (polling).  
   Confirm entries with types **PLAN**, **ACTION**, and **OBSERVE** are all visible before the run completes.

7. Once the run finishes (status = DONE), confirm **at least one chart image** appears in the Charts section.

8. Confirm the **Report** section displays Markdown with all four required headers:
   - `## Dataset Summary`
   - `## Key Trends`
   - `## Anomalies`
   - `## Recommendations`

9. In the **Ask a Follow-up Question** box, type:
   ```
   What were the key sales trends?
   ```
   Click **Ask** and confirm an **"Question queued"** info message appears.

10. Refresh the page (or wait for the next poll cycle).  
    Confirm the question and its answer appear in the Q&A section without a timeout error.

---

## Pass Criteria

All 10 steps complete without unhandled exceptions, blank sections, broken image references, or timeout errors.
