# Invariant Coverage

All 24 hard invariants from CLAUDE.md mapped to their automated test(s).

| Invariant | Description (summary) | Test(s) | Status |
|-----------|----------------------|---------|--------|
| I-01 | Every record carries a `session_id`; queries without `session_id` filter are bugs | `test_db.py::test_job_session_not_null`<br>`test_db.py::test_result_session_not_null`<br>`test_db.py::test_session_isolation_results` | ✅ |
| I-02 | Session IDs are UUID4; collision raises immediately | `test_db.py::test_session_id_format`<br>`test_db.py::test_session_id_uniqueness`<br>`test_db.py::test_session_collision_raises` | ✅ |
| I-03 | Charts written to `outputs/{session_id}/` only; path traversal in session ID is a hard error | `test_executor.py::test_dotdot_session_id`<br>`test_executor.py::test_slash_session_id` | ✅ |
| I-04 | Job holds exactly one status; transitions atomic via conditional `UPDATE WHERE status='PENDING'` | `test_db.py::test_job_status_enum`<br>`test_db.py::test_job_lifecycle`<br>`test_db.py::test_concurrent_claim` | ✅ |
| I-05 | Job transitions PENDING→PROCESSING exactly once; no job claimed twice | `test_db.py::test_concurrent_claim`<br>`test_db.py::test_job_lifecycle` | ✅ |
| I-06 | Stall-detection marks stuck jobs FAILED with reason `"stalled"` | `test_db.py::test_stall_detection` | ✅ |
| I-07 | `write_result` and `write_log_entry` must not appear in `ui/app.py` | `test_loop.py::test_ui_no_write_result_or_log_entry` | ✅ |
| I-08 | `exec()`, `eval()`, `compile()` must not appear in `executor.py` | `test_loop.py::test_executor_no_exec_eval_compile` | ✅ |
| I-09 | Subprocess returns structured JSON with `status`, `output`, `error`, `charts`; silence → TIMEOUT | `test_executor.py::test_simple_exec`<br>`test_executor.py::test_exec_error`<br>`test_executor.py::test_chart_capture`<br>`test_executor.py::test_empty_stdout` | ✅ |
| I-10 | Subprocess killed immediately on timeout — no grace period | `test_executor.py::test_execute_code_timeout`<br>`test_executor.py::test_timeout_enforcement`<br>`test_executor.py::test_infinite_loop_killed` | ✅ |
| I-11 | Subprocess side effects outside `outputs/{session_id}/` detected and deleted | `test_executor.py::test_no_escape_write`<br>`test_executor.py::test_session_isolation_creates_separate_outputs` | ✅ |
| I-12 | Raw DataFrame rows never in LLM prompt; `assert_no_raw_rows()` called in `profile_csv()` | `test_loop.py::test_prompt_no_raw_data`<br>`test_profiler.py::test_no_raw_rows`<br>`test_loop.py::test_prompt_includes_completed` | ✅ |
| I-13 | Planner never called without `completed_analyses` and `all_attempted` from live memory | `test_loop.py::test_prompt_includes_completed`<br>`test_loop.py::test_prompt_no_completed_shows_none`<br>`test_integration.py::test_full_session_mocked_llm` | ✅ |
| I-14 | Message history bounded; `maybe_summarise()` compresses after every third step | `test_loop.py::test_prompt_observation_cap`<br>`test_loop.py::test_summarise_triggers_on_step_3`<br>`test_loop.py::test_summarise_triggers_on_step_6`<br>`test_loop.py::test_history_after_summarise_is_shorter` | ✅ |
| I-15 | COMPLETED analysis never re-scheduled; `memory.is_done()` checked at top of each iteration | `test_db.py::test_memory_completed_done`<br>`test_loop.py::test_skip_completed`<br>`test_integration.py::test_full_session_mocked_llm` | ✅ |
| I-16 | FAILED and TIMEOUT results never treated as COMPLETED; `is_done()` returns `False` for both | `test_db.py::test_memory_failed_not_done`<br>`test_db.py::test_memory_timeout_not_done`<br>`test_executor.py::test_run_analysis_step_timeout_calls_record_timeout`<br>`test_executor.py::test_error_memory_record` | ✅ |
| I-17 | Report generator validates all four sections; missing sections get non-empty placeholders; empty body fails | `test_reporter.py::test_placeholder_nonempty`<br>`test_reporter.py::test_report_all_sections`<br>`test_integration.py::test_validator_catches_empty_body` | ✅ |
| I-18 | Chart references in report verified with `os.path.exists()` at write time; missing → explicit note | `test_reporter.py::test_no_broken_refs`<br>`test_reporter.py::test_chart_missing_no_broken_ref`<br>`test_integration.py::test_validator_catches_broken_ref` | ✅ |
| I-19 | Report written exactly once per session; second call raises `FileExistsError` | `test_reporter.py::test_report_no_overwrite`<br>`test_reporter.py::test_report_written` | ✅ |
| I-20 | ACTION log entry written before `subprocess.run()` is called | `test_executor.py::test_run_analysis_step_action_log_before_execute`<br>`test_executor.py::test_action_logged_before_exec`<br>`test_executor.py::test_observe_after_action` | ✅ |
| I-21 | Reasoning log is append-only; `BEFORE UPDATE` and `BEFORE DELETE` triggers raise `ABORT` | `test_db.py::test_log_no_update`<br>`test_db.py::test_log_no_delete`<br>`test_loop.py::test_logger_append_only` | ✅ |
| I-22 | UI renders waiting state when agent absent; agent polls queue without UI | `test_integration.py::test_polling_no_agent` | ✅ |
| I-23 | All IPC through SQLite; `agent_service` never imported by UI; `streamlit` never imported by agent | `test_loop.py::test_service_no_streamlit_import`<br>`test_loop.py::test_service_no_ui_import`<br>`test_loop.py::test_profiler_loop_executor_importable_without_service` | ✅ |
| I-24 | Output contract: non-empty reasoning log + ≥1 chart + four-section report; `validate_session_output()` asserted post-session | `test_integration.py::test_full_session_mocked_llm`<br>`test_integration.py::test_validator_passes_complete_session` | ✅ |
