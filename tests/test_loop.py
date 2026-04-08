"""Tests for Session 5: Agent Reasoning Loop (tasks 5.1–5.5)."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from agent.planner import (
    MAX_HISTORY_STEPS,
    build_planner_prompt,
    call_planner_llm,
    maybe_summarise,
    summarise_history,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_profile():
    """Minimal profile dict matching profile_csv() output structure."""
    return {
        "file_name": "sales.csv",
        "row_count": 50,
        "col_count": 5,
        "columns": [
            {
                "name": "date",
                "type": "datetime",
                "stats": {"min_date": "2023-01-01", "max_date": "2023-12-01"},
                "missing_pct": 0.0,
            },
            {
                "name": "region",
                "type": "categorical",
                "stats": {"cardinality": 4, "top_values": ["North", "South", "East", "West"]},
                "missing_pct": 0.0,
            },
            {
                "name": "sales",
                "type": "numeric",
                "stats": {"min": 1000.0, "max": 1612.3, "mean": 1306.15, "std": 175.42},
                "missing_pct": 0.0,
            },
            {
                "name": "units",
                "type": "numeric",
                "stats": {"min": 20.0, "max": 28.0, "mean": 24.0, "std": 2.83},
                "missing_pct": 0.0,
            },
            {
                "name": "returned",
                "type": "numeric",
                "stats": {"min": 0.0, "max": 6.0, "mean": 2.98, "std": 2.05},
                "missing_pct": 10.0,
            },
        ],
        "quality_issues": ["column 'returned' has 10.0% missing values"],
    }


# ---------------------------------------------------------------------------
# 5.1 — build_planner_prompt
# ---------------------------------------------------------------------------

def test_prompt_includes_completed(sample_profile):
    """Completed analysis names must appear in the prompt (I-13, I-15)."""
    completed = ["histogram_sales", "count_by_region"]
    prompt = build_planner_prompt(sample_profile, completed, [], [])
    assert "histogram_sales" in prompt
    assert "count_by_region" in prompt


def test_prompt_done_instruction(sample_profile):
    """Prompt must contain the 'DONE' sentinel instruction (I-15)."""
    prompt = build_planner_prompt(sample_profile, [], [], [])
    assert "DONE" in prompt


def test_prompt_observation_cap(sample_profile):
    """Prior observations must be capped at 5 entries (I-14)."""
    observations = [f"obs_{i}" for i in range(10)]
    prompt = build_planner_prompt(sample_profile, [], [], observations)
    # Only the last 5 should appear
    for i in range(5):
        assert f"obs_{i}" not in prompt, f"obs_{i} should have been truncated"
    for i in range(5, 10):
        assert f"obs_{i}" in prompt, f"obs_{i} should be present"


def test_prompt_no_raw_data(sample_profile):
    """build_planner_prompt must not inject any extra raw row data (I-12).

    The function receives a profile dict (which itself is row-free) and must
    only render the metadata it was given. We verify that values we know are
    NOT in the profile stats do not appear in the prompt, proving the function
    does not reach outside the profile to grab raw DataFrame content.
    """
    # These sentinel strings are raw cell values that would never appear in
    # any computed statistic. If they show up in the prompt, the function
    # is pulling data it shouldn't.
    sentinel_raw_values = [
        "RAW_CELL_ALPHA_99XYZ",
        "RAW_CELL_BETA_77ABC",
    ]

    # Inject one sentinel into the profile's quality_issues so we can verify
    # the function does render profile content (positive control), but does
    # NOT render anything else.
    profile_with_sentinel = dict(sample_profile)
    profile_with_sentinel["quality_issues"] = ["RAW_CELL_ALPHA_99XYZ missing data"]

    prompt = build_planner_prompt(profile_with_sentinel, [], [], [])

    # Positive control: content from the profile DOES appear
    assert "RAW_CELL_ALPHA_99XYZ" in prompt

    # Negative control: a sentinel that is NOT in the profile must NOT appear
    assert "RAW_CELL_BETA_77ABC" not in prompt


def test_prompt_dataset_overview_present(sample_profile):
    """Prompt must contain the DATASET OVERVIEW section."""
    prompt = build_planner_prompt(sample_profile, [], [], [])
    assert "DATASET OVERVIEW" in prompt
    assert "sales.csv" in prompt
    assert "50" in prompt  # row_count


def test_prompt_constraints_present(sample_profile):
    """Prompt must list the coding constraints."""
    prompt = build_planner_prompt(sample_profile, [], [], [])
    assert "outputs_dir" in prompt
    assert "plt" in prompt
    assert "sns" in prompt


def test_prompt_no_completed_shows_none(sample_profile):
    """When no analyses completed, prompt notes that."""
    prompt = build_planner_prompt(sample_profile, [], [], [])
    assert "none yet" in prompt.lower()


def test_prompt_observations_listed(sample_profile):
    """Prior observations within cap appear in prompt."""
    observations = ["sales are trending up", "region North dominates"]
    prompt = build_planner_prompt(sample_profile, [], [], observations)
    assert "sales are trending up" in prompt
    assert "region North dominates" in prompt


# ---------------------------------------------------------------------------
# Helpers for 5.2 — mock LLM response builder
# ---------------------------------------------------------------------------

def _mock_completion(content: str) -> MagicMock:
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# 5.2 — call_planner_llm
# ---------------------------------------------------------------------------

@patch("agent.planner._make_client")
def test_llm_response_parsed(mock_make_client):
    """Valid JSON response is parsed and returned as dict (5.2 TC-1)."""
    payload = {"analysis_type": "histogram_sales", "rationale": "check dist", "code": "print(1)"}
    mock_make_client.return_value.chat.completions.create.return_value = (
        _mock_completion(json.dumps(payload))
    )

    result = call_planner_llm("some prompt", [], api_key="test-key")

    assert result["analysis_type"] == "histogram_sales"
    assert result["rationale"] == "check dist"
    assert result["code"] == "print(1)"


@patch("agent.planner._make_client")
def test_llm_parse_failure_returns_done(mock_make_client):
    """JSON parse failure returns the DONE sentinel (5.2 TC-2)."""
    mock_make_client.return_value.chat.completions.create.return_value = (
        _mock_completion("Not valid JSON at all!!!")
    )

    result = call_planner_llm("prompt", [], api_key="test-key")

    assert result["analysis_type"] == "DONE"
    assert result["rationale"] == "parse_error"
    assert result["code"] == ""


@patch("agent.planner._make_client")
def test_llm_strips_markdown_fences(mock_make_client):
    """Model wrapping JSON in ```json fences is handled gracefully."""
    payload = {"analysis_type": "trends", "rationale": "r", "code": "print(2)"}
    wrapped = f"```json\n{json.dumps(payload)}\n```"
    mock_make_client.return_value.chat.completions.create.return_value = (
        _mock_completion(wrapped)
    )

    result = call_planner_llm("prompt", [], api_key="test-key")
    assert result["analysis_type"] == "trends"


@patch("agent.planner._make_client")
def test_llm_missing_analysis_type_returns_done(mock_make_client):
    """JSON dict without 'analysis_type' key falls back to DONE sentinel."""
    mock_make_client.return_value.chat.completions.create.return_value = (
        _mock_completion('{"rationale": "something", "code": ""}')
    )

    result = call_planner_llm("prompt", [], api_key="test-key")
    assert result["analysis_type"] == "DONE"


# ---------------------------------------------------------------------------
# 5.2 — maybe_summarise / summarise_history
# ---------------------------------------------------------------------------

@patch("agent.planner.summarise_history")
def test_summarise_triggers_on_step_3(mock_summarise):
    """maybe_summarise fires when step_count == 3 (5.2 TC-3)."""
    mock_summarise.return_value = [{"role": "system", "content": "summary"}]
    history = [{"role": "user", "content": "x"}]

    result = maybe_summarise(history, step_count=3, api_key="key")

    mock_summarise.assert_called_once_with(history, "key")
    assert result == [{"role": "system", "content": "summary"}]


@patch("agent.planner.summarise_history")
def test_summarise_triggers_on_step_6(mock_summarise):
    """maybe_summarise fires when step_count == 6."""
    mock_summarise.return_value = [{"role": "system", "content": "summary"}]
    history = [{"role": "user", "content": "x"}]

    maybe_summarise(history, step_count=6, api_key="key")
    mock_summarise.assert_called_once()


@patch("agent.planner.summarise_history")
def test_summarise_triggers_on_step_9(mock_summarise):
    """maybe_summarise fires when step_count == 9."""
    mock_summarise.return_value = [{"role": "system", "content": "summary"}]
    history = [{"role": "user", "content": "x"}]

    maybe_summarise(history, step_count=9, api_key="key")
    mock_summarise.assert_called_once()


@patch("agent.planner.summarise_history")
def test_summarise_does_not_trigger_on_step_1(mock_summarise):
    """maybe_summarise does NOT fire on step 1 (5.2 TC-4)."""
    history = [{"role": "user", "content": "x"}]
    result = maybe_summarise(history, step_count=1, api_key="key")
    mock_summarise.assert_not_called()
    assert result is history


@patch("agent.planner.summarise_history")
def test_summarise_does_not_trigger_on_step_2(mock_summarise):
    """maybe_summarise does NOT fire on step 2."""
    history = [{"role": "user", "content": "x"}]
    maybe_summarise(history, step_count=2, api_key="key")
    mock_summarise.assert_not_called()


@patch("agent.planner.summarise_history")
def test_summarise_does_not_trigger_on_step_4(mock_summarise):
    """maybe_summarise does NOT fire on step 4."""
    history = [{"role": "user", "content": "x"}]
    maybe_summarise(history, step_count=4, api_key="key")
    mock_summarise.assert_not_called()


@patch("agent.planner.summarise_history")
def test_summarise_does_not_trigger_on_step_0(mock_summarise):
    """maybe_summarise does NOT fire on step 0 (before any step)."""
    history = [{"role": "user", "content": "x"}]
    maybe_summarise(history, step_count=0, api_key="key")
    mock_summarise.assert_not_called()


@patch("agent.planner._make_client")
def test_history_after_summarise_is_shorter(mock_make_client):
    """History after summarise_history is a single-element list (5.2 TC-5)."""
    mock_make_client.return_value.chat.completions.create.return_value = (
        _mock_completion("A concise summary of all analyses.")
    )
    long_history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]

    new_history = summarise_history(long_history, api_key="test-key")

    assert len(new_history) == 1
    assert new_history[0]["role"] == "system"
    assert "Prior analysis summary:" in new_history[0]["content"]
    assert len(new_history) < len(long_history)


# ---------------------------------------------------------------------------
# Exact test IDs required by the execution plan (aliases)
# ---------------------------------------------------------------------------

# 5.2 TC-2
@patch("agent.planner._make_client")
def test_llm_parse_failure(mock_make_client):
    """Alias: JSON parse failure returns DONE sentinel (plan ID: test_llm_parse_failure)."""
    mock_make_client.return_value.chat.completions.create.return_value = (
        _mock_completion("Not valid JSON!!!")
    )
    result = call_planner_llm("prompt", [], api_key="key")
    assert result["analysis_type"] == "DONE"


# 5.2 TC-3
@patch("agent.planner.summarise_history")
def test_summarise_triggers(mock_summarise):
    """Alias: maybe_summarise triggers on step 3 (plan ID: test_summarise_triggers)."""
    mock_summarise.return_value = [{"role": "system", "content": "s"}]
    maybe_summarise([{"role": "user", "content": "x"}], step_count=3, api_key="key")
    mock_summarise.assert_called_once()


# 5.2 TC-4
@patch("agent.planner.summarise_history")
def test_summarise_skips(mock_summarise):
    """Alias: maybe_summarise does NOT trigger on step 2 (plan ID: test_summarise_skips)."""
    maybe_summarise([{"role": "user", "content": "x"}], step_count=2, api_key="key")
    mock_summarise.assert_not_called()


# 5.2 TC-5
@patch("agent.planner._make_client")
def test_history_compressed(mock_make_client):
    """Alias: history after summarise is shorter (plan ID: test_history_compressed)."""
    mock_make_client.return_value.chat.completions.create.return_value = (
        _mock_completion("Summary text.")
    )
    old_history = [{"role": "user", "content": f"msg {i}"} for i in range(8)]
    new_history = summarise_history(old_history, api_key="key")
    assert len(new_history) < len(old_history)
