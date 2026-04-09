import json
import os

import openai

MAX_HISTORY_STEPS = 3  # summarise after this many steps
MAX_PRIOR_OBSERVATIONS = 5

# Grok (xAI) is OpenAI-compatible — point the SDK at xAI's endpoint.
_DEFAULT_BASE_URL = "https://api.x.ai/v1"
_DEFAULT_MODEL = "grok-3-mini"

DONE_SENTINEL = {"analysis_type": "DONE", "rationale": "parse_error", "code": ""}


def _render_profile(profile: dict) -> str:
    """Render a profile dict as structured text with no raw row data."""
    lines = []
    lines.append(f"File: {profile.get('file_name', 'unknown')}")
    lines.append(f"Rows: {profile.get('row_count', 0)}")
    lines.append(f"Columns: {profile.get('col_count', 0)}")
    lines.append("")
    lines.append("Column Details:")
    for col in profile.get("columns", []):
        name = col.get("name", "")
        col_type = col.get("type", "unknown")
        missing = col.get("missing_pct", 0.0)
        stats = col.get("stats", {})

        col_line = f"  - {name} ({col_type})"
        if missing > 0:
            col_line += f", {missing}% missing"

        stats_parts = []
        if col_type == "numeric":
            for k in ("min", "max", "mean", "std"):
                if k in stats:
                    stats_parts.append(f"{k}={stats[k]}")
        elif col_type == "categorical":
            if "cardinality" in stats:
                stats_parts.append(f"cardinality={stats['cardinality']}")
            if "top_values" in stats:
                stats_parts.append(f"top_values={stats['top_values']}")
        elif col_type == "datetime":
            if "min_date" in stats:
                stats_parts.append(f"min={stats['min_date']}")
            if "max_date" in stats:
                stats_parts.append(f"max={stats['max_date']}")

        if stats_parts:
            col_line += " [" + ", ".join(stats_parts) + "]"
        lines.append(col_line)

    quality_issues = profile.get("quality_issues", [])
    if quality_issues:
        lines.append("")
        lines.append("Quality Issues:")
        for issue in quality_issues:
            lines.append(f"  - {issue}")

    return "\n".join(lines)


def build_planner_prompt(
    profile: dict,
    completed_analyses: list[str],
    all_attempted: list[dict],
    prior_observations: list[str],
) -> str:
    """Build the LLM planner prompt from dataset profile and session memory.

    Args:
        profile: Dataset profile dict from profile_csv() — never contains raw rows.
        completed_analyses: List of analysis_type strings already COMPLETED.
        all_attempted: All result dicts for this session (any status).
        prior_observations: Recent OBSERVE log entries; capped to last 5.

    Returns:
        Prompt string for the planner LLM.
    """
    # Cap observations at MAX_PRIOR_OBSERVATIONS
    capped_observations = prior_observations[-MAX_PRIOR_OBSERVATIONS:]

    sections = []

    # 1. DATASET OVERVIEW
    sections.append("## DATASET OVERVIEW\n" + _render_profile(profile))

    # 2. COMPLETED ANALYSES
    if completed_analyses:
        completed_list = "\n".join(f"  - {a}" for a in completed_analyses)
    else:
        completed_list = "  (none yet)"
    sections.append("## COMPLETED ANALYSES\n" + completed_list)

    # 3. PRIOR OBSERVATIONS
    if capped_observations:
        obs_list = "\n".join(f"  [{i+1}] {obs}" for i, obs in enumerate(capped_observations))
    else:
        obs_list = "  (none yet)"
    sections.append("## PRIOR OBSERVATIONS\n" + obs_list)

    # 4. TASK
    task_text = (
        "## TASK\n"
        "Based on the dataset overview, completed analyses, and prior observations above,\n"
        "decide the SINGLE most valuable next analysis to run.\n\n"
        "Respond with ONLY a valid JSON object — no markdown, no explanation — in one of these formats:\n\n"
        "If there is a useful analysis to run:\n"
        '{\n'
        '  "analysis_type": "<short_snake_case_name>",\n'
        '  "rationale": "<why this analysis is valuable>",\n'
        '  "code": "<python code to execute>"\n'
        '}\n\n'
        "If all useful analyses are done, respond with:\n"
        '{\n'
        '  "analysis_type": "DONE",\n'
        '  "rationale": "<why no further analysis is needed>",\n'
        '  "code": ""\n'
        '}'
    )
    sections.append(task_text)

    # 5. CONSTRAINTS
    constraints_text = (
        "## CONSTRAINTS\n"
        "- Do NOT repeat any completed analysis listed above.\n"
        "- Your code may ONLY use: df (pandas DataFrame), pd (pandas), "
        "plt (matplotlib.pyplot), sns (seaborn).\n"
        "- Save all charts using plt.savefig(f\"{outputs_dir}/<name>.png\") "
        "where outputs_dir is already available in scope.\n"
        "- Print key results so they appear in the observation log.\n"
        "- Keep code concise and self-contained (no imports needed for the above libraries).\n"
        "- analysis_type must be unique across all completed analyses."
    )
    sections.append(constraints_text)

    return "\n\n".join(sections)


def _make_client(api_key: str) -> openai.OpenAI:
    """Create an OpenAI-compatible client pointed at the Grok (xAI) endpoint."""
    base_url = os.getenv("LLM_API_BASE_URL", _DEFAULT_BASE_URL)
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def call_planner_llm(
    prompt: str,
    message_history: list[dict],
    api_key: str,
) -> dict:
    """Call the Grok LLM to decide the next analysis step.

    Args:
        prompt: The planner prompt built by build_planner_prompt().
        message_history: Previous messages in this session (may be summarised).
        api_key: Grok (xAI) API key.

    Returns:
        Parsed JSON dict with keys: analysis_type, rationale, code.
        On any JSON parse failure returns the DONE sentinel.
    """
    model = os.getenv("PLANNER_MODEL", _DEFAULT_MODEL)
    system_msg = {
        "role": "system",
        "content": (
            "You are a data analysis planning assistant. "
            "Respond ONLY with a valid JSON object. No markdown, no explanation."
        ),
    }
    messages = [system_msg] + message_history + [{"role": "user", "content": prompt}]

    client = _make_client(api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""

    # Strip markdown fences in case the model wraps despite json_object mode
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return dict(DONE_SENTINEL)

    if not isinstance(parsed, dict) or "analysis_type" not in parsed:
        return dict(DONE_SENTINEL)

    return parsed


def summarise_history(
    message_history: list[dict],
    api_key: str,
) -> list[dict]:
    """Compress the message history into a single summary system message (I-14).

    Args:
        message_history: Full conversation history so far.
        api_key: Grok (xAI) API key.

    Returns:
        New message_history list with a single system message containing the summary.
    """
    if not message_history:
        return []

    model = os.getenv("PLANNER_MODEL", _DEFAULT_MODEL)
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in message_history
    )
    summarise_prompt = (
        "Summarise the following data analysis conversation in one paragraph. "
        "Capture which analyses were run and what was found. Be concise.\n\n"
        + history_text
    )
    client = _make_client(api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": summarise_prompt}],
    )
    summary = (response.choices[0].message.content or "").strip()

    return [{"role": "system", "content": f"Prior analysis summary: {summary}"}]


def maybe_summarise(
    message_history: list[dict],
    step_count: int,
    api_key: str,
) -> list[dict]:
    """Trigger history summarisation every MAX_HISTORY_STEPS steps (I-14).

    Args:
        message_history: Current message history.
        step_count: Number of analysis steps completed so far.
        api_key: Grok (xAI) API key.

    Returns:
        Either the compressed history (if summarisation triggered) or the
        original history unchanged.
    """
    if step_count > 0 and step_count % MAX_HISTORY_STEPS == 0:
        return summarise_history(message_history, api_key)
    return message_history
