import base64
import io
import json
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no popup windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _scan_files(root_dir: str) -> set[str]:
    files = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            files.add(os.path.abspath(os.path.join(dirpath, filename)))
    return files


def _serialize_df_from_payload(df_payload: dict) -> pd.DataFrame:
    mode = df_payload.get("mode")
    if mode == "bytes":
        data = df_payload.get("data")
        if data is None:
            raise ValueError("df_payload missing required 'data' for bytes mode")
        decoded = base64.b64decode(data)
        buffer = io.BytesIO(decoded)
        return pd.read_parquet(buffer)

    if mode == "path":
        path = df_payload.get("path")
        if path is None:
            raise ValueError("df_payload missing required 'path' for path mode")
        return pd.read_csv(path)

    raise ValueError(f"Unsupported df_payload mode: {mode}")


def _is_under_path(parent: str, child: str) -> bool:
    try:
        parent_abs = os.path.abspath(parent)
        child_abs = os.path.abspath(child)
        return os.path.commonpath([parent_abs, child_abs]) == parent_abs
    except ValueError:
        return False


def _find_new_pngs(outputs_dir: str, before: set[str], after: set[str]) -> list[str]:
    new_files = [path for path in after - before if path.lower().endswith(".png")]
    new_pngs = []
    for path in new_files:
        if _is_under_path(outputs_dir, path):
            new_pngs.append(os.path.basename(path))
    return new_pngs


def _cleanup_external_files(outputs_dir: str, before: set[str], after: set[str]) -> tuple[list[str], str]:
    new_paths = sorted(after - before)
    outside_paths = []
    warnings = []

    for path in new_paths:
        if not _is_under_path(outputs_dir, path):
            outside_paths.append(path)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif os.path.isdir(path):
                try:
                    os.rmdir(path)
                except OSError:
                    pass

    if outside_paths:
        warnings.append(
            "Detected and removed files created outside outputs_dir. "
            "Subprocess must not write outside outputs_dir."
        )
    return outside_paths, "\n".join(warnings)


def main() -> None:
    raw_input = sys.stdin.read()
    try:
        payload = json.loads(raw_input)
    except json.JSONDecodeError as exc:
        result = {
            "status": "error",
            "output": "",
            "error": f"Invalid JSON input: {exc}",
            "charts": [],
        }
        sys.stdout.write(json.dumps(result))
        return

    code = payload.get("code", "")
    df_payload = payload.get("df_payload", {})
    session_id = payload.get("session_id", "")
    outputs_dir = payload.get("outputs_dir", "")

    outputs_dir = os.path.abspath(outputs_dir)
    os.makedirs(outputs_dir, exist_ok=True)

    cwd = os.getcwd()
    before_files = _scan_files(cwd)
    before_outputs_files = _scan_files(outputs_dir)

    try:
        df = _serialize_df_from_payload(df_payload)
    except Exception as exc:
        result = {
            "status": "error",
            "output": "",
            "error": str(exc),
            "charts": [],
        }
        sys.stdout.write(json.dumps(result))
        return

    # Patch plt.savefig so every chart automatically rotates x-axis tick labels
    # before saving — runs in the child process only, no effect on the parent.
    _orig_savefig = plt.savefig

    def _auto_savefig(fname, *args, **kwargs):
        for ax in plt.gcf().get_axes():
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
        plt.tight_layout()
        _orig_savefig(fname, *args, **kwargs)

    plt.savefig = _auto_savefig

    namespace = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "outputs_dir": outputs_dir,
        "session_id": session_id,
    }

    captured = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = captured
        exec(code, namespace)
        exec_error = None
    except Exception:
        exec_error = traceback.format_exc()
    finally:
        sys.stdout = original_stdout

    after_files = _scan_files(cwd)
    after_outputs_files = _scan_files(outputs_dir)

    new_charts = _find_new_pngs(outputs_dir, before_outputs_files, after_outputs_files)
    outside_paths, cleanup_warning = _cleanup_external_files(outputs_dir, before_files, after_files)

    output_text = captured.getvalue().strip()
    if cleanup_warning:
        if output_text:
            output_text = f"{output_text}\n{cleanup_warning}"
        else:
            output_text = cleanup_warning

    if exec_error is not None:
        result = {
            "status": "error",
            "output": output_text,
            "error": exec_error,
            "charts": new_charts,
        }
    else:
        result = {
            "status": "success",
            "output": output_text,
            "error": None,
            "charts": new_charts,
        }

    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
