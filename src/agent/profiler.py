import os
import warnings

import pandas as pd

LARGE_FILE_ROW_THRESHOLD = 50_000


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        path: Path to the CSV file

    Returns:
        DataFrame containing the CSV data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the CSV has 0 rows or 0 columns
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file has 0 columns: {path}")

    if len(df) == 0:
        raise ValueError(f"CSV file has 0 rows: {path}")

    if len(df.columns) == 0:
        raise ValueError(f"CSV file has 0 columns: {path}")

    return df


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Infer column types based on pandas dtypes and data characteristics.

    Args:
        df: DataFrame to analyze

    Returns:
        Dict mapping column names to type labels:
        - "numeric": pandas numeric dtypes
        - "boolean": pandas bool dtype
        - "categorical": object dtype with low cardinality or not datetime
        - "datetime": object dtype that parses as datetime with high success rate
    """
    result = {}

    for col in df.columns:
        dtype = df[col].dtype
        series = df[col]

        # Check for pandas boolean dtype
        if dtype == "bool":
            result[col] = "boolean"
            continue

        # Check for pandas numeric dtypes
        if pd.api.types.is_numeric_dtype(dtype):
            result[col] = "numeric"
            continue

        # For object dtype, check various possibilities
        if dtype == "object":
            # Check if it looks like datetime
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format, so each element will be parsed individually",
                    category=UserWarning
                )
                try:
                    parsed = pd.to_datetime(series, errors="coerce")
                    success_rate = parsed.notna().mean()
                    if success_rate >= 0.8:  # 80% success rate
                        result[col] = "datetime"
                        continue
                except (ValueError, TypeError):
                    pass

            # Check cardinality for categorical
            n_unique = series.nunique()
            n_total = len(series)

            # Low cardinality: <= 50 unique values OR <= 20% of total rows
            if n_unique <= 50 or n_unique <= 0.2 * n_total:
                result[col] = "categorical"
            else:
                # High cardinality object columns default to categorical
                result[col] = "categorical"
            continue

        # Fallback for any other dtypes
        result[col] = "categorical"

    return result


def compute_summary_stats(df: pd.DataFrame, col_types: dict[str, str]) -> dict:
    """Compute summary statistics for each column based on its type.

    Args:
        df: DataFrame to analyze
        col_types: Dict mapping column names to type labels

    Returns:
        Dict mapping column names to their statistics
    """
    result = {}

    for col, col_type in col_types.items():
        if col not in df.columns:
            continue

        series = df[col]

        if col_type == "numeric":
            # Numeric statistics
            stats = {
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
            }
            result[col] = stats

        elif col_type == "categorical":
            # Categorical statistics
            value_counts = series.value_counts()
            top_values = value_counts.head(5).index.tolist()
            stats = {
                "cardinality": int(series.nunique()),
                "top_values": [str(v) for v in top_values],
            }
            result[col] = stats

        elif col_type == "datetime":
            # Datetime statistics - ensure it's parsed as datetime
            try:
                dt_series = pd.to_datetime(series, errors="coerce")
                min_date = dt_series.min()
                max_date = dt_series.max()
                stats = {
                    "min_date": min_date.isoformat() if pd.notna(min_date) else None,
                    "max_date": max_date.isoformat() if pd.notna(max_date) else None,
                }
                result[col] = stats
            except (ValueError, TypeError):
                # If datetime parsing fails, skip this column
                pass

        # Skip boolean and other types for now

    return result


def detect_missing(df: pd.DataFrame) -> dict[str, float]:
    """Detect missing values in DataFrame columns.

    Args:
        df: DataFrame to analyze

    Returns:
        Dict mapping column names to missing percentage (0.0-100.0)
        Only includes columns with missing_pct > 0
    """
    result = {}
    total_rows = len(df)

    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_pct = (missing_count / total_rows) * 100.0
            result[col] = round(missing_pct, 2)

    return result


def profile_csv(path: str) -> dict:
    """Profile a CSV file and return comprehensive metadata.

    Args:
        path: Path to the CSV file

    Returns:
        Dict with keys:
        - row_count: int
        - col_count: int
        - file_name: str (basename only)
        - columns: list of dicts with name, type, stats, missing_pct
        - quality_issues: list of strings describing problems
    """
    df = load_csv(path)
    col_types = infer_column_types(df)
    stats = compute_summary_stats(df, col_types)
    missing = detect_missing(df)

    # Build columns list
    columns = []
    for col in df.columns:
        col_info = {
            "name": col,
            "type": col_types.get(col, "unknown"),
            "stats": stats.get(col, {}),
            "missing_pct": missing.get(col, 0.0),
        }
        columns.append(col_info)

    # Build quality issues
    quality_issues = []
    for col_info in columns:
        if col_info["missing_pct"] > 0:
            quality_issues.append(
                f"column '{col_info['name']}' has {col_info['missing_pct']}% missing values"
            )

    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "file_name": os.path.basename(path),
        "columns": columns,
        "quality_issues": quality_issues,
    }


def assert_no_raw_rows(profile: dict, df: pd.DataFrame):
    """Assert that the profile contains no raw row data from the DataFrame.

    Args:
        profile: Profile dict returned by profile_csv
        df: Original DataFrame

    Raises:
        AssertionError: If raw row data is detected in the profile
    """
    import json

    # Sample up to 10 random rows
    sample_size = min(10, len(df))
    if sample_size > 0:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    # Convert profile to JSON string
    profile_json = json.dumps(profile, default=str)

    # Get all allowed values from stats (these are computed metadata, not raw data)
    allowed_values = set()
    for col_info in profile.get("columns", []):
        stats = col_info.get("stats", {})
        # Add all stat values (min, max, mean, std, cardinality, top_values, dates, etc.)
        for key, value in stats.items():
            if isinstance(value, list):
                allowed_values.update(str(v) for v in value)
            elif isinstance(value, (int, float)):
                # For numeric stats, allow both int and float representations
                allowed_values.add(str(value))
                allowed_values.add(str(int(value)) if value == int(value) else str(value))
            else:
                allowed_values.add(str(value))

    # Also allow the missing_pct values
    for col_info in profile.get("columns", []):
        allowed_values.add(str(col_info.get("missing_pct", 0)))

    # Check each cell in the sample
    for _, row in sample_df.iterrows():
        for cell_value in row:
            cell_str = str(cell_value)
            # Skip NaN values
            if cell_str == 'nan':
                continue
            # Skip if it's in allowed values (computed metadata)
            if cell_str in allowed_values:
                continue
            # Check if this raw cell value appears in the profile JSON
            if cell_str in profile_json:
                # Use word boundaries to avoid false positives (e.g., '4.0' in '24.0')
                import re
                if re.search(r'\b' + re.escape(cell_str) + r'\b', profile_json):
                    raise AssertionError(
                        f"Raw row data detected in profile: '{cell_str}' from DataFrame "
                        f"appears in profile JSON"
                    )


def serialize_df(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to parquet bytes.

    Args:
        df: DataFrame to serialize

    Returns:
        Bytes containing parquet data
    """
    import io
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    return buffer.getvalue()


def deserialize_df(data: bytes) -> pd.DataFrame:
    """Deserialize parquet bytes back to DataFrame.

    Args:
        data: Parquet bytes

    Returns:
        Reconstructed DataFrame
    """
    import io
    buffer = io.BytesIO(data)
    return pd.read_parquet(buffer)


def get_df_transfer_payload(df: pd.DataFrame, csv_path: str) -> dict:
    """Get transfer payload for DataFrame based on size.

    For small DataFrames, serializes to base64-encoded bytes.
    For large DataFrames, uses file path.

    Args:
        df: DataFrame to transfer
        csv_path: Path to the original CSV file

    Returns:
        Dict with mode and data/path
    """
    import base64

    if len(df) <= LARGE_FILE_ROW_THRESHOLD:
        serialized = serialize_df(df)
        encoded = base64.b64encode(serialized).decode()
        return {"mode": "bytes", "data": encoded}
    else:
        return {"mode": "path", "path": csv_path}
