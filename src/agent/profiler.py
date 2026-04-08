import os
import warnings

import pandas as pd


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
