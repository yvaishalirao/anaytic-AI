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
