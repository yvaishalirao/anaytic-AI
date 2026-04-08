import pandas as pd
import pytest

from agent.profiler import (
    assert_no_raw_rows,
    compute_summary_stats,
    deserialize_df,
    detect_missing,
    get_df_transfer_payload,
    infer_column_types,
    load_csv,
    profile_csv,
    serialize_df,
)


def test_infer_datetime(sales_csv):
    """Test that date column is inferred as datetime."""
    df = pd.read_csv(sales_csv)
    types = infer_column_types(df)

    assert "date" in types
    assert types["date"] == "datetime"


def test_infer_categorical(sales_csv):
    """Test that region column is inferred as categorical."""
    df = pd.read_csv(sales_csv)
    types = infer_column_types(df)

    assert "region" in types
    assert types["region"] == "categorical"


def test_infer_numeric(sales_csv):
    """Test that sales column is inferred as numeric."""
    df = pd.read_csv(sales_csv)
    types = infer_column_types(df)

    assert "sales" in types
    assert types["sales"] == "numeric"


def test_infer_boolean(tmp_path):
    """Test that boolean columns are inferred as boolean."""
    # Create a CSV with a boolean column
    csv_path = tmp_path / "boolean_test.csv"
    with open(csv_path, "w") as f:
        f.write("flag,value\n")
        f.write("True,10\n")
        f.write("False,20\n")
        f.write("True,30\n")

    df = pd.read_csv(csv_path)
    types = infer_column_types(df)

    assert "flag" in types
    assert types["flag"] == "boolean"


def test_load_missing_file():
    """Test that load_csv raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent_file.csv")


def test_load_empty_csv(tmp_path):
    """Test that load_csv raises ValueError for empty CSV."""
    # Create an empty CSV file
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w") as f:
        f.write("")  # Empty file

    with pytest.raises(ValueError, match="0 columns"):
        load_csv(str(csv_path))


def test_load_csv_no_columns(tmp_path):
    """Test that load_csv raises ValueError for CSV with no columns."""
    # Create a CSV with no columns (just headers with no data)
    csv_path = tmp_path / "no_columns.csv"
    with open(csv_path, "w") as f:
        f.write("\n")  # Just a newline

    with pytest.raises(ValueError, match="0 columns"):
        load_csv(str(csv_path))


def test_numeric_stats(sales_csv):
    """Test that numeric columns get proper summary statistics."""
    df = pd.read_csv(sales_csv)
    col_types = infer_column_types(df)
    stats = compute_summary_stats(df, col_types)

    assert "sales" in stats
    sales_stats = stats["sales"]
    assert "min" in sales_stats
    assert "max" in sales_stats
    assert "mean" in sales_stats
    assert "std" in sales_stats

    # All should be floats
    assert isinstance(sales_stats["min"], float)
    assert isinstance(sales_stats["max"], float)
    assert isinstance(sales_stats["mean"], float)
    assert isinstance(sales_stats["std"], float)


def test_categorical_stats(sales_csv):
    """Test that categorical columns get cardinality and top values."""
    df = pd.read_csv(sales_csv)
    col_types = infer_column_types(df)
    stats = compute_summary_stats(df, col_types)

    assert "region" in stats
    region_stats = stats["region"]
    assert "cardinality" in region_stats
    assert "top_values" in region_stats

    assert region_stats["cardinality"] == 4  # North, South, East, West
    assert isinstance(region_stats["top_values"], list)
    assert len(region_stats["top_values"]) <= 5


def test_missing_detection(sales_csv):
    """Test that missing values are detected in the returned column."""
    df = pd.read_csv(sales_csv)
    missing = detect_missing(df)

    # The 'returned' column should have some missing values
    assert "returned" in missing
    assert missing["returned"] > 0
    assert missing["returned"] < 50  # Should be around 10-20%


def test_no_missing(wide_csv):
    """Test that clean CSV returns empty missing dict."""
    df = pd.read_csv(wide_csv)
    missing = detect_missing(df)

    assert missing == {}


def test_profile_structure(sales_csv):
    """Test that profile_csv output has required top-level keys."""
    profile = profile_csv(str(sales_csv))

    required_keys = {"row_count", "col_count", "file_name", "columns", "quality_issues"}
    assert set(profile.keys()) == required_keys

    assert isinstance(profile["row_count"], int)
    assert isinstance(profile["col_count"], int)
    assert isinstance(profile["file_name"], str)
    assert isinstance(profile["columns"], list)
    assert isinstance(profile["quality_issues"], list)


def test_no_raw_rows(sales_csv):
    """Test that assert_no_raw_rows passes on valid profile."""
    profile = profile_csv(str(sales_csv))
    df = pd.read_csv(sales_csv)

    # This should not raise an AssertionError
    assert_no_raw_rows(profile, df)


def test_quality_issues(sales_csv):
    """Test that quality_issues lists column with missing values."""
    profile = profile_csv(str(sales_csv))

    # The 'returned' column should have missing values
    assert len(profile["quality_issues"]) > 0
    issue_texts = [issue.lower() for issue in profile["quality_issues"]]
    assert any("returned" in text and "missing" in text for text in issue_texts)


def test_file_name_basename(sales_csv):
    """Test that file_name is basename only."""
    profile = profile_csv(str(sales_csv))

    file_name = profile["file_name"]
    assert file_name == "sales.csv"
    assert "/" not in file_name
    assert "\\" not in file_name


def test_serialize_roundtrip(sales_csv):
    """Test that serialize/deserialize round-trip preserves DataFrame shape."""
    df = pd.read_csv(sales_csv)
    original_shape = df.shape
    original_columns = list(df.columns)
    original_dtypes = df.dtypes.to_dict()

    # Serialize and deserialize
    serialized = serialize_df(df)
    deserialized = deserialize_df(serialized)

    # Check shape preservation
    assert deserialized.shape == original_shape
    assert list(deserialized.columns) == original_columns
    assert deserialized.dtypes.to_dict() == original_dtypes


def test_transfer_small_df(sales_csv):
    """Test that small DataFrame uses bytes mode."""
    df = pd.read_csv(sales_csv)
    # sales_csv has 50 rows, which is < 50_000
    payload = get_df_transfer_payload(df, str(sales_csv))

    assert payload["mode"] == "bytes"
    assert "data" in payload
    assert isinstance(payload["data"], str)  # base64 encoded


def test_transfer_large_df(tmp_path):
    """Test that large DataFrame uses path mode."""
    # Create a large DataFrame with 60,000 rows
    large_df = pd.DataFrame({
        "col1": range(60000),
        "col2": ["value"] * 60000
    })

    csv_path = tmp_path / "large.csv"
    large_df.to_csv(csv_path, index=False)

    payload = get_df_transfer_payload(large_df, str(csv_path))

    assert payload["mode"] == "path"
    assert payload["path"] == str(csv_path)


def test_deserialize_bytes(sales_csv):
    """Test that deserialize handles bytes payload correctly."""
    df = pd.read_csv(sales_csv)
    payload = get_df_transfer_payload(df, str(sales_csv))

    # Extract and decode the bytes
    import base64
    serialized_bytes = base64.b64decode(payload["data"])
    reconstructed = deserialize_df(serialized_bytes)

    # Check reconstruction
    assert reconstructed.shape == df.shape
    assert list(reconstructed.columns) == list(df.columns)
    # Check a few sample values
    assert reconstructed.iloc[0, 0] == df.iloc[0, 0]
