import pandas as pd
import pytest

from agent.profiler import load_csv, infer_column_types


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
