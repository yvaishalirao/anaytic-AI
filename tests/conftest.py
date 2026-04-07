import csv

import pytest


@pytest.fixture
def sales_csv(tmp_path):
    path = tmp_path / "sales.csv"
    header = ["date", "region", "sales", "units", "returned"]
    regions = ["North", "South", "East", "West"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(50):
            month = i % 12 + 1
            date = f"2023-{month:02d}-01"
            region = regions[i % len(regions)]
            sales = round(1000 + i * 12.5 + (i % 4) * 3.3, 2)
            units = 20 + (i * 2) % 10
            returned = "" if i < 5 else (i % 7)
            writer.writerow([date, region, sales, units, returned])
    return path


@pytest.fixture
def wide_csv(tmp_path):
    path = tmp_path / "wide.csv"
    header = [f"col_{i+1}" for i in range(10)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in range(100):
            writer.writerow([row * 0.5 + col * 0.1 for col in range(10)])
    return path


@pytest.fixture
def session_id():
    return "test-session-0001"

