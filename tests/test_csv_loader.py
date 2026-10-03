"""Automated tests for CSVLoader: loading, encoding fallback, and edge case validation."""

import io
import pytest
import pandas as pd
from src.data.csv_loader import CSVLoader
from src.utils.exceptions import CSVLoadError, DatasetValidationError


@pytest.fixture
def loader():
    return CSVLoader(max_rows=1000, max_size_mb=10)


@pytest.fixture
def sample_csv_text():
    return (
        "Name, Department, Salary, Age\n"
        "Alice, Engineering, 90000, 29\n"
        "Bob, Marketing, 65000, 34\n"
        "Charlie, Engineering, 110000, 42\n"
    )


def test_load_valid_csv_stream(loader, sample_csv_text):
    """Verifies that a valid CSV stream loads into a DataFrame with trimmed headers."""
    buf = io.BytesIO(sample_csv_text.encode("utf-8"))
    df = loader.load(buf, filename="test.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["Name", "Department", "Salary", "Age"]
    assert df["Salary"].sum() == 265000


def test_load_empty_file_raises_validation_error(loader):
    """Verifies that an empty 0-byte file raises DatasetValidationError."""
    buf = io.BytesIO(b"")
    with pytest.raises(DatasetValidationError) as exc_info:
        loader.load(buf, filename="empty.csv")
    assert "empty (0 bytes)" in str(exc_info.value)


def test_load_whitespace_only_raises_validation_error(loader):
    """Verifies that a whitespace-only file raises DatasetValidationError."""
    buf = io.BytesIO(b"   \n\t  \n  ")
    with pytest.raises(DatasetValidationError):
        loader.load(buf, filename="whitespace.csv")


def test_load_latin1_encoding(loader):
    """Verifies automatic fallback to latin1 encoding."""
    # Byte sequence containing Latin-1 specific accented characters (é = 0xe9)
    latin1_bytes = "Employee,City\nRené,Montréal\nFrançois,Québec\n".encode("latin1")
    buf = io.BytesIO(latin1_bytes)
    df = loader.load(buf, filename="french.csv")

    assert len(df) == 2
    assert "René" in df["Employee"].values


def test_file_size_limit_enforcement():
    """Verifies that files exceeding max_size_mb are rejected."""
    # Set limit to 1 MB and provide 2 MB
    small_loader = CSVLoader(max_size_mb=1)
    huge_bytes = b"col1,col2\n" + (b"val1,val2\n" * 150000)
    buf = io.BytesIO(huge_bytes)

    with pytest.raises(DatasetValidationError) as exc:
        small_loader.load(buf, filename="huge.csv")
    assert "exceeds the permitted limit" in str(exc.value)


def test_row_limit_truncation():
    """Verifies that datasets with more rows than max_rows are cleanly truncated."""
    lim_loader = CSVLoader(max_rows=5)
    csv_content = "id,val\n" + "\n".join(f"{i},{i*10}" for i in range(20))
    buf = io.BytesIO(csv_content.encode("utf-8"))

    df = lim_loader.load(buf, filename="limit_test.csv")
    assert len(df) == 5
    assert df["id"].tolist() == [0, 1, 2, 3, 4]


def test_nonexistent_file_raises_csv_load_error(loader):
    """Verifies error handling for missing filesystem paths."""
    with pytest.raises(CSVLoadError) as exc:
        loader.load("non_existent_file_path_12345.csv")
    assert "not found" in str(exc.value).lower()
