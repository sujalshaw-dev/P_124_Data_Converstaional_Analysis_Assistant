"""Automated tests for DatasetProfiler: structure, statistics, and LLM context extraction."""

import pandas as pd
import pytest
from src.data.profiler import DatasetProfiler, DatasetProfile


@pytest.fixture
def sample_dataframe():
    """Generates a controlled test DataFrame with known metrics."""
    return pd.DataFrame({
        "ID": [1, 2, 3, 4, 4],  # Row 5 is duplicate of row 4 if all cols match
        "Name": ["Alice", "Bob", "Charlie", "David", "David"],
        "Department": ["IT", "Sales", "IT", "HR", "HR"],
        "Salary": [80000.0, 60000.0, 100000.0, 70000.0, 70000.0],
        "Bonus": [5000.0, None, 8000.0, None, None],  # 3 nulls
    })


def test_profiler_counts_and_duplicates(sample_dataframe):
    """Verifies row, column, and duplicate row detection."""
    profile = DatasetProfiler.profile(sample_dataframe)

    assert profile.row_count == 5
    assert profile.column_count == 5
    assert profile.duplicate_rows_count == 1
    assert profile.duplicate_percentage == 20.0
    assert "ID" in profile.columns
    assert "Salary" in profile.columns


def test_profiler_missing_value_analysis(sample_dataframe):
    """Verifies accurate identification of null counts and percentages."""
    profile = DatasetProfiler.profile(sample_dataframe)

    bonus_col = profile.columns["Bonus"]
    assert bonus_col.null_count == 3
    assert bonus_col.non_null_count == 2
    assert bonus_col.null_percentage == 60.0

    name_col = profile.columns["Name"]
    assert name_col.null_count == 0
    assert name_col.non_null_count == 5
    assert name_col.null_percentage == 0.0


def test_profiler_numerical_metrics(sample_dataframe):
    """Verifies min, max, mean, and median calculations on numerical columns."""
    profile = DatasetProfiler.profile(sample_dataframe)

    salary_col = profile.columns["Salary"]
    assert salary_col.is_numeric is True
    assert salary_col.min_val == 60000.0
    assert salary_col.max_val == 100000.0
    # Values: 80k, 60k, 100k, 70k, 70k -> sum = 380k -> mean = 76000
    assert salary_col.mean_val == 76000.0
    # Sorted: 60k, 70k, 70k, 80k, 100k -> median = 70000
    assert salary_col.median_val == 70000.0


def test_to_llm_schema_context_preserves_privacy(sample_dataframe):
    """Verifies that schema context sends metadata and samples, but not the entire raw data."""
    profile = DatasetProfiler.profile(sample_dataframe)
    context_str = profile.to_llm_schema_context(max_sample_rows=2)

    # Must contain structural summaries
    assert "5 rows, 5 columns" in context_str
    assert "Salary" in context_str
    assert "min=60000.0" in context_str

    # Must include sample head limited to specified rows
    assert "Row 1:" in context_str
    assert "Row 2:" in context_str
    assert "Row 4:" not in context_str  # Row 4 excluded from LLM prompt
