"""Automated tests for CodeExecutor: sandbox safety, runtime error handling, and numerical correctness."""

import time
import pandas as pd
import pytest
from src.analysis.code_executor import CodeExecutor
from src.utils.exceptions import CodeExecutionError, CodeExecutionTimeoutError


@pytest.fixture
def executor():
    return CodeExecutor(timeout_seconds=2)


@pytest.fixture
def employees_df():
    """Deterministic test dataset for numerical correctness verification."""
    return pd.DataFrame({
        "Name": ["Rahul", "Priya", "Amit", "Sneha", "Vikram"],
        "Department": ["Engineering", "Marketing", "Sales", "HR", "Engineering"],
        "Salary": [90000, 80000, 70000, 60000, 100000],
        "Age": [30, 28, 35, 26, 32],
    })


def test_executor_numerical_average(executor, employees_df):
    """Verifies that average salary is computed with deterministic mathematical correctness."""
    code = "result = df['Salary'].mean()"
    res = executor.execute(code, employees_df)

    assert res.success is True
    # (90000 + 80000 + 70000 + 60000 + 100000) / 5 = 400000 / 5 = 80000.0
    assert res.result_value == 80000.0
    assert res.execution_time_ms > 0


def test_executor_highest_salary_filtering(executor, employees_df):
    """Verifies highest salary record extraction."""
    code = "result = df.loc[df['Salary'].idxmax()]['Name']"
    res = executor.execute(code, employees_df)

    assert res.success is True
    assert res.result_value == "Vikram"


def test_executor_department_aggregation(executor, employees_df):
    """Verifies groupby aggregation correctness."""
    code = "result = df.groupby('Department')['Salary'].mean().to_dict()"
    res = executor.execute(code, employees_df)

    assert res.success is True
    res_dict = res.result_value
    # Engineering: (90000 + 100000) / 2 = 95000.0
    assert res_dict["Engineering"] == 95000.0
    assert res_dict["HR"] == 60000.0


def test_executor_dataframe_immutability(executor, employees_df):
    """Verifies that modifications inside sandboxed code do not mutate the original DataFrame."""
    code = "df['Mutated_Col'] = 999\ndf.drop(columns=['Salary'], inplace=True)\nresult = len(df)"
    res = executor.execute(code, employees_df)

    assert res.success is True
    # Original DataFrame must still have its original columns and no 'Mutated_Col'
    assert "Salary" in employees_df.columns
    assert "Mutated_Col" not in employees_df.columns


def test_executor_missing_column_key_error(executor, employees_df):
    """Verifies informative error when referencing nonexistent column."""
    code = "result = df['NonExistentColumn'].mean()"
    with pytest.raises(CodeExecutionError) as exc_info:
        executor.execute(code, employees_df)

    err_str = str(exc_info.value)
    assert "NonExistentColumn" in err_str
    assert "Available columns are" in err_str


def test_executor_division_by_zero(executor, employees_df):
    """Verifies division by zero error capture."""
    code = "result = 100 / 0"
    with pytest.raises(CodeExecutionError) as exc_info:
        executor.execute(code, employees_df)

    assert "division by zero" in str(exc_info.value).lower()


def test_executor_timeout_enforcement(employees_df):
    """Verifies that long-running or infinite loops are halted by the timeout."""
    # Fast 1-second timeout executor
    fast_executor = CodeExecutor(timeout_seconds=1)
    infinite_loop_code = "while True:\n    pass\nresult = 1"

    with pytest.raises(CodeExecutionTimeoutError) as exc_info:
        fast_executor.execute(infinite_loop_code, employees_df)

    assert "timed out" in str(exc_info.value).lower()


def test_executor_missing_result_variable(executor, employees_df):
    """Verifies error raised when code doesn't produce an answer or set 'result'."""
    code = "x = 10\ny = 20"  # No 'result' and multiple variables
    with pytest.raises(CodeExecutionError) as exc_info:
        executor.execute(code, employees_df)

    assert "did not assign an answer to the 'result' variable" in str(exc_info.value)
