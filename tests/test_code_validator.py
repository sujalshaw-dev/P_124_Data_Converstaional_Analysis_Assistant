"""Automated tests for CodeValidator: AST security scanning and dangerous code rejection."""

import pytest
from src.analysis.code_validator import CodeValidator


@pytest.fixture
def validator():
    return CodeValidator()


def test_validator_accepts_valid_pandas_code(validator):
    """Verifies that standard pandas and math operations pass validation."""
    valid_codes = [
        "result = df['Salary'].mean()",
        "result = df.groupby('Department')['Salary'].agg(['mean', 'count']).reset_index()",
        "import numpy as np\nresult = df[df['Salary'] > np.median(df['Salary'])]",
        "result = df.sort_values(by='Salary', ascending=False).head(5)",
        "result = df.isna().sum().to_dict()",
    ]
    for code in valid_codes:
        is_valid, err = validator.validate(code)
        assert is_valid, f"Expected code to pass, but failed with: {err}\nCode: {code}"
        assert err is None


def test_validator_rejects_dangerous_imports(validator):
    """Verifies rejection of dangerous external module imports."""
    dangerous_imports = [
        "import os\nos.system('dir')",
        "import sys\nsys.exit(0)",
        "import subprocess\nsubprocess.run(['ls'])",
        "import shutil\nshutil.rmtree('.')",
        "import socket\ns = socket.socket()",
        "from os import path",
        "from subprocess import Popen",
    ]
    for code in dangerous_imports:
        is_valid, err = validator.validate(code)
        assert not is_valid, f"Dangerous import was not caught: {code}"
        assert "Disallowed import" in err


def test_validator_rejects_dangerous_calls(validator):
    """Verifies rejection of dangerous built-in function calls."""
    dangerous_calls = [
        "eval('2 + 2')",
        "exec('result = 100')",
        "open('some_file.txt', 'w')",
        "__import__('os').system('dir')",
        "input('Enter password:')",
        "compile('x = 1', '', 'exec')",
    ]
    for code in dangerous_calls:
        is_valid, err = validator.validate(code)
        assert not is_valid, f"Dangerous call was not caught: {code}"
        assert "Disallowed function" in err or "Disallowed" in err


def test_validator_rejects_dunder_attribute_traversals(validator):
    """Verifies rejection of dunder attribute lookups used for sandbox escapes."""
    escapes = [
        "result = ().__class__.__bases__[0].__subclasses__()",
        "result = df.__class__.__mro__",
        "result = len.__globals__",
    ]
    for code in escapes:
        is_valid, err = validator.validate(code)
        assert not is_valid, f"Dunder attribute access was not blocked: {code}"
        assert "sensitive internal attribute" in err or "Disallowed" in err


def test_validator_syntax_error_handling(validator):
    """Verifies syntax errors are reported gracefully."""
    bad_syntax = "result = df['Salary'].mean(]"
    is_valid, err = validator.validate(bad_syntax)
    assert not is_valid
    assert "Syntax error" in err


def test_validator_extract_code_from_markdown():
    """Verifies clean extraction of python code from markdown code fences."""
    raw_markdown = (
        "Here is the pandas code to compute the result:\n\n"
        "```python\n"
        "# Calculate mean salary\n"
        "result = df['Salary'].mean()\n"
        "```\n\n"
        "This answers your question."
    )
    extracted = CodeValidator.extract_code(raw_markdown)
    assert extracted == "# Calculate mean salary\nresult = df['Salary'].mean()"

    # Plain code without fences
    plain = "result = df['Salary'].max()"
    assert CodeValidator.extract_code(plain) == plain
