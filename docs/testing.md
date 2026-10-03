# Testing & Quality Assurance Guide

## 1. Testing Philosophy

The test suite is built on **deterministic test data** and **independently verified ground-truth values**.

Because untrusted LLM-generated code poses both runtime and security risks, testing verifies:
1. **Structural Ingestion Integrity**: Encoding fallbacks, empty files, size and row limits.
2. **Deterministic Mathematical Correctness**: Ground-truth calculations for averages, maximums, and aggregations.
3. **AST Security Boundaries**: Exhaustive verification of rejected imports, dangerous built-ins, and dunder escapes.
4. **Sandbox Reliability**: DataFrame immutability, timeout halts on infinite loops, and formatted error messages.
5. **Pipeline Resilience**: Deterministic mock LLM verification, automated self-correction, and graceful handling of provider failures.

---

## 2. Test Suite Structure

```
tests/
├── __init__.py
├── test_csv_loader.py         # CSV parsing, encodings, limits, edge cases
├── test_profiler.py           # Metrics, duplicates, null analysis, schema context
├── test_code_validator.py     # AST security, dangerous code rejection
├── test_code_executor.py      # Sandbox safety, timeouts, numerical correctness
└── test_analysis_pipeline.py  # Mock LLM end-to-end pipeline & live Gemini test
```

---

## 3. Running the Test Suite

Run all automated unit and offline integration tests:

```bash
# Run pytest with verbose test names
python -m pytest -v

# Run with concise test summary
python -m pytest
```

### Running Specific Module Tests

```bash
# Test AST security and validation
python -m pytest tests/test_code_validator.py -v

# Test Sandboxed execution and timeouts
python -m pytest tests/test_code_executor.py -v

# Test CSV Loader
python -m pytest tests/test_csv_loader.py -v

# Test Profiler
python -m pytest tests/test_profiler.py -v

# Test End-to-End Analysis Pipeline
python -m pytest tests/test_analysis_pipeline.py -v
```

---

## 4. Test Categories

### 4.1 Unit Tests (Mocked / Offline)
- **`test_csv_loader.py`**:
  - `test_load_valid_csv_stream`: Validates header trimming and data parsing.
  - `test_load_empty_file_raises_validation_error`: Ensures 0-byte files raise `DatasetValidationError`.
  - `test_load_latin1_encoding`: Confirms Latin-1 character decoding (e.g., accented characters).
  - `test_file_size_limit_enforcement`: Tests rejection when exceeding `MAX_UPLOAD_SIZE_MB`.
  - `test_row_limit_truncation`: Verifies row capping to `MAX_DATASET_ROWS`.
- **`test_profiler.py`**:
  - `test_profiler_counts_and_duplicates`: Mathematical ground truth for duplicate detection.
  - `test_profiler_missing_value_analysis`: Verifies null counts and percentage formulas.
  - `test_profiler_numerical_metrics`: Verifies min, max, mean, and median calculations.
  - `test_to_llm_schema_context_preserves_privacy`: Verifies full datasets are never serialized into LLM prompts.
- **`test_code_validator.py`**:
  - `test_validator_accepts_valid_pandas_code`: Allows legitimate analytical expressions.
  - `test_validator_rejects_dangerous_imports`: Blocks `os`, `sys`, `subprocess`, `shutil`, `socket`.
  - `test_validator_rejects_dangerous_calls`: Blocks `eval()`, `exec()`, `open()`, `input()`, `__import__()`.
  - `test_validator_rejects_dunder_attribute_traversals`: Blocks `__subclasses__`, `__mro__`, `__globals__`.
- **`test_code_executor.py`**:
  - `test_executor_numerical_average`: Ground-truth calculation verification.
  - `test_executor_dataframe_immutability`: Proves original data cannot be mutated in the sandbox.
  - `test_executor_timeout_enforcement`: Proves runaway `while True` loops are halted.
  - `test_executor_missing_column_key_error`: Verifies user-friendly error formatting.

### 4.2 End-to-End Pipeline & Live Integration Tests
- **`test_analysis_pipeline.py`**:
  - `test_pipeline_end_to_end_success`: Employs `MockLLMProvider` to test the full pipeline offline.
  - `test_pipeline_self_correction_on_runtime_error`: Verifies that a failed first attempt triggers a correction prompt and recovers on attempt 2.
  - `test_pipeline_catches_and_aborts_dangerous_code`: Verifies pipeline aborts when the LLM outputs malicious code.
  - `test_live_gemini_integration`: Skipped by default unless a real `GEMINI_API_KEY` is present in the environment.
