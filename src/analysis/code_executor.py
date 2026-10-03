"""Sandboxed execution engine for validated Python/Pandas analysis scripts."""

import contextlib
import io
import math
import time
import datetime
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.config import AppConfig
from src.utils.exceptions import CodeExecutionError, CodeExecutionTimeoutError
from src.utils.logging_config import setup_logger

logger = setup_logger("code_executor")

# Whitelist of strictly safe built-in functions
SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


@dataclass
class ExecutionResult:
    """Encapsulates output and telemetry from running generated analysis code."""

    success: bool
    result_value: Any
    stdout: str
    execution_time_ms: float
    error_message: Optional[str] = None


class CodeExecutor:
    """Executes validated code against a target DataFrame within a constrained environment."""

    def __init__(self, timeout_seconds: Optional[int] = None) -> None:
        self.timeout_seconds = timeout_seconds or AppConfig.EXECUTION_TIMEOUT_SECONDS

    def execute(self, code: str, df: pd.DataFrame) -> ExecutionResult:
        """Executes code against a deep copy of df within the configured timeout.

        Args:
            code: Validated Python code string.
            df: Target pandas DataFrame.

        Returns:
            ExecutionResult containing computed value, execution time, and stdout.

        Raises:
            CodeExecutionError: If a runtime exception occurs.
            CodeExecutionTimeoutError: If execution exceeds the safety timeout.
        """
        logger.info(f"Executing generated code (timeout={self.timeout_seconds}s)...")
        start_time = time.perf_counter()

        container: Dict[str, Any] = {}
        error_container: Dict[str, Exception] = {}

        def worker() -> None:
            try:
                res_val, out_txt = self._run_in_sandbox(code, df)
                container["result"] = res_val
                container["stdout"] = out_txt
            except Exception as e:
                error_container["error"] = e

        # Run inside a daemon thread so it will not block program shutdown if timed out
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=float(self.timeout_seconds))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if thread.is_alive():
            err_msg = (
                f"Execution timed out after {self.timeout_seconds} seconds. "
                "The operation was terminated to preserve system responsiveness."
            )
            logger.error(err_msg)
            raise CodeExecutionTimeoutError(err_msg)

        if "error" in error_container:
            exc = error_container["error"]
            err_msg = self._format_runtime_error(exc, df)
            logger.error(f"Code execution runtime failure: {err_msg}")
            raise CodeExecutionError(err_msg, details=str(exc)) from exc

        logger.info(f"Code executed successfully in {elapsed_ms:.2f} ms.")
        return ExecutionResult(
            success=True,
            result_value=container.get("result"),
            stdout=container.get("stdout", ""),
            execution_time_ms=elapsed_ms,
        )

    def _run_in_sandbox(self, code: str, df: pd.DataFrame) -> tuple[Any, str]:
        """Internal execution within restricted globals and isolated scopes."""
        # Deep copy to ensure dataframe immutability
        df_copy = df.copy(deep=True)

        stdout_buf = io.StringIO()

        # Restrict globals strictly
        safe_globals: Dict[str, Any] = {
            "__builtins__": SAFE_BUILTINS,
            "pd": pd,
            "np": np,
            "math": math,
            "datetime": datetime,
            "df": df_copy,
        }

        local_scope: Dict[str, Any] = {}

        with contextlib.redirect_stdout(stdout_buf):
            compiled = compile(code, "<assistant_code>", "exec")
            exec(compiled, safe_globals, local_scope)

        stdout_text = stdout_buf.getvalue()
        # Cap captured stdout length to 50,000 characters
        if len(stdout_text) > 50000:
            stdout_text = stdout_text[:50000] + "\n...[stdout truncated for display]"

        # Check for 'result' variable
        if "result" in local_scope:
            return local_scope["result"], stdout_text
        elif "result" in safe_globals and safe_globals["result"] is not df_copy:
            return safe_globals["result"], stdout_text

        # Fallback: check if single variable was defined
        user_vars = {k: v for k, v in local_scope.items() if not k.startswith("_")}
        if len(user_vars) == 1:
            var_name, val = next(iter(user_vars.items()))
            logger.info(f"Variable 'result' was not explicitly set; using single assigned variable '{var_name}'.")
            return val, stdout_text

        raise CodeExecutionError(
            "Code executed but did not assign an answer to the 'result' variable.",
            details="Make sure the script ends with `result = ...`.",
        )

    def _format_runtime_error(self, exc: Exception, df: pd.DataFrame) -> str:
        """Translates Python runtime exceptions into human-friendly error messages."""
        if isinstance(exc, KeyError):
            missing_col = str(exc).strip("'\"")
            return (
                f"Column '{missing_col}' does not exist in the dataset. "
                f"Available columns are: {list(df.columns)}."
            )
        elif isinstance(exc, ZeroDivisionError):
            return "Calculation encountered a division by zero error."
        elif isinstance(exc, TypeError):
            return f"Type mismatch error during calculation: {exc}."
        elif isinstance(exc, ValueError):
            return f"Value error in pandas operation: {exc}."
        elif isinstance(exc, AttributeError):
            return f"Attribute error: {exc}."
        elif isinstance(exc, IndexError):
            return f"Index out of bounds error: {exc}."
        return f"Runtime error: {exc}"
