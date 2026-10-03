"""AST-based static code validator for untrusted LLM-generated code."""

import ast
import re
from typing import Tuple, Optional, Set
from src.utils.exceptions import CodeValidationError
from src.utils.logging_config import setup_logger

logger = setup_logger("code_validator")

# Whitelist of allowed modules if imports appear in generated code
ALLOWED_MODULES: Set[str] = {
    "pandas",
    "numpy",
    "math",
    "datetime",
    "re",
    "collections",
    "itertools",
}

# Explicit blacklist of prohibited function calls and names
DISALLOWED_CALLS: Set[str] = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "__import__",
    "exit",
    "quit",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "help",
}

# Forbidden dunder attributes that could facilitate sandbox escapes
DISALLOWED_ATTRIBUTES: Set[str] = {
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
    "__builtins__",
    "__import__",
    "__class__",
    "__dict__",
    "__qualname__",
}


class CodeValidator(ast.NodeVisitor):
    """Inspects Abstract Syntax Tree of generated code to enforce security boundaries."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.has_result_assignment: bool = False

    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validates Python code against security constraints.

        Args:
            code: Python code string.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        self.errors = []
        self.has_result_assignment = False

        if not code or not code.strip():
            return False, "Code snippet is empty."

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            msg = f"Syntax error in generated code at line {e.lineno}, col {e.offset}: {e.msg}"
            logger.warning(msg)
            return False, msg

        self.visit(tree)

        if self.errors:
            error_summary = "; ".join(self.errors)
            logger.warning(f"Code validation failed: {error_summary}")
            return False, error_summary

        return True, None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod not in ALLOWED_MODULES:
                self.errors.append(f"Disallowed import '{alias.name}'. Only {sorted(ALLOWED_MODULES)} are permitted.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod not in ALLOWED_MODULES:
                self.errors.append(
                    f"Disallowed import from '{node.module}'. Only {sorted(ALLOWED_MODULES)} are permitted."
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct calls like open(), eval()
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in DISALLOWED_CALLS:
                self.errors.append(f"Disallowed function call '{func_name}()'.")
        # Check method calls like os.system()
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name in DISALLOWED_CALLS or attr_name in DISALLOWED_ATTRIBUTES:
                self.errors.append(f"Disallowed function or attribute access '{attr_name}'.")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in DISALLOWED_ATTRIBUTES:
            self.errors.append(f"Access to sensitive internal attribute '{node.attr}' is prohibited.")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "result":
                self.has_result_assignment = True
        self.generic_visit(node)

    @classmethod
    def extract_code(cls, raw_text: str) -> str:
        """Extracts python code from LLM response markdown or plain text."""
        # Find ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        matches = re.findall(pattern, raw_text, flags=re.DOTALL)
        if matches:
            # Use the first or longest match
            return max(matches, key=len).strip()
        # Fallback: if no markdown fences, return stripped string
        return raw_text.strip()
