"""Prompt construction module for LLM code generation and result explanation."""

from typing import Optional, Any
from src.data.profiler import DatasetProfile


class PromptBuilder:
    """Constructs structured, privacy-preserving prompts for code generation and insight synthesis."""

    SYSTEM_CODE_GENERATION = (
        "You are an expert Data Analyst and Python/Pandas specialist.\n"
        "Your task is to write clean, secure, and idiomatic Python code that queries an existing pandas DataFrame `df` "
        "to answer the user's analytical question.\n\n"
        "STRICT EXECUTION CONSTRAINTS:\n"
        "1. The DataFrame is ALREADY loaded in variable `df`. Never call pd.read_csv() or read files.\n"
        "2. Store the final answer in a variable named `result`.\n"
        "   - For tabular results, `result` should be a pandas DataFrame or Series.\n"
        "   - For single metrics, `result` can be a scalar (float, int, str).\n"
        "   - For multi-part answers, `result` can be a dictionary or DataFrame.\n"
        "3. Only use `pandas` (as `pd`), `numpy` (as `np`), `math`, and standard Python built-ins.\n"
        "4. NEVER import or use `os`, `sys`, `subprocess`, `shutil`, `socket`, `urllib`, `requests`, `builtins`.\n"
        "5. NEVER invoke `open()`, `eval()`, `exec()`, `input()`, `compile()`, or `__import__()`.\n"
        "6. Do not modify `df` in-place unless making a copy.\n"
        "7. Format your response STRICTLY as a markdown Python code block:\n"
        "```python\n"
        "# your code here\n"
        "result = ...\n"
        "```\n"
        "Do not include any conversational explanation or preamble outside the code block."
    )

    SYSTEM_EXPLANATION = (
        "You are an executive data analyst communicating findings to business stakeholders and students.\n"
        "Explain the computational results accurately, concisely, and clearly.\n"
        "Base your explanation solely on the provided actual execution output.\n"
        "Highlight significant numbers, top/bottom performers, trends, or notable findings.\n"
        "Do not invent facts or make claims not supported by the calculation results."
    )

    @classmethod
    def build_code_prompt(
        cls,
        user_question: str,
        profile: DatasetProfile,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> str:
        """Builds prompt for generating pandas analysis code."""
        schema_context = profile.to_llm_schema_context(max_sample_rows=3)

        prompt_parts = [
            "### DATASET SCHEMA & METADATA",
            schema_context,
            "",
            "### USER QUESTION",
            f'"{user_question}"',
            "",
            "### TASK",
            "Write Python code using DataFrame `df` to answer the question. Assign your final answer to variable `result`.",
        ]

        if previous_error and previous_code:
            prompt_parts.extend([
                "",
                "### PREVIOUS ATTEMPT FAILED",
                "Your previous code produced an error:",
                f"```python\n{previous_code}\n```",
                f"Error details: {previous_error}",
                "Please fix the code, ensuring correct column names and valid operations.",
            ])

        prompt_parts.append("\nReturn ONLY the executable ```python ... ``` block.")
        return "\n".join(prompt_parts)

    @classmethod
    def build_explanation_prompt(
        cls,
        user_question: str,
        executed_code: str,
        execution_output_str: str,
    ) -> str:
        """Builds prompt for interpreting real execution results."""
        return (
            f"User Question: {user_question}\n\n"
            f"Code Executed:\n```python\n{executed_code}\n```\n\n"
            f"Actual Execution Result:\n{execution_output_str}\n\n"
            "Please provide a concise, natural-language explanation of these results. "
            "Cite key figures and summarize what this answer means in plain business language."
        )
