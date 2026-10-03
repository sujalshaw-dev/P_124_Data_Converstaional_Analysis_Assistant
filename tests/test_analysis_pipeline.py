"""Automated end-to-end tests for the AnalysisPipeline using deterministic mock LLMs."""

import os
from typing import Optional, List
import pandas as pd
import pytest

from src.analysis.code_generator import AnalysisPipeline
from src.data.profiler import DatasetProfiler
from src.llm.base import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Deterministic Mock LLM for offline pipeline and self-correction testing."""

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self.responses = responses or []
        self.call_count = 0
        self.prompts_received: List[str] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-deterministic"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        self.prompts_received.append(prompt)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return LLMResponse(content=resp, model=self.model_name, provider=self.provider_name)
        return LLMResponse(
            content="```python\nresult = df['Salary'].mean()\n```",
            model=self.model_name,
            provider=self.provider_name,
        )

    def validate_connection(self) -> tuple[bool, str]:
        return True, "Mock connected"


@pytest.fixture
def test_dataset():
    return pd.DataFrame({
        "Employee_ID": [101, 102, 103],
        "Name": ["Rahul", "Priya", "Amit"],
        "Department": ["Engineering", "Marketing", "Sales"],
        "Salary": [90000, 80000, 70000],
    })


def test_pipeline_end_to_end_success(test_dataset):
    """Tests complete happy path: Prompt -> Mock LLM Code -> AST Validation -> Execution -> Explanation."""
    profile = DatasetProfiler.profile(test_dataset)

    mock_llm = MockLLMProvider(responses=[
        "```python\nresult = df['Salary'].mean()\n```",  # 1st call: code gen
        "The computed average salary across the company is 80,000.",  # 2nd call: explanation
    ])

    pipeline = AnalysisPipeline(llm_provider=mock_llm)
    result = pipeline.run("What is the average salary?", test_dataset, profile)

    assert result.success is True
    assert result.execution_result.result_value == 80000.0
    assert result.processed_result.display_scalar == 80000.0
    assert "80,000" in result.explanation
    assert result.error_message is None
    assert result.attempts_made == 1


def test_pipeline_self_correction_on_runtime_error(test_dataset):
    """Verifies that the pipeline feeds runtime errors back to the LLM and recovers on retry."""
    profile = DatasetProfiler.profile(test_dataset)

    mock_llm = MockLLMProvider(responses=[
        "```python\nresult = df['NonExistentSalaryColumn'].mean()\n```",  # Attempt 1: Fails
        "```python\nresult = df['Salary'].mean()\n```",  # Attempt 2 (Correction): Succeeds
        "Average salary is 80,000 after correction.",  # Explanation
    ])

    pipeline = AnalysisPipeline(llm_provider=mock_llm, max_retries=1)
    result = pipeline.run("What is the average salary?", test_dataset, profile)

    assert result.success is True
    assert result.execution_result.result_value == 80000.0
    assert result.attempts_made == 2
    # Check that the second prompt included error feedback
    assert "PREVIOUS ATTEMPT FAILED" in mock_llm.prompts_received[1]


def test_pipeline_catches_and_aborts_dangerous_code(test_dataset):
    """Verifies that when LLM returns dangerous code, the pipeline blocks it and fails safely."""
    profile = DatasetProfiler.profile(test_dataset)

    mock_llm = MockLLMProvider(responses=[
        "```python\nimport os\nos.system('echo dangerous')\nresult = 0\n```",
        "```python\nimport os\nos.system('echo dangerous again')\nresult = 0\n```",
    ])

    pipeline = AnalysisPipeline(llm_provider=mock_llm, max_retries=1)
    result = pipeline.run("Do something malicious", test_dataset, profile)

    assert result.success is False
    assert result.error_stage == "validation"
    assert "Disallowed import" in result.error_message


# Clearly distinguished live Gemini test (run only if real GEMINI_API_KEY is present)
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="Live Gemini integration test skipped: GEMINI_API_KEY environment variable is not set.",
)
def test_live_gemini_integration(test_dataset):
    """Live integration test against Google Gemini API (only executed when credentials exist)."""
    from src.llm.gemini_provider import GeminiProvider

    profile = DatasetProfiler.profile(test_dataset)
    gemini = GeminiProvider()
    pipeline = AnalysisPipeline(llm_provider=gemini)

    result = pipeline.run("What is the maximum salary?", test_dataset, profile)
    assert result.success is True
    assert result.execution_result.result_value == 90000
