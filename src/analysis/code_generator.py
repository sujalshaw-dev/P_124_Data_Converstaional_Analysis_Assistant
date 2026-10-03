"""Analysis pipeline orchestrating LLM code generation, AST validation, execution, and interpretation."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from src.analysis.prompt_builder import PromptBuilder
from src.analysis.code_validator import CodeValidator
from src.analysis.code_executor import CodeExecutor, ExecutionResult
from src.analysis.result_processor import ResultProcessor, ProcessedResult
from src.data.profiler import DatasetProfile
from src.llm.base import BaseLLMProvider
from src.utils.exceptions import (
    CodeValidationError,
    CodeExecutionError,
    LLMProviderError,
    LLMConfigurationError,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("analysis_pipeline")


@dataclass
class AnalysisPipelineResult:
    """Complete result container encapsulating all pipeline stages."""

    success: bool
    question: str
    generated_code: str
    execution_result: Optional[ExecutionResult] = None
    processed_result: Optional[ProcessedResult] = None
    explanation: str = ""
    error_message: Optional[str] = None
    error_stage: Optional[str] = None  # 'generation', 'validation', 'execution', 'explanation'
    attempts_made: int = 1


class AnalysisPipeline:
    """Orchestrates natural language to validated code execution and result interpretation."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        code_validator: Optional[CodeValidator] = None,
        code_executor: Optional[CodeExecutor] = None,
        max_retries: int = 1,
    ) -> None:
        self.llm = llm_provider
        self.validator = code_validator or CodeValidator()
        self.executor = code_executor or CodeExecutor()
        self.max_retries = max_retries

    def run(self, question: str, df: pd.DataFrame, profile: DatasetProfile) -> AnalysisPipelineResult:
        """Executes the full conversational analysis pipeline.

        1. Builds code generation prompt with schema metadata.
        2. Calls LLM to generate pandas analysis script.
        3. Extracts and performs static AST validation.
        4. Executes code against DataFrame within isolated sandbox.
        5. Automatically retries with error feedback if execution fails.
        6. Processes outputs into standardized display formats.
        7. Calls LLM to produce a natural-language explanation of actual results.
        """
        logger.info(f"Initiating analysis pipeline for query: '{question}'")

        last_code = ""
        last_error = ""
        attempts = 0

        while attempts <= self.max_retries:
            attempts += 1
            logger.info(f"Code generation attempt {attempts}/{self.max_retries + 1}")

            # 1. Prompt Construction
            code_prompt = PromptBuilder.build_code_prompt(
                user_question=question,
                profile=profile,
                previous_error=last_error if attempts > 1 else None,
                previous_code=last_code if attempts > 1 else None,
            )

            # 2. LLM Code Generation
            try:
                llm_response = self.llm.generate(
                    prompt=code_prompt,
                    system_instruction=PromptBuilder.SYSTEM_CODE_GENERATION,
                    temperature=0.1,  # Low temperature for deterministic code
                )
            except (LLMConfigurationError, LLMProviderError) as e:
                logger.error(f"LLM generation failed: {e}")
                return AnalysisPipelineResult(
                    success=False,
                    question=question,
                    generated_code="",
                    error_message=str(e),
                    error_stage="generation",
                    attempts_made=attempts,
                )

            # Extract raw code from markdown fences
            extracted_code = CodeValidator.extract_code(llm_response.content)
            last_code = extracted_code

            # 3. Static AST Validation
            is_valid, val_error = self.validator.validate(extracted_code)
            if not is_valid:
                last_error = f"Static security/syntax validation failed: {val_error}"
                logger.warning(last_error)
                if attempts <= self.max_retries:
                    continue
                return AnalysisPipelineResult(
                    success=False,
                    question=question,
                    generated_code=extracted_code,
                    error_message=val_error,
                    error_stage="validation",
                    attempts_made=attempts,
                )

            # 4. Code Execution
            try:
                exec_result = self.executor.execute(extracted_code, df)
                break  # Successful execution, break retry loop
            except CodeExecutionError as e:
                last_error = str(e)
                logger.warning(f"Execution error on attempt {attempts}: {last_error}")
                if attempts <= self.max_retries:
                    continue
                return AnalysisPipelineResult(
                    success=False,
                    question=question,
                    generated_code=extracted_code,
                    error_message=str(e),
                    error_stage="execution",
                    attempts_made=attempts,
                )

        # 5. Result Processing
        processed_res = ResultProcessor.process(exec_result.result_value)

        # 6. LLM Explanation of Actual Results
        explanation_text = ""
        try:
            explanation_prompt = PromptBuilder.build_explanation_prompt(
                user_question=question,
                executed_code=extracted_code,
                execution_output_str=processed_res.summary_text,
            )
            exp_response = self.llm.generate(
                prompt=explanation_prompt,
                system_instruction=PromptBuilder.SYSTEM_EXPLANATION,
                temperature=0.3,
            )
            explanation_text = exp_response.content.strip()
        except Exception as e:
            logger.warning(f"Explanation generation failed ({e}); falling back to formatted summary.")
            explanation_text = (
                f"**Computed Result:**\n\n{processed_res.summary_text}\n\n"
                f"*(Automated interpretation was unavailable: {e})*"
            )

        return AnalysisPipelineResult(
            success=True,
            question=question,
            generated_code=extracted_code,
            execution_result=exec_result,
            processed_result=processed_res,
            explanation=explanation_text,
            attempts_made=attempts,
        )
