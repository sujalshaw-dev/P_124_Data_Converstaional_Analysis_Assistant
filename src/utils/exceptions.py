"""Custom exceptions for the Conversational Data Analysis Assistant."""


class AppBaseException(Exception):
    """Base exception for all application-specific errors."""

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class CSVLoadError(AppBaseException):
    """Raised when CSV parsing, reading, or decoding fails."""
    pass


class DatasetValidationError(AppBaseException):
    """Raised when an uploaded dataset fails structural, size, or content validation."""
    pass


class LLMConfigurationError(AppBaseException):
    """Raised when an LLM provider is misconfigured (e.g., missing API key or model name)."""
    pass


class LLMProviderError(AppBaseException):
    """Raised when an external LLM request fails due to API errors, rate limits, or network."""
    pass


class CodeValidationError(AppBaseException):
    """Raised when generated code fails static AST validation (security or safety check)."""
    pass


class CodeExecutionError(AppBaseException):
    """Raised when permitted code encounters a runtime error during execution against the DataFrame."""
    pass


class CodeExecutionTimeoutError(CodeExecutionError):
    """Raised when code execution exceeds the allocated safety timeout."""
    pass


class ChartGenerationError(AppBaseException):
    """Raised when visualization generation encounters invalid columns or data shapes."""
    pass
