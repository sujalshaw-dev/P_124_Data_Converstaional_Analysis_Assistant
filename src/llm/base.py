"""Abstract Base Class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMResponse:
    """Structured response container from an LLM call."""

    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract interface defining required behaviors for LLM integrations."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'ollama')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generates text from the LLM given a user prompt and optional system instructions.

        Args:
            prompt: The user query and context.
            system_instruction: Guiding persona/system constraints.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).

        Returns:
            LLMResponse containing generated text and metadata.

        Raises:
            LLMConfigurationError: If provider credentials or configuration are invalid.
            LLMProviderError: If remote generation fails (rate limits, timeouts, API errors).
        """
        pass

    @abstractmethod
    def validate_connection(self) -> tuple[bool, str]:
        """Tests connectivity and authentication with the model service.

        Returns:
            Tuple of (is_connected: bool, status_message: str).
        """
        pass
