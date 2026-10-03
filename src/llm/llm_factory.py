"""Factory for creating and configuring LLM provider instances."""

from typing import Optional
from src.llm.base import BaseLLMProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.ollama_provider import OllamaProvider
from src.config import AppConfig
from src.utils.exceptions import LLMConfigurationError
from src.utils.logging_config import setup_logger

logger = setup_logger("llm_factory")


class LLMFactory:
    """Instantiates configured LLM providers."""

    @staticmethod
    def create_provider(
        provider_name: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> BaseLLMProvider:
        """Creates an LLM provider based on name and parameters.

        Args:
            provider_name: 'gemini' or 'ollama'. Defaults to AppConfig.DEFAULT_PROVIDER.
            api_key: Optional Gemini API key override.
            model: Optional model name override.
            base_url: Optional base URL for Ollama.

        Returns:
            An active BaseLLMProvider implementation.

        Raises:
            LLMConfigurationError: If unknown provider or missing mandatory credentials.
        """
        chosen_provider = (provider_name or AppConfig.DEFAULT_PROVIDER).lower().strip()

        logger.info(f"Instantiating LLM provider: '{chosen_provider}'")

        if chosen_provider == "gemini":
            return GeminiProvider(api_key=api_key, model=model)
        elif chosen_provider == "ollama":
            return OllamaProvider(base_url=base_url, model=model)
        else:
            raise LLMConfigurationError(
                f"Unsupported LLM provider: '{chosen_provider}'.",
                details="Supported providers are 'gemini' and 'ollama'.",
            )
