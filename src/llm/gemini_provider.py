"""Google Gemini provider implementation using the official Google Gen AI SDK."""

from typing import Optional
from src.llm.base import BaseLLMProvider, LLMResponse
from src.config import AppConfig
from src.utils.exceptions import LLMConfigurationError, LLMProviderError
from src.utils.logging_config import setup_logger

logger = setup_logger("gemini_provider")


class GeminiProvider(BaseLLMProvider):
    """Integrates Google Gemini via official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = AppConfig.get_gemini_key(api_key)
        self.model = model or AppConfig.GEMINI_MODEL

        if not self.api_key:
            raise LLMConfigurationError(
                "Gemini API key is not configured.",
                details=(
                    "Please set GEMINI_API_KEY in your '.env' file or provide it "
                    "directly in the Streamlit application sidebar."
                ),
            )

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Google Gen AI Client: {e}")
            raise LLMConfigurationError(
                "Failed to initialize Google Gen AI client.",
                details=str(e),
            ) from e

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Invokes the Gemini model to generate content."""
        if not prompt or not prompt.strip():
            raise LLMProviderError("Cannot send an empty prompt to Gemini.")

        logger.info(f"Dispatching prompt to Gemini model '{self.model}' (temp={temperature})...")

        try:
            from google.genai import types, errors

            config_args = {"temperature": temperature}
            if system_instruction:
                config_args["system_instruction"] = system_instruction

            config = types.GenerateContentConfig(**config_args)

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            # Extract content text
            content_text = getattr(response, "text", "")
            if not content_text and hasattr(response, "candidates") and response.candidates:
                first_cand = response.candidates[0]
                if hasattr(first_cand, "content") and hasattr(first_cand.content, "parts"):
                    content_text = "".join(getattr(p, "text", "") for p in first_cand.content.parts)

            if not content_text:
                raise LLMProviderError(
                    "Gemini returned an empty response. The content may have been blocked by safety filters."
                )

            # Token usage if available
            tokens_used = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens_used = getattr(response.usage_metadata, "total_token_count", None)

            return LLMResponse(
                content=content_text,
                model=self.model,
                provider=self.provider_name,
                tokens_used=tokens_used,
            )

        except Exception as e:
            error_str = str(e)
            logger.error(f"Gemini API request failed: {error_str}")

            # Specific handling for common error scenarios
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                raise LLMProviderError(
                    "Gemini API quota or rate limit exceeded.",
                    details="Please check your rate limits or Google AI Studio billing/quota.",
                ) from e
            elif "400" in error_str or "INVALID_ARGUMENT" in error_str:
                raise LLMProviderError(
                    f"Invalid request sent to Gemini: {error_str}",
                    details="The model name or parameters may be unsupported.",
                ) from e
            elif "401" in error_str or "403" in error_str or "PERMISSION_DENIED" in error_str:
                raise LLMProviderError(
                    "Gemini API authentication failed.",
                    details="The supplied GEMINI_API_KEY is invalid or lacks necessary permissions.",
                ) from e
            elif "Timeout" in error_str or "deadline exceeded" in error_str.lower():
                raise LLMProviderError(
                    "Gemini API request timed out.",
                    details="The network request exceeded the timeout threshold.",
                ) from e
            else:
                raise LLMProviderError(
                    f"Gemini provider error: {error_str}",
                    details=error_str,
                ) from e

    def validate_connection(self) -> tuple[bool, str]:
        """Performs a minimal health check query against the Gemini API."""
        try:
            from google.genai import types
            config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=10)
            res = self.client.models.generate_content(
                model=self.model,
                contents="ping",
                config=config,
            )
            if res and res.text:
                return True, f"Successfully connected to Gemini ({self.model})."
            return True, f"Connected to Gemini ({self.model})."
        except Exception as e:
            return False, f"Failed connecting to Gemini: {str(e)}"
