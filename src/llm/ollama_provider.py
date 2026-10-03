"""Local LLM provider integration using Ollama's HTTP API."""

from typing import Optional
import requests

from src.llm.base import BaseLLMProvider, LLMResponse
from src.config import AppConfig
from src.utils.exceptions import LLMProviderError
from src.utils.logging_config import setup_logger

logger = setup_logger("ollama_provider")


class OllamaProvider(BaseLLMProvider):
    """Integrates locally hosted LLMs running under Ollama."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.base_url = (base_url or AppConfig.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or AppConfig.OLLAMA_MODEL
        self.timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Sends generation request to local Ollama instance."""
        if not prompt or not prompt.strip():
            raise LLMProviderError("Cannot send an empty prompt to Ollama.")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system_instruction:
            payload["system"] = system_instruction

        logger.info(f"Sending request to Ollama ({self.model}) at {url}...")

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise LLMProviderError(
                f"Cannot connect to Ollama at {self.base_url}.",
                details="Make sure Ollama is installed and running (`ollama serve`).",
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMProviderError(
                f"Ollama request timed out after {self.timeout}s.",
                details="The local model may be overloaded or requires too many resources.",
            ) from e
        except Exception as e:
            raise LLMProviderError(f"Ollama request error: {e}") from e

        if resp.status_code == 404:
            raise LLMProviderError(
                f"Ollama model '{self.model}' was not found.",
                details=f"Please pull the model first using terminal: `ollama pull {self.model}`.",
            )

        if resp.status_code != 200:
            raise LLMProviderError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text}",
                details="Check local Ollama server logs.",
            )

        try:
            data = resp.json()
            content = data.get("response", "")
            tokens_used = data.get("eval_count", None)
            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.provider_name,
                tokens_used=tokens_used,
            )
        except Exception as e:
            raise LLMProviderError(f"Failed to parse Ollama JSON response: {e}") from e

    def validate_connection(self) -> tuple[bool, str]:
        """Checks if the local Ollama daemon is reachable and has the model."""
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                # Check model matching (e.g. "llama3.2" matches "llama3.2:latest")
                matched = any(self.model in m for m in models)
                if matched:
                    return True, f"Connected to Ollama. Model '{self.model}' is available."
                return (
                    False,
                    f"Ollama server is active, but model '{self.model}' is not installed. Run `ollama pull {self.model}`.",
                )
            return False, f"Ollama returned HTTP status {resp.status_code}."
        except Exception as e:
            return (
                False,
                f"Ollama service is unreachable at {self.base_url}. Ensure it is running or switch to Gemini.",
            )
