"""Application configuration management for the Conversational Data Analysis Assistant."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


class AppConfig:
    """Central application configuration with environment fallback and validation."""

    PROJECT_ROOT: Path = PROJECT_ROOT
    SAMPLE_DATA_PATH: Path = PROJECT_ROOT / "data" / "sample_employees.csv"

    # LLM Settings
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2").strip()

    # Security & Execution Controls
    EXECUTION_TIMEOUT_SECONDS: int = int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "10"))
    MAX_DATASET_ROWS: int = int(os.getenv("MAX_DATASET_ROWS", "100000"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @classmethod
    def get_gemini_key(cls, runtime_key: Optional[str] = None) -> str:
        """Retrieves Gemini API key with runtime override support."""
        if runtime_key and runtime_key.strip():
            return runtime_key.strip()
        return cls.GEMINI_API_KEY

    @classmethod
    def reload(cls) -> None:
        """Reloads settings from the .env file."""
        load_dotenv(dotenv_path=cls.PROJECT_ROOT / ".env", override=True)
        cls.DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
        cls.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
        cls.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        cls.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        cls.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip()
        cls.EXECUTION_TIMEOUT_SECONDS = int(os.getenv("EXECUTION_TIMEOUT_SECONDS", "10"))
        cls.MAX_DATASET_ROWS = int(os.getenv("MAX_DATASET_ROWS", "100000"))
        cls.MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
        cls.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
