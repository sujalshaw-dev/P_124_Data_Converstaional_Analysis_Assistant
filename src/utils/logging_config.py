"""Logging configuration for the Conversational Data Analysis Assistant."""

import logging
import os
import re
import sys
from typing import Optional


class SensitiveFilter(logging.Filter):
    """Filter that scrubs potential API keys and secrets from log messages."""

    API_KEY_PATTERN = re.compile(r"(AIza[0-9A-Za-z-_]{35}|[a-zA-Z0-9_-]{32,})")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            # Mask API keys if they look like Google API keys or long random tokens
            record.msg = self.API_KEY_PATTERN.sub("[REDACTED_SECRET]", record.msg)
        return True


def setup_logger(
    name: str = "data_analysis_assistant",
    log_level: Optional[str] = None,
) -> logging.Logger:
    """Configures and returns a logger instance with secure formatting.

    Args:
        name: Name of the logger module.
        log_level: Desired log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level_name = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(SensitiveFilter())

    logger.addHandler(handler)
    logger.propagate = False

    return logger
