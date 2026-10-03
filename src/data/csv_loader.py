"""CSV loader module for secure, resilient dataset ingestion."""

import io
from pathlib import Path
from typing import Union, BinaryIO, Optional
import pandas as pd

from src.config import AppConfig
from src.utils.exceptions import CSVLoadError, DatasetValidationError
from src.utils.logging_config import setup_logger

logger = setup_logger("csv_loader")

SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]


class CSVLoader:
    """Loads and validates CSV files from filesystem paths or uploaded file streams."""

    def __init__(
        self,
        max_rows: Optional[int] = None,
        max_size_mb: Optional[int] = None,
    ) -> None:
        self.max_rows = max_rows or AppConfig.MAX_DATASET_ROWS
        self.max_size_mb = max_size_mb or AppConfig.MAX_UPLOAD_SIZE_MB

    def load(
        self,
        source: Union[str, Path, BinaryIO, io.BytesIO, io.StringIO],
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads a CSV into a pandas DataFrame with encoding fallback and validation.

        Args:
            source: File path or file-like binary/string buffer.
            filename: Optional filename for logging and error reporting.

        Returns:
            pd.DataFrame: Cleaned pandas DataFrame.

        Raises:
            CSVLoadError: If the file cannot be read, decoded, or parsed.
            DatasetValidationError: If the file violates size, empty, or structure rules.
        """
        display_name = filename or (str(source) if isinstance(source, (str, Path)) else "uploaded_file.csv")
        logger.info(f"Initiating CSV load for: {display_name}")

        raw_bytes: bytes

        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise CSVLoadError(f"File not found: '{path}'")
            if not path.is_file():
                raise CSVLoadError(f"Path is not a regular file: '{path}'")

            file_size_bytes = path.stat().st_size
            self._validate_size(file_size_bytes, display_name)

            try:
                with open(path, "rb") as f:
                    raw_bytes = f.read()
            except Exception as e:
                raise CSVLoadError(f"Failed to read file from disk '{display_name}': {e}") from e
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            try:
                # Seek to beginning if possible
                if hasattr(source, "seek"):
                    source.seek(0)
                content = source.read()
                if isinstance(content, str):
                    raw_bytes = content.encode("utf-8")
                elif isinstance(content, bytes):
                    raw_bytes = content
                else:
                    raise CSVLoadError(f"Unsupported content type from stream: {type(content)}")

                if hasattr(source, "seek"):
                    source.seek(0)
            except Exception as e:
                raise CSVLoadError(f"Failed reading stream for '{display_name}': {e}") from e

            self._validate_size(len(raw_bytes), display_name)

        if not raw_bytes or len(raw_bytes.strip()) == 0:
            raise DatasetValidationError(f"The CSV file '{display_name}' is empty (0 bytes).")

        df = self._parse_bytes(raw_bytes, display_name)
        df = self._post_process(df, display_name)

        logger.info(
            f"Successfully loaded '{display_name}' with shape {df.shape} "
            f"({len(df)} rows, {len(df.columns)} columns)."
        )
        return df

    def _validate_size(self, size_bytes: int, display_name: str) -> None:
        """Validates file byte size against configured thresholds."""
        max_bytes = self.max_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise DatasetValidationError(
                f"File '{display_name}' size ({size_bytes / (1024 * 1024):.2f} MB) exceeds "
                f"the permitted limit of {self.max_size_mb} MB."
            )

    def _parse_bytes(self, raw_bytes: bytes, display_name: str) -> pd.DataFrame:
        """Attempts parsing CSV bytes across multiple encodings."""
        last_error: Optional[Exception] = None

        for encoding in SUPPORTED_ENCODINGS:
            try:
                buffer = io.BytesIO(raw_bytes)
                # First attempt standard parse with C engine
                df = pd.read_csv(buffer, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                raise DatasetValidationError(f"The CSV file '{display_name}' contains no readable columns or rows.")
            except pd.errors.ParserError:
                # Try fallback with python engine and delimiter detection
                try:
                    buffer = io.BytesIO(raw_bytes)
                    df = pd.read_csv(buffer, encoding=encoding, engine="python", on_bad_lines="skip")
                    logger.warning(f"File '{display_name}' parsed using fallback python engine with bad lines skipped.")
                    return df
                except Exception as inner_e:
                    last_error = inner_e
            except Exception as e:
                last_error = e

        error_msg = f"Failed to parse CSV file '{display_name}'. "
        if last_error:
            error_msg += f"Reason: {last_error}"
        raise CSVLoadError(error_msg)

    def _post_process(self, df: pd.DataFrame, display_name: str) -> pd.DataFrame:
        """Cleans and validates the parsed DataFrame."""
        if df.empty:
            raise DatasetValidationError(f"The dataset '{display_name}' contains 0 rows.")

        if len(df.columns) == 0:
            raise DatasetValidationError(f"The dataset '{display_name}' contains no columns.")

        # Clean column names: strip leading/trailing whitespace
        df.columns = [str(c).strip() for c in df.columns]

        # Enforce max row limit if dataset exceeds it
        if len(df) > self.max_rows:
            logger.warning(
                f"Dataset '{display_name}' has {len(df)} rows, exceeding limit {self.max_rows}. "
                f"Truncating to top {self.max_rows} rows."
            )
            df = df.iloc[: self.max_rows].copy()

        return df
