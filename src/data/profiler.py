"""Dataset profiling module for extracting structural, statistical, and LLM-ready metadata."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger("profiler")


@dataclass
class ColumnProfile:
    """Metadata and descriptive metrics for a single DataFrame column."""

    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any]
    is_numeric: bool
    is_datetime: bool
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    mean_val: Optional[float] = None
    median_val: Optional[float] = None


@dataclass
class DatasetProfile:
    """Comprehensive profiling report for an uploaded dataset."""

    row_count: int
    column_count: int
    column_names: List[str]
    memory_usage_bytes: int
    memory_usage_str: str
    duplicate_rows_count: int
    duplicate_percentage: float
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    summary_stats_numeric: Dict[str, Any] = field(default_factory=dict)
    summary_stats_categorical: Dict[str, Any] = field(default_factory=dict)
    sample_head: List[Dict[str, Any]] = field(default_factory=list)

    def to_llm_schema_context(self, max_sample_rows: int = 3) -> str:
        """Serializes privacy-conscious schema and summary info for LLM prompt context.

        Only column names, types, non-null ratios, and minimal sample rows are provided,
        preserving privacy by never transmitting entire datasets.
        """
        lines = [
            f"Dataset Summary: {self.row_count} rows, {self.column_count} columns.",
            f"Duplicate Rows: {self.duplicate_rows_count} ({self.duplicate_percentage:.1f}%).",
            "\nColumns & Data Types:",
        ]

        for col_name, prof in self.columns.items():
            extra_stat = ""
            if prof.is_numeric:
                mean_str = f"{prof.mean_val:.2f}" if prof.mean_val is not None else "N/A"
                extra_stat = f", min={prof.min_val}, max={prof.max_val}, mean={mean_str}"
            elif prof.unique_count <= 10:
                unique_str = ", ".join(str(v) for v in prof.sample_values[:5])
                extra_stat = f", unique_values=[{unique_str}]"

            null_info = f", nulls={prof.null_count} ({prof.null_percentage:.1f}%)" if prof.null_count > 0 else ""
            lines.append(f"  - `{col_name}` ({prof.dtype}): {prof.unique_count} unique values{null_info}{extra_stat}")

        lines.append(f"\nFirst {min(max_sample_rows, len(self.sample_head))} sample records (schema illustration only):")
        for i, row in enumerate(self.sample_head[:max_sample_rows]):
            lines.append(f"  Row {i+1}: {row}")

        return "\n".join(lines)


class DatasetProfiler:
    """Profiles a pandas DataFrame to generate statistical summaries and LLM context."""

    @staticmethod
    def profile(df: pd.DataFrame) -> DatasetProfile:
        """Generates a complete DatasetProfile from a DataFrame."""
        logger.info(f"Generating profile for DataFrame with shape {df.shape}...")
        row_count, col_count = df.shape

        # Memory usage
        mem_bytes = int(df.memory_usage(deep=True).sum())
        mem_str = DatasetProfiler._format_bytes(mem_bytes)

        # Duplicate rows
        duplicates = int(df.duplicated().sum())
        dup_pct = (duplicates / row_count * 100) if row_count > 0 else 0.0

        # Column-level profiles
        columns_profile: Dict[str, ColumnProfile] = {}
        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            null_count = int(series.isna().sum())
            non_null_count = row_count - null_count
            null_pct = (null_count / row_count * 100) if row_count > 0 else 0.0
            unique_count = int(series.nunique(dropna=True))

            is_numeric = bool(pd.api.types.is_numeric_dtype(series))
            is_datetime = bool(pd.api.types.is_datetime64_any_dtype(series))

            # Non-null samples for preview
            non_null_series = series.dropna()
            sample_vals = non_null_series.head(5).tolist()

            min_val = None
            max_val = None
            mean_val = None
            median_val = None

            if is_numeric and len(non_null_series) > 0:
                try:
                    min_val = float(non_null_series.min())
                    max_val = float(non_null_series.max())
                    mean_val = float(non_null_series.mean())
                    median_val = float(non_null_series.median())
                except Exception:
                    pass

            columns_profile[col] = ColumnProfile(
                name=str(col),
                dtype=dtype_str,
                non_null_count=non_null_count,
                null_count=null_count,
                null_percentage=null_pct,
                unique_count=unique_count,
                sample_values=sample_vals,
                is_numeric=is_numeric,
                is_datetime=is_datetime,
                min_val=min_val,
                max_val=max_val,
                mean_val=mean_val,
                median_val=median_val,
            )

        # Numerical descriptive stats
        numeric_df = df.select_dtypes(include=[np.number])
        summary_stats_numeric: Dict[str, Any] = {}
        if not numeric_df.empty:
            summary_stats_numeric = numeric_df.describe().to_dict()

        # Categorical / object descriptive stats
        cat_df = df.select_dtypes(include=["object", "category"])
        summary_stats_cat: Dict[str, Any] = {}
        if not cat_df.empty:
            try:
                summary_stats_cat = cat_df.describe().to_dict()
            except Exception:
                pass

        # Sample head records
        sample_head: List[Dict[str, Any]] = []
        for _, row in df.head(5).iterrows():
            sample_head.append({k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()})

        return DatasetProfile(
            row_count=row_count,
            column_count=col_count,
            column_names=list(df.columns),
            memory_usage_bytes=mem_bytes,
            memory_usage_str=mem_str,
            duplicate_rows_count=duplicates,
            duplicate_percentage=dup_pct,
            columns=columns_profile,
            summary_stats_numeric=summary_stats_numeric,
            summary_stats_categorical=summary_stats_cat,
            sample_head=sample_head,
        )

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        """Formats byte count to human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
