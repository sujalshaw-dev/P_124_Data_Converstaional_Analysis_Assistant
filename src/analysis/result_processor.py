"""Result processing module for formatting, inspecting, and standardizing execution outputs."""

from dataclasses import dataclass
from typing import Any, Optional, Dict
import pandas as pd
import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger("result_processor")


@dataclass
class ProcessedResult:
    """Standardized representation of execution results for UI rendering and LLM interpretation."""

    result_type: str  # 'dataframe', 'series', 'scalar', 'dict', 'empty', 'none'
    display_df: Optional[pd.DataFrame] = None
    display_scalar: Optional[Any] = None
    display_dict: Optional[Dict[str, Any]] = None
    summary_text: str = ""
    is_empty: bool = False
    can_visualize: bool = False
    suggested_chart_type: Optional[str] = None
    suggested_x: Optional[str] = None
    suggested_y: Optional[str] = None


class ResultProcessor:
    """Processes, cleans, and analyzes raw execution outputs."""

    @classmethod
    def process(cls, raw_result: Any) -> ProcessedResult:
        """Processes any raw Python output into a structured ProcessedResult."""
        if raw_result is None:
            return ProcessedResult(
                result_type="none",
                summary_text="The code executed successfully but returned None.",
                is_empty=True,
            )

        # Handle Pandas DataFrame
        if isinstance(raw_result, pd.DataFrame):
            return cls._process_dataframe(raw_result)

        # Handle Pandas Series
        if isinstance(raw_result, pd.Series):
            return cls._process_series(raw_result)

        # Handle Numpy scalar
        if isinstance(raw_result, (np.generic, np.ndarray)):
            if isinstance(raw_result, np.ndarray):
                if raw_result.size == 1:
                    return cls._process_scalar(raw_result.item())
                else:
                    return cls._process_dataframe(pd.DataFrame(raw_result))
            return cls._process_scalar(raw_result.item())

        # Handle Python Dictionary
        if isinstance(raw_result, dict):
            return cls._process_dict(raw_result)

        # Handle Python List / Tuple / Set
        if isinstance(raw_result, (list, tuple, set)):
            return cls._process_sequence(raw_result)

        # Handle Standard Python Scalars
        if isinstance(raw_result, (int, float, str, bool)):
            return cls._process_scalar(raw_result)

        # Fallback for unexpected objects
        return ProcessedResult(
            result_type="scalar",
            display_scalar=str(raw_result),
            summary_text=str(raw_result),
        )

    @classmethod
    def _process_dataframe(cls, df: pd.DataFrame) -> ProcessedResult:
        """Processes DataFrame outputs."""
        if df.empty:
            return ProcessedResult(
                result_type="empty",
                display_df=df,
                summary_text="DataFrame is empty (0 rows matching query).",
                is_empty=True,
            )

        # Check for visualization potential
        can_vis = False
        chart_type = None
        sug_x = None
        sug_y = None

        cols = list(df.columns)
        if len(cols) >= 2:
            first_col = cols[0]
            second_col = cols[1]
            first_is_num = pd.api.types.is_numeric_dtype(df[first_col])
            second_is_num = pd.api.types.is_numeric_dtype(df[second_col])

            if not first_is_num and second_is_num:
                can_vis = True
                chart_type = "bar" if len(df) <= 30 else "line"
                sug_x = first_col
                sug_y = second_col
            elif first_is_num and second_is_num:
                can_vis = True
                chart_type = "scatter"
                sug_x = first_col
                sug_y = second_col

        summary_preview = df.head(10).to_string(index=True)
        summary_text = (
            f"DataFrame with {len(df)} rows and {len(df.columns)} columns.\n"
            f"Preview:\n{summary_preview}"
        )

        return ProcessedResult(
            result_type="dataframe",
            display_df=df,
            summary_text=summary_text,
            can_visualize=can_vis,
            suggested_chart_type=chart_type,
            suggested_x=sug_x,
            suggested_y=sug_y,
        )

    @classmethod
    def _process_series(cls, series: pd.Series) -> ProcessedResult:
        """Processes Series outputs."""
        if series.empty:
            return ProcessedResult(
                result_type="empty",
                summary_text="Series is empty (0 elements).",
                is_empty=True,
            )

        # Single-element series can be treated as a scalar
        if len(series) == 1:
            val = series.iloc[0]
            name = str(series.name or series.index[0])
            return ProcessedResult(
                result_type="scalar",
                display_scalar=val,
                summary_text=f"{name}: {val}",
            )

        # Convert multi-element series to a DataFrame for clean presentation
        df_series = series.reset_index()
        if len(df_series.columns) == 2:
            df_series.columns = ["Index", series.name or "Value"]

        can_vis = False
        chart_type = None
        sug_x = None
        sug_y = None

        if len(df_series.columns) == 2:
            can_vis = True
            chart_type = "bar"
            sug_x = df_series.columns[0]
            sug_y = df_series.columns[1]

        summary_text = f"Series with {len(series)} items:\n{series.head(10).to_string()}"

        return ProcessedResult(
            result_type="series",
            display_df=df_series,
            summary_text=summary_text,
            can_visualize=can_vis,
            suggested_chart_type=chart_type,
            suggested_x=sug_x,
            suggested_y=sug_y,
        )

    @classmethod
    def _process_dict(cls, data: dict) -> ProcessedResult:
        """Processes dictionary outputs."""
        summary = "\n".join(f"{k}: {v}" for k, v in list(data.items())[:15])

        # Attempt to convert simple flat dict to DataFrame for table presentation
        can_convert_df = False
        display_df = None
        can_vis = False
        chart_type = None

        try:
            # If values are all primitives
            if all(isinstance(v, (int, float, str, bool)) for v in data.values()):
                display_df = pd.DataFrame(list(data.items()), columns=["Category", "Value"])
                can_convert_df = True
                if any(isinstance(v, (int, float)) for v in data.values()):
                    can_vis = True
                    chart_type = "bar"
        except Exception:
            pass

        return ProcessedResult(
            result_type="dataframe" if can_convert_df else "dict",
            display_df=display_df,
            display_dict=data,
            summary_text=f"Dictionary result:\n{summary}",
            can_visualize=can_vis,
            suggested_chart_type=chart_type,
            suggested_x="Category" if can_vis else None,
            suggested_y="Value" if can_vis else None,
        )

    @classmethod
    def _process_sequence(cls, seq: Any) -> ProcessedResult:
        """Processes list, tuple, or set outputs."""
        items = list(seq)
        if not items:
            return ProcessedResult(
                result_type="empty",
                summary_text="List/sequence is empty.",
                is_empty=True,
            )

        if len(items) == 1:
            return cls._process_scalar(items[0])

        # Try converting sequence of dicts to DataFrame
        if all(isinstance(i, dict) for i in items):
            try:
                df = pd.DataFrame(items)
                return cls._process_dataframe(df)
            except Exception:
                pass

        summary = ", ".join(str(i) for i in items[:20])
        return ProcessedResult(
            result_type="scalar",
            display_scalar=items,
            summary_text=f"List of {len(items)} items: [{summary}]",
        )

    @classmethod
    def _process_scalar(cls, val: Any) -> ProcessedResult:
        """Processes scalar numerical or textual results."""
        if isinstance(val, float):
            formatted_str = f"{val:,.4f}".rstrip("0").rstrip(".")
        elif isinstance(val, int):
            formatted_str = f"{val:,}"
        else:
            formatted_str = str(val)

        return ProcessedResult(
            result_type="scalar",
            display_scalar=val,
            summary_text=formatted_str,
        )
