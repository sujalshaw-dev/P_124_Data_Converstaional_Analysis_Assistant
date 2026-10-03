"""Interactive Plotly chart generation with schema and data validation."""

from typing import Optional, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.analysis.result_processor import ProcessedResult
from src.utils.exceptions import ChartGenerationError
from src.utils.logging_config import setup_logger

logger = setup_logger("charts")

CHART_THEME = "plotly_white"
DEFAULT_COLOR_PALETTE = px.colors.qualitative.Prism


class ChartGenerator:
    """Generates validated, styled Plotly charts for data analysis results."""

    @classmethod
    def create_bar_chart(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: Optional[str] = None,
        orientation: str = "v",
    ) -> go.Figure:
        """Generates a styled bar chart."""
        cls._validate_columns(df, [x_col, y_col])
        fig = px.bar(
            df,
            x=x_col if orientation == "v" else y_col,
            y=y_col if orientation == "v" else x_col,
            orientation=orientation,
            title=title or f"{y_col} by {x_col}",
            template=CHART_THEME,
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def create_line_chart(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Generates a styled line chart."""
        cls._validate_columns(df, [x_col, y_col])
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title or f"{y_col} Trend over {x_col}",
            template=CHART_THEME,
            markers=True,
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def create_histogram(
        cls,
        df: pd.DataFrame,
        col: str,
        nbins: int = 25,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Generates a styled histogram distribution."""
        cls._validate_columns(df, [col])
        fig = px.histogram(
            df,
            x=col,
            nbins=nbins,
            title=title or f"Distribution of {col}",
            template=CHART_THEME,
            marginal="box",
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )
        fig.update_layout(xaxis_title=col, yaxis_title="Count", margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def create_scatter_plot(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Generates a scatter plot."""
        needed_cols = [x_col, y_col]
        if color_col and color_col in df.columns:
            needed_cols.append(color_col)
        cls._validate_columns(df, needed_cols)

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col if color_col in df.columns else None,
            title=title or f"{y_col} vs {x_col}",
            template=CHART_THEME,
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
            trendline="ols" if len(df) > 5 and color_col is None else None,
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def create_box_plot(
        cls,
        df: pd.DataFrame,
        y_col: str,
        x_col: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Generates a box plot."""
        needed = [y_col]
        if x_col and x_col in df.columns:
            needed.append(x_col)
        cls._validate_columns(df, needed)

        fig = px.box(
            df,
            x=x_col if x_col in df.columns else None,
            y=y_col,
            title=title or f"Box Plot of {y_col}" + (f" by {x_col}" if x_col else ""),
            template=CHART_THEME,
            color=x_col if x_col in df.columns else None,
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )
        fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def create_pie_chart(
        cls,
        df: pd.DataFrame,
        names_col: str,
        values_col: str,
        title: Optional[str] = None,
    ) -> go.Figure:
        """Generates a pie / donut chart."""
        cls._validate_columns(df, [names_col, values_col])
        fig = px.pie(
            df,
            names=names_col,
            values=values_col,
            hole=0.4,
            title=title or f"{values_col} Distribution by {names_col}",
            template=CHART_THEME,
            color_discrete_sequence=DEFAULT_COLOR_PALETTE,
        )
        fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def create_correlation_heatmap(
        cls,
        df: pd.DataFrame,
        title: Optional[str] = None,
    ) -> Optional[go.Figure]:
        """Generates a correlation heatmap for numerical columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None

        corr = numeric_df.corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title=title or "Feature Correlation Heatmap",
            color_continuous_scale="Blues",
            template=CHART_THEME,
        )
        fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @classmethod
    def auto_visualize(
        cls,
        active_df: pd.DataFrame,
        processed_result: ProcessedResult,
        question: str,
    ) -> Optional[go.Figure]:
        """Infers and produces the most suitable visualization based on question intent and result structure."""
        q_lower = question.lower()
        wants_chart = any(
            w in q_lower
            for w in [
                "chart",
                "plot",
                "graph",
                "visualize",
                "visualization",
                "distribution",
                "histogram",
                "heatmap",
                "scatter",
                "box",
                "pie",
                "trend",
            ]
        )

        # 1. Correlation heatmap request
        if "correlation" in q_lower or "heatmap" in q_lower:
            return cls.create_correlation_heatmap(active_df)

        # 2. Direct visualization from execution result DataFrame
        if processed_result.display_df is not None and not processed_result.display_df.empty:
            res_df = processed_result.display_df
            cols = list(res_df.columns)

            # If question specifically asked for pie chart and 2 columns exist
            if "pie" in q_lower and len(cols) >= 2:
                try:
                    return cls.create_pie_chart(res_df, cols[0], cols[1])
                except Exception:
                    pass

            # If question asked for line chart
            if "line" in q_lower and len(cols) >= 2:
                try:
                    return cls.create_line_chart(res_df, cols[0], cols[1])
                except Exception:
                    pass

            # If result suggests a chart type
            if processed_result.can_visualize and processed_result.suggested_chart_type:
                chart_t = processed_result.suggested_chart_type
                sx = processed_result.suggested_x or cols[0]
                sy = processed_result.suggested_y or cols[1]

                try:
                    if chart_t == "bar":
                        return cls.create_bar_chart(res_df, sx, sy)
                    elif chart_t == "line":
                        return cls.create_line_chart(res_df, sx, sy)
                    elif chart_t == "scatter":
                        return cls.create_scatter_plot(res_df, sx, sy)
                except Exception as e:
                    logger.warning(f"Auto-visualize from result failed: {e}")

        # 3. If user asked for visualization from active dataset directly
        if wants_chart and not active_df.empty:
            num_cols = active_df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = active_df.select_dtypes(include=["object", "category"]).columns.tolist()

            # Histogram
            if "histogram" in q_lower or "distribution" in q_lower:
                for col in num_cols:
                    if col.lower() in q_lower:
                        return cls.create_histogram(active_df, col)
                if num_cols:
                    return cls.create_histogram(active_df, num_cols[0])

            # Box plot
            if "box" in q_lower:
                for col in num_cols:
                    if col.lower() in q_lower:
                        cat_match = next((c for c in cat_cols if c.lower() in q_lower), None)
                        return cls.create_box_plot(active_df, col, cat_match)
                if num_cols:
                    return cls.create_box_plot(active_df, num_cols[0])

            # Scatter plot
            if "scatter" in q_lower and len(num_cols) >= 2:
                matched = [c for c in num_cols if c.lower() in q_lower]
                if len(matched) >= 2:
                    return cls.create_scatter_plot(active_df, matched[0], matched[1])
                return cls.create_scatter_plot(active_df, num_cols[0], num_cols[1])

        return None

    @staticmethod
    def _validate_columns(df: pd.DataFrame, columns: List[str]) -> None:
        """Validates that requested columns exist in the DataFrame."""
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ChartGenerationError(
                f"Missing column(s) for chart generation: {missing}. Available: {list(df.columns)}"
            )
