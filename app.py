"""Conversational Data Analysis Assistant - Main Streamlit Application."""

import io
import time
from datetime import datetime
from typing import Optional, List
import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.data.csv_loader import CSVLoader
from src.data.profiler import DatasetProfiler, DatasetProfile
from src.analysis.prompt_builder import PromptBuilder
from src.analysis.code_validator import CodeValidator
from src.analysis.code_executor import CodeExecutor
from src.analysis.result_processor import ResultProcessor
from src.analysis.code_generator import AnalysisPipeline, AnalysisPipelineResult
from src.visualization.charts import ChartGenerator
from src.llm.llm_factory import LLMFactory
from src.llm.base import BaseLLMProvider
from src.utils.exceptions import (
    AppBaseException,
    CSVLoadError,
    DatasetValidationError,
    LLMConfigurationError,
    LLMProviderError,
    CodeValidationError,
    CodeExecutionError,
)
from src.utils.logging_config import setup_logger

logger = setup_logger("app")

# Page Configuration
st.set_page_config(
    page_title="Conversational Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished typography, card design, and badge styles
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .insight-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 1.05rem;
        line-height: 1.5;
        color: #1E293B;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initializes Streamlit session state variables."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = "None"
    if "history" not in st.session_state:
        st.session_state.history = []
    if "query_input_text" not in st.session_state:
        st.session_state.query_input_text = ""
    if "last_pipeline_result" not in st.session_state:
        st.session_state.last_pipeline_result = None


def load_dataset(source, display_name: str) -> bool:
    """Loads and profiles dataset into session state."""
    loader = CSVLoader()
    try:
        df = loader.load(source, filename=display_name)
        profile = DatasetProfiler.profile(df)
        st.session_state.df = df
        st.session_state.profile = profile
        st.session_state.dataset_name = display_name
        logger.info(f"Dataset '{display_name}' loaded successfully into session.")
        return True
    except (CSVLoadError, DatasetValidationError) as e:
        st.error(f"⚠️ **Failed to load dataset:** {e}")
        return False
    except Exception as e:
        st.error(f"⚠️ **Unexpected error loading dataset:** {e}")
        return False


def get_llm_provider(
    provider_type: str,
    api_key_override: str,
    model_override: str,
    ollama_url: str,
) -> Optional[BaseLLMProvider]:
    """Instantiates the selected LLM provider with error handling."""
    try:
        if provider_type == "Google Gemini":
            key = api_key_override.strip() or AppConfig.GEMINI_API_KEY
            if not key:
                return None
            return LLMFactory.create_provider(
                provider_name="gemini",
                api_key=key,
                model=model_override.strip() or AppConfig.GEMINI_MODEL,
            )
        elif provider_type == "Ollama (Local)":
            return LLMFactory.create_provider(
                provider_name="ollama",
                model=model_override.strip() or AppConfig.OLLAMA_MODEL,
                base_url=ollama_url.strip() or AppConfig.OLLAMA_BASE_URL,
            )
    except LLMConfigurationError as e:
        st.sidebar.error(f"Provider configuration error: {e}")
        return None
    except Exception as e:
        st.sidebar.error(f"Failed to create provider: {e}")
        return None
    return None


def main() -> None:
    init_session_state()

    # --- TOP HEADER ---
    st.markdown('<div class="main-header">📊 Conversational Data Analysis Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Natural Language to Executable Code & Data Analytics '
        'using Large Language Models (LLMs) with AST Verification and Sandboxed Execution.</div>',
        unsafe_allow_html=True,
    )

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Configuration & Data")

        # 1. Dataset Selection
        st.subheader("1. Dataset Source")
        data_source = st.radio(
            "Select Data Input:",
            ["Sample Employee Dataset", "Upload Custom CSV"],
            index=0,
            help="Choose between built-in sample employees dataset or upload your own CSV.",
        )

        if data_source == "Sample Employee Dataset":
            if st.session_state.df is None or st.session_state.dataset_name != "sample_employees.csv":
                if AppConfig.SAMPLE_DATA_PATH.exists():
                    load_dataset(AppConfig.SAMPLE_DATA_PATH, "sample_employees.csv")
                else:
                    st.error(f"Sample data file not found at {AppConfig.SAMPLE_DATA_PATH}")
            if st.button("🔄 Reload Sample Dataset", use_container_width=True):
                load_dataset(AppConfig.SAMPLE_DATA_PATH, "sample_employees.csv")
                st.rerun()

        else:
            uploaded_file = st.file_uploader(
                "Upload CSV File",
                type=["csv"],
                help=f"Upload a CSV dataset up to {AppConfig.MAX_UPLOAD_SIZE_MB} MB.",
            )
            if uploaded_file is not None:
                if st.session_state.dataset_name != uploaded_file.name:
                    with st.spinner("Parsing and profiling uploaded CSV..."):
                        if load_dataset(uploaded_file, uploaded_file.name):
                            st.success(f"Loaded '{uploaded_file.name}'!")
                            st.rerun()

        st.divider()

        # 2. LLM Provider Selection
        st.subheader("2. LLM Provider")
        provider_choice = st.selectbox(
            "Select Provider:",
            ["Google Gemini", "Ollama (Local)"],
            index=0,
            help="Choose Google Gemini (Cloud API) or Ollama (Local LLM).",
        )

        gemini_key_input = ""
        gemini_model_choice = "gemini-2.5-flash"
        ollama_url_input = "http://localhost:11434"
        ollama_model_input = "llama3.2"

        if provider_choice == "Google Gemini":
            gemini_key_input = st.text_input(
                "Gemini API Key:",
                type="password",
                value=AppConfig.GEMINI_API_KEY,
                help="Enter your Google Gemini API key or set GEMINI_API_KEY in .env.",
            )
            gemini_model_choice = st.selectbox(
                "Gemini Model:",
                ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
                index=0,
            )

            # Test connection button
            if st.button("🔌 Test Gemini Connection", use_container_width=True):
                if not gemini_key_input.strip():
                    st.warning("Please enter a Gemini API Key to test connection.")
                else:
                    with st.spinner("Testing API connection..."):
                        try:
                            test_provider = LLMFactory.create_provider(
                                provider_name="gemini",
                                api_key=gemini_key_input,
                                model=gemini_model_choice,
                            )
                            connected, msg = test_provider.validate_connection()
                            if connected:
                                st.success(f"✅ {msg}")
                            else:
                                st.error(f"❌ {msg}")
                        except Exception as e:
                            st.error(f"❌ Connection error: {e}")

            if not gemini_key_input.strip():
                st.info(
                    "💡 **No API key?** Get a free API key at [Google AI Studio](https://aistudio.google.com/). "
                    "You can still preview and explore the dataset without an API key!"
                )

        else:
            ollama_url_input = st.text_input("Ollama Server URL:", value=AppConfig.OLLAMA_BASE_URL)
            ollama_model_input = st.text_input("Ollama Model Name:", value=AppConfig.OLLAMA_MODEL)
            if st.button("🔌 Test Ollama Connection", use_container_width=True):
                with st.spinner("Testing Ollama connection..."):
                    try:
                        test_provider = LLMFactory.create_provider(
                            provider_name="ollama",
                            model=ollama_model_input,
                            base_url=ollama_url_input,
                        )
                        connected, msg = test_provider.validate_connection()
                        if connected:
                            st.success(f"✅ {msg}")
                        else:
                            st.error(f"❌ {msg}")
                    except Exception as e:
                        st.error(f"❌ Connection error: {e}")

        st.divider()

        # 3. Session Management
        st.subheader("3. Session Controls")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.history = []
                st.session_state.last_pipeline_result = None
                st.rerun()
        with col_c2:
            if st.button("♻️ Reset All", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        # System Info Card
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #6B7280; margin-top: 15px;">
            <b>Security & Architecture:</b><br/>
            • AST-based static code validation<br/>
            • Restricted Python runtime sandbox<br/>
            • Privacy-conscious schema-only prompts<br/>
            • Verified interactive visualizations
            </div>
            """,
            unsafe_allow_html=True,
        )

    # If no dataset is loaded, prompt user
    if st.session_state.df is None:
        st.info("👋 Welcome! Please select a dataset in the sidebar to begin analysis.")
        return

    df: pd.DataFrame = st.session_state.df
    profile: DatasetProfile = st.session_state.profile

    # --- TOP METRIC SUMMARY CARDS ---
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Dataset</div>'
            f'<div class="metric-value" style="font-size:1.1rem;word-break:break-all;">{st.session_state.dataset_name}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Rows</div>'
            f'<div class="metric-value">{profile.row_count:,}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Columns</div>'
            f'<div class="metric-value">{profile.column_count}</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        total_nulls = sum(p.null_count for p in profile.columns.values())
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Missing Values</div>'
            f'<div class="metric-value">{total_nulls}</div></div>',
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Duplicates</div>'
            f'<div class="metric-value">{profile.duplicate_rows_count}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # --- TABS INTERFACE ---
    tab_query, tab_data, tab_charts, tab_history = st.tabs([
        "💬 Conversational Query & Analysis",
        "📋 Dataset Explorer & Profiling",
        "📊 Visual Analytics Studio",
        "📜 Query History & Audit Log",
    ])

    # =========================================================================
    # TAB 1: CONVERSATIONAL QUERY & ANALYSIS
    # =========================================================================
    with tab_query:
        st.subheader("💡 Suggested Analysis Queries")
        st.write("Click any sample query below or type your own question in the input field:")

        # Example question pills / buttons
        sample_queries = [
            "What is the average salary?",
            "Find the employee with the highest salary.",
            "Show the top 5 employees by salary.",
            "Find the salary of employee Rahul.",
            "Calculate department-wise average salary.",
            "Count employees in every department.",
            "Show missing values in each column.",
            "Identify duplicate records.",
            "Generate a chart showing average salary by department.",
            "Explain the key insights in this dataset.",
        ]

        q_cols = st.columns(5)
        for idx, q_text in enumerate(sample_queries):
            col_target = q_cols[idx % 5]
            if col_target.button(q_text, key=f"sample_btn_{idx}", use_container_width=True):
                st.session_state.query_input_text = q_text
                st.rerun()

        st.markdown("<br/>", unsafe_allow_html=True)

        # Question input form
        with st.form("query_form", clear_on_submit=False):
            user_question = st.text_input(
                "Ask a question about the active dataset in natural language:",
                value=st.session_state.query_input_text,
                placeholder="e.g., Which department has the highest average salary?",
                key="query_text_field",
            )
            submit_btn = st.form_submit_button("🚀 Run Analysis", use_container_width=True)

        if submit_btn and user_question.strip():
            # Clear previous preset input state
            st.session_state.query_input_text = user_question

            # Check LLM provider availability
            active_provider = get_llm_provider(
                provider_choice, gemini_key_input, gemini_model_choice, ollama_url_input
            )

            if active_provider is None:
                if provider_choice == "Google Gemini":
                    st.error(
                        "🔒 **Gemini API Key Required:** Please configure your `GEMINI_API_KEY` in `.env` "
                        "or enter it in the sidebar on the left. You can obtain a free key at "
                        "[Google AI Studio](https://aistudio.google.com/)."
                    )
                else:
                    st.error(
                        "🔒 **Ollama Provider Unreachable:** Please ensure your local Ollama instance is running "
                        "at the specified URL."
                    )
            else:
                pipeline = AnalysisPipeline(llm_provider=active_provider, max_retries=1)

                with st.status("🔍 Analyzing query and executing code...", expanded=True) as status:
                    st.write("1️⃣ Formulating prompt with dataset schema and privacy boundaries...")
                    time.sleep(0.1)

                    st.write("2️⃣ Requesting Python/Pandas code from LLM...")
                    pipeline_res = pipeline.run(user_question, df, profile)

                    if not pipeline_res.success:
                        status.update(label=f"❌ Analysis failed at {pipeline_res.error_stage} stage.", state="error")
                    else:
                        st.write("3️⃣ Passed AST static security & syntax checks.")
                        st.write("4️⃣ Executed code against dataset in isolated sandbox.")
                        st.write("5️⃣ Synthesized business insights and verified outputs.")
                        status.update(label="✅ Analysis completed successfully!", state="complete")

                st.session_state.last_pipeline_result = pipeline_res
                # Append to history
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": user_question,
                    "result": pipeline_res,
                })

        # --- DISPLAY CURRENT RESULT ---
        curr_res: Optional[AnalysisPipelineResult] = st.session_state.last_pipeline_result
        if curr_res is not None:
            st.divider()

            if not curr_res.success:
                st.error(
                    f"### ❌ Analysis Could Not Be Completed\n\n"
                    f"**Stage:** `{curr_res.error_stage}`\n\n"
                    f"**Details:** {curr_res.error_message}\n\n"
                    f"**Troubleshooting Tips:**\n"
                    f"- Verify that requested column names exist in the dataset.\n"
                    f"- Check your API key and network connection.\n"
                    f"- Try rephrasing your question to be more specific."
                )
                if curr_res.generated_code:
                    with st.expander("Inspect Generated Code (Failed)"):
                        st.code(curr_res.generated_code, language="python")
            else:
                # 1. Natural Language Explanation
                st.markdown(
                    f'<div class="insight-box"><b>💡 Analysis Insight:</b><br/>{curr_res.explanation}</div>',
                    unsafe_allow_html=True,
                )

                # 2. Execution Results Presentation
                st.subheader("📋 Computed Result")
                proc = curr_res.processed_result

                if proc.result_type == "empty":
                    st.warning("Query produced an empty result set (0 records).")

                elif proc.display_df is not None:
                    st.dataframe(proc.display_df, use_container_width=True)

                    # Export button for computed table
                    csv_data = proc.display_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Result as CSV",
                        data=csv_data,
                        file_name=f"result_{int(time.time())}.csv",
                        mime="text/csv",
                    )

                elif proc.display_scalar is not None:
                    st.metric(label="Answer Metric", value=str(proc.display_scalar))

                elif proc.display_dict is not None:
                    st.json(proc.display_dict)

                # 3. Interactive Visualization (Auto or Result-Driven)
                fig = ChartGenerator.auto_visualize(df, proc, curr_res.question)
                if fig is not None:
                    st.subheader("📊 Interactive Visualization")
                    st.plotly_chart(fig, use_container_width=True)

                # 4. Expandable Code Viewer & Telemetry
                with st.expander("💻 View Executed Python / Pandas Code & Sandbox Telemetry", expanded=False):
                    st.markdown("**LLM-Generated Pandas Code (Validated via AST):**")
                    st.code(curr_res.generated_code, language="python")

                    if curr_res.execution_result:
                        ex = curr_res.execution_result
                        t_cols = st.columns(3)
                        t_cols[0].metric("Execution Time", f"{ex.execution_time_ms:.2f} ms")
                        t_cols[1].metric("Security Check", "AST Verified ✅")
                        t_cols[2].metric("Sandbox Status", "Safe Execution ✅")

                        if ex.stdout.strip():
                            st.markdown("**Standard Output Capture:**")
                            st.text(ex.stdout)

    # =========================================================================
    # TAB 2: DATASET EXPLORER & PROFILING
    # =========================================================================
    with tab_data:
        st.subheader(f"Dataset Overview: {st.session_state.dataset_name}")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.write("**Dataset Shape & Structure:**")
            st.write(f"- Rows: **{profile.row_count:,}**")
            st.write(f"- Columns: **{profile.column_count}**")
            st.write(f"- Deep Memory Footprint: **{profile.memory_usage_str}**")
            st.write(f"- Duplicate Records: **{profile.duplicate_rows_count}** ({profile.duplicate_percentage:.2f}%)")

        with p_col2:
            st.write("**Data Types Breakdown:**")
            dtypes_count = df.dtypes.value_map_str = df.dtypes.astype(str).value_counts()
            st.write(dtypes_count.to_dict())

        st.divider()

        # Preview Data Table
        st.subheader("Data Preview (First 100 Rows)")
        st.dataframe(df.head(100), use_container_width=True)

        st.divider()

        # Detailed Column Metadata Table
        st.subheader("Column Metadata & Health Report")
        col_summary_data = []
        for col_name, c_prof in profile.columns.items():
            col_summary_data.append({
                "Column": col_name,
                "Data Type": c_prof.dtype,
                "Non-Null Count": c_prof.non_null_count,
                "Null Count": c_prof.null_count,
                "Null %": f"{c_prof.null_percentage:.1f}%",
                "Unique Values": c_prof.unique_count,
                "Min": c_prof.min_val if c_prof.min_val is not None else "-",
                "Max": c_prof.max_val if c_prof.max_val is not None else "-",
                "Mean": f"{c_prof.mean_val:.2f}" if c_prof.mean_val is not None else "-",
            })
        st.dataframe(pd.DataFrame(col_summary_data), use_container_width=True)

        # Numerical Descriptive Statistics
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            st.divider()
            st.subheader("Descriptive Statistics (Numerical Columns)")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # =========================================================================
    # TAB 3: VISUAL ANALYTICS STUDIO
    # =========================================================================
    with tab_charts:
        st.subheader("🎨 Custom Interactive Chart Studio")
        st.write("Generate interactive Plotly visualizations directly on the active dataset:")

        v_c1, v_c2, v_c3 = st.columns([1, 1, 1])

        with v_c1:
            chart_type_sel = st.selectbox(
                "Chart Type:",
                ["Bar Chart", "Line Chart", "Histogram", "Scatter Plot", "Box Plot", "Correlation Heatmap"],
            )

        all_cols = list(df.columns)
        num_cols = list(df.select_dtypes(include=["number"]).columns)

        with v_c2:
            if chart_type_sel in ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"]:
                x_choice = st.selectbox("X-Axis Feature:", all_cols, index=0)
            elif chart_type_sel == "Histogram":
                x_choice = st.selectbox("Numerical Feature for Distribution:", num_cols or all_cols, index=0)
            else:
                x_choice = None

        with v_c3:
            if chart_type_sel in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                y_choice = st.selectbox(
                    "Y-Axis Feature:",
                    num_cols or all_cols,
                    index=min(1, len(num_cols or all_cols) - 1),
                )
            elif chart_type_sel == "Box Plot":
                y_choice = st.selectbox("Value Feature (Y):", num_cols or all_cols, index=0)
            else:
                y_choice = None

        if st.button("Generate Chart", use_container_width=True):
            try:
                custom_fig = None
                if chart_type_sel == "Bar Chart":
                    # Aggregate if many records
                    agg_df = df.groupby(x_choice)[y_choice].mean().reset_index()
                    custom_fig = ChartGenerator.create_bar_chart(
                        agg_df, x_choice, y_choice, title=f"Average {y_choice} by {x_choice}"
                    )
                elif chart_type_sel == "Line Chart":
                    custom_fig = ChartGenerator.create_line_chart(df, x_choice, y_choice)
                elif chart_type_sel == "Histogram":
                    custom_fig = ChartGenerator.create_histogram(df, x_choice)
                elif chart_type_sel == "Scatter Plot":
                    custom_fig = ChartGenerator.create_scatter_plot(df, x_choice, y_choice)
                elif chart_type_sel == "Box Plot":
                    custom_fig = ChartGenerator.create_box_plot(df, y_choice, x_choice)
                elif chart_type_sel == "Correlation Heatmap":
                    custom_fig = ChartGenerator.create_correlation_heatmap(df)

                if custom_fig is not None:
                    st.plotly_chart(custom_fig, use_container_width=True)
                else:
                    st.warning("Could not generate chart with selected features.")
            except Exception as e:
                st.error(f"Failed to generate visualization: {e}")

    # =========================================================================
    # TAB 4: QUERY HISTORY & AUDIT LOG
    # =========================================================================
    with tab_history:
        st.subheader("📜 Session Query History & Audit Trail")

        if not st.session_state.history:
            st.info("No queries have been run in this session yet.")
        else:
            for idx, item in enumerate(reversed(st.session_state.history)):
                q_res: AnalysisPipelineResult = item["result"]
                with st.expander(f"Query #{len(st.session_state.history) - idx}: {item['question']} ({item['timestamp']})", expanded=(idx == 0)):
                    st.markdown(f"**Question:** `{item['question']}`")
                    st.markdown(f"**Status:** {'✅ Succeeded' if q_res.success else '❌ Failed'}")
                    if q_res.success:
                        st.markdown(f"**Explanation:**\n{q_res.explanation}")
                        st.markdown("**Executed Code:**")
                        st.code(q_res.generated_code, language="python")
                        if q_res.execution_result:
                            st.markdown(f"**Execution Time:** {q_res.execution_result.execution_time_ms:.2f} ms")
                    else:
                        st.error(f"Error ({q_res.error_stage}): {q_res.error_message}")


if __name__ == "__main__":
    main()
