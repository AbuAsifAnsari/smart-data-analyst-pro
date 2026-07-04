from utils.data_manipulator import show_manipulate_page
import streamlit as st
import pandas as pd

from utils.data_loader import load_file, get_data_summary
from utils.ollama_chat import (ask_gemma, get_llm_status, find_best_column,
                                detect_date_column, compute_table)
from utils.memory_engine import save_qa, get_relevant_memory
from utils.dashboard import auto_detect_kpi_columns, compute_kpis
from utils.chart_agent import generate_chart
from utils.history_manager import save_message, load_history, clear_history, get_all_sessions
from utils.report_generator import generate_excel, generate_pdf
from utils.sql_engine import show_sql_page
from utils.ml_engine import show_ml_page

st.set_page_config(page_title="Smart Data Analyst Pro", layout="wide")

# ── ID column filter helper ───────────────────────────────────────────────────
_ID_KW = ["id", "_id", "code", "no.", "number", "num",
          "invoice", "serial", "index", "key", "order"]

def _useful_cats(df):
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    filtered = [c for c in cat_cols if not any(kw in c.lower() for kw in _ID_KW)]
    return filtered if filtered else cat_cols

def _smart_num(df):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    rev  = next((c for c in num_cols if any(k in c.lower()
                 for k in ["revenue","sales","amount","income","profit"])), None)
    qty  = next((c for c in num_cols if any(k in c.lower()
                 for k in ["quantity","qty","units","volume"])), None)
    main = rev or qty or (num_cols[0] if num_cols else None)
    sec  = (qty if qty and qty != main else
            rev  if rev  and rev  != main else
            (num_cols[1] if len(num_cols) >= 2 else None))
    return main, sec, num_cols


# ══════════════════════════════════════════════════════════════
# V12 — MULTI-FILE UPLOAD
# ══════════════════════════════════════════════════════════════

# Multiple file uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV / Excel (multiple files supported)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True
)

# Load all uploaded files
loaded_files = {}  # {filename: df}
for uf in uploaded_files:
    d, err = load_file(uf)
    if err:
        st.sidebar.error(f"{uf.name}: {err}")
    else:
        loaded_files[uf.name] = d

# ── File selector (agar multiple files hain) ─────────────────
df           = None
dataset_name = "unknown"

if loaded_files:
    file_names = list(loaded_files.keys())

    if len(file_names) == 1:
        # Single file — same as before
        dataset_name = file_names[0]
        df           = loaded_files[dataset_name]
        st.session_state["df"]           = df
        st.session_state["dataset_name"] = dataset_name
        st.session_state["loaded_files"] = loaded_files

    else:
        # Multiple files — selector + merge option
        st.session_state["loaded_files"] = loaded_files

        st.sidebar.markdown("**📂 Loaded Files:**")
        for fname, fdf in loaded_files.items():
            st.sidebar.caption(f"✅ {fname} — {fdf.shape[0]}R × {fdf.shape[1]}C")

        st.sidebar.divider()

        # Active file selector
        selected = st.sidebar.selectbox(
            "Active file for analysis",
            file_names,
            key="active_file"
        )
        dataset_name = selected
        df           = loaded_files[selected]
        st.session_state["df"]           = df
        st.session_state["dataset_name"] = dataset_name

        # Merge option
        st.sidebar.markdown("**🔗 Merge Files:**")
        merge_f1 = st.sidebar.selectbox("File 1", file_names, key="mf1")
        merge_f2 = st.sidebar.selectbox("File 2",
                    [f for f in file_names if f != merge_f1],
                    key="mf2")

        if merge_f1 and merge_f2:
            df1 = loaded_files[merge_f1]
            df2 = loaded_files[merge_f2]
            common = [c for c in df1.columns if c in df2.columns]

            if common:
                merge_on  = st.sidebar.selectbox("Join on", common, key="merge_on")
                merge_how = st.sidebar.selectbox("Join type",
                             ["inner","left","right","outer"], key="merge_how")

                if st.sidebar.button("🔗 Merge & Use", use_container_width=True):
                    merged = pd.merge(df1, df2, on=merge_on, how=merge_how)
                    mname  = f"merged_{merge_f1.split('.')[0]}_{merge_f2.split('.')[0]}.csv"
                    st.session_state["df"]           = merged
                    st.session_state["dataset_name"] = mname
                    st.session_state["loaded_files"][mname] = merged
                    st.sidebar.success(f"Merged! {merged.shape[0]} rows × {merged.shape[1]} cols")
                    st.rerun()
            else:
                st.sidebar.warning("No common columns found for joining.")

elif "df" in st.session_state:
    df           = st.session_state["df"]
    dataset_name = st.session_state.get("dataset_name", "unknown")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Smart Data Analyst Pro")
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "💬 Chat",
        "🛠️ Manipulate",
        "🗄️ SQL",
        "🤖 ML"
    ], index=0)
    st.divider()

    status = get_llm_status()
    if status["gemini"]:
        st.success("🤖 Gemini API connected")
    elif status["ollama"]:
        st.warning("🦙 Ollama fallback active")
    else:
        st.error("⚠️ No LLM — Python compute only")
    st.caption(f"Mode: {status['mode']}")
    st.divider()

    sessions = get_all_sessions()
    if sessions:
        st.markdown("**📂 Past Sessions:**")
        for ds_name, info in sessions.items():
            short = ds_name[:22] + "..." if len(ds_name) > 25 else ds_name
            if st.button(f"📄 {short} ({info['questions']}Q)",
                         key=f"sess_{ds_name}", use_container_width=True,
                         help=f"Last: {info['last_seen']}\n{info['preview']}"):
                st.session_state["load_session"] = ds_name
                st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        if dataset_name and dataset_name != "unknown":
            clear_history(dataset_name)
        st.rerun()
    if st.session_state.get("messages"):
        total_q = len(st.session_state.messages) // 2
        st.caption(f"{total_q} question{'s' if total_q != 1 else ''} asked")


# ── Dashboard Page ─────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.header("📊 Dashboard")

    if df is None:
        st.info("⬅️ Upload a CSV or Excel file from the sidebar.")
    else:
        # V12 — multi-file badge
        loaded = st.session_state.get("loaded_files", {})
        if len(loaded) > 1:
            st.success(f"📂 {len(loaded)} files loaded — Active: `{dataset_name}`")

        kpi_cols = auto_detect_kpi_columns(df)
        kpis     = compute_kpis(df, kpi_cols)

        if kpis:
            card_cols = st.columns(len(kpis))
            for col_widget, (label, value) in zip(card_cols, kpis.items()):
                col_widget.metric(label=label, value=value)
        else:
            st.warning("KPI columns could not be auto-detected — please check the dataset.")

        st.divider()
        st.subheader("📈 Data Charts")

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols     = df.select_dtypes(include='object').columns.tolist()

        if cat_cols and numeric_cols:
            rev_col = (find_best_column("revenue",  list(df.columns))
                       or find_best_column("profit",  list(df.columns))
                       or numeric_cols[0])
            prod_col = (find_best_column("product",  list(df.columns))
                        or find_best_column("category", list(df.columns))
                        or cat_cols[0])
            region_col = (find_best_column("region",   list(df.columns))
                          or find_best_column("city",    list(df.columns))
                          or find_best_column("customer", list(df.columns)))

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.caption(f"📦 {rev_col} by {prod_col}")
                fig1 = generate_chart(
                    question=f"top {prod_col} by {rev_col}", df=df,
                    intent="ranking", cols={"primary": prod_col, "secondary": rev_col})
                if fig1: st.plotly_chart(fig1, use_container_width=True)
                else:    st.info("Chart could not be generated.")

            with chart_col2:
                if region_col:
                    st.caption(f"🗺️ {rev_col} by {region_col}")
                    fig2 = generate_chart(
                        question=f"top {region_col} by {rev_col}", df=df,
                        intent="ranking", cols={"primary": region_col, "secondary": rev_col})
                elif cat_cols and cat_cols[0] != prod_col:
                    alt = cat_cols[0]
                    st.caption(f"🏷️ {rev_col} by {alt}")
                    fig2 = generate_chart(
                        question=f"top {alt} by {rev_col}", df=df,
                        intent="ranking", cols={"primary": alt, "secondary": rev_col})
                else:
                    num2 = numeric_cols[1] if len(numeric_cols) >= 2 else numeric_cols[0]
                    st.caption(f"📊 {num2} by {prod_col}")
                    fig2 = generate_chart(
                        question=f"top {prod_col} by {num2}", df=df,
                        intent="ranking", cols={"primary": prod_col, "secondary": num2})
                if fig2: st.plotly_chart(fig2, use_container_width=True)
                else:    st.info("Chart could not be generated.")

            date_col = detect_date_column(df)
            if date_col:
                st.subheader("📅 Trend Over Time")
                fig3 = generate_chart(
                    question=f"monthly trend of {rev_col}", df=df,
                    intent="trend", cols={"primary": rev_col, "secondary": None})
                if fig3: st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Both numeric and categorical columns are needed for charts.")

        st.divider()
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(50), use_container_width=True)

        with st.expander("🔍 Auto-detected KPI columns"):
            for k, v in kpi_cols.items():
                st.write(f"**{k}** → `{v}`" if v else f"**{k}** → ⚠️ not found")

        with st.expander("📊 Dataset Summary"):
            st.text(get_data_summary(df))

        st.divider()
        st.subheader("📥 Download Report")
        rep_col1, rep_col2 = st.columns(2)

        report_charts = []
        if cat_cols and numeric_cols:
            fig_r1 = generate_chart(
                question=f"top {prod_col} by {rev_col}", df=df,
                intent="ranking", cols={"primary": prod_col, "secondary": rev_col})
            if fig_r1: report_charts.append((fig_r1, f"{rev_col} by {prod_col}"))
            if region_col:
                fig_r2 = generate_chart(
                    question=f"top {region_col} by {rev_col}", df=df,
                    intent="ranking", cols={"primary": region_col, "secondary": rev_col})
                if fig_r2: report_charts.append((fig_r2, f"{rev_col} by {region_col}"))

        with rep_col1:
            try:
                excel_bytes = generate_excel(
                    df=df, kpis=kpis,
                    messages=st.session_state.get("messages", []),
                    dataset_name=dataset_name)
                st.download_button(
                    label="⬇️ Download Excel Report", data=excel_bytes,
                    file_name=f"report_{dataset_name.split('.')[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True)
            except Exception as e:
                st.error(f"Excel error: {e}")

        with rep_col2:
            try:
                pdf_bytes = generate_pdf(
                    df=df, kpis=kpis,
                    messages=st.session_state.get("messages", []),
                    dataset_name=dataset_name, charts=report_charts)
                st.download_button(
                    label="⬇️ Download PDF Report", data=pdf_bytes,
                    file_name=f"report_{dataset_name.split('.')[0]}.pdf",
                    mime="application/pdf",
                    use_container_width=True)
            except Exception as e:
                st.error(f"PDF error: {e}")


# ── Chat Page ──────────────────────────────────────────────────────────────────
elif page == "💬 Chat":
    st.header("💬 Chat with your Data")

    if df is None:
        st.info("⬅️ First upload data from sidebar.")
    else:
        st.caption(f"📂 Loaded: `{dataset_name}` — {df.shape[0]} rows × {df.shape[1]} columns")

        if ("messages" not in st.session_state
                or st.session_state.get("last_dataset") != dataset_name
                or "load_session" in st.session_state):
            load_ds = st.session_state.pop("load_session", dataset_name)
            saved   = load_history(load_ds)
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"],
                 "query": m.get("query", ""), "had_chart": False}
                for m in saved
            ]
            st.session_state["last_dataset"] = dataset_name

        with st.expander("📋 Sample Data & Suggested Questions", expanded=True):
            col_left, col_right = st.columns([3, 2])

            with col_left:
                st.markdown("**First 5 rows:**")
                st.dataframe(df.head(5), use_container_width=True)

            with col_right:
                st.markdown("**Columns in your dataset:**")
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    icon  = ("🔢" if "int" in dtype or "float" in dtype
                             else "📅" if "date" in dtype else "🔤")
                    st.caption(f"{icon} `{col}` — {dtype}")

                st.divider()
                st.markdown("**Try asking:**")

                useful_cat = _useful_cats(df)
                main_num, sec_num, all_nums = _smart_num(df)
                main_cat = useful_cat[0] if useful_cat else None
                sec_cat  = useful_cat[1] if len(useful_cat) >= 2 else None

                suggestions = []
                if main_num:
                    suggestions.append(f"What is total {main_num}?")
                    suggestions.append(f"What is average {main_num}?")
                if main_cat and main_num:
                    suggestions.append(f"Top 5 {main_cat} by {main_num}")
                if main_cat and main_num and sec_num:
                    suggestions.append(f"{main_cat} wise {main_num} vs {sec_num}")
                elif main_cat and main_num:
                    suggestions.append(f"{main_cat} wise {main_num}")
                if sec_cat and main_num:
                    suggestions.append(f"{sec_cat} wise {main_num}")
                if main_cat:
                    suggestions.append(f"How many unique values in {main_cat}?")
                if len(all_nums) >= 2:
                    suggestions.append(f"Correlation between {all_nums[0]} and {all_nums[1]}?")

                for s in suggestions:
                    if st.button(s, key=f"suggest_{s}", use_container_width=True):
                        st.session_state["prefill_question"] = s
                        st.rerun()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        col_clear, col_info = st.columns([1, 4])
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_info:
            if st.session_state.messages:
                total_q = len(st.session_state.messages) // 2
                st.caption(f"{total_q} question{'s' if total_q != 1 else ''} asked")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and msg.get("had_chart"):
                    st.caption("📊 Chart was shown for this answer.")
                if msg["role"] == "assistant" and msg.get("query"):
                    with st.expander("🐍 Python query used"):
                        st.code(msg["query"], language="python")

        prefill         = st.session_state.pop("prefill_question", "")
        question        = st.chat_input("Ask anything about your data...")
        active_question = prefill or question

        if active_question:
            st.session_state.messages.append({"role": "user", "content": active_question})
            save_message(dataset_name, "user", active_question)
            with st.chat_message("user"):
                st.write(active_question)

            context = get_relevant_memory(active_question, dataset_name)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, query, intent, cols = ask_gemma(active_question, df, context)
                    tbl = compute_table(active_question, df, intent, cols)
                    fig = generate_chart(active_question, df, intent, cols)

                st.write(answer)
                if tbl is not None:
                    st.dataframe(tbl, use_container_width=True)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                if query:
                    with st.expander("🐍 Python query used"):
                        st.code(query, language="python")

            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "query": query, "had_chart": fig is not None,
            })
            save_message(dataset_name, "assistant", answer, query)
            save_qa(active_question, answer, dataset_name)


elif page == "🛠️ Manipulate":
    if df is None:
        st.info("⬅️ Please upload a dataset from the sidebar.")
    else:
        show_manipulate_page(df, dataset_name)


elif page == "🗄️ SQL":
    if df is None:
        st.info("⬅️ Please upload a dataset from the sidebar.")
    else:
        show_sql_page(df, dataset_name)


elif page == "🤖 ML":
    if df is None:
        st.info("⬅️ Please upload a dataset from the sidebar.")
    else:
        show_ml_page(df, dataset_name)