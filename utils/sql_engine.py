# utils/sql_engine.py — V10
# SQL Engine using DuckDB
# Auto SQL generation + Manual SQL Editor

import pandas as pd
import streamlit as st
import duckdb
import io


# ══════════════════════════════════════════════════════════════
#  CORE SQL ENGINE
# ══════════════════════════════════════════════════════════════

def run_sql(df: pd.DataFrame, query: str) -> tuple:
    """
    Run SQL query on DataFrame using DuckDB.
    Table name: 'data' (always)
    Returns: (result_df, error_string)
    """
    try:
        conn   = duckdb.connect()
        conn.register("data", df)
        result = conn.execute(query).df()
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)


def generate_sql(question: str, df: pd.DataFrame) -> str:
    """
    Generate SQL query from natural language question.
    Python-first — no LLM needed for common patterns.
    Table name is always 'data'.
    """
    q        = question.lower().strip()
    cols     = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    def best_num(detected=None):
        if detected:
            for c in detected:
                if c in num_cols:
                    return c
        return num_cols[0] if num_cols else None

    def best_cat(detected=None):
        if detected:
            for c in detected:
                if c in cat_cols:
                    return c
        return cat_cols[0] if cat_cols else None

    # Detect mentioned columns
    mentioned = [c for c in cols if c.lower() in q]
    num_mentioned = [c for c in mentioned if c in num_cols]
    cat_mentioned = [c for c in mentioned if c in cat_cols]

    nc = best_num(num_mentioned) or best_num()
    cc = best_cat(cat_mentioned) or best_cat()

    import re
    n_match = re.search(r"top\s+(\d+)|bottom\s+(\d+)", q)
    n       = int(n_match.group(1) or n_match.group(2)) if n_match else 10
    order   = "ASC" if any(w in q for w in
                           ["bottom","lowest","worst","least"]) else "DESC"

    # ── Pattern matching ──────────────────────────────────────

    # Total / Sum
    if any(k in q for k in ["total", "sum"]) and nc:
        if cc and any(k in q for k in ["by", "wise", "per", "each"]):
            return (f'SELECT "{cc}", SUM("{nc}") AS "Total {nc}"\n'
                    f'FROM data\n'
                    f'GROUP BY "{cc}"\n'
                    f'ORDER BY "Total {nc}" DESC')
        return f'SELECT SUM("{nc}") AS "Total {nc}" FROM data'

    # Average / Mean
    if any(k in q for k in ["average", "mean", "avg"]) and nc:
        if cc and any(k in q for k in ["by", "wise", "per", "each"]):
            return (f'SELECT "{cc}", ROUND(AVG("{nc}"), 2) AS "Avg {nc}"\n'
                    f'FROM data\n'
                    f'GROUP BY "{cc}"\n'
                    f'ORDER BY "Avg {nc}" DESC')
        return f'SELECT ROUND(AVG("{nc}"), 2) AS "Avg {nc}" FROM data'

    # Count
    if any(k in q for k in ["count", "how many", "number of"]):
        if cc:
            return (f'SELECT "{cc}", COUNT(*) AS "Count"\n'
                    f'FROM data\n'
                    f'GROUP BY "{cc}"\n'
                    f'ORDER BY "Count" DESC')
        return "SELECT COUNT(*) AS Total_Rows FROM data"

    # Top N
    if re.search(r"top\s+\d+|bottom\s+\d+", q):
        if cc and nc:
            return (f'SELECT "{cc}", SUM("{nc}") AS "Total {nc}"\n'
                    f'FROM data\n'
                    f'GROUP BY "{cc}"\n'
                    f'ORDER BY "Total {nc}" {order}\n'
                    f'LIMIT {n}')
        elif nc:
            return (f'SELECT *\n'
                    f'FROM data\n'
                    f'ORDER BY "{nc}" {order}\n'
                    f'LIMIT {n}')

    # Group by / Wise
    if any(k in q for k in ["by", "wise", "group", "per", "each"]) and cc and nc:
        nums = num_mentioned if num_mentioned else [nc]
        aggs = ", ".join([f'SUM("{c}") AS "Total {c}"' for c in nums[:3]])
        return (f'SELECT "{cc}", {aggs}\n'
                f'FROM data\n'
                f'GROUP BY "{cc}"\n'
                f'ORDER BY "Total {nums[0]}" DESC')

    # Max / Min
    if any(k in q for k in ["maximum", "max", "highest", "largest"]) and nc:
        return f'SELECT MAX("{nc}") AS "Max {nc}" FROM data'
    if any(k in q for k in ["minimum", "min", "lowest", "smallest"]) and nc:
        return f'SELECT MIN("{nc}") AS "Min {nc}" FROM data'

    # Missing / Null
    if any(k in q for k in ["missing", "null", "nan", "empty"]):
        null_checks = " +\n       ".join(
            [f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in cols[:8]]
        )
        col_checks = ",\n       ".join(
            [f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}_nulls"'
             for c in cols[:8]]
        )
        return f'SELECT\n       {col_checks}\nFROM data'

    # Unique / Distinct
    if any(k in q for k in ["unique", "distinct"]) and cc:
        return (f'SELECT DISTINCT "{cc}"\n'
                f'FROM data\n'
                f'ORDER BY "{cc}"')

    # Correlation
    if any(k in q for k in ["correlation", "corr"]) and len(num_cols) >= 2:
        c1 = num_cols[0]
        c2 = num_mentioned[1] if len(num_mentioned) >= 2 else num_cols[1]
        return (f'SELECT\n'
                f'  CORR("{c1}", "{c2}") AS "Correlation"\n'
                f'FROM data')

    # Summary / Describe
    if any(k in q for k in ["summary", "describe", "overview"]):
        stats = ",\n  ".join([
            f'MIN("{c}") AS "{c}_min", MAX("{c}") AS "{c}_max", '
            f'ROUND(AVG("{c}"), 2) AS "{c}_avg"'
            for c in num_cols[:4]
        ])
        return f'SELECT\n  {stats}\nFROM data'

    # Default — show all
    return "SELECT * FROM data LIMIT 20"


# ══════════════════════════════════════════════════════════════
#  SQL TAB UI
# ══════════════════════════════════════════════════════════════

def show_sql_page(df: pd.DataFrame, dataset_name: str):
    """Main V10 SQL page — call from app.py."""
    from utils.chart_agent import generate_chart

    st.header("🗄️ SQL Mode")
    st.caption(f"📂 Table name: `data` — {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Column reference ──────────────────────────────────────
    with st.expander("📋 Available Columns", expanded=False):
        col_data = []
        for c in df.columns:
            dtype = str(df[c].dtype)
            kind  = "🔢 Numeric" if "int" in dtype or "float" in dtype \
                    else "📅 Date" if "date" in dtype else "🔤 Text"
            col_data.append({"Column": c, "Type": dtype, "Kind": kind})
        st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Two modes ─────────────────────────────────────────────
    mode = st.radio("Mode", ["✨ Auto SQL (Question → SQL)",
                              "⌨️ Manual SQL Editor"],
                    horizontal=True)

    st.divider()

    # ══════════════════════════════════════════════════════════
    # MODE 1: Auto SQL
    # ══════════════════════════════════════════════════════════
    if mode == "✨ Auto SQL (Question → SQL)":
        st.subheader("Ask a Question — Get SQL + Result")

        # Suggested questions
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        main_num = num_cols[0] if num_cols else ""
        main_cat = cat_cols[0] if cat_cols else ""

        suggestions = []
        if main_num:
            suggestions.append(f"Total {main_num}")
            suggestions.append(f"Average {main_num}")
        if main_cat and main_num:
            suggestions.append(f"Top 10 {main_cat} by {main_num}")
            suggestions.append(f"{main_cat} wise {main_num}")
        if main_cat:
            suggestions.append(f"Count by {main_cat}")
        if len(num_cols) >= 2:
            suggestions.append(f"Correlation between {num_cols[0]} and {num_cols[1]}")

        st.markdown("**Quick questions:**")
        sug_cols = st.columns(len(suggestions[:4]))
        for i, sug in enumerate(suggestions[:4]):
            if sug_cols[i].button(sug, key=f"sql_sug_{i}", use_container_width=True):
                st.session_state["sql_question"] = sug
                st.rerun()

        question = st.text_input(
            "Your question",
            value=st.session_state.get("sql_question", ""),
            placeholder="e.g. Top 5 products by revenue",
            key="sql_q_input"
        )

        if st.button("Generate SQL & Run", type="primary", use_container_width=True):
            if question:
                st.session_state["sql_question"] = question
                sql = generate_sql(question, df)
                st.session_state["last_sql"] = sql

                # Show generated SQL
                st.markdown("**Generated SQL:**")
                st.code(sql, language="sql")

                # Run it
                result, err = run_sql(df, sql)
                if err:
                    st.error(f"SQL Error: {err}")
                else:
                    st.success(f"✅ {len(result)} rows returned")
                    st.dataframe(result, use_container_width=True)

                    # Auto chart
                    from utils.ollama_chat import detect_intent, detect_cols
                    intent = detect_intent(question)
                    cols   = detect_cols(question, df)
                    fig    = generate_chart(question, df, intent, cols)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # Download result
                    csv = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Result as CSV",
                        data=csv,
                        file_name="sql_result.csv",
                        mime="text/csv"
                    )

    # ══════════════════════════════════════════════════════════
    # MODE 2: Manual SQL Editor
    # ══════════════════════════════════════════════════════════
    else:
        st.subheader("Write Your Own SQL Query")
        st.info("💡 Table name is always `data`")

        # Sample queries
        with st.expander("📝 Sample Queries"):
            num_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            nc = num_cols[0] if num_cols else "column"
            cc = cat_cols[0] if cat_cols else "column"

            samples = {
                "Select All"          : "SELECT * FROM data LIMIT 20",
                "Count Rows"          : "SELECT COUNT(*) AS total_rows FROM data",
                f"Sum {nc}"           : f'SELECT SUM("{nc}") AS total FROM data',
                f"Group by {cc}"      : f'SELECT "{cc}", SUM("{nc}") AS total\nFROM data\nGROUP BY "{cc}"\nORDER BY total DESC',
                f"Top 10 by {nc}"     : f'SELECT *\nFROM data\nORDER BY "{nc}" DESC\nLIMIT 10',
                "Null Check"          : f'SELECT\n  ' + ',\n  '.join([
                                         f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}_nulls"'
                                         for c in df.columns[:5]]) + '\nFROM data',
            }

            for name, query in samples.items():
                if st.button(name, key=f"sample_{name}", use_container_width=False):
                    st.session_state["manual_sql"] = query
                    st.rerun()

        # SQL Editor
        default_sql = st.session_state.get("manual_sql",
                                            "SELECT * FROM data LIMIT 20")
        sql_input = st.text_area(
            "SQL Query",
            value=default_sql,
            height=160,
            placeholder="SELECT * FROM data LIMIT 20",
            key="sql_editor"
        )

        run_col, clear_col = st.columns([3, 1])
        with run_col:
            run_clicked = st.button("▶️ Run Query",
                                     type="primary", use_container_width=True)
        with clear_col:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["manual_sql"] = "SELECT * FROM data LIMIT 20"
                st.rerun()

        if run_clicked and sql_input.strip():
            st.session_state["manual_sql"] = sql_input
            result, err = run_sql(df, sql_input)

            if err:
                st.error(f"❌ SQL Error: {err}")
                st.caption("Check column names — use double quotes around column names with spaces.")
            else:
                st.success(f"✅ {len(result)} rows returned")
                st.dataframe(result, use_container_width=True)

                # Try chart
                if len(result.columns) >= 2:
                    num_r = result.select_dtypes(include="number").columns.tolist()
                    cat_r = result.select_dtypes(include="object").columns.tolist()
                    if cat_r and num_r:
                        import plotly.express as px
                        try:
                            fig = px.bar(
                                result.head(15),
                                x=cat_r[0], y=num_r[0],
                                title=f"{num_r[0]} by {cat_r[0]}",
                                color=num_r[0],
                                color_continuous_scale="Teal"
                            )
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                coloraxis_showscale=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass

                # Download
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Result as CSV",
                    data=csv,
                    file_name="sql_result.csv",
                    mime="text/csv"
                )