# utils/data_manipulator.py — V9
# Data Manipulation Engine
# Filter, Sort, Rename, Drop, Clean, Merge, Add Column, Type Change

import pandas as pd
import numpy as np
import streamlit as st
import io


# ══════════════════════════════════════════════════════════════
#  CORE OPERATIONS
# ══════════════════════════════════════════════════════════════

def filter_rows(df: pd.DataFrame, column: str, operator: str, value) -> pd.DataFrame:
    """Filter rows based on condition."""
    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            value = float(value)
            ops = {
                "equals"              : df[column] == value,
                "not equals"          : df[column] != value,
                "greater than"        : df[column] >  value,
                "less than"           : df[column] <  value,
                "greater than or equal": df[column] >= value,
                "less than or equal"  : df[column] <= value,
            }
        else:
            ops = {
                "equals"      : df[column].astype(str).str.lower() == str(value).lower(),
                "not equals"  : df[column].astype(str).str.lower() != str(value).lower(),
                "contains"    : df[column].astype(str).str.lower().str.contains(str(value).lower(), na=False),
                "starts with" : df[column].astype(str).str.lower().str.startswith(str(value).lower()),
                "ends with"   : df[column].astype(str).str.lower().str.endswith(str(value).lower()),
            }
        mask = ops.get(operator)
        if mask is None:
            return df
        return df[mask].reset_index(drop=True)
    except Exception as e:
        st.error(f"Filter error: {e}")
        return df


def sort_data(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    """Sort dataframe by column."""
    try:
        return df.sort_values(column, ascending=ascending).reset_index(drop=True)
    except Exception as e:
        st.error(f"Sort error: {e}")
        return df


def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    """Rename a column."""
    try:
        return df.rename(columns={old_name: new_name})
    except Exception as e:
        st.error(f"Rename error: {e}")
        return df


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop selected columns."""
    try:
        return df.drop(columns=columns, errors="ignore")
    except Exception as e:
        st.error(f"Drop error: {e}")
        return df


def handle_missing(df: pd.DataFrame, column: str, method: str,
                   fill_value=None) -> pd.DataFrame:
    """Handle missing values in a column."""
    try:
        df = df.copy()
        if method == "Drop rows":
            df = df.dropna(subset=[column]).reset_index(drop=True)
        elif method == "Fill with mean":
            df[column] = df[column].fillna(df[column].mean())
        elif method == "Fill with median":
            df[column] = df[column].fillna(df[column].median())
        elif method == "Fill with mode":
            df[column] = df[column].fillna(df[column].mode()[0])
        elif method == "Fill with value":
            df[column] = df[column].fillna(fill_value)
        elif method == "Forward fill":
            df[column] = df[column].ffill()
        elif method == "Backward fill":
            df[column] = df[column].bfill()
        return df
    except Exception as e:
        st.error(f"Missing value error: {e}")
        return df


def add_calculated_column(df: pd.DataFrame, new_col_name: str,
                           expression: str) -> pd.DataFrame:
    """
    Add new column from expression.
    Expression uses column names directly.
    e.g. "revenue / quantity" or "price * 0.9"
    """
    try:
        df = df.copy()
        # Safe eval with df columns as variables
        local_vars = {col: df[col] for col in df.columns}
        df[new_col_name] = eval(expression, {"__builtins__": {}}, local_vars)
        return df
    except Exception as e:
        st.error(f"Calculated column error: {e}. Check expression syntax.")
        return df


def change_dtype(df: pd.DataFrame, column: str, new_type: str) -> pd.DataFrame:
    """Change column data type."""
    try:
        df = df.copy()
        if new_type == "Integer":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        elif new_type == "Float":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif new_type == "String":
            df[column] = df[column].astype(str)
        elif new_type == "DateTime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif new_type == "Boolean":
            df[column] = df[column].astype(bool)
        return df
    except Exception as e:
        st.error(f"Type change error: {e}")
        return df


def remove_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """Remove duplicate rows."""
    try:
        before = len(df)
        df = df.drop_duplicates(subset=subset if subset else None).reset_index(drop=True)
        after = len(df)
        st.success(f"Removed {before - after} duplicate rows.")
        return df
    except Exception as e:
        st.error(f"Duplicate removal error: {e}")
        return df


def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame,
                     on: str, how: str = "inner") -> pd.DataFrame:
    """Merge two dataframes."""
    try:
        return pd.merge(df1, df2, on=on, how=how)
    except Exception as e:
        st.error(f"Merge error: {e}")
        return df1


# ══════════════════════════════════════════════════════════════
#  DOWNLOAD HELPER
# ══════════════════════════════════════════════════════════════

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Manipulated Data")
    output.seek(0)
    return output.read()


# ══════════════════════════════════════════════════════════════
#  MAIN UI — Streamlit Page
# ══════════════════════════════════════════════════════════════

def show_manipulate_page(df: pd.DataFrame, dataset_name: str):
    """
    Main V9 page — call this from app.py.
    Returns modified df so app.py can update session state.
    """
    st.header("🛠️ Data Manipulation")
    st.caption(f"📂 {dataset_name} — {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Session state for undo ────────────────────────────────
    if "manip_history" not in st.session_state:
        st.session_state.manip_history = []

    # Current working df
    if "working_df" not in st.session_state or \
       st.session_state.get("manip_dataset") != dataset_name:
        st.session_state.working_df   = df.copy()
        st.session_state.manip_dataset = dataset_name
        st.session_state.manip_history = []

    wdf = st.session_state.working_df

    # ── Top controls ──────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Working dataset:** {wdf.shape[0]} rows × {wdf.shape[1]} cols")
    with col2:
        if st.button("↩️ Undo Last", use_container_width=True):
            if st.session_state.manip_history:
                st.session_state.working_df = st.session_state.manip_history.pop()
                wdf = st.session_state.working_df
                st.success("Undo successful!")
                st.rerun()
            else:
                st.warning("Nothing to undo.")
    with col3:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.working_df   = df.copy()
            st.session_state.manip_history = []
            st.success("Reset to original!")
            st.rerun()

    st.divider()

    # ── Data preview ──────────────────────────────────────────
    with st.expander("📋 Current Data Preview", expanded=True):
        st.dataframe(wdf.head(20), use_container_width=True)

        # Missing values summary
        miss = wdf.isnull().sum()
        miss = miss[miss > 0]
        if not miss.empty:
            st.warning(f"⚠️ Missing values found in: {', '.join(miss.index.tolist())}")
        else:
            st.success("✅ No missing values")

    st.divider()

    # ── Operations tabs ───────────────────────────────────────
    tabs = st.tabs([
        "🔍 Filter", "↕️ Sort", "✏️ Rename",
        "🗑️ Drop", "🩹 Missing", "➕ Add Column",
        "🔄 Type", "👥 Duplicates", "🔗 Merge"
    ])

    # ── Helper: save to history before operation ──────────────
    def _save_history():
        st.session_state.manip_history.append(
            st.session_state.working_df.copy()
        )

    # ── Tab 1: Filter ─────────────────────────────────────────
    with tabs[0]:
        st.subheader("Filter Rows")
        col = st.selectbox("Select column", wdf.columns, key="filt_col")

        if pd.api.types.is_numeric_dtype(wdf[col]):
            ops = ["equals", "not equals", "greater than", "less than",
                   "greater than or equal", "less than or equal"]
        else:
            ops = ["equals", "not equals", "contains", "starts with", "ends with"]

        op  = st.selectbox("Condition", ops, key="filt_op")
        val = st.text_input("Value", key="filt_val")

        if st.button("Apply Filter", use_container_width=True):
            if val:
                _save_history()
                st.session_state.working_df = filter_rows(wdf, col, op, val)
                st.success(f"Filter applied! {len(st.session_state.working_df)} rows remaining.")
                st.rerun()

    # ── Tab 2: Sort ───────────────────────────────────────────
    with tabs[1]:
        st.subheader("Sort Data")
        col  = st.selectbox("Sort by column", wdf.columns, key="sort_col")
        asc  = st.radio("Order", ["Ascending", "Descending"], key="sort_ord")

        if st.button("Apply Sort", use_container_width=True):
            _save_history()
            st.session_state.working_df = sort_data(
                wdf, col, ascending=(asc == "Ascending"))
            st.success("Sort applied!")
            st.rerun()

    # ── Tab 3: Rename ─────────────────────────────────────────
    with tabs[2]:
        st.subheader("Rename Column")
        old = st.selectbox("Select column to rename", wdf.columns, key="ren_old")
        new = st.text_input("New name", key="ren_new")

        if st.button("Rename", use_container_width=True):
            if new and new != old:
                _save_history()
                st.session_state.working_df = rename_column(wdf, old, new)
                st.success(f"'{old}' renamed to '{new}'!")
                st.rerun()

    # ── Tab 4: Drop ───────────────────────────────────────────
    with tabs[3]:
        st.subheader("Drop Columns")
        to_drop = st.multiselect("Select columns to drop", wdf.columns, key="drop_cols")

        if st.button("Drop Selected", use_container_width=True):
            if to_drop:
                _save_history()
                st.session_state.working_df = drop_columns(wdf, to_drop)
                st.success(f"Dropped: {', '.join(to_drop)}")
                st.rerun()

    # ── Tab 5: Missing Values ─────────────────────────────────
    with tabs[4]:
        st.subheader("Handle Missing Values")
        col = st.selectbox("Select column", wdf.columns, key="miss_col")

        null_count = wdf[col].isnull().sum()
        st.info(f"Missing values in '{col}': {null_count}")

        if pd.api.types.is_numeric_dtype(wdf[col]):
            methods = ["Drop rows", "Fill with mean", "Fill with median",
                       "Fill with mode", "Fill with value",
                       "Forward fill", "Backward fill"]
        else:
            methods = ["Drop rows", "Fill with mode", "Fill with value",
                       "Forward fill", "Backward fill"]

        method = st.selectbox("Method", methods, key="miss_method")
        fill_val = None
        if method == "Fill with value":
            fill_val = st.text_input("Fill value", key="miss_fill")

        if st.button("Apply", use_container_width=True):
            _save_history()
            st.session_state.working_df = handle_missing(
                wdf, col, method, fill_val)
            st.success("Missing values handled!")
            st.rerun()

    # ── Tab 6: Add Column ─────────────────────────────────────
    with tabs[5]:
        st.subheader("Add Calculated Column")
        st.caption("Use column names in expression. Example: `revenue / quantity` or `price * 0.9`")

        st.markdown("**Available columns:**")
        st.code(" | ".join(wdf.columns.tolist()))

        new_col  = st.text_input("New column name", key="calc_name")
        expr     = st.text_input("Expression", key="calc_expr",
                                  placeholder="e.g. revenue / quantity")

        if st.button("Add Column", use_container_width=True):
            if new_col and expr:
                _save_history()
                st.session_state.working_df = add_calculated_column(
                    wdf, new_col, expr)
                st.success(f"Column '{new_col}' added!")
                st.rerun()

    # ── Tab 7: Change Type ────────────────────────────────────
    with tabs[6]:
        st.subheader("Change Data Type")
        col      = st.selectbox("Select column", wdf.columns, key="type_col")
        cur_type = str(wdf[col].dtype)
        st.info(f"Current type: `{cur_type}`")
        new_type = st.selectbox("Convert to",
                                ["Integer", "Float", "String",
                                 "DateTime", "Boolean"], key="type_new")

        if st.button("Convert", use_container_width=True):
            _save_history()
            st.session_state.working_df = change_dtype(wdf, col, new_type)
            st.success(f"'{col}' converted to {new_type}!")
            st.rerun()

    # ── Tab 8: Duplicates ─────────────────────────────────────
    with tabs[7]:
        st.subheader("Remove Duplicates")
        dup_count = wdf.duplicated().sum()
        st.info(f"Duplicate rows found: {dup_count}")

        use_subset = st.checkbox("Check duplicates based on specific columns only")
        subset = None
        if use_subset:
            subset = st.multiselect("Select columns", wdf.columns, key="dup_sub")

        if st.button("Remove Duplicates", use_container_width=True):
            _save_history()
            st.session_state.working_df = remove_duplicates(
                wdf, subset if use_subset and subset else None)
            st.rerun()

    # ── Tab 9: Merge ─────────────────────────────────────────
    with tabs[8]:
        st.subheader("Merge Another File")
        uploaded2 = st.file_uploader(
            "Upload second CSV/Excel", type=["csv","xlsx","xls"],
            key="merge_file")

        if uploaded2:
            from utils.data_loader import load_file
            df2, err = load_file(uploaded2)
            if err:
                st.error(err)
            else:
                st.success(f"File loaded: {df2.shape[0]} rows × {df2.shape[1]} cols")
                st.dataframe(df2.head(3), use_container_width=True)

                common_cols = [c for c in wdf.columns if c in df2.columns]
                if common_cols:
                    on  = st.selectbox("Join on column", common_cols, key="merge_on")
                    how = st.selectbox("Join type",
                                       ["inner","left","right","outer"],
                                       key="merge_how")
                    if st.button("Merge", use_container_width=True):
                        _save_history()
                        st.session_state.working_df = merge_dataframes(
                            wdf, df2, on=on, how=how)
                        st.success("Merge successful!")
                        st.rerun()
                else:
                    st.warning("No common columns found for joining.")

    st.divider()

    # ── Download section ──────────────────────────────────────
    st.subheader("📥 Download Modified Data")
    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            label="⬇️ Download as CSV",
            data=df_to_csv_bytes(st.session_state.working_df),
            file_name=f"modified_{dataset_name.split('.')[0]}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl2:
        st.download_button(
            label="⬇️ Download as Excel",
            data=df_to_excel_bytes(st.session_state.working_df),
            file_name=f"modified_{dataset_name.split('.')[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # ── Update main df ────────────────────────────────────────
    st.divider()
    if st.button("✅ Use Modified Data in Dashboard & Chat",
                 use_container_width=True, type="primary"):
        st.session_state.df = st.session_state.working_df.copy()
        st.success("Dashboard aur Chat ab modified data use karenge!")

    return st.session_state.working_df