import pandas as pd


def load_file(uploaded_file):
    filename = uploaded_file.name

    if filename.endswith(".csv"):
        # Try encodings in order — most CSVs are UTF-8, but files exported
        # from Excel (like the common "Superstore" dataset) are often
        # Windows-1252 / Latin-1, which breaks pd.read_csv with a
        # UnicodeDecodeError if we don't fall back.
        encodings_to_try = ["utf-8", "latin1", "cp1252"]
        df = None
        last_error = None

        for enc in encodings_to_try:
            try:
                uploaded_file.seek(0)  # reset pointer before each attempt
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
            except Exception as e:
                # Non-encoding errors (bad delimiter, corrupt file, etc.)
                return None, f"❌ Could not read CSV file: {e}"

        if df is None:
            return None, f"❌ Could not read file — unsupported encoding: {last_error}"

    elif filename.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            return None, f"❌ Could not read Excel file: {e}"

    else:
        return None, "❌ Sirf CSV ya Excel file upload karein."

    return df, None


def get_data_summary(df):
    basic = f"""
DATASET OVERVIEW:
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}
- Columns: {list(df.columns)}
"""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_summary = "\nNUMERIC COLUMNS FULL STATS:\n"
    if numeric_cols:
        numeric_summary += df[numeric_cols].describe().to_string()
        numeric_summary += "\n\nCOLUMN TOTALS:\n"
        for col in numeric_cols:
            numeric_summary += f"- {col}: Total = {df[col].sum():.2f}, Avg = {df[col].mean():.2f}\n"

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_summary = "\nCATEGORICAL COLUMNS (value counts):\n"
    for col in cat_cols:
        cat_summary += f"\n{col} unique values ({df[col].nunique()} total):\n"
        cat_summary += df[col].value_counts().to_string() + "\n"

    group_summary = "\nGROUP-WISE AGGREGATIONS (actual computed):\n"
    for cat in cat_cols:
        for num in numeric_cols:
            grouped = df.groupby(cat)[num].sum().sort_values(ascending=False)
            group_summary += f"\n{num} by {cat}:\n{grouped.to_string()}\n"

    return basic + numeric_summary + cat_summary + group_summary