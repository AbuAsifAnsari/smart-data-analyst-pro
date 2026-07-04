# utils/ml_engine.py — V12
# Machine Learning Engine
# Regression, Classification, Clustering, Forecasting
#
# V12 CHANGELOG (from V11):
#   - Confusion matrix now shows actual class names instead of 0/1/2
#   - Warning added when a continuous numeric column is picked as a
#     classification target (likely user error -> should use Regression)
#   - Trained model can now be downloaded (pickle) after Regression/
#     Classification runs
#   - Forecasting now reports in-sample MAPE / accuracy so results
#     aren't shown blind
#   - DBSCAN cluster slider is hidden (not just disabled) since it
#     doesn't apply, to avoid misleading UI

import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score,
                              accuracy_score, classification_report,
                              confusion_matrix)
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        coloraxis_showscale=False,
    )
    return fig


def _encode_df(df: pd.DataFrame):
    """Encode categorical columns for ML."""
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


def _mape(y_true, y_pred):
    """Mean Absolute Percentage Error, safe against zero values."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return None
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ══════════════════════════════════════════════════════════════
#  REGRESSION
# ══════════════════════════════════════════════════════════════

def run_regression(df: pd.DataFrame, target: str,
                   features: list, model_name: str):
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    df_clean = df[features + [target]].dropna()
    df_enc, _ = _encode_df(df_clean)

    X = df_enc[features]
    y = df_enc[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression"       : LinearRegression(),
        "Ridge Regression"        : Ridge(alpha=1.0),
        "Random Forest"           : RandomForestRegressor(
                                     n_estimators=100, random_state=42),
        "Gradient Boosting"       : GradientBoostingRegressor(
                                     n_estimators=100, random_state=42),
    }
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature"   : features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
    elif hasattr(model, "coef_"):
        imp = pd.DataFrame({
            "Feature"   : features,
            "Importance": np.abs(model.coef_)
        }).sort_values("Importance", ascending=False)
    else:
        imp = None

    return {
        "model"    : model,
        "y_test"   : y_test,
        "y_pred"   : y_pred,
        "rmse"     : rmse,
        "r2"       : r2,
        "importance": imp,
    }


# ══════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def run_classification(df: pd.DataFrame, target: str,
                       features: list, model_name: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    df_clean = df[features + [target]].dropna()
    df_enc, encoders = _encode_df(df_clean)

    X = df_enc[features]
    y = df_enc[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    models = {
        "Logistic Regression" : LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest"       : RandomForestClassifier(
                                 n_estimators=100, random_state=42),
        "Gradient Boosting"   : GradientBoostingClassifier(
                                 n_estimators=100, random_state=42),
    }
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature"   : features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
    else:
        imp = None

    return {
        "model"    : model,
        "accuracy" : accuracy,
        "y_test"   : y_test,
        "y_pred"   : y_pred,
        "cm"       : cm,
        "importance": imp,
        "encoder"  : encoders.get(target),
    }


# ══════════════════════════════════════════════════════════════
#  CLUSTERING
# ══════════════════════════════════════════════════════════════

def run_clustering(df: pd.DataFrame, features: list,
                   n_clusters: int, model_name: str):
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler

    df_clean = df[features].dropna()
    df_enc, _ = _encode_df(df_clean)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_enc)

    if model_name == "K-Means":
        model  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    else:  # DBSCAN
        model  = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    df_result            = df_clean.copy()
    df_result["Cluster"] = [f"Cluster {l}" if l >= 0
                             else "Noise" for l in labels]

    # Cluster summary
    summary = df_result.groupby("Cluster")[features].mean().round(2)

    return {
        "df_result" : df_result,
        "labels"    : labels,
        "n_clusters": n_clusters,
        "summary"   : summary,
        "features"  : features,
    }


# ══════════════════════════════════════════════════════════════
#  FORECASTING
# ══════════════════════════════════════════════════════════════

def run_forecasting(df: pd.DataFrame, date_col: str,
                    value_col: str, periods: int):
    try:
        from prophet import Prophet
    except ImportError:
        return None, "Prophet not installed. Run: pip install prophet"

    try:
        df_prophet = df[[date_col, value_col]].copy()
        df_prophet.columns = ["ds", "y"]
        df_prophet["ds"]   = pd.to_datetime(df_prophet["ds"], errors="coerce")
        df_prophet          = df_prophet.dropna()
        df_prophet          = df_prophet.groupby("ds")["y"].sum().reset_index()

        model = Prophet(yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False)
        model.fit(df_prophet)

        future   = model.make_future_dataframe(periods=periods, freq="ME")
        forecast = model.predict(future)

        # In-sample accuracy check (V12): compare model's fitted values
        # against the historical actuals it was trained on.
        in_sample = forecast[forecast["ds"].isin(df_prophet["ds"])] \
                    .merge(df_prophet, on="ds", how="inner")
        mape = _mape(in_sample["y"], in_sample["yhat"])

        return {"model": model, "forecast": forecast,
                "df_prophet": df_prophet, "mape": mape}, None
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════

def show_ml_page(df: pd.DataFrame, dataset_name: str):
    """Main V12 ML page — call from app.py."""
    st.header("🤖 Machine Learning")
    st.caption(f"📂 {dataset_name} — {df.shape[0]} rows × {df.shape[1]} cols")

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    all_cols  = df.columns.tolist()

    # ── ML Type selector ──────────────────────────────────────
    ml_type = st.radio(
        "Select ML Task",
        ["📈 Regression", "🎯 Classification",
         "🔵 Clustering", "📅 Forecasting"],
        horizontal=True
    )
    st.divider()

    # ══════════════════════════════════════════════════════════
    # REGRESSION
    # ══════════════════════════════════════════════════════════
    if ml_type == "📈 Regression":
        st.subheader("📈 Regression — Predict a Number")
        st.caption("e.g. Predict revenue, price, quantity")

        col1, col2 = st.columns(2)
        with col1:
            target   = st.selectbox("Target column (what to predict)",
                                     num_cols, key="reg_target")
        with col2:
            model_name = st.selectbox("Model",
                ["Linear Regression", "Ridge Regression",
                 "Random Forest", "Gradient Boosting"],
                key="reg_model")

        feat_opts = [c for c in all_cols if c != target]
        features  = st.multiselect("Feature columns (inputs)",
                                    feat_opts,
                                    default=feat_opts[:4],
                                    key="reg_feats")

        if st.button("🚀 Train Model", type="primary",
                     use_container_width=True, key="reg_run"):
            if not features:
                st.warning("Please select at least one feature column.")
            else:
                with st.spinner("Training model..."):
                    res = run_regression(df, target, features, model_name)

                # Metrics
                st.subheader("📊 Model Performance")
                m1, m2 = st.columns(2)
                m1.metric("R² Score",
                           f"{res['r2']:.4f}",
                           help="1.0 = perfect, 0 = no predictive power")
                m2.metric("RMSE",
                           f"{res['rmse']:,.2f}",
                           help="Lower is better")

                # Actual vs Predicted
                pred_df = pd.DataFrame({
                    "Actual"   : res["y_test"].values,
                    "Predicted": res["y_pred"]
                }).head(50)
                fig = px.scatter(
                    pred_df, x="Actual", y="Predicted",
                    title="Actual vs Predicted",
                    color_discrete_sequence=["#1D9E75"]
                )
                fig.add_shape(type="line",
                              x0=pred_df["Actual"].min(),
                              y0=pred_df["Actual"].min(),
                              x1=pred_df["Actual"].max(),
                              y1=pred_df["Actual"].max(),
                              line=dict(color="#E24B4A", dash="dash"))
                st.plotly_chart(_style(fig), use_container_width=True)

                # Feature importance
                if res["importance"] is not None:
                    st.subheader("🎯 Feature Importance")
                    fig2 = px.bar(
                        res["importance"], x="Importance", y="Feature",
                        orientation="h",
                        title="Feature Importance",
                        color="Importance",
                        color_continuous_scale="Teal"
                    )
                    st.plotly_chart(_style(fig2), use_container_width=True)

                # Insight
                st.info(
                    f"**Model:** {model_name} | "
                    f"**R²:** {res['r2']:.3f} | "
                    f"**RMSE:** {res['rmse']:,.2f}\n\n"
                    f"{'✅ Good fit!' if res['r2'] > 0.7 else '⚠️ Model needs improvement — try more features or different model.'}"
                )

                # V12: model export
                model_bytes = pickle.dumps(res["model"])
                st.download_button(
                    "⬇️ Download Trained Model (.pkl)",
                    data=model_bytes,
                    file_name=f"regression_{model_name.replace(' ', '_').lower()}.pkl",
                    mime="application/octet-stream"
                )

    # ══════════════════════════════════════════════════════════
    # CLASSIFICATION
    # ══════════════════════════════════════════════════════════
    elif ml_type == "🎯 Classification":
        st.subheader("🎯 Classification — Predict a Category")
        st.caption("e.g. Predict product category, customer segment")

        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Target column (what to predict)",
                                   cat_cols + num_cols, key="clf_target")
        with col2:
            model_name = st.selectbox("Model",
                ["Logistic Regression", "Decision Tree",
                 "Random Forest", "Gradient Boosting"],
                key="clf_model")

        # V12: warn if a continuous numeric column is picked as target
        if target in num_cols and df[target].nunique() > 15:
            st.warning(
                f"⚠️ '{target}' looks like a continuous variable "
                f"({df[target].nunique()} unique values) rather than a "
                f"category. Classification may give meaningless results — "
                f"consider using **Regression** instead, or pick a column "
                f"with fewer distinct values."
            )

        feat_opts = [c for c in num_cols if c != target]
        features  = st.multiselect("Feature columns (numeric only)",
                                    feat_opts,
                                    default=feat_opts[:4],
                                    key="clf_feats")

        if st.button("🚀 Train Model", type="primary",
                     use_container_width=True, key="clf_run"):
            if not features:
                st.warning("Please select at least one numeric feature.")
            else:
                with st.spinner("Training model..."):
                    res = run_classification(df, target, features, model_name)

                # Accuracy
                st.subheader("📊 Model Performance")
                st.metric("Accuracy",
                           f"{res['accuracy']*100:.2f}%",
                           help="Percentage of correct predictions")

                # Confusion matrix (V12: real class names instead of 0/1/2)
                if res["encoder"] is not None:
                    class_names = list(res["encoder"].classes_)
                else:
                    class_names = [str(c) for c in sorted(pd.unique(
                        pd.concat([res["y_test"], pd.Series(res["y_pred"])])
                    ))]

                cm_df = pd.DataFrame(res["cm"],
                                      index=class_names,
                                      columns=class_names)
                fig   = px.imshow(
                    cm_df,
                    title="Confusion Matrix",
                    color_continuous_scale="Blues",
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual")
                )
                st.plotly_chart(_style(fig), use_container_width=True)

                # Feature importance
                if res["importance"] is not None:
                    st.subheader("🎯 Feature Importance")
                    fig2 = px.bar(
                        res["importance"], x="Importance", y="Feature",
                        orientation="h",
                        title="Feature Importance",
                        color="Importance",
                        color_continuous_scale="Teal"
                    )
                    st.plotly_chart(_style(fig2), use_container_width=True)

                # Insight
                acc = res["accuracy"]
                st.info(
                    f"**Model:** {model_name} | "
                    f"**Accuracy:** {acc*100:.2f}%\n\n"
                    f"{'✅ Excellent accuracy!' if acc > 0.85 else '✅ Good accuracy!' if acc > 0.7 else '⚠️ Try Random Forest or more features.'}"
                )

                # V12: model export
                model_bytes = pickle.dumps(res["model"])
                st.download_button(
                    "⬇️ Download Trained Model (.pkl)",
                    data=model_bytes,
                    file_name=f"classification_{model_name.replace(' ', '_').lower()}.pkl",
                    mime="application/octet-stream"
                )

    # ══════════════════════════════════════════════════════════
    # CLUSTERING
    # ══════════════════════════════════════════════════════════
    elif ml_type == "🔵 Clustering":
        st.subheader("🔵 Clustering — Find Groups in Data")
        st.caption("e.g. Customer segmentation, product grouping")

        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox("Algorithm",
                                       ["K-Means", "DBSCAN"],
                                       key="clust_model")
        with col2:
            # V12: slider is hidden (not just disabled) for DBSCAN,
            # since cluster count isn't a DBSCAN parameter at all.
            if model_name == "K-Means":
                n_clusters = st.slider("Number of clusters", 2, 8, 3,
                                        key="clust_n")
            else:
                n_clusters = 3  # unused, DBSCAN infers cluster count itself
                st.caption("DBSCAN finds the number of clusters automatically.")

        features = st.multiselect("Feature columns",
                                   num_cols,
                                   default=num_cols[:3],
                                   key="clust_feats")

        if st.button("🚀 Run Clustering", type="primary",
                     use_container_width=True, key="clust_run"):
            if len(features) < 2:
                st.warning("Please select at least 2 feature columns.")
            else:
                with st.spinner("Finding clusters..."):
                    res = run_clustering(df, features, n_clusters, model_name)

                st.success(f"Found {res['n_clusters']} clusters!")

                # Scatter plot (first 2 features)
                fig = px.scatter(
                    res["df_result"],
                    x=features[0],
                    y=features[1] if len(features) > 1 else features[0],
                    color="Cluster",
                    title=f"Clusters — {features[0]} vs {features[1] if len(features) > 1 else features[0]}",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(_style(fig), use_container_width=True)

                # Cluster summary
                st.subheader("📊 Cluster Summary")
                st.dataframe(res["summary"], use_container_width=True)

                # Download
                csv = res["df_result"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Clustered Data",
                    data=csv,
                    file_name="clustered_data.csv",
                    mime="text/csv"
                )

                st.info(
                    f"**Algorithm:** {model_name} | "
                    f"**Clusters found:** {res['n_clusters']}\n\n"
                    "Each cluster represents a group of similar records. "
                    "Use the summary table to understand what makes each group unique."
                )

    # ══════════════════════════════════════════════════════════
    # FORECASTING
    # ══════════════════════════════════════════════════════════
    elif ml_type == "📅 Forecasting":
        st.subheader("📅 Forecasting — Predict Future Values")
        st.caption("e.g. Next month revenue, future sales trend")

        from utils.ollama_chat import detect_date_column
        date_col_auto = detect_date_column(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.selectbox(
                "Date column",
                all_cols,
                index=all_cols.index(date_col_auto)
                      if date_col_auto in all_cols else 0,
                key="fc_date"
            )
        with col2:
            value_col = st.selectbox("Value to forecast",
                                      num_cols, key="fc_val")
        with col3:
            periods = st.slider("Forecast months", 1, 24, 6,
                                 key="fc_periods")

        if st.button("🚀 Generate Forecast", type="primary",
                     use_container_width=True, key="fc_run"):
            with st.spinner("Forecasting... (this may take 30-60 seconds)"):
                res, err = run_forecasting(df, date_col, value_col, periods)

            if err:
                st.error(f"Forecasting error: {err}")
            else:
                forecast = res["forecast"]
                df_hist  = res["df_prophet"]
                mape     = res.get("mape")

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_hist["ds"], y=df_hist["y"],
                    name="Historical",
                    line=dict(color="#1D9E75", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=forecast["ds"].tail(periods),
                    y=forecast["yhat"].tail(periods),
                    name="Forecast",
                    line=dict(color="#378ADD", width=2, dash="dash")
                ))
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast["ds"].tail(periods),
                                  forecast["ds"].tail(periods).iloc[::-1]]),
                    y=pd.concat([forecast["yhat_upper"].tail(periods),
                                  forecast["yhat_lower"].tail(periods).iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(55,138,221,0.1)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Interval"
                ))
                fig.update_layout(
                    title=f"{value_col} Forecast — Next {periods} months",
                    xaxis_title="Date",
                    yaxis_title=value_col,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

                # V12: accuracy metric shown before the table
                if mape is not None:
                    acc_label = ("✅ High confidence" if mape < 10 else
                                 "✅ Reasonable confidence" if mape < 20 else
                                 "⚠️ Low confidence — historical fit is noisy")
                    st.metric("Historical Fit (MAPE)", f"{mape:.1f}%",
                               help="Average % error of the model on data it "
                                    "already saw. Lower is better.")
                    st.caption(acc_label)

                # Forecast table
                st.subheader("📋 Forecast Values")
                fc_table = forecast[["ds","yhat","yhat_lower","yhat_upper"]]\
                           .tail(periods).copy()
                fc_table.columns = ["Date","Forecast","Lower Bound","Upper Bound"]
                fc_table["Date"] = fc_table["Date"].dt.strftime("%Y-%m")
                fc_table = fc_table.round(2)
                st.dataframe(fc_table, use_container_width=True, hide_index=True)

                # Download
                csv = fc_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Forecast",
                    data=csv,
                    file_name="forecast.csv",
                    mime="text/csv"
                )

                next_val = forecast["yhat"].iloc[-periods]
                st.info(
                    f"**Next month forecast:** {next_val:,.2f}\n\n"
                    f"Forecast based on {len(df_hist)} historical data points "
                    f"using Prophet model."
                    + (f" Model's historical fit MAPE: {mape:.1f}%." if mape is not None else "")
                )