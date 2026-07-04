# utils/ml_engine.py — V14
# Machine Learning Engine
# Regression, Classification, Clustering, Forecasting
#
# V14 CHANGELOG (from V13):
#   - DBSCAN now has tunable eps / min_samples sliders instead of
#     hardcoded values (fixes the "everything becomes Noise" issue
#     seen during testing with the Superstore dataset)
#   - Identifier-like columns (Row ID, Order ID, Postal Code, etc. —
#     anything with ~as many unique values as rows) are now auto-
#     detected and excluded from default feature selections across
#     Regression, Classification, and Clustering, with a note shown
#     to the user. They can still be manually re-added if wanted.
#
# (V13 changes carried over: hyperparameter controls, cross-validation.
#  V12 changes carried over: confusion matrix class names, continuous-
#  target warning, model export, forecast MAPE, DBSCAN slider hidden
#  when not needed.)

import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold, KFold)
from sklearn.pipeline import Pipeline
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


def _detect_identifier_columns(df: pd.DataFrame, threshold: float = 0.95):
    """V14: Flag columns that look like unique identifiers (Row ID,
    Order ID, Postal Code, etc.) — i.e. columns whose number of unique
    values is close to the row count. These are almost never useful
    as ML features/targets and just add noise or leak row identity."""
    n = len(df)
    if n == 0:
        return []
    id_cols = []
    for col in df.columns:
        try:
            nunique = df[col].nunique()
        except TypeError:
            continue
        if nunique >= threshold * n and nunique > 1:
            id_cols.append(col)
    return id_cols


# ══════════════════════════════════════════════════════════════
#  REGRESSION
# ══════════════════════════════════════════════════════════════

def run_regression(df: pd.DataFrame, target: str,
                   features: list, model_name: str,
                   hyperparams: dict = None):
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    hyperparams = hyperparams or {}

    df_clean = df[features + [target]].dropna()
    df_enc, _ = _encode_df(df_clean)

    X = df_enc[features]
    y = df_enc[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(
            alpha=hyperparams.get("alpha", 1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 3),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            random_state=42),
    }
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    cv_result = None
    try:
        cv_folds = min(5, len(X) // 2)
        if cv_folds >= 2:
            cv_scores = cross_val_score(
                models[model_name], X, y, cv=cv_folds, scoring="r2")
            cv_result = {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "folds": cv_folds,
            }
    except Exception:
        cv_result = None

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
        "cv"       : cv_result,
    }


# ══════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def run_classification(df: pd.DataFrame, target: str,
                       features: list, model_name: str,
                       hyperparams: dict = None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    hyperparams = hyperparams or {}

    df_clean = df[features + [target]].dropna()
    df_enc, encoders = _encode_df(df_clean)

    X = df_enc[features]
    y = df_enc[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train)
    X_test_sc    = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            C=hyperparams.get("C", 1.0),
            max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=hyperparams.get("max_depth", 5),
            random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 3),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            random_state=42),
    }
    model = models[model_name]
    model.fit(X_train_sc, y_train)
    y_pred   = model.predict(X_test_sc)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    cv_result = None
    try:
        class_counts = y.value_counts()
        cv_folds = min(5, class_counts.min(), len(X) // 2)
        if cv_folds >= 2:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", models[model_name]),
            ])
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                   random_state=42)
            cv_scores = cross_val_score(pipe, X, y, cv=skf,
                                         scoring="accuracy")
            cv_result = {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "folds": cv_folds,
            }
    except Exception:
        cv_result = None

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "Feature"   : features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
    else:
        imp = None

    return {
        "model"    : model,
        "scaler"   : scaler,
        "accuracy" : accuracy,
        "y_test"   : y_test,
        "y_pred"   : y_pred,
        "cm"       : cm,
        "importance": imp,
        "encoder"  : encoders.get(target),
        "cv"       : cv_result,
    }


# ══════════════════════════════════════════════════════════════
#  CLUSTERING
# ══════════════════════════════════════════════════════════════

def run_clustering(df: pd.DataFrame, features: list,
                   n_clusters: int, model_name: str,
                   eps: float = 0.5, min_samples: int = 5):
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler

    df_clean = df[features].dropna()
    df_enc, _ = _encode_df(df_clean)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_enc)

    if model_name == "K-Means":
        model  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
    else:  # DBSCAN — V14: eps/min_samples now configurable from the UI
        model  = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    df_result            = df_clean.copy()
    df_result["Cluster"] = [f"Cluster {l}" if l >= 0
                             else "Noise" for l in labels]

    noise_count = int((labels == -1).sum()) if model_name == "DBSCAN" else 0
    noise_pct   = (noise_count / len(labels) * 100) if len(labels) else 0

    summary = df_result.groupby("Cluster")[features].mean().round(2)

    return {
        "df_result" : df_result,
        "labels"    : labels,
        "n_clusters": n_clusters,
        "summary"   : summary,
        "features"  : features,
        "noise_pct" : noise_pct,
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
    """Main V14 ML page — call from app.py."""
    st.header("🤖 Machine Learning")
    st.caption(f"📂 {dataset_name} — {df.shape[0]} rows × {df.shape[1]} cols")

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    all_cols  = df.columns.tolist()

    # V14: detect identifier-like columns once, reuse everywhere below
    id_cols = _detect_identifier_columns(df)

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

        hyperparams = {}
        with st.expander("⚙️ Advanced settings (hyperparameters)"):
            if model_name == "Ridge Regression":
                hyperparams["alpha"] = st.slider(
                    "Alpha (regularization strength)", 0.01, 10.0, 1.0,
                    key="reg_alpha")
            elif model_name in ("Random Forest", "Gradient Boosting"):
                hyperparams["n_estimators"] = st.slider(
                    "Number of trees (n_estimators)", 50, 300, 100, step=10,
                    key="reg_n_est")
                hyperparams["max_depth"] = st.slider(
                    "Max tree depth", 2, 20,
                    5 if model_name == "Gradient Boosting" else 10,
                    key="reg_depth")
                if model_name == "Gradient Boosting":
                    hyperparams["learning_rate"] = st.slider(
                        "Learning rate", 0.01, 0.3, 0.1, step=0.01,
                        key="reg_lr")
            else:
                st.caption("Linear Regression has no tunable hyperparameters.")

        # V14: exclude identifier-like columns from default features
        feat_opts     = [c for c in all_cols if c != target]
        default_feats = [c for c in feat_opts if c not in id_cols][:4]
        features  = st.multiselect("Feature columns (inputs)",
                                    feat_opts,
                                    default=default_feats,
                                    key="reg_feats")
        excluded_ids = [c for c in id_cols if c in feat_opts and c not in features]
        if excluded_ids:
            st.caption(
                f"ℹ️ Not included by default (look like unique IDs): "
                f"{', '.join(excluded_ids)}. Add manually above if you "
                f"really want to use them."
            )

        if st.button("🚀 Train Model", type="primary",
                     use_container_width=True, key="reg_run"):
            if not features:
                st.warning("Please select at least one feature column.")
            else:
                with st.spinner("Training model..."):
                    res = run_regression(df, target, features, model_name,
                                          hyperparams)

                st.subheader("📊 Model Performance")
                m1, m2 = st.columns(2)
                m1.metric("R² Score (test split)",
                           f"{res['r2']:.4f}",
                           help="1.0 = perfect, 0 = no predictive power")
                m2.metric("RMSE",
                           f"{res['rmse']:,.2f}",
                           help="Lower is better")

                if res["cv"] is not None:
                    cv = res["cv"]
                    st.metric(
                        f"Cross-Validated R² ({cv['folds']}-fold)",
                        f"{cv['mean']:.4f} ± {cv['std']:.4f}",
                        help="Average R² across multiple train/test splits — "
                             "more reliable than a single split."
                    )
                    if cv["std"] > 0.15:
                        st.caption("⚠️ High variance across folds — results "
                                   "may be unstable with this data size.")

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

                st.info(
                    f"**Model:** {model_name} | "
                    f"**R²:** {res['r2']:.3f} | "
                    f"**RMSE:** {res['rmse']:,.2f}\n\n"
                    f"{'✅ Good fit!' if res['r2'] > 0.7 else '⚠️ Model needs improvement — try more features, different hyperparameters, or a different model.'}"
                )

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

        if target in num_cols and df[target].nunique() > 15:
            st.warning(
                f"⚠️ '{target}' looks like a continuous variable "
                f"({df[target].nunique()} unique values) rather than a "
                f"category. Classification may give meaningless results — "
                f"consider using **Regression** instead, or pick a column "
                f"with fewer distinct values."
            )

        hyperparams = {}
        with st.expander("⚙️ Advanced settings (hyperparameters)"):
            if model_name == "Logistic Regression":
                hyperparams["C"] = st.slider(
                    "C (inverse regularization strength)", 0.01, 10.0, 1.0,
                    key="clf_c")
            elif model_name == "Decision Tree":
                hyperparams["max_depth"] = st.slider(
                    "Max tree depth", 2, 20, 5, key="clf_depth_dt")
            elif model_name in ("Random Forest", "Gradient Boosting"):
                hyperparams["n_estimators"] = st.slider(
                    "Number of trees (n_estimators)", 50, 300, 100, step=10,
                    key="clf_n_est")
                hyperparams["max_depth"] = st.slider(
                    "Max tree depth", 2, 20,
                    5 if model_name == "Gradient Boosting" else 10,
                    key="clf_depth")
                if model_name == "Gradient Boosting":
                    hyperparams["learning_rate"] = st.slider(
                        "Learning rate", 0.01, 0.3, 0.1, step=0.01,
                        key="clf_lr")

        # V14: exclude identifier-like columns from default features
        feat_opts     = [c for c in num_cols if c != target]
        default_feats = [c for c in feat_opts if c not in id_cols][:4]
        features  = st.multiselect("Feature columns (numeric only)",
                                    feat_opts,
                                    default=default_feats,
                                    key="clf_feats")
        excluded_ids = [c for c in id_cols if c in feat_opts and c not in features]
        if excluded_ids:
            st.caption(
                f"ℹ️ Not included by default (look like unique IDs): "
                f"{', '.join(excluded_ids)}. Add manually above if you "
                f"really want to use them."
            )

        if st.button("🚀 Train Model", type="primary",
                     use_container_width=True, key="clf_run"):
            if not features:
                st.warning("Please select at least one numeric feature.")
            else:
                with st.spinner("Training model..."):
                    res = run_classification(df, target, features, model_name,
                                              hyperparams)

                st.subheader("📊 Model Performance")
                m1, m2 = st.columns(2)
                m1.metric("Accuracy (test split)",
                           f"{res['accuracy']*100:.2f}%",
                           help="Percentage of correct predictions")

                if res["cv"] is not None:
                    cv = res["cv"]
                    m2.metric(
                        f"Cross-Validated Accuracy ({cv['folds']}-fold)",
                        f"{cv['mean']*100:.2f}% ± {cv['std']*100:.2f}%",
                        help="Average accuracy across multiple train/test "
                             "splits — more reliable than a single split."
                    )
                    if cv["std"] > 0.1:
                        st.caption("⚠️ High variance across folds — results "
                                   "may be unstable with this data size.")
                else:
                    st.caption("Cross-validation skipped — not enough data "
                               "or class examples for reliable folds.")

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

                acc = res["accuracy"]
                st.info(
                    f"**Model:** {model_name} | "
                    f"**Accuracy:** {acc*100:.2f}%\n\n"
                    f"{'✅ Excellent accuracy!' if acc > 0.85 else '✅ Good accuracy!' if acc > 0.7 else '⚠️ Try different hyperparameters, Random Forest, or more features.'}"
                )

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
            if model_name == "K-Means":
                n_clusters = st.slider("Number of clusters", 2, 8, 3,
                                        key="clust_n")
                eps, min_samples = 0.5, 5  # unused for K-Means
            else:
                n_clusters = 3  # unused, DBSCAN infers cluster count itself
                st.caption("DBSCAN finds the number of clusters automatically.")

        # V14: DBSCAN tuning controls (shown only when DBSCAN is selected)
        if model_name == "DBSCAN":
            c1, c2 = st.columns(2)
            with c1:
                eps = st.slider(
                    "eps (neighborhood distance)", 0.1, 3.0, 0.5, step=0.05,
                    key="clust_eps",
                    help="Smaller = stricter grouping (more 'Noise'). "
                         "Larger = looser grouping (fewer, bigger clusters)."
                )
            with c2:
                min_samples = st.slider(
                    "min_samples (min points per cluster)", 2, 20, 5,
                    key="clust_min_samples",
                    help="Minimum number of nearby points needed to form "
                         "a cluster. Lower = easier to form small clusters."
                )
            st.caption(
                "💡 If most points show up as 'Noise' after running, "
                "try increasing eps first."
            )

        # V14: exclude identifier-like columns from default features
        default_feats = [c for c in num_cols if c not in id_cols][:3]
        features = st.multiselect("Feature columns",
                                   num_cols,
                                   default=default_feats,
                                   key="clust_feats")
        excluded_ids = [c for c in id_cols if c in num_cols and c not in features]
        if excluded_ids:
            st.caption(
                f"ℹ️ Not included by default (look like unique IDs): "
                f"{', '.join(excluded_ids)}. Add manually above if you "
                f"really want to use them."
            )

        if st.button("🚀 Run Clustering", type="primary",
                     use_container_width=True, key="clust_run"):
            if len(features) < 2:
                st.warning("Please select at least 2 feature columns.")
            else:
                with st.spinner("Finding clusters..."):
                    res = run_clustering(df, features, n_clusters, model_name,
                                          eps=eps, min_samples=min_samples)

                st.success(f"Found {res['n_clusters']} clusters!")

                if model_name == "DBSCAN" and res["noise_pct"] > 40:
                    st.warning(
                        f"⚠️ {res['noise_pct']:.0f}% of points were classified "
                        f"as 'Noise' (not part of any cluster). Try increasing "
                        f"eps or lowering min_samples above and run again."
                    )

                fig = px.scatter(
                    res["df_result"],
                    x=features[0],
                    y=features[1] if len(features) > 1 else features[0],
                    color="Cluster",
                    title=f"Clusters — {features[0]} vs {features[1] if len(features) > 1 else features[0]}",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(_style(fig), use_container_width=True)

                st.subheader("📊 Cluster Summary")
                st.dataframe(res["summary"], use_container_width=True)

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

                if mape is not None:
                    acc_label = ("✅ High confidence" if mape < 10 else
                                 "✅ Reasonable confidence" if mape < 20 else
                                 "⚠️ Low confidence — historical fit is noisy")
                    st.metric("Historical Fit (MAPE)", f"{mape:.1f}%",
                               help="Average % error of the model on data it "
                                    "already saw. Lower is better.")
                    st.caption(acc_label)

                st.subheader("📋 Forecast Values")
                fc_table = forecast[["ds","yhat","yhat_lower","yhat_upper"]]\
                           .tail(periods).copy()
                fc_table.columns = ["Date","Forecast","Lower Bound","Upper Bound"]
                fc_table["Date"] = fc_table["Date"].dt.strftime("%Y-%m")
                fc_table = fc_table.round(2)
                st.dataframe(fc_table, use_container_width=True, hide_index=True)

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