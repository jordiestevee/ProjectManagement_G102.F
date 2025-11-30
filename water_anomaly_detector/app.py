import os
import glob
from typing import Dict, Any

import pandas as pd
import streamlit as st

# Project imports (your existing modules)
from config import COLUMN_CONFIG, PIPELINE_CONFIG
from main import WaterAnomalyDetectionPipeline
from utils import validate_input_data, generate_summary_report

# -----------------------------------------------------------------------------
# Streamlit basic setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Water Anomaly Detection",
    layout="wide",
)

st.title("💧 Water Anomaly Detection Dashboard")

st.markdown(
    """
    Upload a CSV file with the same structure as your current dataset
    (`POLIZA_SUMINISTRO`, `FECHA`, `CONSUMO_REAL`, etc.), and run the full
    anomaly detection pipeline:
    
    1. **Feature Engineering**  
    2. **K-Means static anomalies**  
    3. **LSTM training**  
    4. **LSTM prediction**  

    The app shows intermediate information, generated plots, and a summary of
    the current run.

    The **Daily Tracker** page lets you pick a date and see:
    - Static anomalies for that **day**  
    - Static anomalies in the **week** and **month** starting that day  
    - LSTM risk & anomaly metrics “as of” that day
    """
)

# -----------------------------------------------------------------------------
# Session state initialisation
# -----------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.raw_df = None
    st.session_state.df_features = None
    st.session_state.kmeans_df = None
    st.session_state.lstm_predictions = None
    st.session_state.kmeans_feature_cols = None
    st.session_state.results: Dict[str, Any] = {}
    st.session_state.summary_text = ""


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("1. Upload data")

uploaded_file = st.sidebar.file_uploader(
    "Input data (CSV with the usual schema)",
    type=["csv"],
)

st.sidebar.header("2. Steps to run")

default_steps = ["Feature Engineering", "K-Means", "LSTM Training", "LSTM Prediction"]

selected_steps = st.sidebar.multiselect(
    "Select pipeline steps",
    options=default_steps,
    default=default_steps,
)

run_button = st.sidebar.button("🚀 Run selected steps")

# -----------------------------------------------------------------------------
# Helper: discover plot directories from results
# -----------------------------------------------------------------------------
def get_kmeans_plots_dir(results: Dict[str, Any]) -> str | None:
    """Infer the k-means plots directory from the results dict."""
    km_res = results.get("kmeans")
    if not km_res:
        return None
    output_path = km_res.get("output_path")
    if not output_path or not os.path.exists(output_path):
        return None

    # output_path = <km_run_dir>/data/labeled_data_*.parquet
    data_dir = os.path.dirname(output_path)
    km_run_dir = os.path.dirname(data_dir)
    plots_dir = os.path.join(km_run_dir, "plots")
    return plots_dir if os.path.isdir(plots_dir) else None


def get_lstm_training_plots_dir(results: Dict[str, Any]) -> str | None:
    """Infer the LSTM training plots directory from the results dict."""
    lstm_res = results.get("lstm_training")
    if not lstm_res:
        return None
    run_dir = lstm_res.get("output_path")
    if not run_dir:
        return None
    plots_dir = os.path.join(run_dir, "plots")
    return plots_dir if os.path.isdir(plots_dir) else None


def get_lstm_prediction_plots_dir(results: Dict[str, Any]) -> str | None:
    """
    Infer the LSTM prediction plots directory from the results dict.

    Note: currently the LSTM prediction pipeline only saves a CSV.
    If you implement prediction plots there (e.g. histograms), they should
    end up in ./lstm_predictions/pred_*/plots, and this helper will find them.
    """
    lstm_pred = results.get("lstm_prediction")
    if not lstm_pred:
        return None
    csv_path = lstm_pred.get("output_path")
    if not csv_path or not os.path.exists(csv_path):
        return None

    # csv_path = <pred_dir>/reports/lstm_predictions.csv
    reports_dir = os.path.dirname(csv_path)
    pred_dir = os.path.dirname(reports_dir)
    plots_dir = os.path.join(pred_dir, "plots")
    return plots_dir if os.path.isdir(plots_dir) else None


def show_png_folder(plots_dir: str, columns: int = 2):
    """Render all PNGs in a folder as a simple gallery."""
    if not plots_dir or not os.path.isdir(plots_dir):
        st.info("No plots directory found.")
        return

    png_files = sorted(glob.glob(os.path.join(plots_dir, "*.png")))
    if not png_files:
        st.info("No PNG plots found in this run directory.")
        return

    st.write(f"Plots from `{plots_dir}`")

    # Display in a grid
    cols = st.columns(columns)
    for i, path in enumerate(png_files):
        with cols[i % columns]:
            st.image(path, caption=os.path.basename(path), use_container_width=True)


# -----------------------------------------------------------------------------
# Run pipeline when the user clicks the button
# -----------------------------------------------------------------------------
if run_button:
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    else:
        try:
            # Load data directly from the uploaded buffer
            df = pd.read_csv(uploaded_file)

            # Validate schema
            is_valid, missing_cols = validate_input_data(
                df, COLUMN_CONFIG["required_columns"]
            )
            if not is_valid:
                st.error(
                    f"Input file is missing required columns: {missing_cols}. "
                    f"Please upload a CSV with the same format as the current one."
                )
            else:
                st.session_state.raw_df = df

                # Initialise a fresh pipeline
                st.session_state.pipeline = WaterAnomalyDetectionPipeline()
                pipe = st.session_state.pipeline

                # Local alias for results dict
                results = pipe.results

                # --------------------------------------------
                # Step 1: Feature engineering
                # --------------------------------------------
                if "Feature Engineering" in selected_steps:
                    with st.spinner("Running feature engineering..."):
                        df_features, kmeans_feature_cols = pipe.run_feature_engineering(
                            df
                        )

                    st.session_state.df_features = df_features
                    st.session_state.kmeans_feature_cols = kmeans_feature_cols

                    fe_res = results.get("feature_engineering", {})
                    st.success(
                        f"Feature engineering complete – "
                        f"{fe_res.get('output_records', len(df_features))} records, "
                        f"{fe_res.get('features_created', len(df_features.columns))} features."
                    )
                else:
                    st.warning(
                        "You skipped feature engineering. "
                        "Downstream steps may fail if required features are missing."
                    )

                # --------------------------------------------
                # Step 2: K-Means clustering
                # --------------------------------------------
                if "K-Means" in selected_steps:
                    if st.session_state.df_features is None:
                        st.error(
                            "K-Means requested but feature engineering data is missing."
                        )
                    else:
                        with st.spinner("Running k-means clustering..."):
                            labeled_df = pipe.run_kmeans_clustering(
                                st.session_state.df_features,
                                st.session_state.kmeans_feature_cols,
                            )

                        st.session_state.kmeans_df = labeled_df

                        km_res = results.get("kmeans", {})
                        st.success(
                            f"K-Means clustering complete – "
                            f"{km_res.get('total_anomalies', 0)} anomalies "
                            f"({km_res.get('anomaly_rate', '0.00%')})."
                        )

                # --------------------------------------------
                # Step 3: LSTM training
                # --------------------------------------------
                if "LSTM Training" in selected_steps:
                    if st.session_state.kmeans_df is None:
                        st.error(
                            "LSTM training requested but k-means labeled data is missing."
                        )
                    else:
                        with st.spinner("Running LSTM training (this may take a while)..."):
                            lstm_train_metrics = pipe.run_lstm_training(
                                st.session_state.kmeans_df
                            )

                        lstm_res = results.get("lstm_training", {})
                        st.success(
                            f"LSTM training complete – "
                            f"{lstm_res.get('models_trained', 0)} models trained."
                        )

                # --------------------------------------------
                # Step 4: LSTM prediction
                # --------------------------------------------
                if "LSTM Prediction" in selected_steps:
                    if st.session_state.kmeans_df is None:
                        st.error(
                            "LSTM prediction requested but k-means labeled data is missing."
                        )
                    else:
                        with st.spinner("Running LSTM prediction..."):
                            pred_df = pipe.run_lstm_prediction(
                                st.session_state.kmeans_df
                            )

                        st.session_state.lstm_predictions = pred_df

                        lstm_pred = results.get("lstm_prediction", {})
                        st.success(
                            f"LSTM prediction complete – "
                            f"{lstm_pred.get('total_sequences', len(pred_df))} sequences scored."
                        )

                # --------------------------------------------
                # Generate summary text for the overview tab
                # --------------------------------------------
                if PIPELINE_CONFIG.get("generate_reports", True):
                    st.session_state.summary_text = generate_summary_report(
                        results, pipe.run_dir
                    )
                else:
                    st.session_state.summary_text = ""

        except Exception as e:
            st.exception(e)

# -----------------------------------------------------------------------------
# Tabs for visualisation (added 📅 Daily Tracker)
# -----------------------------------------------------------------------------
(
    tab_overview,
    tab_data,
    tab_kmeans,
    tab_lstm_train,
    tab_lstm_pred,
    tab_daily,
) = st.tabs(
    [
        "📊 Overview",
        "📁 Raw & Features",
        "📌 K-Means Anomalies",
        "🧠 LSTM Training",
        "🔮 LSTM Predictions",
        "📅 Daily Tracker",
    ]
)

# --------------------------------------------
# Overview tab
# --------------------------------------------
with tab_overview:
    st.subheader("Run Overview")

    if st.session_state.pipeline is None:
        st.info("Upload data and click **Run selected steps** to see results.")
    else:
        pipe = st.session_state.pipeline
        results = pipe.results

        # High-level metrics
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if st.session_state.raw_df is not None:
                st.metric("Input records", f"{len(st.session_state.raw_df):,}")
        with c2:
            fe_res = results.get("feature_engineering", {})
            st.metric(
                "Engineered records", f"{fe_res.get('output_records', 0):,}"
            )
        with c3:
            km_res = results.get("kmeans", {})
            st.metric(
                "Static anomalies",
                f"{km_res.get('total_anomalies', 0):,}",
                km_res.get("anomaly_rate", "0.00%"),
            )
        with c4:
            lstm_res = results.get("lstm_training", {})
            st.metric(
                "LSTM models",
                f"{lstm_res.get('models_trained', 0):,}",
            )

        st.markdown("---")
        st.markdown(f"**Run directory:** `{pipe.run_dir}`")

        # Printable summary report
        if st.session_state.summary_text:
            st.subheader("Pipeline summary report")
            st.text(st.session_state.summary_text)
        else:
            st.info("Summary report not generated (PIPELINE_CONFIG['generate_reports'] is False).")

# --------------------------------------------
# Data tab
# --------------------------------------------
with tab_data:
    st.subheader("Raw input data")

    if st.session_state.raw_df is None:
        st.info("No data loaded yet.")
    else:
        df = st.session_state.raw_df
        st.write("First 10 rows of the input data:")
        st.dataframe(df.head(10))

        # Basic info
        if COLUMN_CONFIG.get("date") in df.columns:
            date_col = COLUMN_CONFIG["date"]
            try:
                dates = pd.to_datetime(df[date_col])
                st.write(
                    f"Date range: **{dates.min().date()}** → **{dates.max().date()}**"
                )
            except Exception:
                pass

        if COLUMN_CONFIG.get("meter_id") in df.columns:
            meter_col = COLUMN_CONFIG["meter_id"]
            st.write(f"Unique meters: **{df[meter_col].nunique():,}**")

        st.markdown("---")
        st.subheader("Feature-engineered data")

        if st.session_state.df_features is None:
            st.info("Feature engineering not run yet.")
        else:
            fe_df = st.session_state.df_features
            st.write(
                f"Feature-engineered dataset: {len(fe_df):,} rows, "
                f"{len(fe_df.columns):,} columns."
            )
            st.dataframe(fe_df.head(10))

# --------------------------------------------
# K-Means tab
# --------------------------------------------
with tab_kmeans:
    st.subheader("K-Means clustering & static anomalies")

    if st.session_state.kmeans_df is None:
        st.info("K-Means has not been run yet.")
    else:
        labeled_df = st.session_state.kmeans_df
        st.write(
            f"Labeled dataset: {len(labeled_df):,} rows "
            f"({labeled_df['is_anomaly'].sum():,} anomalies)."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write("Sample of anomalies:")
            st.dataframe(labeled_df[labeled_df["is_anomaly"] == 1].head(20))
        with col2:
            st.write("Random non-anomalous sample:")
            st.dataframe(
                labeled_df[labeled_df["is_anomaly"] == 0].sample(
                    min(20, (labeled_df["is_anomaly"] == 0).sum())
                )
            )

        st.markdown("---")
        st.subheader("K-Means diagnostic plots (from pipeline run)")

        if st.session_state.pipeline is not None:
            plots_dir = get_kmeans_plots_dir(st.session_state.pipeline.results)
            show_png_folder(plots_dir)

# --------------------------------------------
# LSTM training tab
# --------------------------------------------
with tab_lstm_train:
    st.subheader("LSTM training")

    pipe = st.session_state.pipeline
    if pipe is None or "lstm_training" not in pipe.results:
        st.info("LSTM training has not been run yet.")
    else:
        lstm_res = pipe.results["lstm_training"]
        metrics_nested = lstm_res.get("metrics", {}) or {}

        st.write(
            f"Models trained: **{lstm_res.get('models_trained', 0)}** "
            f"– run directory: `{lstm_res.get('output_path', '')}`"
        )

        # Flatten metrics into a table for quick inspection
        rows = []
        for brand, horizons in metrics_nested.items():
            for horizon, m in horizons.items():
                row = {
                    "brand": brand,
                    "horizon": horizon,
                    "auc": m.get("auc"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1": m.get("f1"),
                    "accuracy": m.get("accuracy"),
                    "best_threshold": m.get("best_threshold"),
                    "best_epoch": m.get("best_epoch"),
                }
                rows.append(row)

        if rows:
            metrics_df = pd.DataFrame(rows)
            st.write("Per brand / horizon validation metrics:")
            st.dataframe(metrics_df)

        st.markdown("---")
        st.subheader("LSTM training plots")

        plots_dir = get_lstm_training_plots_dir(pipe.results)
        show_png_folder(plots_dir)

        st.info(
            "If you don't see LSTM plots, ensure that your "
            "`LSTMAnomalyTrainingPipeline.run()` calls "
            "`create_training_plots(model_key)` for each trained model."
        )

# --------------------------------------------
# LSTM prediction tab
# --------------------------------------------
with tab_lstm_pred:
    st.subheader("LSTM predictions")

    if st.session_state.lstm_predictions is None:
        st.info("LSTM prediction has not been run yet.")
    else:
        pred_df = st.session_state.lstm_predictions

        st.write(
            f"Prediction dataset: {len(pred_df):,} sequences, "
            f"{pred_df['meter_id'].nunique():,} meters (if `meter_id` is present)."
        )

        st.write("Head of predictions:")
        st.dataframe(pred_df.head(20))

        # Risk-level distribution quick view (full predictions)
        if "risk_level" in pred_df.columns:
            st.markdown("**Risk level distribution (all predictions)**")
            st.bar_chart(pred_df["risk_level"].value_counts().sort_index())

        st.markdown("---")
        st.subheader("Prediction-time plots (if implemented in your pipeline)")

        pipe = st.session_state.pipeline
        if pipe is not None:
            plots_dir = get_lstm_prediction_plots_dir(pipe.results)
            show_png_folder(plots_dir)

        st.info(
            "If you add plots to `LSTMAnomalyPredictionPipeline` "
            "(e.g. probability histograms, risk counts), "
            "save them into `self.output_dir/plots` and they will appear here."
        )

# --------------------------------------------
# 📅 Daily Tracker tab
# --------------------------------------------
with tab_daily:
    st.subheader("Daily anomaly & risk tracker")

    if st.session_state.raw_df is None:
        st.info("Run the pipeline on a dataset first to use the Daily Tracker.")
    else:
        raw_df = st.session_state.raw_df.copy()
        date_col = COLUMN_CONFIG["date"]

        # Parse dates safely
        try:
            raw_df[date_col] = pd.to_datetime(raw_df[date_col])
        except Exception:
            st.error(f"Could not parse date column `{date_col}` in raw data.")
            st.stop()

        min_date = raw_df[date_col].min().date()
        max_date = raw_df[date_col].max().date()

        # Date selector
        selected_date = st.date_input(
            "Reference date for monitoring",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            help="Daily / weekly / monthly status will be computed starting from this date.",
        )
        ref_ts = pd.to_datetime(selected_date)

        st.markdown(
            f"""
            **Selected date:** `{selected_date}`  
            - **Day:** anomalies on this calendar day  
            - **Week:** anomalies in the 7 days `[D, D+7)`  
            - **Month:** anomalies in the 30 days `[D, D+30)`  
            LSTM metrics are computed from predictions whose `end_date` is the latest
            available **on or before** this reference date.
            """
        )

        # ---------------------------------------------------------------------
        # Static anomalies (K-Means) around the selected day
        # ---------------------------------------------------------------------
        st.markdown("### Static anomalies (K-Means)")

        if st.session_state.kmeans_df is None:
            st.info("K-Means results are not available. Run K-Means to see static anomalies.")
        else:
            kdf = st.session_state.kmeans_df.copy()

            if date_col not in kdf.columns:
                st.warning(
                    f"Column `{date_col}` not found in k-means labeled data; "
                    "cannot compute daily static anomalies."
                )
            else:
                try:
                    kdf[date_col] = pd.to_datetime(kdf[date_col])
                except Exception:
                    st.error(f"Could not parse date column `{date_col}` in k-means data.")
                    kdf = None

            if kdf is not None:
                day_mask = kdf[date_col] == ref_ts
                week_mask = (kdf[date_col] >= ref_ts) & (
                    kdf[date_col] < ref_ts + pd.Timedelta(days=7)
                )
                month_mask = (kdf[date_col] >= ref_ts) & (
                    kdf[date_col] < ref_ts + pd.Timedelta(days=30)
                )

                day_df = kdf[day_mask]
                week_df = kdf[week_mask]
                month_df = kdf[month_mask]

                c_day, c_week, c_month = st.columns(3)

                def _static_summary_block(col, label, df_window):
                    with col:
                        total = len(df_window)
                        anomalies = int(df_window["is_anomaly"].sum()) if total > 0 else 0
                        rate = anomalies / total * 100 if total > 0 else 0.0
                        st.metric(
                            f"{label} – anomalies",
                            f"{anomalies:,}",
                            f"{rate:0.2f}% of {total:,}",
                        )

                _static_summary_block(c_day, "Day", day_df)
                _static_summary_block(c_week, "Week", week_df)
                _static_summary_block(c_month, "Month", month_df)

                # Details for the selected day
                with st.expander("Show detailed anomalies for selected day"):
                    if day_df.empty or day_df["is_anomaly"].sum() == 0:
                        st.write("No static anomalies detected on this day.")
                    else:
                        # Show anomalies and a few useful columns if present
                        cols_to_show = [
                            c
                            for c in [
                                COLUMN_CONFIG.get("meter_id"),
                                COLUMN_CONFIG.get("date"),
                                COLUMN_CONFIG.get("consumption"),
                                "BRAND",
                                "cluster",
                                "distance",
                                "anomaly_score",
                                "is_anomaly",
                            ]
                            if c in day_df.columns
                        ]
                        st.dataframe(day_df[day_df["is_anomaly"] == 1][cols_to_show].head(100))

        # ---------------------------------------------------------------------
        # LSTM predicted risk & anomalies as of the selected day
        # ---------------------------------------------------------------------
        st.markdown("### LSTM predicted risk (as of selected day)")

        if st.session_state.lstm_predictions is None:
            st.info("LSTM predictions are not available. Run LSTM prediction to see this section.")
        else:
            ldf = st.session_state.lstm_predictions.copy()

            # Parse prediction dates
            if "end_date" not in ldf.columns:
                st.warning(
                    "`end_date` column not found in LSTM predictions; "
                    "cannot align predictions with calendar."
                )
            else:
                try:
                    ldf["end_date"] = pd.to_datetime(ldf["end_date"])
                except Exception:
                    st.error("Could not parse `end_date` in LSTM predictions.")
                    ldf = None

            if ldf is not None:
                # Use predictions whose end_date is <= selected day;
                # pick the latest such date as the "current" status.
                valid = ldf[ldf["end_date"] <= ref_ts]
                if valid.empty:
                    st.info(
                        "No LSTM predictions with `end_date` on or before "
                        f"{selected_date}. Try a later date."
                    )
                else:
                    anchor_end_date = valid["end_date"].max()
                    today_preds = valid[valid["end_date"] == anchor_end_date].copy()

                    st.write(
                        f"Using predictions with `end_date = {anchor_end_date.date()}` "
                        "(closest available on/before the selected date)."
                    )

                    # Determine available horizons
                    horizons = []
                    for h in ["day", "week", "month"]:
                        if f"prob_anomaly_{h}" in today_preds.columns and f"pred_anomaly_{h}" in today_preds.columns:
                            horizons.append(h)

                    # Summary metrics per horizon
                    if horizons:
                        rows = []
                        for h in horizons:
                            p_col = f"prob_anomaly_{h}"
                            y_col = f"pred_anomaly_{h}"
                            total_seq = len(today_preds)
                            preds_anom = int(today_preds[y_col].sum())
                            mean_prob = float(today_preds[p_col].mean()) if total_seq > 0 else 0.0
                            rows.append(
                                {
                                    "horizon": h,
                                    "sequences": total_seq,
                                    "predicted_anomalies": preds_anom,
                                    "mean_probability": round(mean_prob, 4),
                                }
                            )
                        st.write("Per-horizon LSTM anomaly metrics (as of anchor date):")
                        st.dataframe(pd.DataFrame(rows))

                    # Risk-level distribution (only for anchor set)
                    if "risk_level" in today_preds.columns:
                        st.markdown("**Risk level distribution (LSTM, anchor date only)**")
                        st.bar_chart(today_preds["risk_level"].value_counts().sort_index())

                    # Top high-risk meters table
                    with st.expander("Show top high-risk meters (anchor predictions)"):
                        if not horizons:
                            st.write("No horizon columns found in predictions.")
                        else:
                            # Use max probability across horizons as ranking score
                            prob_cols = [f"prob_anomaly_{h}" for h in horizons]
                            today_preds["max_prob"] = today_preds[prob_cols].max(axis=1)
                            top = today_preds.sort_values("max_prob", ascending=False).head(100)

                            cols_to_show = ["meter_id", "brand_model", "risk_level", "start_date", "end_date", "max_prob"]
                            cols_to_show += prob_cols
                            cols_to_show = [c for c in cols_to_show if c in top.columns]

                            st.dataframe(top[cols_to_show])
