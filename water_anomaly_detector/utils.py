"""
Utility functions for the Water Anomaly Detection project.

This module centralises:

- Logging configuration
- I/O helpers (load/save data, run directories, config dumps)
- High-level summary report generation for the full pipeline

The new summary machinery is *step-aware*: it produces a different, richer
section for each pipeline stage (feature engineering, k-means, LSTM train,
LSTM prediction) instead of one very generic block.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging & basic helpers
# ---------------------------------------------------------------------------

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure the root logger using the provided config dict.

    Parameters
    ----------
    config : Dict[str, Any]
        Dictionary with logging configuration. Expected keys:
        - level: str, e.g. "INFO"
        - format: str, log format
        - datefmt: str, date format
        - log_file: Optional[str], if given logs will also be written there.
    """
    level = getattr(logging, str(config.get("level", "INFO")).upper(), logging.INFO)
    fmt = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    datefmt = config.get("datefmt", "%Y-%m-%d %H:%M:%S")

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    log_file = config.get("log_file")
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,  # overwrite any previous basicConfig
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging initialised (level=%s)", logging.getLevelName(level))
    return logger


def validate_input_data(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that all required columns exist in the dataframe.

    Returns
    -------
    is_valid : bool
    missing_columns : List[str]
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.warning("Missing required columns: %s", missing)
    return len(missing) == 0, missing


def load_data(file_path: str, date_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from CSV, Parquet or Excel.

    Parameters
    ----------
    file_path : str
        Path to input file.
    date_column : Optional[str]
        If provided, attempt to parse this column as datetime.

    Returns
    -------
    df : pd.DataFrame
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(file_path)
    elif ext in [".parquet"]:
        df = pd.read_parquet(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    logger.info("Loaded data from %s with shape %s", file_path, df.shape)
    return df


def save_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to CSV or Parquet depending on extension.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()

    if ext == ".csv":
        df.to_csv(output_path, index=False)
    elif ext == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        # Default to Parquet for unknown extension
        output_path = output_path + ".parquet"
        df.to_parquet(output_path, index=False)

    logger.info("Saved results to %s", output_path)


def create_run_directory(base_output_dir: str) -> str:
    """
    Create a timestamped run directory inside base_output_dir.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Created run directory: %s", run_dir)
    return run_dir


def save_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Persist configuration used for a pipeline run.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config_used.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    logger.info("Configuration saved to %s", path)


# ---------------------------------------------------------------------------
# Detailed summary generation
# ---------------------------------------------------------------------------

def _summarise_feature_engineering(fe_result: Dict[str, Any]) -> str:
    """
    Detailed summary for the feature engineering step.

    Expected keys in fe_result:
        - input_records
        - output_records
        - features_created
        - output_path (optional)
    """
    input_records = int(fe_result.get("input_records", 0) or 0)
    output_records = int(fe_result.get("output_records", 0) or 0)
    features_created = int(fe_result.get("features_created", 0) or 0)
    output_path = fe_result.get("output_path")

    dropped = max(0, input_records - output_records)
    dropped_pct = (dropped / input_records) * 100.0 if input_records > 0 else 0.0

    lines: List[str] = []
    lines.append("1) Feature Engineering")
    lines.append("-" * 60)
    lines.append(f"Input rows          : {input_records:,}")
    lines.append(f"Output rows         : {output_records:,}")
    if dropped > 0:
        lines.append(f"Rows dropped        : {dropped:,} ({dropped_pct:.2f}%)")
    else:
        lines.append("Rows dropped        : 0")

    lines.append(f"Features generated  : {features_created:,}")

    # Simple qualitative assessment
    if dropped_pct > 20.0:
        lines.append(
            "⚠ High row drop detected (>20%). You may be losing a lot of meters "
            "due to missing critical fields or date conversion issues."
        )
    elif 0.0 < dropped_pct <= 20.0:
        lines.append(
            "ℹ Some rows were dropped. This is usually fine, but you may want to "
            "verify that important customer segments are not systematically removed."
        )
    else:
        lines.append("✓ No data loss at this stage.")

    if output_path:
        lines.append(f"Saved feature-engineered data to: {output_path}")

    return "\n".join(lines)


def _load_json_safely(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load JSON from %s: %s", path, e)
        return None


def _summarise_kmeans(kmeans_result: Dict[str, Any]) -> str:
    """
    Detailed summary for the k-means clustering + anomaly detection step.

    Expected keys in kmeans_result:
        - models_trained
        - total_anomalies
        - anomaly_rate  (string like '2.34%')
        - output_path   (path to labeled parquet)
    Additionally we try to read per-group clustering metrics from the
    KMeansAnomalyDetectionPipeline metadata.json, if available.
    """
    models_trained = int(kmeans_result.get("models_trained", 0) or 0)
    total_anomalies = int(kmeans_result.get("total_anomalies", 0) or 0)
    anomaly_rate_str = str(kmeans_result.get("anomaly_rate", "0.00%") or "0.00%")
    output_path = kmeans_result.get("output_path")

    try:
        numeric_rate = float(anomaly_rate_str.replace("%", "").strip()) / 100.0
    except Exception:
        numeric_rate = None

    lines: List[str] = []
    lines.append("2) K-Means Clustering & Static Anomaly Detection")
    lines.append("-" * 60)
    lines.append(f"Models trained      : {models_trained}")
    lines.append(f"Total anomalies     : {total_anomalies:,} ({anomaly_rate_str})")
    if output_path:
        lines.append(f"Labeled data        : {output_path}")

    # Quick sanity check on anomaly rate
    if numeric_rate is not None:
        if numeric_rate < 0.001:
            lines.append(
                "⚠ Anomaly rate is extremely low (<0.1%). Models may be too "
                "conservative – useful anomalies could be missed."
            )
        elif numeric_rate > 0.20:
            lines.append(
                "⚠ Anomaly rate is very high (>20%). Clusters or thresholds "
                "may be too aggressive, leading to many false positives."
            )
        else:
            lines.append(
                "✓ Anomaly rate is within a reasonable range for rare-event detection."
            )

    # Attempt to pull detailed clustering metrics from metadata.json
    # metadata.json lives two levels above the labeled parquet:
    # <kmeans_run_dir>/data/labeled_data_xxx.parquet
    if output_path:
        kmeans_run_dir = os.path.dirname(os.path.dirname(output_path))
        meta_path = os.path.join(kmeans_run_dir, "metadata.json")
        meta = _load_json_safely(meta_path)
    else:
        meta = None

    if meta and "groups" in meta and meta["groups"]:
        groups = meta["groups"]
        n_groups = len(groups)
        lines.append("")
        lines.append(f"Number of brand groups modelled: {n_groups}")

        silhouettes: List[Tuple[str, float]] = []
        davies: List[Tuple[str, float]] = []
        k_values: List[Tuple[str, int]] = []

        for group_name, info in groups.items():
            m = info.get("metrics", {})
            s = m.get("silhouette_score")
            d = m.get("davies_bouldin_score")
            k = m.get("k") or m.get("k_opt")
            if s is not None:
                silhouettes.append((group_name, float(s)))
            if d is not None:
                davies.append((group_name, float(d)))
            if k is not None:
                try:
                    k_values.append((group_name, int(k)))
                except Exception:
                    pass

        if silhouettes:
            avg_sil = sum(v for _, v in silhouettes) / len(silhouettes)
            best_group, best_sil = max(silhouettes, key=lambda t: t[1])
            worst_group, worst_sil = min(silhouettes, key=lambda t: t[1])

            lines.append(
                f"Average silhouette score        : {avg_sil:.3f} "
                f"(best={best_sil:.3f} for '{best_group}', "
                f"worst={worst_sil:.3f} for '{worst_group}')"
            )
            if avg_sil < 0.25:
                lines.append(
                    "⚠ Overall clustering quality is weak (silhouette < 0.25). "
                    "Clusters are not well separated; consider revisiting features "
                    "or k-selection."
                )
            elif avg_sil < 0.4:
                lines.append(
                    "ℹ Clustering separation is moderate. This can still be useful "
                    "for anomaly scoring, but cluster boundaries are fuzzy."
                )
            else:
                lines.append("✓ Clusters are reasonably well separated on average.")

        if k_values:
            k_list = [k for _, k in k_values]
            lines.append(
                f"Optimal k per group (min/median/max): "
                f"{min(k_list)} / "
                f"{int(pd.Series(k_list).median())} / "
                f"{max(k_list)}"
            )

    else:
        lines.append("")
        lines.append(
            "ℹ No detailed metadata.json found for k-means run. "
            "Only global anomaly counts are summarised."
        )

    return "\n".join(lines)


def _summarise_lstm_training(lstm_result: Dict[str, Any]) -> str:
    """
    Detailed summary for the LSTM training step.

    Expected keys in lstm_result:
        - models_trained
        - output_path  (LSTM run_dir)
        - metrics      (nested dict: {brand: {horizon: metric_dict}})
    """
    models_trained = int(lstm_result.get("models_trained", 0) or 0)
    run_dir = lstm_result.get("output_path")
    metrics_nested: Dict[str, Dict[str, Dict[str, Any]]] = lstm_result.get("metrics", {}) or {}

    lines: List[str] = []
    lines.append("3) LSTM Training (Temporal Anomaly Risk Models)")
    lines.append("-" * 60)
    lines.append(f"Models trained (brand × horizon): {models_trained}")
    if run_dir:
        lines.append(f"Run directory                   : {run_dir}")

    if not metrics_nested:
        lines.append("⚠ No detailed metrics available for LSTM training.")
        return "\n".join(lines)

    # Aggregate metrics across brand/horizon models
    rows = []
    for brand, per_horizon in metrics_nested.items():
        for horizon, m in per_horizon.items():
            row = {
                "brand": brand,
                "horizon": horizon,
                "val_loss": float(m.get("val_loss", 0.0) or 0.0),
                "precision": float(m.get("precision", 0.0) or 0.0),
                "recall": float(m.get("recall", 0.0) or 0.0),
                "f1": float(m.get("f1", 0.0) or 0.0),
                "auc": float(m.get("auc", 0.0) or 0.0),
                "epochs_trained": int(m.get("epochs_trained", 0) or 0),
                "best_epoch": int(m.get("best_epoch", 0) or 0),
            }
            rows.append(row)

    if not rows:
        lines.append("⚠ Metrics dictionary is empty; cannot summarise LSTM performance.")
        return "\n".join(lines)

    df = pd.DataFrame(rows)

    # Overall statistics
    lines.append("")
    lines.append("Overall validation performance (across all brand/horizon models):")
    lines.append(
        f"  AUC   : mean={df['auc'].mean():.3f}, "
        f"min={df['auc'].min():.3f}, max={df['auc'].max():.3f}"
    )
    lines.append(
        f"  F1    : mean={df['f1'].mean():.3f}, "
        f"min={df['f1'].min():.3f}, max={df['f1'].max():.3f}"
    )
    lines.append(
        f"  Prec. : mean={df['precision'].mean():.3f}, "
        f"Rec. mean={df['recall'].mean():.3f}"
    )

    # Best / worst models by AUC
    best_row = df.loc[df["auc"].idxmax()]
    worst_row = df.loc[df["auc"].idxmin()]

    lines.append("")
    lines.append(
        "Best model by AUC: "
        f"{best_row['brand']} ({best_row['horizon']}) "
        f"– AUC={best_row['auc']:.3f}, F1={best_row['f1']:.3f}"
    )
    lines.append(
        "Weakest model by AUC: "
        f"{worst_row['brand']} ({worst_row['horizon']}) "
        f"– AUC={worst_row['auc']:.3f}, F1={worst_row['f1']:.3f}"
    )

    # Horizon-wise comparison
    lines.append("")
    lines.append("Average metrics per prediction horizon:")
    by_h = df.groupby("horizon").agg(
        auc_mean=("auc", "mean"),
        f1_mean=("f1", "mean"),
        prec_mean=("precision", "mean"),
        rec_mean=("recall", "mean"),
    )

    for horizon, row in by_h.iterrows():
        lines.append(
            f"  - {horizon:<6}: "
            f"AUC={row['auc_mean']:.3f}, "
            f"F1={row['f1_mean']:.3f}, "
            f"P={row['prec_mean']:.3f}, "
            f"R={row['rec_mean']:.3f}"
        )

    # Simple qualitative assessment / drawbacks
    auc_mean = df["auc"].mean()
    if auc_mean < 0.6:
        lines.append("")
        lines.append(
            "⚠ LSTM models show limited discriminative power on average (AUC < 0.60). "
            "You may need more informative features, different horizons, or a simpler model."
        )
    elif auc_mean < 0.75:
        lines.append("")
        lines.append(
            "ℹ LSTM models are moderately effective (0.60 ≤ AUC < 0.75). "
            "They should add value on top of static k-means scoring, but there is room for improvement."
        )
    else:
        lines.append("")
        lines.append(
            "✓ LSTM models achieve good discriminative performance on average (AUC ≥ 0.75)."
        )

    return "\n".join(lines)


def _summarise_lstm_prediction(pred_result: Dict[str, Any]) -> str:
    """
    Detailed summary for the LSTM prediction step.

    Expected keys in pred_result:
        - total_sequences
        - output_path (CSV with per-sequence predictions)
    """
    total_sequences = int(pred_result.get("total_sequences", 0) or 0)
    output_path = pred_result.get("output_path")

    lines: List[str] = []
    lines.append("4) LSTM Prediction (Scoring Latest Sequences)")
    lines.append("-" * 60)
    lines.append(f"Sequences scored : {total_sequences:,}")
    if output_path:
        lines.append(f"Predictions file : {output_path}")

    if not output_path or not os.path.exists(output_path):
        lines.append("⚠ Predictions CSV not found; cannot analyse prediction distribution.")
        return "\n".join(lines)

    try:
        df = pd.read_csv(output_path)
    except Exception as e:
        lines.append(f"⚠ Failed to read predictions CSV: {e}")
        return "\n".join(lines)

    if df.empty:
        lines.append("⚠ Predictions file is empty.")
        return "\n".join(lines)

    # Basic distribution stats
    if "meter_id" in df.columns:
        lines.append(f"Unique meters    : {df['meter_id'].nunique():,}")

    risk_col = "risk_level" if "risk_level" in df.columns else None
    pred_cols = [c for c in df.columns if c.startswith("pred_anomaly_")]
    prob_cols = [c for c in df.columns if c.startswith("prob_anomaly_")]

    if risk_col:
        lines.append("")
        lines.append("Risk level distribution:")
        counts = df[risk_col].value_counts()
        total = counts.sum()
        for level, cnt in counts.items():
            frac = cnt / total if total > 0 else 0.0
            lines.append(f"  - {level:<9}: {cnt:>6,} ({frac:.2%})")

    if pred_cols:
        lines.append("")
        lines.append("Predicted anomaly frequency by horizon:")
        for c in sorted(pred_cols):
            frac = df[c].mean()
            lines.append(f"  - {c}: {frac:.2%} of sequences flagged")

    if prob_cols:
        lines.append("")
        lines.append("Average predicted anomaly probability by horizon:")
        for c in sorted(prob_cols):
            mean_p = df[c].mean()
            p95 = df[c].quantile(0.95)
            lines.append(f"  - {c}: mean={mean_p:.3f}, 95th pct={p95:.3f}")

    # Simple qualitative comment on usage
    if pred_cols:
        overall_flag_rate = df[pred_cols].max(axis=1).mean()
        if overall_flag_rate < 0.01:
            lines.append("")
            lines.append(
                "⚠ Very few sequences are being flagged by LSTM (<1%). "
                "This could indicate thresholds that are too strict for day-to-day monitoring."
            )
        elif overall_flag_rate > 0.3:
            lines.append("")
            lines.append(
                "⚠ A large fraction of sequences are flagged (>30%). "
                "Consider increasing thresholds or reviewing training labels."
            )
        else:
            lines.append("")
            lines.append(
                "✓ LSTM flags a reasonable proportion of sequences for further review."
            )

    return "\n".join(lines)


def generate_summary_report(results: Dict[str, Any], run_dir: str) -> str:
    """
    Generate a rich, human-readable summary of the full pipeline execution.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary populated by WaterAnomalyDetectionPipeline with keys like:
        - 'feature_engineering'
        - 'kmeans'
        - 'lstm_training'
        - 'lstm_prediction'
    run_dir : str
        Top-level pipeline run directory; included for context in the header.

    Returns
    -------
    summary : str
        Multi-section text report.
    """
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("WATER ANOMALY DETECTION – PIPELINE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Run directory: {run_dir}")
    lines.append(f"Generated at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 1) Feature engineering
    fe_res = results.get("feature_engineering")
    if fe_res:
        lines.append(_summarise_feature_engineering(fe_res))
        lines.append("")
    else:
        lines.append("1) Feature Engineering\n" + "-" * 60)
        lines.append("Step not executed or no results recorded.\n")

    # 2) K-means
    km_res = results.get("kmeans")
    if km_res:
        lines.append(_summarise_kmeans(km_res))
        lines.append("")
    else:
        lines.append("2) K-Means Clustering & Static Anomaly Detection\n" + "-" * 60)
        lines.append("Step not executed or no results recorded.\n")

    # 3) LSTM training
    lstm_train_res = results.get("lstm_training")
    if lstm_train_res:
        lines.append(_summarise_lstm_training(lstm_train_res))
        lines.append("")
    else:
        lines.append("3) LSTM Training\n" + "-" * 60)
        lines.append("Step not executed or no results recorded.\n")

    # 4) LSTM prediction
    lstm_pred_res = results.get("lstm_prediction")
    if lstm_pred_res:
        lines.append(_summarise_lstm_prediction(lstm_pred_res))
        lines.append("")
    else:
        lines.append("4) LSTM Prediction\n" + "-" * 60)
        lines.append("Step not executed or no results recorded.\n")

    summary = "\n".join(lines)
    logger.info("Summary report generated.")
    return summary
