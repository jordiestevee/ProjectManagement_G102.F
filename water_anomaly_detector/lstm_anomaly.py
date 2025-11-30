# lstm_anomaly.py

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, callbacks

from config import COLUMN_CONFIG  # meter_id, date_col, brand_col names

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. SEQUENCE / DATASET BUILDER
# ---------------------------------------------------------------------------

def _infer_feature_columns_for_lstm(
    df: pd.DataFrame,
    extra_exclude: Optional[List[str]] = None,
) -> List[str]:
    """
    Build the list of feature columns for the LSTM.

    Idea:
    - Use *all* numeric/boolean features that describe the context of the
      observation, including:
        - consumption level
        - z-scores / robust z-scores
        - temporal encodings
        - rolling stats
        - anomaly scores from k-means (distance_to_centroid, anomaly_score, is_anomaly, etc.)
    - Exclude obvious identifiers and any future-target labels.
    """
    meter_col = COLUMN_CONFIG.get("meter_id", "POLIZA_SUMINISTRO")
    date_col = COLUMN_CONFIG.get("date_col", "FECHA")

    # Columns we never want as numeric features
    base_exclude = {
        meter_col,
        date_col,
        "BRAND",
        "MODEL",
        "brand_model",             # used for grouping, not as numeric feature
        "CONSUMO_REAL",           # we usually use CONSUMO_LOG instead
        "anomaly_day",
        "anomaly_week",
        "anomaly_month",
    }

    if extra_exclude:
        base_exclude.update(extra_exclude)

    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    # Keep everything numeric except the forbidden ones above
    features = [c for c in numeric_cols if c not in base_exclude]

    logger.info(
        "LSTM feature selection: %d numeric features selected out of %d columns",
        len(features),
        len(df.columns),
    )
    return sorted(features)


def _build_targets_for_horizons(
    meter_df: pd.DataFrame,
    is_anomaly_col: str,
    idx_start: int,
    idx_end: int,
    target_horizons: Dict[str, int],
    min_anomaly_counts: Dict[str, int],
) -> Dict[str, int]:
    """
    For a sequence ending at idx_end (exclusive), build binary targets for each horizon.

    For each horizon h (days_ahead):
      target_h = 1 if the number of anomalies in [idx_end, idx_end + h)
                  is >= min_anomaly_counts[h] else 0.
    """
    targets = {}
    n = len(meter_df)

    for h_name, h_days in target_horizons.items():
        look_ahead_start = idx_end
        look_ahead_end = min(idx_end + h_days, n)
        if look_ahead_start >= look_ahead_end:
            targets[h_name] = 0
        else:
            future_slice = meter_df.iloc[look_ahead_start:look_ahead_end][is_anomaly_col].values
            anomaly_count = int(np.sum(future_slice > 0))
            required = int(min_anomaly_counts.get(h_name, 1))
            targets[h_name] = int(anomaly_count >= required)

    return targets


def prepare_lstm_dataset(
    df: pd.DataFrame,
    sequence_length: int,
    target_horizons: Dict[str, int],
    min_anomaly_count: Union[int, Dict[str, int]] = 1,
    stride: int = 1,
    max_sequences_per_meter: Optional[int] = None,
    meters_sample_frac: float = 1.0,
    output_format: str = "dataframe",
    random_state: int = 42,
) -> Tuple[Union[pd.DataFrame, tf.data.Dataset], List[str]]:
    """
    Build LSTM sequences from the labeled (k-means) dataframe.

    IMPORTANT DESIGN CHOICES:
    - LSTM sees *all* relevant engineered features (not just k-means subset).
    - We still use k-means outputs (cluster_id, distances, is_anomaly) as
      features, because they are known at prediction time and carry signal.
    - We keep per-sequence tags: meter_id, brand_model (now brand-level),
      start_date, end_date, so later training / prediction can group correctly.

    Parameters
    ----------
    df : pd.DataFrame
        Output from the k-means stage (already labeled with 'is_anomaly', cluster info, etc.)
    sequence_length : int
        Window length in days.
    target_horizons : Dict[str, int]
        Mapping horizon_name -> days ahead (e.g. {'day': 1, 'week': 7, 'month': 30}).
    stride : int
        Step used when sliding the window along each meter.
    max_sequences_per_meter : Optional[int]
        Cap on sequences per meter to avoid explosion.
    meters_sample_frac : float
        Fraction of meters to keep (efficiency knob).
    output_format : {'dataframe', 'tfdata'}
        - 'dataframe': returns a DataFrame (for the training pipeline).
        - 'tfdata': returns a tf.data.Dataset (optional usage).
    random_state : int
        Random seed used for sampling meters.
    min_anomaly_count : int or Dict[str, int]
        - If int: same minimum number of anomaly observations required
          in the look-ahead window for all horizons.
        - If dict: mapping horizon_name -> minimum anomaly count.

    Returns
    -------
    sequences_df or tf.data.Dataset, feature_names
    """
    meter_col = COLUMN_CONFIG.get("meter_id", "POLIZA_SUMINISTRO")
    date_col = COLUMN_CONFIG.get("date_col", "FECHA")
    brand_col = COLUMN_CONFIG.get("brand_col", "BRAND")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([meter_col, date_col])

    if "is_anomaly" not in df.columns:
        raise ValueError("LSTM dataset builder expects an 'is_anomaly' column from k-means labeling.")

    # Ensure brand and brand_model tags exist
    if brand_col not in df.columns:
        df[brand_col] = "UNKNOWN"

    if "brand_model" not in df.columns:
        df["brand_model"] = df[brand_col].astype(str)

    rng = np.random.RandomState(random_state)

    # Optionally sample meters to reduce dataset size
    unique_meters = df[meter_col].unique()
    n_meters_total = len(unique_meters)
    if 0 < meters_sample_frac < 1.0:
        n_keep = max(1, int(n_meters_total * meters_sample_frac))
        sampled_meters = rng.choice(unique_meters, size=n_keep, replace=False)
    else:
        sampled_meters = unique_meters

    df = df[df[meter_col].isin(sampled_meters)].copy()
    logger.info(
        "LSTM dataset: using %d/%d meters (%.1f%%)",
        len(sampled_meters),
        n_meters_total,
        100.0 * len(sampled_meters) / max(1, n_meters_total),
    )

    # Build feature list (use all numeric features except obvious IDs / future labels)
    feature_cols = _infer_feature_columns_for_lstm(df)
    sequences: List[Dict] = []
    horizon_names = list(target_horizons.keys())

    # NEW: resolve per-horizon minimum anomaly counts
    if isinstance(min_anomaly_count, dict):
        min_counts = {
            h: int(min_anomaly_count.get(h, min_anomaly_count.get("default", 1)))
            for h in target_horizons.keys()
        }
    else:
        min_counts = {h: int(min_anomaly_count) for h in target_horizons.keys()}

    for meter_id, meter_df in df.groupby(meter_col):
        meter_df = meter_df.sort_values(date_col).reset_index(drop=True)

        brand_values = meter_df[brand_col].dropna().unique()
        brand_value = brand_values[0] if len(brand_values) > 0 else "UNKNOWN"

        n = len(meter_df)
        if n < sequence_length + min(target_horizons.values()):
            continue

        seq_count = 0
        max_seq_this_meter = max_sequences_per_meter if max_sequences_per_meter is not None else np.inf

        for start_idx in range(0, n - sequence_length - min(target_horizons.values()) + 1, stride):
            if seq_count >= max_seq_this_meter:
                break

            end_idx = start_idx + sequence_length

            window = meter_df.iloc[start_idx:end_idx]
            x = window[feature_cols].to_numpy(dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            targets = _build_targets_for_horizons(
                meter_df=meter_df,
                is_anomaly_col="is_anomaly",
                idx_start=start_idx,
                idx_end=end_idx,
                target_horizons=target_horizons,
                min_anomaly_counts=min_counts,
            )

            sequences.append(
                {
                    "meter_id": meter_id,
                    "brand_model": brand_value,
                    "start_date": meter_df.loc[start_idx, date_col],
                    "end_date": meter_df.loc[end_idx - 1, date_col],
                    "sequence": x,
                    **{f"anomaly_{h_name}": targets[h_name] for h_name in horizon_names},
                }
            )

            seq_count += 1

    if not sequences:
        logger.warning("No sequences were created for LSTM. Check sequence_length, horizons and data coverage.")
        sequences_df = pd.DataFrame(columns=["meter_id", "brand_model", "start_date", "end_date", "sequence"])
        return sequences_df, feature_cols

    if output_format == "tfdata":
        # Optional path: build tf.data.Dataset (not used by main pipeline now)
        n_features = len(feature_cols)

        def gen():
            for row in sequences:
                x = row["sequence"].astype(np.float32)
                y = np.array(
                    [
                        row.get("anomaly_day", 0),
                        row.get("anomaly_week", 0),
                        row.get("anomaly_month", 0),
                    ],
                    dtype=np.float32,
                )
                yield x, y

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(sequence_length, n_features), dtype=tf.float32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32),
            ),
        )

        opts = tf.data.Options()
        opts.experimental_deterministic = False
        ds = ds.with_options(opts)
        ds = ds.cache()
        ds = ds.batch(64).prefetch(tf.data.AUTOTUNE)

        logger.info(
            "Built tf.data Dataset: sequences=%d, sequence_length=%d, n_features=%d",
            len(sequences),
            sequence_length,
            n_features,
        )
        return ds, feature_cols

    # Default: DataFrame for training / prediction pipelines
    sequences_df = pd.DataFrame(sequences)
    logger.info(
        "Built sequences dataframe: %d sequences for %d meters (%d features)",
        len(sequences_df),
        sequences_df["meter_id"].nunique(),
        len(feature_cols),
    )
    return sequences_df, feature_cols


# ---------------------------------------------------------------------------
# 2. TRAINING PIPELINE
# ---------------------------------------------------------------------------

class LSTMAnomalyTrainingPipeline:
    """
    LSTM training pipeline for water consumption anomaly prediction.

    Key ideas:
    - One model per brand (via 'brand_model', which now stores BRAND).
    - No leakage between train/val: we split by *meters* (no overlapping sequences
      from the same meter across splits).
    - Class imbalance handled via per-horizon sample weights.
    - All contextual + k-means features available to the model.
    """

    def __init__(
        self,
        sequence_length: int = 90,
        target_horizons: Optional[Dict[str, int]] = None,
        units: Union[int, List[int]] = 128,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        validation_split: float = 0.2,  # fraction of *meters* used as validation
        output_dir: str = "./lstm_output",
    ):
        self.sequence_length = sequence_length
        self.target_horizons = target_horizons or {"day": 1, "week": 7, "month": 30}
        self.horizon_names = list(self.target_horizons.keys())

        # Allow either a single int (e.g. 128) or a list like [128, 64]
        if isinstance(units, (list, tuple)):
            self.lstm_units = list(units)
        else:
            # Default behavior: 2-layer LSTM with units and units // 2
            self.lstm_units = [int(units), int(units) // 2]

        self.dropout_rate = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.output_dir = output_dir

        # Output dirs
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"lstm_run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "reports"), exist_ok=True)

        # State
        self.models: Dict[str, tf.keras.Model] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.histories: Dict[str, Dict] = {}
        self.metrics: Dict[str, Dict] = {}
        self.feature_names: Optional[List[str]] = None

    # ------------------ model architecture ------------------

    def build_model(self, input_shape, n_outputs: int = 1) -> tf.keras.Model:
        logger.info(
            "Building LSTM model for %s with units=%s and n_outputs=%d",
            "%s",  # placeholder, or pass group/horizon if you like
            self.lstm_units,
            n_outputs,
        )

        model = tf.keras.Sequential(name="lstm_anomaly")
        model.add(layers.Input(shape=input_shape))

        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                )
            )
            model.add(layers.BatchNormalization())

        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dropout(self.dropout_rate))

        # <<< NEW: configurable output dimension
        model.add(
            layers.Dense(
                n_outputs,
                activation="sigmoid",
                name="anomaly_probabilities",
            )
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        return model


    # ------------------ data preparation ------------------

    def _split_train_val_by_meter(
        self, group_df: pd.DataFrame, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into train/val by *meters* to avoid overlapping windows.

        Returns
        -------
        X_train, X_val, y_train, y_val, sample_weight_train
        """
        rng = np.random.RandomState(random_state)
        meters = group_df["meter_id"].unique()
        rng.shuffle(meters)

        n_meters = len(meters)
        n_val = max(1, int(n_meters * self.validation_split))
        val_meters = set(meters[:n_val])
        train_meters = set(meters[n_val:]) if n_meters > n_val else set()

        # Fallback: if we end up with 0 train meters, use all but last meter as train
        if len(train_meters) == 0 and n_meters > 1:
            train_meters = set(meters[:-1])
            val_meters = set(meters[-1:])

        train_mask = group_df["meter_id"].isin(train_meters).values
        val_mask = group_df["meter_id"].isin(val_meters).values

        X = np.stack(group_df["sequence"].values).astype(np.float32)
        y = group_df[[f"anomaly_{h}" for h in self.horizon_names]].values.astype(np.float32)

        # Scale features based on *train* data only
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_reshaped = np.nan_to_num(X_reshaped, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        scaler.fit(X_reshaped[train_mask.repeat(n_timesteps)])

        mean = np.nan_to_num(scaler.mean_, nan=0.0, posinf=0.0, neginf=0.0)
        scale = np.nan_to_num(scaler.scale_, nan=1.0, posinf=1.0, neginf=1.0)
        scale[scale == 0] = 1.0

        X_reshaped = (X_reshaped - mean) / scale

        scaler.mean_ = mean
        scaler.scale_ = scale

        self.scalers[group_df["brand_model"].iloc[0]] = scaler
        X_scaled = X_reshaped.reshape(n_samples, n_timesteps, n_features)

        X_train = X_scaled[train_mask]
        X_val = X_scaled[val_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]

        # Build per-sample, per-horizon weights for imbalance + horizon importance
        n_train = y_train.shape[0]
        n_h = len(self.horizon_names)
        sample_weight = np.ones((n_train, n_h), dtype=np.float32)

        # Default horizon weights: day > week > month
        base_horizon_weights = {h: 1.0 for h in self.horizon_names}
        if "day" in base_horizon_weights:
            base_horizon_weights["day"] = 1.0
        if "week" in base_horizon_weights:
            base_horizon_weights["week"] = 0.8
        if "month" in base_horizon_weights:
            base_horizon_weights["month"] = 0.6

        for i, h in enumerate(self.horizon_names):
            y_i = y_train[:, i]
            pos = y_i.sum()
            neg = len(y_i) - pos
            if pos == 0:
                pos_weight = 1.0
                neg_weight = 1.0
            else:
                pos_weight = max(1.0, neg / (pos + 1e-6))
                neg_weight = 1.0

            horizon_w = base_horizon_weights[h]

            # sample_weight[j,i] = (pos/neg weight) * horizon weight
            sample_weight[:, i] = (y_i * pos_weight + (1.0 - y_i) * neg_weight) * horizon_w

        logger.info(
            "Train/val split for brand %s: meters train=%d, val=%d, sequences train=%d, val=%d",
            group_df["brand_model"].iloc[0],
            len(train_meters),
            len(val_meters),
            len(X_train),
            len(X_val),
        )

        return X_train, X_val, y_train, y_val, sample_weight

    def _split_train_val_by_meter_single_horizon(
        self,
        group_df: pd.DataFrame,
        group_name: str,
        horizon: str,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Train/val split by meters for a SINGLE horizon (e.g. 'day').

        Returns
        -------
        X_train, X_val, y_train, y_val, sample_weight (1D)
        """
        rng = np.random.RandomState(random_state)
        meters = group_df["meter_id"].unique()
        rng.shuffle(meters)

        n_meters = len(meters)
        n_val = max(1, int(n_meters * self.validation_split))
        val_meters = set(meters[:n_val])
        train_meters = set(meters[n_val:]) if n_meters > n_val else set()

        # Fallback: ensure we have at least one train meter if possible
        if len(train_meters) == 0 and n_meters > 1:
            train_meters = set(meters[:-1])
            val_meters = set(meters[-1:])

        train_mask = group_df["meter_id"].isin(train_meters).values
        val_mask = group_df["meter_id"].isin(val_meters).values

        # Stack sequences
        X = np.stack(group_df["sequence"].values).astype(np.float32)
        n_samples, n_timesteps, n_features = X.shape

        # Flatten for scaling
        X_reshaped = X.reshape(-1, n_features)
        X_reshaped = np.nan_to_num(X_reshaped, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit scaler on TRAIN ONLY (no leakage)
        scaler = StandardScaler()
        scaler.fit(X_reshaped[train_mask.repeat(n_timesteps)])

        mean = np.nan_to_num(scaler.mean_, nan=0.0, posinf=0.0, neginf=0.0)
        scale = np.nan_to_num(scaler.scale_, nan=1.0, posinf=1.0, neginf=1.0)
        scale[scale == 0] = 1.0

        X_reshaped = (X_reshaped - mean) / scale

        # Fix scaler stats to the cleaned ones
        scaler.mean_ = mean
        scaler.scale_ = scale

        # Store scaler under the SAME key as the model: "<brand>_<horizon>"
        model_key = f"{group_name}_{horizon}"
        self.scalers[model_key] = scaler

        # Reshape back to 3D
        X_scaled = X_reshaped.reshape(n_samples, n_timesteps, n_features)

        X_train = X_scaled[train_mask]
        X_val = X_scaled[val_mask]

        # Labels for this horizon
        y_all = group_df[f"anomaly_{horizon}"].values.astype(np.float32)
        y_train = y_all[train_mask]
        y_val = y_all[val_mask]

        # Simple class-imbalance weights (1D)
        pos = y_train.sum()
        neg = len(y_train) - pos
        if pos == 0:
            pos_weight = 1.0
            neg_weight = 1.0
        else:
            pos_weight = max(1.0, neg / (pos + 1e-6))
            neg_weight = 1.0

        sample_weight = np.where(y_train == 1.0, pos_weight, neg_weight).astype(np.float32)

        logger.info(
            "Train/val split for brand %s, horizon %s: meters train=%d, val=%d, sequences train=%d, val=%d",
            group_name,
            horizon,
            len(train_meters),
            len(val_meters),
            len(X_train),
            len(X_val),
        )
        logger.info(
            "%s-%s target prevalence (train)=%.2f%%, (val)=%.2f%%",
            group_name,
            horizon,
            100.0 * y_train.mean(),
            100.0 * y_val.mean(),
        )

        return X_train, X_val, y_train, y_val, sample_weight

    def prepare_data_for_training(
        self,
        sequences_df: pd.DataFrame,
        group_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Filter sequences for a brand_model group and prepare X/y + weights.
        """
        if group_name != "all":
            group_df = sequences_df[sequences_df["brand_model"] == group_name].copy()
        else:
            group_df = sequences_df.copy()

        if len(group_df) < 100:
            logger.warning("Insufficient sequences for %s: %d samples", group_name, len(group_df))
            return None, None, None, None, None

        X_train, X_val, y_train, y_val, sample_weight = self._split_train_val_by_meter(group_df)

        logger.info(
            "%s target prevalence (train) - %s",
            group_name,
            ", ".join(
                f"{h}: {y_train[:, i].mean():.2%}"
                for i, h in enumerate(self.horizon_names)
            ),
        )
        logger.info(
            "%s target prevalence (val)   - %s",
            group_name,
            ", ".join(
                f"{h}: {y_val[:, i].mean():.2%}"
                for i, h in enumerate(self.horizon_names)
            ),
        )

        return X_train, X_val, y_train, y_val, sample_weight

    # ------------------ training ------------------

    def train_model(
        self,
        group_name: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray,
    ) -> Dict:
        """
        Train one LSTM model for a specific brand group.
        """
        logger.info("=" * 60)
        logger.info("Training LSTM model for group: %s", group_name)
        logger.info("=" * 60)

        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_model(input_shape, group_name)

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1,
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        )
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=os.path.join(self.run_dir, "models", f"{group_name}_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        )

        history = model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr, model_checkpoint],
            verbose=1,
        )

        self.models[group_name] = model
        self.histories[group_name] = history.history

        # Evaluate
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)

        y_pred_val = model.predict(X_val, verbose=0)

        metrics = {
            "train_loss": float(train_loss[0]),
            "val_loss": float(val_loss[0]),
            "epochs_trained": len(history.history["loss"]),
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        }

        # Per-horizon metrics
        for i, h in enumerate(self.horizon_names):
            y_true = y_val[:, i]
            y_scores = y_pred_val[:, i]
            y_pred = (y_scores > 0.5).astype(int)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(y_true, y_scores)
            except Exception:
                auc = 0.0

            metrics[f"{h}_precision"] = float(precision)
            metrics[f"{h}_recall"] = float(recall)
            metrics[f"{h}_f1"] = float(f1)
            metrics[f"{h}_auc"] = float(auc)

        self.metrics[group_name] = metrics

        logger.info(
            "Training finished for %s. Val loss=%.4f, best_epoch=%d/%d",
            group_name,
            metrics["val_loss"],
            metrics["best_epoch"],
            metrics["epochs_trained"],
        )

        return metrics

    def train_model_single_horizon(
        self,
        group_name: str,
        horizon: str,
        model: tf.keras.Model,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray,
    ) -> Dict:
        logger.info("=" * 60)
        logger.info("Training LSTM model for group=%s, horizon=%s", group_name, horizon)
        logger.info("=" * 60)

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1,
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        )
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=os.path.join(self.run_dir, "models", f"{group_name}_{horizon}_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        )

        history = model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, reduce_lr, model_checkpoint],
            verbose=1,
        )

        key = f"{group_name}_{horizon}"
        self.models[key] = model
        self.histories[key] = history.history

        train_loss = model.evaluate(X_train, y_train, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)

        y_scores = model.predict(X_val, verbose=0).ravel()
        y_pred = (y_scores > 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_val, y_scores)
        except Exception:
            auc = 0.0

        metrics = {
            "train_loss": float(train_loss[0]),
            "val_loss": float(val_loss[0]),
            "epochs_trained": len(history.history["loss"]),
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        }

        self.metrics[key] = metrics

        logger.info(
            "Training finished for %s-%s. Val loss=%.4f, best_epoch=%d/%d",
            group_name,
            horizon,
            metrics["val_loss"],
            metrics["best_epoch"],
            metrics["epochs_trained"],
        )

        return metrics


    # ------------------ plots & reporting ------------------

    def create_training_plots(self, group_name: str):
        """
        Create training plots: loss + metrics + probability histograms on val.
        """
        history = self.histories[group_name]
        metrics = self.metrics[group_name]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"LSTM Training Results - {group_name}", fontsize=16)

        # (0,0) Loss
        axes[0, 0].plot(history["loss"], label="Train Loss", linewidth=2)
        axes[0, 0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0, 0].axvline(
            x=metrics["best_epoch"] - 1,
            color="r",
            linestyle="--",
            label=f"Best Epoch ({metrics['best_epoch']})",
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # (0,1) Accuracy
        if "accuracy" in history and "val_accuracy" in history:
            axes[0, 1].plot(history["accuracy"], label="Train Acc", linewidth=2)
            axes[0, 1].plot(history["val_accuracy"], label="Val Acc", linewidth=2)
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].set_title("Accuracy")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # (0,2) AUC
        if "auc" in history and "val_auc" in history:
            axes[0, 2].plot(history["auc"], label="Train AUC", linewidth=2)
            axes[0, 2].plot(history["val_auc"], label="Val AUC", linewidth=2)
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("AUC")
            axes[0, 2].set_title("AUC")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # (1,0) Precision
        if "precision" in history and "val_precision" in history:
            axes[1, 0].plot(history["precision"], label="Train Prec", linewidth=2)
            axes[1, 0].plot(history["val_precision"], label="Val Prec", linewidth=2)
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Precision")
            axes[1, 0].set_title("Precision")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # (1,1) Recall
        if "recall" in history and "val_recall" in history:
            axes[1, 1].plot(history["recall"], label="Train Recall", linewidth=2)
            axes[1, 1].plot(history["val_recall"], label="Val Recall", linewidth=2)
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Recall")
            axes[1, 1].set_title("Recall")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # (1,2) Bar chart: per-horizon metrics
        horizon_metrics = ["precision", "recall", "f1", "auc"]
        x = np.arange(len(self.horizon_names))
        width = 0.2

        for j, m in enumerate(horizon_metrics):
            vals = [metrics.get(f"{h}_{m}", 0.0) for h in self.horizon_names]
            axes[1, 2].bar(x + j * width, vals, width, label=m.upper())

        axes[1, 2].set_xticks(x + width * (len(horizon_metrics) - 1) / 2)
        axes[1, 2].set_xticklabels(self.horizon_names)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title("Val metrics per horizon")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(self.run_dir, "plots", f"{group_name}_training.png")
        plt.savefig(plot_path)
        plt.close(fig)

    def save_models_and_reports(self, feature_names: List[str]) -> None:
        """
        Save each trained model + scaler + metadata, and a global CSV report.

        New design (after per-horizon refactor):

        - Each entry in self.models has a key like "<brand_model>_<horizon>".
          Example: "unknown_day", "unknown_week", "unknown_month".

        - For each such key we create:

            run_dir/
                models/
                    <key>/
                        model.h5
                        scaler.pkl
                        metadata.json

        - metadata.json includes:
            - model_id       : "<brand_model>_<horizon>"
            - brand_model    : e.g. "unknown"
            - horizon        : e.g. "day"
            - horizon_days   : e.g. 1, 7, 30
            - sequence_length
            - feature_names
            - all_target_horizons : full dict used at training
            - metrics        : metrics for this model only
            - timestamp      : training run timestamp
        """
        self.feature_names = feature_names
        report_rows: List[Dict] = []

        for model_key, model in self.models.items():
            # model_key is something like "<brand_model>_<horizon>"
            brand_model = None
            horizon = None

            # Infer horizon from the suffix based on known horizon names
            for h in self.horizon_names:
                suffix = f"_{h}"
                if model_key.endswith(suffix):
                    horizon = h
                    brand_model = model_key[: -len(suffix)]
                    break

            # Fallback in weird cases
            if brand_model is None or horizon is None:
                logger.warning(
                    "Could not infer brand/horizon from model key '%s'. "
                    "Using full key as brand_model and horizon='unknown'.",
                    model_key,
                )
                brand_model = model_key
                horizon = "unknown"

            group_dir = os.path.join(self.run_dir, "models", model_key)
            os.makedirs(group_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(group_dir, "model.h5")
            model.save(model_path)

            # Save scaler (keyed by the same model_key as in _split_train_val_by_meter_single_horizon)
            scaler = self.scalers.get(model_key)
            if scaler is not None:
                scaler_path = os.path.join(group_dir, "scaler.pkl")
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
            else:
                logger.warning(
                    "No scaler found for model key '%s'. "
                    "Predictions may fail if scaler is required.",
                    model_key,
                )

            # Collect metrics for this model
            metrics = self.metrics.get(model_key, {})

            # Save metadata
            meta = {
                "model_id": model_key,
                "brand_model": brand_model,
                "horizon": horizon,
                "horizon_days": self.target_horizons.get(horizon),
                "sequence_length": self.sequence_length,
                "feature_names": feature_names,
                "all_target_horizons": self.target_horizons,
                "metrics": metrics,
                "timestamp": self.timestamp,
            }
            meta_path = os.path.join(group_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

            # Add row to global training report
            row = {
                "model_id": model_key,
                "brand_model": brand_model,
                "horizon": horizon,
            }
            row.update(metrics)
            report_rows.append(row)

        # Global training report across all brand+horizon models
        if report_rows:
            report_df = pd.DataFrame(report_rows)
            report_path = os.path.join(self.run_dir, "reports", "lstm_training_report.csv")
            report_df.to_csv(report_path, index=False)
        else:
            logger.warning("No models to save in save_models_and_reports; report will be empty.")

        logger.info("LSTM training finished. Models and report stored in %s", self.run_dir)

    # ------------------ public API ------------------

    def run(self, sequences_df: pd.DataFrame, feature_names: List[str]) -> Dict:
        """
        Train one LSTM per (brand_model, horizon).

        Parameters
        ----------
        sequences_df : pd.DataFrame
            Output from prepare_lstm_dataset (output_format='dataframe').
        feature_names : List[str]
            Feature names used inside 'sequence' arrays.

        Returns
        -------
        Dict[group_name][horizon] -> metrics dict
        """
        self.feature_names = feature_names

        groups = sorted(sequences_df["brand_model"].unique())
        logger.info("Training LSTM for %d brand groups: %s", len(groups), groups)

        results: Dict[str, Dict[str, Dict]] = {}

        for group_name in groups:
            group_df = sequences_df[sequences_df["brand_model"] == group_name].copy()
            if len(group_df) < 100:
                logger.warning(
                    "Insufficient sequences for %s: %d samples",
                    group_name,
                    len(group_df),
                )
                continue

            results[group_name] = {}

            for h in self.horizon_names:
                logger.info("=" * 60)
                logger.info("Preparing data for group=%s, horizon=%s", group_name, h)
                logger.info("=" * 60)

                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    sample_weight,
                ) = self._split_train_val_by_meter_single_horizon(
                    group_df, group_name, h
                )

                if X_train is None or len(X_train) == 0:
                    logger.warning("No train data for %s-%s", group_name, h)
                    continue

                # Build a single-output model for this horizon
                input_shape = (X_train.shape[1], X_train.shape[2])
                model = self.build_model(input_shape, n_outputs=1)

                metrics = self.train_model_single_horizon(
                    group_name,
                    h,
                    model,
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    sample_weight,
                )

                results[group_name][h] = metrics

        # Persist all trained models, scalers and metadata to disk
        self.save_models_and_reports(feature_names)

        # Create training plots for each model
        for model_key in self.models.keys():
            try:
                self.create_training_plots(model_key)
            except Exception as e:
                logger.warning("Failed to create LSTM training plot for %s: %s", model_key, e)

        return results

# ---------------------------------------------------------------------------
# 3. PREDICTION PIPELINE
# ---------------------------------------------------------------------------

class LSTMAnomalyPredictionPipeline:
    """
    LSTM prediction pipeline for water consumption anomaly detection.

    Important notes (per-horizon design)
    ------------------------------------
    - Each trained model is stored under a key "<brand_model>_<horizon>"
      (e.g. "unknown_day", "unknown_week", "unknown_month").
    - Each such model has its own scaler and metadata.json.
    - At prediction time we:
        * group sequences by brand_model
        * for each horizon look for a model with id f"{brand_model}_{horizon}"
        * if present, use it to compute probabilities for that horizon
        * aggregate all horizon probabilities per sequence + risk level.
    """

    def __init__(
        self,
        models_dir: str,
        sequence_length: int = 90,
        threshold: Union[float, Dict[str, float]] = 0.5,
        version: str = "latest",
    ):
        """
        Parameters
        ----------
        models_dir : str
            Root directory containing trained LSTM runs (the parent of
            'lstm_run_YYYYMMDD_HHMMSS' folders or one such folder itself).
        sequence_length : int
            Sequence length (must match training).
        threshold : float or dict
            - If float: same threshold for all horizons.
            - If dict: mapping {horizon_name: threshold}.
        version : str
            'latest' or specific run folder name.
        """
        self.models_dir = models_dir
        self.sequence_length = sequence_length
        self.version = version

        # Thresholds per horizon will be filled when metadata is loaded.
        # For now we store raw input and resolve later.
        self.global_threshold = threshold
        self.horizon_thresholds: Dict[str, float] = {}

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("./lstm_predictions", f"pred_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)

        self.models: Dict[str, tf.keras.Model] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metadata: Dict[str, Dict] = {}

        self._load_models()

    def _resolve_thresholds_from_meta(self) -> None:
        """
        Use training metadata & global config to set per-horizon thresholds.

        Logic:
        - If global_threshold is a dict, use it directly.
        - Else, apply the same scalar to all horizons discovered in metadata.
        """
        if isinstance(self.global_threshold, dict):
            self.horizon_thresholds = {
                str(k): float(v) for k, v in self.global_threshold.items()
            }
        else:
            # Discover horizon names from metadata
            horizon_names = set()
            for meta in self.metadata.values():
                # Explicit horizon for this model (e.g. "day")
                h = meta.get("horizon")
                if h is not None:
                    horizon_names.add(str(h))

                # Full horizons dict used at training time
                horizons_dict = meta.get("all_target_horizons") or meta.get("target_horizons")
                if isinstance(horizons_dict, dict):
                    for key in horizons_dict.keys():
                        horizon_names.add(str(key))

            if not horizon_names:
                # Fallback to the standard three horizons if metadata is missing
                horizon_names = {"day", "week", "month"}

            self.horizon_thresholds = {
                h: float(self.global_threshold) for h in sorted(horizon_names)
            }

        logger.info("Prediction thresholds per horizon: %s", self.horizon_thresholds)

    def _load_models(self) -> None:
        """
        Load all models, their scalers and metadata.

        Expected directory structure (from training pipeline):

            <models_dir>/
                lstm_run_YYYYMMDD_HHMMSS/   # or models_dir IS this path
                    models/
                        <brand>_<horizon>/
                            model.h5
                            scaler.pkl
                            metadata.json
        """
        # Resolve the actual run folder to use
        if self.version == "latest":
            base_name = os.path.basename(self.models_dir)
            if base_name.startswith("lstm_run_"):
                runs_path = self.models_dir
            else:
                runs = [
                    d for d in os.listdir(self.models_dir)
                    if d.startswith("lstm_run_") and os.path.isdir(os.path.join(self.models_dir, d))
                ]
                if not runs:
                    raise ValueError(f"No LSTM runs found in {self.models_dir}")
                latest_run = sorted(runs)[-1]
                runs_path = os.path.join(self.models_dir, latest_run)
        else:
            base_name = os.path.basename(self.models_dir)
            if base_name.startswith("lstm_run_"):
                runs_path = self.models_dir
            else:
                runs_path = os.path.join(self.models_dir, self.version)

        models_path = os.path.join(runs_path, "models")
        logger.info("Loading LSTM models from %s", models_path)

        if not os.path.isdir(models_path):
            raise ValueError(f"Models path does not exist or is not a directory: {models_path}")

        for model_key in os.listdir(models_path):
            full_group_path = os.path.join(models_path, model_key)
            if not os.path.isdir(full_group_path):
                continue

            model_file = os.path.join(full_group_path, "model.h5")
            scaler_file = os.path.join(full_group_path, "scaler.pkl")
            metadata_file = os.path.join(full_group_path, "metadata.json")

            if not os.path.exists(model_file):
                continue

            model = tf.keras.models.load_model(model_file)

            # Scaler is required for correct feature scaling
            if not os.path.exists(scaler_file):
                logger.warning(
                    "Scaler file missing for model '%s'. Predictions may be incorrect.",
                    model_key,
                )
                scaler = None
            else:
                with open(scaler_file, "rb") as f:
                    scaler = pickle.load(f)

            # Metadata (feature names, horizons, metrics...)
            if not os.path.exists(metadata_file):
                meta = {}
                logger.warning(
                    "Metadata file missing for model '%s'. Some functionality may be limited.",
                    model_key,
                )
            else:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)

            self.models[model_key] = model
            if scaler is not None:
                self.scalers[model_key] = scaler
            self.metadata[model_key] = meta

            logger.info("Loaded LSTM model '%s'", model_key)

        if not self.models:
            raise ValueError("No models could be loaded for prediction.")

        self._resolve_thresholds_from_meta()

    def _get_risk_level(self, probs: Dict[str, float]) -> str:
        """
        Map horizon probabilities to a qualitative risk level.

        Simple, interpretable rule:
        - Use the maximum probability across horizons.
        - Thresholds:
            [0, t_low)      -> 'LOW'
            [t_low, t_med)  -> 'MEDIUM'
            [t_med, t_high) -> 'HIGH'
            >= t_high       -> 'CRITICAL'
        - Base thresholds derived from the per-horizon classification thresholds.
        """
        # Base thresholds from config threshold
        base_thr = float(np.mean(list(self.horizon_thresholds.values()))) if self.horizon_thresholds else 0.5
        t_low = 0.5 * base_thr
        t_med = base_thr
        t_high = min(1.0, base_thr + 0.25)

        p_max = max(probs.values()) if probs else 0.0

        if p_max >= t_high:
            return "CRITICAL"
        elif p_max >= t_med:
            return "HIGH"
        elif p_max >= t_low:
            return "MEDIUM"
        else:
            return "LOW"

    def _create_prediction_plots(self, pred_df: pd.DataFrame) -> None:
        plots_dir = os.path.join(self.output_dir, "plots")

        # 1) Risk level distribution
        risk_counts = pred_df["risk_level"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        risk_counts.plot(kind="bar", ax=ax)
        ax.set_title("Risk level distribution")
        ax.set_xlabel("Risk level")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "risk_level_distribution.png"))
        plt.close(fig)

        # 2) Probability histograms per horizon
        for h in self.horizon_thresholds.keys():
            col = f"prob_anomaly_{h}"
            if col not in pred_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            pred_df[col].hist(bins=30, ax=ax)
            ax.set_title(f"Predicted anomaly probability ({h})")
            ax.set_xlabel("Probability")
            ax.set_ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"prob_anomaly_{h}_hist.png"))
            plt.close(fig)

    def predict(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run prediction on sequences_df (output of prepare_lstm_dataset).

        For each brand_model and each horizon:
          - look for a model with id "<brand_model>_<horizon>"
          - if present, use it to compute probabilities for that horizon
          - aggregate probabilities across horizons for each sequence.
        """
        if not len(sequences_df):
            logger.warning("Empty sequences dataframe passed to LSTM prediction.")
            return sequences_df.copy()

        all_rows: List[Dict] = []

        for brand_model, group_df in sequences_df.groupby("brand_model"):
            group_df = group_df.reset_index(drop=True)

            # Find all horizon-specific models available for this brand
            horizon_model_keys = {}
            for h in self.horizon_thresholds.keys():
                key = f"{brand_model}_{h}"
                if key in self.models:
                    horizon_model_keys[h] = key

            if not horizon_model_keys:
                logger.warning(
                    "No LSTM models found for brand '%s'. Skipping these sequences.",
                    brand_model,
                )
                continue

            # Stack sequences once (we'll reuse for each horizon model)
            sequences = np.stack(group_df["sequence"].values).astype(np.float32)
            n_samples, n_timesteps, n_features = sequences.shape

            if n_timesteps != self.sequence_length:
                logger.warning(
                    "Sequence length mismatch for brand '%s': expected=%d, got=%d. "
                    "Predictions may be invalid.",
                    brand_model,
                    self.sequence_length,
                    n_timesteps,
                )

            # Store probabilities per horizon: dict[horizon] -> np.array of shape (n_samples,)
            probs_by_horizon: Dict[str, np.ndarray] = {}

            for h, model_key in horizon_model_keys.items():
                model = self.models[model_key]
                scaler = self.scalers.get(model_key)

                meta = self.metadata.get(model_key, {})
                feature_names = meta.get("feature_names")

                if feature_names is not None and len(feature_names) != n_features:
                    logger.warning(
                        "Feature count mismatch for model '%s' (horizon=%s): "
                        "metadata=%d, sequence=%d.",
                        model_key,
                        h,
                        len(feature_names),
                        n_features,
                    )

                # Scale using the stored scaler (if present)
                seq_reshaped = sequences.reshape(-1, n_features)
                seq_reshaped = np.nan_to_num(seq_reshaped, nan=0.0, posinf=0.0, neginf=0.0)

                if scaler is not None:
                    seq_scaled_flat = scaler.transform(seq_reshaped)
                else:
                    seq_scaled_flat = seq_reshaped  # fallback: no scaling

                seq_scaled = seq_scaled_flat.reshape(n_samples, n_timesteps, n_features)

                # Model outputs a single probability per sequence
                y_pred = model.predict(seq_scaled, verbose=0).ravel()
                probs_by_horizon[h] = y_pred

            # Build per-sequence records
            for i, (_, row) in enumerate(group_df.iterrows()):
                # Collect probabilities for this sequence across horizons
                seq_probs = {
                    h: float(probs_by_horizon[h][i]) for h in probs_by_horizon.keys()
                }

                # Binary predictions per horizon
                labels = {
                    f"pred_anomaly_{h}": int(
                        seq_probs[h] >= self.horizon_thresholds.get(h, 0.5)
                    )
                    for h in seq_probs.keys()
                }

                risk = self._get_risk_level(seq_probs) if seq_probs else "LOW"

                record: Dict = {
                    "meter_id": row["meter_id"],
                    "brand_model": brand_model,
                    "start_date": row["start_date"],
                    "end_date": row["end_date"],
                    "risk_level": risk,
                }

                # Attach probs and labels
                for h, p in seq_probs.items():
                    record[f"prob_anomaly_{h}"] = p
                    record[f"pred_anomaly_{h}"] = labels[f"pred_anomaly_{h}"]

                all_rows.append(record)

        if not all_rows:
            logger.warning("No predictions generated; check that groups and models exist.")
            return pd.DataFrame()

        pred_df = pd.DataFrame(all_rows)

        # Save report
        report_path = os.path.join(self.output_dir, "reports", "lstm_predictions.csv")
        pred_df.to_csv(report_path, index=False)

        logger.info("LSTM predictions saved to %s", report_path)

        # Create prediction plots
        self._create_prediction_plots(pred_df)

        return pred_df

