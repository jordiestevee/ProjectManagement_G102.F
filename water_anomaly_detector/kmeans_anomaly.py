import os
import warnings
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Force non-interactive backend for safety in batch environments
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import MiniBatchKMeans, KMeans  # KMeans kept just in case
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.spatial.distance import euclidean

import pickle
import json


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# KMeans anomaly detection pipeline
# ---------------------------------------------------------------------------

class KMeansAnomalyDetectionPipeline:
    """
    K-Means pipeline for anomaly detection with support for:
      - Grouping by brand (model as a feature, not a group key)
      - Explicit selection of feature columns for clustering
      - Scalable K optimization with subsampling
      - Vectorised distance computation (Euclidean / Mahalanobis)
      - Quantile-based anomaly thresholds
      - Inference / scoring on new data via score_df
    """

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 10,
        distance_metric: str = "mahalanobis",
        random_state: int = 42,
        output_dir: str = "./kmeans_output",
        train_sample_fraction: Optional[float] = None,
        train_sample_cap: Optional[int] = None,
        k_opt_sample_size: Optional[int] = 10_000,
        silhouette_sample_size: Optional[int] = 5_000,
        anomaly_tail_fraction: float = 0.025,
        std_threshold: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        min_clusters, max_clusters
            Min / max number of clusters to consider per group. If they are equal,
            no K optimization is performed and that fixed k is used.
        distance_metric
            'euclidean' or 'mahalanobis'.
        random_state
            Random seed for reproducibility.
        output_dir
            Base directory where models, plots and data will be stored.
        train_sample_fraction
            Optional fraction of rows per group to use for fitting K-means.
        train_sample_cap
            Optional hard cap on number of rows per group to use for fitting.
        k_opt_sample_size
            Max number of samples per group to use during K search.
        silhouette_sample_size
            Max number of samples to use inside silhouette_score.
        anomaly_tail_fraction
            Fraction of farthest points in each cluster to mark as anomalies
            (e.g. 0.025 => top 2.5% farthest distances).
        std_threshold
            Optional multiplier for a std-based distance threshold:
            mean_distance + std_threshold * std_distance.
            If provided, it is combined with anomaly_tail_fraction to form
            a more conservative anomaly cut-off.
        """
        self.min_clusters = int(min_clusters)
        self.max_clusters = int(max_clusters)
        self.distance_metric = distance_metric.lower()
        self.random_state = int(random_state)
        self.output_dir = output_dir
        self.train_sample_fraction = train_sample_fraction
        self.train_sample_cap = train_sample_cap
        self.k_opt_sample_size = k_opt_sample_size
        self.silhouette_sample_size = silhouette_sample_size

        self.anomaly_tail_fraction = float(anomaly_tail_fraction) if anomaly_tail_fraction is not None else 0.0
        self.std_threshold = float(std_threshold) if std_threshold is not None else None

        # Will be filled during run()
        self.timestamp: Optional[str] = None
        self.run_dir: Optional[str] = None
        self.models_dir: Optional[str] = None
        self.plots_dir: Optional[str] = None
        self.data_dir: Optional[str] = None

        self.models: Dict[str, Dict] = {}  # group_name -> model info
        self.metadata: Dict = {}

        if self.distance_metric not in {"euclidean", "mahalanobis"}:
            raise ValueError("distance_metric must be 'euclidean' or 'mahalanobis'")

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _infer_brand_column(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Build a 'BRAND' column from MARCA_COMP_* one-hot columns.
        If no brand columns are found, all rows are assigned BRAND='unknown'.
        """
        brand_cols = [c for c in df.columns if c.startswith("MARCA_COMP_")]
        df = df.copy()

        if not brand_cols:
            logger.warning("No MARCA_COMP_* columns found; using BRAND='unknown' for all rows.")
            df["BRAND"] = "unknown"
            return df, "BRAND"

        df["BRAND"] = "unknown"
        for col in brand_cols:
            brand_name = col.replace("MARCA_COMP_", "")
            mask = df[col] == True  # noqa: E712
            df.loc[mask, "BRAND"] = brand_name

        return df, "BRAND"

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[Dict[str, pd.DataFrame], List[str], List[Tuple[str, int]]]:
        """
        Prepare data for k-means clustering.

        - Enforces the use of an explicit list of feature columns (no more
          "all numeric columns" auto-selection).
        - Builds a BRAND grouping column (model is expected to be in features).

        Parameters
        ----------
        df
            Feature-engineered dataframe.
        feature_cols
            Columns to be used for clustering (must be present in df).

        Returns
        -------
        grouped_data
            Dict: brand -> dataframe for that brand.
        feature_cols_used
            Sanitised list of feature columns actually used.
        skipped_groups
            List of (brand, size) groups that were skipped due to insufficient data.
        """
        id_cols = ["POLIZA_SUMINISTRO", "FECHA", "CONSUMO_REAL"]

        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"The following feature_cols are missing in df: {missing_features}")

        df, brand_col = self._infer_brand_column(df)

        # Ensure identifiers and features are not overlapping
        feature_cols_used = [c for c in feature_cols if c not in id_cols and c != brand_col]

        grouped_data: Dict[str, pd.DataFrame] = {}
        skipped_groups: List[Tuple[str, int]] = []

        for brand, group in df.groupby(brand_col):
            n = len(group)
            if n >= self.min_clusters * 30:
                grouped_data[brand] = group.copy()
                logger.info(f"Group {brand}: {n:,} records - included for clustering.")
            else:
                skipped_groups.append((str(brand), int(n)))
                logger.warning(
                    f"Skipping brand={brand}: not enough records ({n}) "
                    f"for min_clusters={self.min_clusters}"
                )

        if not grouped_data:
            logger.error("No groups met the minimum size requirement for clustering.")
            raise RuntimeError("No groups to cluster.")

        return grouped_data, feature_cols_used, skipped_groups

    # ------------------------------------------------------------------
    # K optimization
    # ------------------------------------------------------------------

    def _subsample_for_k_opt(self, X: np.ndarray) -> np.ndarray:
        """Return index array for subsampling X during K search."""
        n = X.shape[0]
        if self.k_opt_sample_size is None or n <= self.k_opt_sample_size:
            return np.arange(n)
        rng = np.random.default_rng(self.random_state)
        return rng.choice(n, size=self.k_opt_sample_size, replace=False)

    def _compute_k_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        kmeans_model: MiniBatchKMeans,
        sample_size_for_silhouette: Optional[int],
    ) -> Tuple[float, float, float, float]:
        """
        Compute clustering metrics on X / labels:

        - silhouette_score (higher is better)
        - davies_bouldin_score (lower is better)
        - calinski_harabasz_score (higher is better)
        - inertia (lower is better)
        """
        # Silhouette can be extremely expensive; use sample_size if provided.
        try:
            if sample_size_for_silhouette is not None:
                sample_size = min(sample_size_for_silhouette, X.shape[0])
            else:
                sample_size = None

            sil = silhouette_score(
                X,
                labels,
                metric="euclidean",  # silhouette is defined in Euclidean space
                sample_size=sample_size,
                random_state=self.random_state,
            )
        except Exception as exc:
            logger.warning(f"Failed to compute silhouette_score: {exc}")
            sil = np.nan

        try:
            db = davies_bouldin_score(X, labels)
        except Exception as exc:
            logger.warning(f"Failed to compute davies_bouldin_score: {exc}")
            db = np.nan

        try:
            ch = calinski_harabasz_score(X, labels)
        except Exception as exc:
            logger.warning(f"Failed to compute calinski_harabasz_score: {exc}")
            ch = np.nan

        inertia = float(getattr(kmeans_model, "inertia_", np.nan))

        return sil, db, ch, inertia

    def optimize_clusters(
        self,
        X_scaled: np.ndarray,
        group_name: str,
    ) -> Dict:
        """
        Find a good number of clusters k for a given group using subsampled
        K search and multiple evaluation metrics.

        Returns a dictionary with:
          - 'k_values', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'inertia'
          - 'best_k'
          - 'optimization_performed' flag
        """
        min_k = self.min_clusters
        max_k = self.max_clusters

        metrics = {
            "k_values": [],
            "silhouette": [],
            "davies_bouldin": [],
            "calinski_harabasz": [],
            "inertia": [],
            "best_k": min_k,
            "optimization_performed": False,
        }

        # If fixed k, skip optimization
        if min_k == max_k:
            logger.info(f"{group_name}: Using fixed k={min_k} (no optimization).")
            metrics["best_k"] = min_k
            return metrics

        n_samples = X_scaled.shape[0]
        max_k_by_size = max(min(max_k, n_samples // 30), min_k)
        if max_k_by_size < min_k:
            logger.warning(
                f"{group_name}: Not enough samples ({n_samples}) to explore k in "
                f"[{min_k}, {max_k}]; falling back to k={min_k}."
            )
            metrics["best_k"] = min_k
            return metrics

        k_range = list(range(min_k, max_k_by_size + 1))
        logger.info(f"{group_name}: Optimizing k over {k_range}.")

        # Subsample for K optimization
        idx_sub = self._subsample_for_k_opt(X_scaled)
        X_sub = X_scaled[idx_sub]

        start_time = time.time()
        metrics["optimization_performed"] = True

        for idx, k in enumerate(k_range, start=1):
            iter_start = time.time()
            logger.info(f"{group_name}: [k={k}] Fitting MiniBatchKMeans on {X_sub.shape[0]:,} samples...")

            kmeans = MiniBatchKMeans(
                n_clusters=k,
                init="k-means++",
                random_state=self.random_state,
                batch_size=2048,
                n_init=3,
            )
            labels = kmeans.fit_predict(X_sub)

            sil, db, ch, inertia = self._compute_k_metrics(
                X_sub, labels, kmeans, self.silhouette_sample_size
            )

            metrics["k_values"].append(k)
            metrics["silhouette"].append(sil)
            metrics["davies_bouldin"].append(db)
            metrics["calinski_harabasz"].append(ch)
            metrics["inertia"].append(inertia)

            iter_time = time.time() - iter_start
            elapsed_time = time.time() - start_time
            avg_time_per_k = elapsed_time / idx
            remaining_k = len(k_range) - idx
            est_remaining = avg_time_per_k * remaining_k

            logger.info(
                f"{group_name}: [k={k}] Metrics: Sil={sil:.4f}, DB={db:.4f}, "
                f"CH={ch:.2f}, Inertia={inertia:.1f} "
                f"(time {iter_time:.1f}s, est. remaining {est_remaining:.1f}s)"
            )

        # Choose best k via combined normalized score
        best_k = self._select_best_k_from_metrics(metrics, group_name)
        metrics["best_k"] = best_k
        logger.info(f"{group_name}: Selected best_k={best_k} based on combined metrics.")

        return metrics

    def _select_best_k_from_metrics(self, metrics: Dict, group_name: str) -> int:
        """
        Combine silhouette, Davies–Bouldin, Calinski–Harabasz and inertia into
        a single normalized score and pick the best k. Favour smaller k when
        scores are very close.
        """
        ks = np.array(metrics["k_values"], dtype=float)
        sil = np.array(metrics["silhouette"], dtype=float)
        db = np.array(metrics["davies_bouldin"], dtype=float)
        ch = np.array(metrics["calinski_harabasz"], dtype=float)
        inertia = np.array(metrics["inertia"], dtype=float)

        if len(ks) == 0:
            logger.warning(f"{group_name}: No K metrics available, defaulting to min_clusters={self.min_clusters}.")
            return self.min_clusters

        eps = 1e-8

        def norm_pos(x: np.ndarray) -> np.ndarray:
            # Normalize so that larger values are better => [0,1]
            mask = ~np.isnan(x)
            if not np.any(mask):
                return np.zeros_like(x)
            x_min, x_max = x[mask].min(), x[mask].max()
            if x_max - x_min < eps:
                return np.ones_like(x) * 0.5
            out = np.zeros_like(x)
            out[mask] = (x[mask] - x_min) / (x_max - x_min + eps)
            return out

        def norm_neg(x: np.ndarray) -> np.ndarray:
            # For metrics where smaller is better: invert sign before norm_pos
            return norm_pos(-x)

        sil_n = norm_pos(sil)          # higher better
        ch_n = norm_pos(ch)            # higher better
        db_n = norm_neg(db)            # lower better
        inertia_n = norm_neg(inertia)  # lower better

        # Weights: silhouette most important
        w_sil, w_ch, w_db, w_in = 0.4, 0.3, 0.2, 0.1
        score = w_sil * sil_n + w_ch * ch_n + w_db * db_n + w_in * inertia_n

        best_idx = int(np.nanargmax(score))
        best_k = int(ks[best_idx])

        # Prefer smaller k if scores are very close
        best_score = score[best_idx]
        close_mask = score >= best_score - 0.02  # within 0.02 of best score
        if np.any(close_mask):
            candidate_ks = ks[close_mask]
            best_k = int(candidate_ks.min())

        logger.info(
            f"{group_name}: K selection scores:\n"
            f"  k={ks.tolist()}\n"
            f"  combined_score={score.round(3).tolist()}\n"
            f"  chosen_k={best_k}"
        )
        return best_k

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def _compute_inv_cov_matrices(
        self, X_scaled: np.ndarray, labels: np.ndarray, n_clusters: int
    ) -> Dict[int, np.ndarray]:
        """
        Compute per-cluster inverse covariance matrices for Mahalanobis distance.
        """
        inv_covs: Dict[int, np.ndarray] = {}
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_points = X_scaled[mask]
            if cluster_points.shape[0] <= 1:
                continue
            cov = np.cov(cluster_points, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6
            try:
                inv_covs[cluster_id] = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                logger.warning(f"Cluster {cluster_id}: covariance matrix singular; skipping Mahalanobis.")
        return inv_covs

    def calculate_distances(
        self,
        X_scaled: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        inv_covs: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Vectorised computation of distances from each point to its cluster centroid.
        """
        n_samples = X_scaled.shape[0]
        distances = np.zeros(n_samples, dtype=float)
        n_clusters = centroids.shape[0]

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                continue
            cluster_points = X_scaled[mask]
            centroid = centroids[cluster_id]

            if self.distance_metric == "euclidean" or inv_covs is None:
                diff = cluster_points - centroid
                d = np.linalg.norm(diff, axis=1)
                distances[mask] = d
            else:
                inv_cov = inv_covs.get(cluster_id)
                if inv_cov is None or cluster_points.shape[0] <= 1:
                    # Fallback to Euclidean
                    diff = cluster_points - centroid
                    d = np.linalg.norm(diff, axis=1)
                    distances[mask] = d
                else:
                    delta = cluster_points - centroid
                    # Mahalanobis distance in vectorised form
                    d_sq = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
                    d = np.sqrt(np.maximum(d_sq, 0.0))
                    distances[mask] = d

        return distances

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        group_name: str,
        group_df: pd.DataFrame,
        X_scaled: np.ndarray,
        labels: np.ndarray,
        kmeans_model: MiniBatchKMeans,
    ) -> Tuple[pd.DataFrame, Dict, np.ndarray]:
        """
        Mark anomalies using a combination of:
          - a quantile-based tail threshold (anomaly_tail_fraction), and
          - a std-based threshold: mean + std_threshold * std.

        The final threshold per cluster is the *maximum* of the valid
        candidates (i.e. the more conservative of the two).
        """
        n_clusters = kmeans_model.n_clusters

        # Precompute inverse covariances for Mahalanobis (optional)
        inv_covs = None
        if self.distance_metric == "mahalanobis":
            inv_covs = self._compute_inv_cov_matrices(X_scaled, labels, n_clusters)

        distances = self.calculate_distances(
            X_scaled, labels, kmeans_model.cluster_centers_, inv_covs
        )

        thresholds: Dict[int, Dict] = {}
        is_anomaly = np.zeros_like(labels, dtype=bool)

        tail = float(self.anomaly_tail_fraction) if self.anomaly_tail_fraction is not None else 0.0
        use_tail = tail > 0.0
        use_std = self.std_threshold is not None and self.std_threshold > 0.0

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_size = int(mask.sum())
            if cluster_size == 0:
                continue

            cluster_distances = distances[mask]
            mean_dist = float(np.mean(cluster_distances))
            std_dist = float(np.std(cluster_distances))

            quantile_thr = None
            if use_tail and cluster_size > 1:
                q = max(0.0, min(1.0, 1.0 - tail))
                quantile_thr = float(np.quantile(cluster_distances, q))

            std_thr = None
            if use_std and std_dist > 0.0:
                std_thr = mean_dist + float(self.std_threshold) * std_dist

            # Combine thresholds: pick the most conservative (largest)
            candidates = [
                t for t in (quantile_thr, std_thr)
                if t is not None and np.isfinite(t)
            ]
            if candidates:
                threshold = float(max(candidates))
            else:
                # No meaningful threshold => mark no anomalies in this cluster
                threshold = float("inf")

            cluster_anomaly_mask = cluster_distances >= threshold
            anomaly_count = int(cluster_anomaly_mask.sum())
            anomaly_rate = anomaly_count / cluster_size if cluster_size > 0 else 0.0

            is_anomaly[mask] = cluster_anomaly_mask

            thresholds[int(cluster_id)] = {
                "mean_distance": mean_dist,
                "std_distance": std_dist,
                "threshold": threshold,
                "quantile_threshold": quantile_thr,
                "std_based_threshold": std_thr,
                "cluster_size": cluster_size,
                "anomalies": anomaly_count,
                "anomaly_rate": anomaly_rate,
            }

            logger.info(
                f"{group_name}: Cluster {cluster_id}: size={cluster_size:,}, "
                f"mean={mean_dist:.4f}, std={std_dist:.4f}, "
                f"thr={threshold:.4f} (q_thr={quantile_thr}, std_thr={std_thr}), "
                f"anomalies={anomaly_count} ({anomaly_rate:.2%})"
            )

        group_df = group_df.copy()
        group_df["cluster"] = labels
        group_df["distance"] = distances
        group_df["is_anomaly"] = is_anomaly

        # Simple anomaly_score: distance / threshold of cluster (capped)
        anomaly_scores = np.zeros_like(distances)
        for cluster_id, info in thresholds.items():
            mask = labels == cluster_id
            thr = info["threshold"]
            if thr is None or thr <= 0 or not np.any(mask):
                continue
            anomaly_scores[mask] = np.clip(distances[mask] / thr, 0.0, 10.0)
        group_df["anomaly_score"] = anomaly_scores

        return group_df, thresholds, distances

    # ------------------------------------------------------------------
    # Training per group
    # ------------------------------------------------------------------

    def _sample_for_training(self, X: np.ndarray) -> np.ndarray:
        """Return index array of rows to use for K-means fitting."""
        n = X.shape[0]
        n_sample = n

        if self.train_sample_fraction is not None:
            n_sample = max(1, int(n * float(self.train_sample_fraction)))

        if self.train_sample_cap is not None:
            n_sample = min(n_sample, int(self.train_sample_cap))

        n_sample = min(n_sample, n)

        if n_sample == n:
            return np.arange(n)

        rng = np.random.default_rng(self.random_state)
        return rng.choice(n, size=n_sample, replace=False)

    def train_model(
        self,
        group_name: str,
        group_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict:
        """
        Train K-means for a given brand group and detect anomalies.

        Returns
        -------
        model_info : dict
            {
                'model': fitted MiniBatchKMeans,
                'scaler': fitted RobustScaler,
                'feature_cols': feature_cols,
                'metrics': final_metrics,
                'thresholds': thresholds,
                'inv_covs_path': optional path to pickled inv_covs (if mahalanobis),
                'labeled_data': group_df_with_labels
            }
        """
        logger.info(f"{group_name}: Starting training on {len(group_df):,} rows.")

        X = group_df[feature_cols].to_numpy(dtype=float)
        train_idx = self._sample_for_training(X)
        X_train = X[train_idx]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_scaled = scaler.transform(X)

        # K optimization on training subset
        metrics = self.optimize_clusters(X_train_scaled, group_name)
        best_k = int(metrics["best_k"])

        logger.info(f"{group_name}: Fitting final MiniBatchKMeans with k={best_k}.")
        kmeans = MiniBatchKMeans(
            n_clusters=best_k,
            init="k-means++",
            random_state=self.random_state,
            batch_size=2048,
            n_init=5,
        )
        kmeans.fit(X_train_scaled)

        labels = kmeans.predict(X_scaled)

        # Compute final metrics on a (possibly) larger sample of full data
        final_sil, final_db, final_ch, final_inertia = self._compute_k_metrics(
            X_scaled, labels, kmeans, self.silhouette_sample_size
        )
        final_metrics = {
            "n_samples": int(X_scaled.shape[0]),
            "n_features": int(X_scaled.shape[1]),
            "k": best_k,
            "silhouette_score": float(final_sil),
            "davies_bouldin_score": float(final_db),
            "calinski_harabasz_score": float(final_ch),
            "inertia": float(final_inertia),
        }

        group_df_labeled, thresholds, distances = self.detect_anomalies(
            group_name, group_df, X_scaled, labels, kmeans
        )

        # If Mahalanobis, precompute and save inverse covariances for reuse in scoring
        inv_covs_path = None
        if self.distance_metric == "mahalanobis":
            inv_covs = self._compute_inv_cov_matrices(X_scaled, labels, best_k)
            inv_covs_path = os.path.join(self.models_dir, f"{group_name}_inv_covs.pkl")
            with open(inv_covs_path, "wb") as f:
                pickle.dump(inv_covs, f)

        # Save model & scaler
        model_path = os.path.join(self.models_dir, f"{group_name}_model.pkl")
        scaler_path = os.path.join(self.models_dir, f"{group_name}_scaler.pkl")
        thresholds_path = os.path.join(self.models_dir, f"{group_name}_thresholds.json")

        with open(model_path, "wb") as f:
            pickle.dump(kmeans, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)

        logger.info(
            f"{group_name}: Saved model -> {model_path}, scaler -> {scaler_path}, "
            f"thresholds -> {thresholds_path}, inv_covs -> {inv_covs_path}"
        )

        # Create plots
        self.create_plots(group_name, metrics, thresholds, group_df_labeled)

        model_info = {
            "model": kmeans,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "metrics": final_metrics,
            "thresholds": thresholds,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "thresholds_path": thresholds_path,
            "inv_covs_path": inv_covs_path,
        }

        self.models[group_name] = model_info

        return {
            "model_info": model_info,
            "labeled_data": group_df_labeled,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def create_plots(
        self,
        group_name: str,
        optimization_metrics: Dict,
        thresholds: Dict[int, Dict],
        group_df_labeled: pd.DataFrame,
    ) -> None:
        """
        Generate diagnostic plots for a brand group:
          - K optimization curves (if optimization was performed)
          - Cluster size / anomaly count bar plots
          - Distance distributions per cluster
          - Time series sample with anomalies highlighted
        """
        plots_dir = self.plots_dir

        # 1) K optimization curves
        if optimization_metrics.get("optimization_performed") and optimization_metrics.get("k_values"):
            ks = optimization_metrics["k_values"]
            sil = optimization_metrics["silhouette"]
            db = optimization_metrics["davies_bouldin"]
            ch = optimization_metrics["calinski_harabasz"]
            inertia = optimization_metrics["inertia"]
            best_k = optimization_metrics.get("best_k")

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.ravel()

            axs[0].plot(ks, sil, marker="o")
            axs[0].set_title("Silhouette score vs k")
            axs[0].set_xlabel("k")
            axs[0].set_ylabel("Silhouette")
            if best_k is not None:
                axs[0].axvline(best_k, color="r", linestyle="--", label=f"best k={best_k}")
                axs[0].legend()

            axs[1].plot(ks, db, marker="o")
            axs[1].set_title("Davies-Bouldin score vs k (lower better)")
            axs[1].set_xlabel("k")
            axs[1].set_ylabel("DB score")
            if best_k is not None:
                axs[1].axvline(best_k, color="r", linestyle="--")

            axs[2].plot(ks, ch, marker="o")
            axs[2].set_title("Calinski-Harabasz score vs k")
            axs[2].set_xlabel("k")
            axs[2].set_ylabel("CH score")
            if best_k is not None:
                axs[2].axvline(best_k, color="r", linestyle="--")

            axs[3].plot(ks, inertia, marker="o")
            axs[3].set_title("Inertia vs k (lower better)")
            axs[3].set_xlabel("k")
            axs[3].set_ylabel("Inertia")
            if best_k is not None:
                axs[3].axvline(best_k, color="r", linestyle="--")

            plt.tight_layout()
            path = os.path.join(plots_dir, f"{group_name}_k_optimization.png")
            plt.savefig(path)
            plt.close(fig)
            logger.info(f"{group_name}: Saved K optimization plot -> {path}")

        # 2) Cluster sizes and anomaly counts
        cluster_info = pd.DataFrame.from_dict(thresholds, orient="index")
        cluster_info.index.name = "cluster_id"
        cluster_info.reset_index(inplace=True)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=cluster_info, x="cluster_id", y="cluster_size", ax=ax1)
        ax1.set_ylabel("Cluster size")
        ax1.set_xlabel("Cluster")

        ax2 = ax1.twinx()
        sns.lineplot(
            data=cluster_info,
            x="cluster_id",
            y="anomalies",
            marker="o",
            color="red",
            ax=ax2,
        )
        ax2.set_ylabel("Anomaly count")
        ax1.set_title(f"{group_name}: Cluster sizes and anomalies")

        plt.tight_layout()
        path = os.path.join(plots_dir, f"{group_name}_cluster_distribution.png")
        plt.savefig(path)
        plt.close(fig)
        logger.info(f"{group_name}: Saved cluster distribution plot -> {path}")

        # 3) Distance distributions per cluster
        # Subsample to keep plots readable
        df_plot = group_df_labeled.copy()
        max_points = 50_000
        if len(df_plot) > max_points:
            df_plot = df_plot.sample(max_points, random_state=self.random_state)

        clusters = sorted(df_plot["cluster"].unique())
        n_clusters = len(clusters)
        ncols = min(3, n_clusters)
        nrows = int(np.ceil(n_clusters / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

        for ax, cluster_id in zip(axes.ravel(), clusters):
            subset = df_plot[df_plot["cluster"] == cluster_id]
            thr = thresholds[int(cluster_id)]["threshold"]

            sns.histplot(subset["distance"], bins=30, kde=True, ax=ax)
            ax.axvline(thr, color="red", linestyle="--", label="threshold")
            ax.set_title(f"Cluster {cluster_id}")
            ax.set_xlabel("Distance")
            ax.legend()

        # Hide unused axes if any
        for ax in axes.ravel()[len(clusters):]:
            ax.axis("off")

        plt.tight_layout()
        path = os.path.join(plots_dir, f"{group_name}_distance_distributions.png")
        plt.savefig(path)
        plt.close(fig)
        logger.info(f"{group_name}: Saved distance distributions plot -> {path}")

        # 4) Time series sample for a few meters
        if {"POLIZA_SUMINISTRO", "FECHA", "CONSUMO_REAL", "is_anomaly"}.issubset(group_df_labeled.columns):
            meters = group_df_labeled["POLIZA_SUMINISTRO"].dropna().unique()
            if len(meters) > 0:
                rng = np.random.default_rng(self.random_state)
                n_meters = min(5, len(meters))
                sampled_meters = rng.choice(meters, size=n_meters, replace=False)

                nrows = n_meters
                fig, axes = plt.subplots(nrows, 1, figsize=(10, 3 * nrows), sharex=True)
                if nrows == 1:
                    axes = [axes]

                for ax, meter in zip(axes, sampled_meters):
                    sub = group_df_labeled[group_df_labeled["POLIZA_SUMINISTRO"] == meter].copy()
                    sub = sub.sort_values("FECHA")

                    ax.plot(sub["FECHA"], sub["CONSUMO_REAL"], label="Consumption")
                    anom = sub[sub["is_anomaly"]]
                    ax.scatter(anom["FECHA"], anom["CONSUMO_REAL"], color="red", label="Anomaly")
                    ax.set_title(f"Meter {meter}")
                    ax.set_ylabel("CONSUMO_REAL")
                    ax.legend()

                axes[-1].set_xlabel("Date")
                plt.tight_layout()
                path = os.path.join(plots_dir, f"{group_name}_timeseries_sample.png")
                plt.savefig(path)
                plt.close(fig)
                logger.info(f"{group_name}: Saved time series sample plot -> {path}")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, labeled_df: pd.DataFrame, skipped_groups: List[Tuple[str, int]]) -> None:
        """
        Generate a text report summarizing the training and anomaly statistics.
        """
        report_path = os.path.join(self.run_dir, "training_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"K-Means Anomaly Detection Report - {self.timestamp}\n")
            f.write("=" * 80 + "\n\n")

            total_records = len(labeled_df)
            total_anomalies = int(labeled_df["is_anomaly"].sum())
            anomaly_rate = total_anomalies / total_records if total_records > 0 else 0.0

            f.write(f"Total records processed: {total_records:,}\n")
            f.write(f"Total anomalies detected: {total_anomalies:,} ({anomaly_rate:.2%})\n\n")

            if skipped_groups:
                f.write("Skipped groups (insufficient data):\n")
                for brand, size in skipped_groups:
                    f.write(f"  - {brand}: {size} records\n")
                f.write("\n")

            f.write("Per-group metrics:\n")
            for group_name, info in self.models.items():
                m = info["metrics"]
                f.write(f"\nGroup: {group_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Samples: {m['n_samples']:,}, Features: {m['n_features']}\n")
                f.write(f"  k: {m['k']}\n")
                f.write(f"  Silhouette: {m['silhouette_score']:.4f}\n")
                f.write(f"  Davies-Bouldin: {m['davies_bouldin_score']:.4f}\n")
                f.write(f"  Calinski-Harabasz: {m['calinski_harabasz_score']:.2f}\n")
                f.write(f"  Inertia: {m['inertia']:.1f}\n")

                # Anomaly stats
                group_mask = labeled_df["BRAND"] == group_name
                sub = labeled_df[group_mask]
                if len(sub) > 0:
                    anomalies = int(sub["is_anomaly"].sum())
                    rate = anomalies / len(sub)
                    f.write(f"  Group anomalies: {anomalies:,} ({rate:.2%})\n")

            f.write("\nEnd of report.\n")

        logger.info(f"Training report written to: {report_path}")

    # ------------------------------------------------------------------
    # Run / training entrypoint
    # ------------------------------------------------------------------

    def _setup_output_dirs(self) -> None:
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"run_{self.timestamp}")
        self.models_dir = os.path.join(self.run_dir, "models")
        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.data_dir = os.path.join(self.run_dir, "data")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def run(
        self,
        df_features: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """
        Main training entrypoint.

        Parameters
        ----------
        df_features
            Feature-engineered dataframe.
        feature_cols
            Columns to use for K-means clustering (no "all numeric" auto-selection).

        Returns
        -------
        labeled_df
            Input dataframe with anomaly labels and cluster/distance info.
        """
        self._setup_output_dirs()
        logger.info("=" * 60)
        logger.info("STEP 2: K-MEANS CLUSTERING")
        logger.info("=" * 60)

        grouped_data, feature_cols_used, skipped_groups = self.prepare_data(df_features, feature_cols)

        labeled_parts = []
        self.metadata = {
            "timestamp": self.timestamp,
            "min_clusters": self.min_clusters,
            "max_clusters": self.max_clusters,
            "distance_metric": self.distance_metric,
            "anomaly_tail_fraction": self.anomaly_tail_fraction,
            "groups": {},
        }

        for group_name, group_df in grouped_data.items():
            result = self.train_model(group_name, group_df, feature_cols_used)
            labeled_parts.append(result["labeled_data"])

            self.metadata["groups"][group_name] = {
                "model_path": result["model_info"]["model_path"],
                "scaler_path": result["model_info"]["scaler_path"],
                "thresholds_path": result["model_info"]["thresholds_path"],
                "inv_covs_path": result["model_info"]["inv_covs_path"],
                "metrics": result["model_info"]["metrics"],
            }

        labeled_df = pd.concat(labeled_parts, axis=0, ignore_index=True)

        # Save labeled data
        data_path = os.path.join(self.data_dir, f"labeled_data_{self.timestamp}.parquet")
        labeled_df.to_parquet(data_path, index=False)
        logger.info(f"Saved labeled data -> {data_path}")

        # Save metadata
        meta_path = os.path.join(self.run_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved metadata -> {meta_path}")

        # Generate report
        self.generate_report(labeled_df, skipped_groups)

        return labeled_df

    # ------------------------------------------------------------------
    # Inference / scoring on new data
    # ------------------------------------------------------------------

    def score_df(
        self,
        df_features: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """
        Score a new feature-engineered dataframe using already trained models.
        It expects that `run()` has been called before on this instance, so
        self.models is populated, or that you have manually populated
        self.models with loaded artifacts.

        Parameters
        ----------
        df_features
            New feature-engineered dataframe.
        feature_cols
            Same K-means feature columns used during training.

        Returns
        -------
        scored_df
            Copy of df_features with BRAND, cluster, distance, anomaly_score, is_anomaly.
        """
        if not self.models:
            raise RuntimeError("No trained models found in self.models. Call run() first or load models manually.")

        df, brand_col = self._infer_brand_column(df_features)
        scored_parts = []

        for brand, group in df.groupby(brand_col):
            group_name = str(brand)
            group_copy = group.copy()

            if group_name not in self.models:
                logger.warning(f"Brand {group_name} not present in trained models; marking all as non-anomalous.")
                group_copy["cluster"] = -1
                group_copy["distance"] = 0.0
                group_copy["anomaly_score"] = 0.0
                group_copy["is_anomaly"] = False
                scored_parts.append(group_copy)
                continue

            model_info = self.models[group_name]
            kmeans = model_info["model"]
            scaler = model_info["scaler"]
            thresholds = model_info["thresholds"]

            missing = [c for c in feature_cols if c not in group_copy.columns]
            if missing:
                raise ValueError(f"Missing features for brand {brand}: {missing}")

            X = group_copy[feature_cols].to_numpy(dtype=float)
            X_scaled = scaler.transform(X)
            labels = kmeans.predict(X_scaled)

            inv_covs = None
            if self.distance_metric == "mahalanobis" and model_info.get("inv_covs_path"):
                with open(model_info["inv_covs_path"], "rb") as f:
                    inv_covs = pickle.load(f)

            distances = self.calculate_distances(X_scaled, labels, kmeans.cluster_centers_, inv_covs)

            is_anomaly = np.zeros_like(labels, dtype=bool)
            anomaly_scores = np.zeros_like(distances)

            for cluster_id, info in thresholds.items():
                cid = int(cluster_id)
                mask = labels == cid
                if not np.any(mask):
                    continue
                thr = info["threshold"]
                if thr <= 0:
                    continue
                is_anomaly[mask] = distances[mask] >= thr
                anomaly_scores[mask] = np.clip(distances[mask] / thr, 0.0, 10.0)

            group_copy["cluster"] = labels
            group_copy["distance"] = distances
            group_copy["anomaly_score"] = anomaly_scores
            group_copy["is_anomaly"] = is_anomaly

            scored_parts.append(group_copy)

        scored_df = pd.concat(scored_parts, axis=0, ignore_index=True)
        return scored_df
