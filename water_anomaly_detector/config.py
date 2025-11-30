"""
Configuration file for Water Anomaly Detection System
"""
from datetime import datetime
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'rolling_windows': [7, 14, 30, 90],
    'lag_periods': [1, 7, 30],
    'use_cyclical_encoding': True,
    'handle_zeros': 'log1p'  # log1p for log(1+x) to handle zeros
}

# K-Means Configuration
KMEANS_CONFIG = {
    'min_clusters': 2,
    'max_clusters': 8,

    # How many std deviations above the mean distance we require
    # (combined with the tail fraction below)
    'std_threshold': 2.5,

    # Fraction of farthest points per cluster to consider for anomalies
    # (used together with std_threshold inside k-means)
    'anomaly_tail_fraction': 0.025,

    'distance_metric': 'euclidean',  # 'euclidean' or 'mahalanobis'
    'min_samples_per_group': 100,
    'random_state': 42,
    'train_sample_fraction': 0.40,   # train on 40% of rows
    'train_sample_cap': 1500000,     # and never exceed 1.5M rows
}


# LSTM Configuration
LSTM_CONFIG = {
    'sequence_length': 60,
    'lstm_units': [64, 32, 16],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 250,
    'patience': 15,
    'validation_split': 0.2,
    'use_tfdata': False,
    'shuffle_buffer': 10000,
    'max_steps_per_epoch': 300,
    'max_val_steps': 80,

    # Threshold used at prediction time to turn probabilities into 0/1
    'anomaly_threshold': 0.5,

    # NEW: how many anomaly observations must appear in the window
    # for the window to be labeled as positive.
    # This can be:
    #   - an integer (same for all horizons)
    #   - or a dict like {'day': 1, 'week': 1, 'month': 2}
    'min_anomaly_count': 2,

    'target_horizons': {'day': 1, 'week': 7, 'month': 30},

    'stride': 14,
    'max_sequences_per_meter': 50,
    'meters_sample_frac': 0.5,
    "meters_sample_frac_prediction": 1.0,
}




# Pipeline Configuration
PIPELINE_CONFIG = {
    'run_feature_engineering': True,
    'run_kmeans': True,
    'run_lstm_training': True,
    'run_lstm_prediction': True,
    'save_intermediate_results': True,
    'generate_reports': True,
    'create_visualizations': True
}

# Column Mappings (adjust based on your data)
COLUMN_CONFIG = {
    'meter_id': 'POLIZA_SUMINISTRO',
    'date': 'FECHA',        # used by main/load_data
    'date_col': 'FECHA',    # used inside lstm_anomaly
    'consumption': 'CONSUMO_REAL',
    'timestamp_format': '%Y-%m-%d',
    'required_columns': [
        'POLIZA_SUMINISTRO', 'FECHA', 'CONSUMO_REAL'
    ],
    # NEW: name of the brand column (created from MARCA_COMP_* in k-means)
    'brand_col': 'BRAND',
}


# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}