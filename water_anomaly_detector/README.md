# Water Consumption Anomaly Detection System

A comprehensive machine learning pipeline for detecting anomalies in water consumption patterns using K-means clustering and LSTM neural networks.

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete pipeline:
```bash
python main.py data/water_consumption.csv
```

### Advanced Examples

1. **Run specific steps only:**
```bash
# Only feature engineering and k-means
python main.py data/water_consumption.csv --steps feature_engineering kmeans
```

2. **Custom parameters:**
```bash
# Adjust k-means threshold and LSTM sequence length
python main.py data/water_consumption.csv --kmeans-threshold 3.0 --lstm-sequence-length 60
```

3. **Use pre-trained models for prediction:**
```bash
# Skip training and use existing models
python main.py data/labeled_data.parquet --steps lstm_prediction --models-dir models/lstm_run_20240101
```

4. **Filter by date range:**
```bash
# Process only 2021 data
python main.py data/water_consumption.csv --start-date 2021-01-01 --end-date 2021-12-31
```

5. **Custom configuration file:**
```bash
# Use custom configuration
python main.py data/water_consumption.csv --config-file config/custom_config.json
```

## Input Data Format

The input CSV/Parquet file should contain at minimum:
- `POLIZA_SUMINISTRO`: Water meter ID
- `FECHA`: Date (YYYY-MM-DD format)
- `CONSUMO_REAL`: Daily consumption value

## Pipeline Steps

1. **Feature Engineering**: Creates temporal and statistical features
2. **K-means Clustering**: Identifies consumption patterns and anomalies
3. **LSTM Training**: Trains predictive models for future anomalies
4. **LSTM Prediction**: Predicts anomaly probabilities for next day/week/month

## Output Structure
```
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── pipeline_summary.txt
    ├── feature_engineered_data.parquet
    ├── data/
    │   └── labeled_data_*.parquet
    ├── models/
    │   └── brand_model_*/
    ├── plots/
    └── reports/
```