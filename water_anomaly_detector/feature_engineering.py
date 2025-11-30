import pandas as pd
import numpy as np
from scipy import stats

import numpy as np
import pandas as pd

def feature_engineering_kmeans(df, logger):
    """
    Feature engineering for k-means clustering on water consumption data.
    Returns:
        df_final: full feature dataframe (for LSTM / downstream)
        kmeans_feature_columns: list of columns recommended for k-means
    """
    logger.info("Starting feature engineering for k-means")
    logger.info(f"Input dataframe shape: {df.shape}")
    logger.info(f"Input columns: {df.columns.tolist()}")
    
    # Check for nulls in critical columns upfront
    critical_cols = ['POLIZA_SUMINISTRO', 'FECHA', 'CONSUMO_REAL']
    null_counts = df[critical_cols].isna().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Null values in critical columns:\n{null_counts[null_counts > 0]}")
        logger.info("Dropping rows with nulls in critical columns")
        df = df.dropna(subset=critical_cols)
        logger.info(f"Shape after dropping critical nulls: {df.shape}")
    
    df_feat = df.copy()
    df_feat['FECHA'] = pd.to_datetime(df_feat['FECHA'])
    df_feat = df_feat.sort_values(['POLIZA_SUMINISTRO', 'FECHA'])
    logger.info("Converted FECHA to datetime and sorted data")

    # ===============================
    # 1. CONSUMPTION TRANSFORMATIONS
    # ===============================
    logger.info("Section 1: Starting consumption transformations")
    try:
        df_feat['CONSUMO_LOG'] = np.log1p(df_feat['CONSUMO_REAL'])
        logger.info("Created CONSUMO_LOG")

        df_feat['CONSUMO_ZSCORE'] = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        logger.info("Created CONSUMO_ZSCORE")

        def robust_zscore(x):
            median = x.median()
            mad = np.median(np.abs(x - median))
            return (x - median) / (mad * 1.4826 + 1e-8)

        df_feat['CONSUMO_ROBUST_ZSCORE'] = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL'].transform(robust_zscore)
        logger.info("Created CONSUMO_ROBUST_ZSCORE")

        df_feat['LOG_CONSUMO'] = np.log1p(df_feat['CONSUMO_REAL'])
        logger.info("Created LOG_CONSUMO")
    except Exception as e:
        logger.error(f"Error in consumption transformations: {str(e)}")
        raise

    # ===============================
    # 2. TEMPORAL FEATURES
    # ===============================
    logger.info("Section 2: Starting temporal features")
    try:
        df_feat['DAY_OF_WEEK'] = df_feat['FECHA'].dt.dayofweek
        df_feat['DAY_OF_MONTH'] = df_feat['FECHA'].dt.day
        df_feat['MONTH'] = df_feat['FECHA'].dt.month
        df_feat['QUARTER'] = df_feat['FECHA'].dt.quarter
        df_feat['WEEK_OF_YEAR'] = df_feat['FECHA'].dt.isocalendar().week
        logger.info("Created basic temporal features")

        df_feat['IS_WEEKEND'] = (df_feat['DAY_OF_WEEK'] >= 5).astype(int)
        logger.info("Created IS_WEEKEND")

        def get_season(month):
            if month in [12, 1, 2]:
                return 1  # Winter
            elif month in [3, 4, 5]:
                return 2  # Spring
            elif month in [6, 7, 8]:
                return 3  # Summer
            else:
                return 4  # Fall

        df_feat['SEASON'] = df_feat['MONTH'].apply(get_season)
        logger.info("Created SEASON")

        df_feat['MONTH_SIN'] = np.sin(2 * np.pi * df_feat['MONTH'] / 12)
        df_feat['MONTH_COS'] = np.cos(2 * np.pi * df_feat['MONTH'] / 12)
        df_feat['DAY_OF_WEEK_SIN'] = np.sin(2 * np.pi * df_feat['DAY_OF_WEEK'] / 7)
        df_feat['DAY_OF_WEEK_COS'] = np.cos(2 * np.pi * df_feat['DAY_OF_WEEK'] / 7)
        logger.info("Created cyclical temporal features")
    except Exception as e:
        logger.error(f"Error in temporal features: {str(e)}")
        raise

    # ===============================
    # 3. WEATHER FEATURES
    # ===============================
    logger.info("Section 3: Starting weather features")
    try:
        weather_cols = ['tavg', 'tmin', 'tmax', 'prcp']
        has_weather = all(col in df_feat.columns for col in weather_cols)
        logger.info(f"Weather columns present: {has_weather}")
        
        if has_weather:
            df_feat['TAVG'] = df_feat['tavg']
            df_feat['TMIN'] = df_feat['tmin']
            df_feat['TMAX'] = df_feat['tmax']
            df_feat['PRCP'] = df_feat['prcp']
            logger.info("Copied weather columns")

            df_feat['TEMP_RANGE'] = df_feat['TMAX'] - df_feat['TMIN']
            logger.info("Created TEMP_RANGE")

            # Rolling weather context (no groupby: assumed same weather for all meters)
            df_feat = df_feat.sort_values('FECHA')
            df_feat['TAVG_7D_MEAN'] = df_feat['TAVG'].rolling(window=7, min_periods=3).mean()
            df_feat['PRCP_7D_SUM'] = df_feat['PRCP'].rolling(window=7, min_periods=3).sum()
            df_feat = df_feat.sort_values(['POLIZA_SUMINISTRO', 'FECHA'])
            logger.info("Created rolling weather features")
        else:
            for col in ['TAVG', 'TMIN', 'TMAX', 'PRCP', 'TEMP_RANGE', 'TAVG_7D_MEAN', 'PRCP_7D_SUM']:
                df_feat[col] = 0.0
            logger.info("Weather columns not found, filled with zeros")
    except Exception as e:
        logger.error(f"Error in weather features: {str(e)}")
        raise

    # ===============================
    # 4. ROLLING STATISTICS (per meter)
    # ===============================
    logger.info("Section 4: Starting rolling statistics")

    long_windows = [30, 90]
    global_min_hist = 60  # "ideal" minimum history
    logger.info(f"Using windows: {long_windows}, global_min_hist: {global_min_hist}")

    try:
        for window in long_windows:
            # Ensure min_hist is never larger than the window itself
            min_hist = min(global_min_hist, window)
            logger.info(f"Processing {window}D rolling window (min_hist={min_hist})...")

            roll_mean = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL'].transform(
                lambda x, w=window, m=min_hist: x.rolling(window=w, min_periods=m).mean()
            )
            logger.info(f"Created ROLLING_MEAN_{window}D")

            roll_std = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL'].transform(
                lambda x, w=window, m=min_hist: x.rolling(window=w, min_periods=m).std()
            )
            logger.info(f"Created ROLLING_STD_{window}D")

            df_feat[f'ROLLING_MEAN_{window}D'] = roll_mean
            df_feat[f'ROLLING_STD_{window}D'] = roll_std
            df_feat[f'ROLLING_MEDIAN_{window}D'] = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL'].transform(
                lambda x, w=window, m=min_hist: x.rolling(window=w, min_periods=m).median()
            )
            logger.info(f"Created ROLLING_MEDIAN_{window}D")

            df_feat[f'ROLLING_CV_{window}D'] = roll_std / (roll_mean + 1e-8)
            df_feat[f'DEVIATION_FROM_MEAN_{window}D'] = (
                df_feat['CONSUMO_REAL'] - roll_mean
            ) / (roll_std + 1e-8)
            logger.info(f"Created CV and deviation features for {window}D")

        # Shorter window for MACD / sudden change (LSTM context only)
        logger.info("Creating shorter rolling windows (7D, 30D)...")
        df_feat['ROLLING_MEAN_7D'] = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL'].transform(
            lambda x: x.rolling(window=7, min_periods=3).mean()
        )
        df_feat['ROLLING_MEAN_30D'] = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL'].transform(
            lambda x: x.rolling(window=30, min_periods=10).mean()
        )
        df_feat['ROLLING_STD_7D'] = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL'].transform(
            lambda x: x.rolling(window=7, min_periods=3).std()
        )
        df_feat['ROLLING_MEDIAN_7D'] = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL'].transform(
            lambda x: x.rolling(window=7, min_periods=3).median()
        )
        logger.info("Created all short rolling window features")
    except Exception as e:
        logger.error(f"Error in rolling statistics: {str(e)}")
        raise

    # ===============================
    # 5. LOG-DIFF DYNAMICS (instead of pct_change)
    # ===============================
    logger.info("Section 5: Starting log-diff dynamics")
    try:
        df_feat['LOG_DIFF_1D'] = df_feat.groupby('POLIZA_SUMINISTRO')['LOG_CONSUMO'].diff(1)
        logger.info("Created LOG_DIFF_1D")

        df_feat['AVG_LOG_DIFF_7D'] = df_feat.groupby('POLIZA_SUMINISTRO')['LOG_DIFF_1D'].transform(
            lambda x: x.rolling(window=7, min_periods=3).mean()
        )
        df_feat['AVG_LOG_DIFF_30D'] = df_feat.groupby('POLIZA_SUMINISTRO')['LOG_DIFF_1D'].transform(
            lambda x: x.rolling(window=30, min_periods=10).mean()
        )
        logger.info("Created average log diff features")

        for col in ['AVG_LOG_DIFF_7D', 'AVG_LOG_DIFF_30D']:
            df_feat[col] = df_feat[col].clip(lower=-1.5, upper=1.5)
        logger.info("Clipped log diff extremes")
    except Exception as e:
        logger.error(f"Error in log-diff dynamics: {str(e)}")
        raise

    # ===============================
    # 6. METER CHARACTERISTICS
    # ===============================
    logger.info("Section 6: Starting meter characteristics")
    try:
        diameter_cols = [col for col in df.columns if col.startswith('DIAM_COMP_')]
        logger.info(f"Found {len(diameter_cols)} diameter columns: {diameter_cols}")
        df_feat['DIAMETER'] = np.nan

        for col in diameter_cols:
            diameter_value = float(col.replace('DIAM_COMP_', '').replace('.0', ''))
            df_feat.loc[df_feat[col] == True, 'DIAMETER'] = diameter_value
        logger.info("Extracted DIAMETER values")

        df_feat['DIAMETER_MISSING'] = df_feat['DIAMETER'].isna().astype(int)
        df_feat['DIAMETER'] = df_feat['DIAMETER'].fillna(15)
        logger.info(f"Filled missing diameters. Missing count: {df_feat['DIAMETER_MISSING'].sum()}")

        df_feat['DIAMETER_RISK'] = df_feat['DIAMETER'].apply(
            lambda x: 1.0 if x == 15 else 0.5 if x == 20 else 0.2
        )
        logger.info("Created DIAMETER_RISK")

        model_cols = [col for col in df.columns if col.startswith('CODI_MODEL_')]
        logger.info(f"Found {len(model_cols)} model columns")
        if model_cols:
            df_feat['NUM_MODELS'] = df_feat[model_cols].sum(axis=1)
        else:
            df_feat['NUM_MODELS'] = 0
        logger.info("Created NUM_MODELS")
    except Exception as e:
        logger.error(f"Error in meter characteristics: {str(e)}")
        raise

    # ===============================
    # 7. CONSUMPTION PATTERNS & ANOMALY SCORES
    # ===============================
    logger.info("Section 7: Starting consumption patterns and anomaly scores")
    try:
        # Simple zero-consumption flag (already vectorized)
        df_feat['IS_ZERO_CONSUMPTION'] = (df_feat['CONSUMO_REAL'] == 0).astype(int)
        logger.info(f"Created IS_ZERO_CONSUMPTION. Zero count: {df_feat['IS_ZERO_CONSUMPTION'].sum()}")

        # Use rolling quantiles instead of a Python-level rolling.apply
        logger.info("Computing 365D rolling 5th and 95th percentiles per meter (optimized)...")
        grp = df_feat.groupby('POLIZA_SUMINISTRO')['CONSUMO_REAL']

        # These are cythonized and much faster than a custom apply + rank
        rolling_p95 = grp.transform(
            lambda x: x.rolling(window=365, min_periods=60).quantile(0.95)
        )
        rolling_p05 = grp.transform(
            lambda x: x.rolling(window=365, min_periods=60).quantile(0.05)
        )

        df_feat['ROLLING_P95_365'] = rolling_p95
        df_feat['ROLLING_P05_365'] = rolling_p05
        logger.info("Created ROLLING_P95_365 and ROLLING_P05_365")

        # High / low consumption flags via quantile thresholds
        df_feat['IS_HIGH_CONSUMPTION'] = (df_feat['CONSUMO_REAL'] > rolling_p95).astype(int)
        df_feat['IS_LOW_CONSUMPTION'] = (
            (df_feat['CONSUMO_REAL'] < rolling_p05) &
            (df_feat['CONSUMO_REAL'] > 0)
        ).astype(int)
        logger.info(
            "Created consumption flags. "
            f"High: {df_feat['IS_HIGH_CONSUMPTION'].sum()}, "
            f"Low: {df_feat['IS_LOW_CONSUMPTION'].sum()}"
        )

        # OPTIONAL: approximate percentile feature from the 5th/95th band
        # (keeps CONSUMPTION_PERCENTILE_365 as a smooth [0, 1] feature)
        df_feat['CONSUMPTION_PERCENTILE_365'] = (
            (df_feat['CONSUMO_REAL'] - df_feat['ROLLING_P05_365']) /
            (df_feat['ROLLING_P95_365'] - df_feat['ROLLING_P05_365'] + 1e-8)
        ).clip(0.0, 1.0)
        logger.info("Created approximate CONSUMPTION_PERCENTILE_365 from rolling quantiles")

        # MACD-like and sudden change scores (already vectorized)
        df_feat['MACD'] = df_feat['ROLLING_MEAN_7D'] - df_feat['ROLLING_MEAN_30D']
        df_feat['SUDDEN_CHANGE_SCORE'] = np.abs(
            df_feat['CONSUMO_REAL'] - df_feat['ROLLING_MEDIAN_7D']
        ) / (df_feat['ROLLING_STD_7D'] + 1e-8)
        logger.info("Created MACD and SUDDEN_CHANGE_SCORE")
    except Exception as e:
        logger.error(f"Error in consumption patterns: {str(e)}")
        raise


    # ===============================
    # 8. ASSEMBLE FINAL DATAFRAME
    # ===============================
    logger.info("Section 8: Assembling final dataframe")
    try:
        # Keep MARCA_COMP_* columns so k-means can infer BRAND later
        brand_indicator_columns = [
            c for c in df_feat.columns if c.startswith('MARCA_COMP_')
        ]

        identifier_columns = ['POLIZA_SUMINISTRO', 'FECHA', 'CONSUMO_REAL'] + brand_indicator_columns

        kmeans_feature_columns = [
            'CONSUMO_LOG',
            'SEASON',
            'IS_WEEKEND',
            'MONTH_SIN', 'MONTH_COS',
            'DAY_OF_WEEK_SIN', 'DAY_OF_WEEK_COS',
            'ROLLING_CV_30D', 'ROLLING_CV_90D',
            'DEVIATION_FROM_MEAN_30D', 'DEVIATION_FROM_MEAN_90D',
            'DIAMETER_RISK',
            'NUM_MODELS',
            'TAVG', 'TEMP_RANGE',
            'TAVG_7D_MEAN', 'PRCP_7D_SUM',
            'AVG_LOG_DIFF_7D', 'AVG_LOG_DIFF_30D'
        ]

        extra_context_columns = [
            'CONSUMO_ZSCORE', 'CONSUMO_ROBUST_ZSCORE',
            'DAY_OF_MONTH', 'WEEK_OF_YEAR', 'QUARTER',
            'TMIN', 'TMAX', 'PRCP',
            'ROLLING_MEAN_7D', 'ROLLING_MEAN_30D', 'ROLLING_MEAN_90D',
            'ROLLING_STD_7D', 'ROLLING_MEDIAN_7D',
            'ROLLING_MEDIAN_30D', 'ROLLING_MEDIAN_90D',
            'DIAMETER', 'DIAMETER_MISSING',
            'IS_ZERO_CONSUMPTION',
            'CONSUMPTION_PERCENTILE_365',
            'IS_HIGH_CONSUMPTION', 'IS_LOW_CONSUMPTION',
            'MACD', 'SUDDEN_CHANGE_SCORE',
            'LOG_CONSUMO', 'LOG_DIFF_1D'
        ]

        all_keep = identifier_columns + kmeans_feature_columns + extra_context_columns
        all_keep = [c for c in all_keep if c in df_feat.columns]

        # remove duplicates, keep first occurrence
        seen = set()
        all_keep_unique = []
        for c in all_keep:
            if c not in seen:
                seen.add(c)
                all_keep_unique.append(c)
        all_keep = all_keep_unique

        df_final = df_feat[all_keep].copy()

        logger.info(f"Keeping {len(all_keep)} columns total")

        missing_cols = set(identifier_columns + kmeans_feature_columns + extra_context_columns) - set(df_feat.columns)
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")

        df_final = df_feat[all_keep].copy()
        logger.info(f"Final dataframe shape: {df_final.shape}")

        # Handle NaNs
        numeric_columns = df_final.select_dtypes(include=[np.number]).columns
        nan_counts_before = df_final[numeric_columns].isna().sum()
        logger.info(f"NaN counts before filling:\n{nan_counts_before[nan_counts_before > 0]}")

        for col in numeric_columns:
            median_value = df_final[col].median()
            df_final[col] = df_final[col].fillna(median_value)

        logger.info("Filled NaNs with median values")

        # Replace infs
        inf_counts = np.isinf(df_final[numeric_columns]).sum()
        if inf_counts.sum() > 0:
            logger.warning(f"Infinite values found:\n{inf_counts[inf_counts > 0]}")
        df_final.replace([np.inf, -np.inf], [999.0, -999.0], inplace=True)
        logger.info("Replaced infinite values")

        kmeans_feature_columns = [c for c in kmeans_feature_columns if c in df_final.columns]
        logger.info(f"Final k-means feature columns: {len(kmeans_feature_columns)} features")
        logger.info(f"K-means features: {kmeans_feature_columns}")

        logger.info("Feature engineering completed successfully")
        return df_final, kmeans_feature_columns
    except Exception as e:
        logger.error(f"Error in final assembly: {str(e)}")
        raise

from sklearn.preprocessing import RobustScaler

def prepare_for_kmeans(df_features, kmeans_feature_columns):
    """
    Prepare the feature-engineered dataframe for k-means clustering.

    Returns:
        scaled_features: ndarray for k-means
        scaler: fitted scaler
        feature_names: columns used for k-means
        original_data: POLIZA_SUMINISTRO, FECHA, CONSUMO_REAL
    """
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(df_features[kmeans_feature_columns])

    original_data = df_features[['POLIZA_SUMINISTRO', 'FECHA', 'CONSUMO_REAL']]

    return scaled_features, scaler, kmeans_feature_columns, original_data
