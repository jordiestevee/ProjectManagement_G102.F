"""
Main orchestrator for Water Anomaly Detection System
Executes the complete pipeline: Feature Engineering -> K-Means -> LSTM
"""
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Import configurations
from config import (
    FEATURE_CONFIG, KMEANS_CONFIG, LSTM_CONFIG, 
    PIPELINE_CONFIG, COLUMN_CONFIG, LOGGING_CONFIG,
    DATA_DIR, OUTPUT_DIR, MODELS_DIR
)

# Import utilities
from utils import (
    setup_logging, validate_input_data, load_data,
    save_results, create_run_directory, save_config,
    generate_summary_report
)

# Import pipeline components
from feature_engineering import feature_engineering_kmeans
from kmeans_anomaly import KMeansAnomalyDetectionPipeline
from lstm_anomaly import (
    prepare_lstm_dataset, 
    LSTMAnomalyTrainingPipeline,
    LSTMAnomalyPredictionPipeline
)

# Setup logging
logger = setup_logging(LOGGING_CONFIG)

class WaterAnomalyDetectionPipeline:
    """
    Main pipeline orchestrator for water consumption anomaly detection.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the pipeline with configuration.
        
        Parameters:
        -----------
        config : Dict
            Custom configuration dictionary (optional)
        """
        self.config = config or {}
        self.run_dir = create_run_directory(OUTPUT_DIR)
        self.results = {}
        
        logger.info(f"Pipeline initialized. Run directory: {self.run_dir}")
        
        # Save configuration
        save_config({
            'feature_config': FEATURE_CONFIG,
            'kmeans_config': KMEANS_CONFIG,
            'lstm_config': LSTM_CONFIG,
            'pipeline_config': PIPELINE_CONFIG,
            'custom_config': self.config
        }, self.run_dir)
    
    def run_feature_engineering(self, df):
        """Run feature engineering step."""
        logger.info("="*60)
        logger.info("STEP 1: FEATURE ENGINEERING")
        logger.info("="*60)
        
        input_records = len(df)
        
        # Apply feature engineering
        df_features, kmeans_feature_columns = feature_engineering_kmeans(df, logger=logger)

        output_records = len(df_features)
        feature_columns = [col for col in df_features.columns 
                          if col not in COLUMN_CONFIG['required_columns']]
        
        # Save results
        if PIPELINE_CONFIG['save_intermediate_results']:
            output_path = os.path.join(self.run_dir, 'feature_engineered_data.parquet')
            save_results(df_features, output_path)
            logger.info(f"Feature engineered data saved to: {output_path}")
        
        # Store results
        self.results['feature_engineering'] = {
            'input_records': input_records,
            'output_records': output_records,
            'features_created': len(feature_columns),
            'output_path': output_path if PIPELINE_CONFIG['save_intermediate_results'] else None
        }
        
        logger.info(f"Feature engineering complete: {output_records} records, {len(feature_columns)} features")
        
        return df_features, kmeans_feature_columns
    
    def run_kmeans_clustering(self, df, kmeans_feature_columns):
        """Run K-means clustering step."""
        logger.info("=" * 60)
        logger.info("STEP 2: K-MEANS CLUSTERING")
        logger.info("=" * 60)

        kmeans_pipeline = KMeansAnomalyDetectionPipeline(
            min_clusters=KMEANS_CONFIG['min_clusters'],
            max_clusters=KMEANS_CONFIG['max_clusters'],
            distance_metric=KMEANS_CONFIG['distance_metric'],
            random_state=KMEANS_CONFIG['random_state'],
            output_dir=self.run_dir,
            train_sample_fraction=KMEANS_CONFIG.get('train_sample_fraction'),
            train_sample_cap=KMEANS_CONFIG.get('train_sample_cap'),
            k_opt_sample_size=KMEANS_CONFIG.get('k_opt_sample_size', 10_000),
            silhouette_sample_size=KMEANS_CONFIG.get('silhouette_sample_size', 5_000),

            # NEW: quantile + std-based thresholds
            anomaly_tail_fraction=KMEANS_CONFIG.get('anomaly_tail_fraction', 0.025),
            std_threshold=KMEANS_CONFIG.get('std_threshold'),
        )

        labeled_df = kmeans_pipeline.run(
            df_features=df,
            feature_cols=kmeans_feature_columns,
        )

        self.results['kmeans'] = {
            'models_trained': len(kmeans_pipeline.models),
            'total_anomalies': int(labeled_df['is_anomaly'].sum()),
            'anomaly_rate': f"{labeled_df['is_anomaly'].mean():.2%}",
            'output_path': os.path.join(
                kmeans_pipeline.data_dir,
                f"labeled_data_{kmeans_pipeline.timestamp}.parquet"
            ),
        }

        logger.info(
            f"K-means clustering complete: {self.results['kmeans']['total_anomalies']} anomalies detected"
        )
        
        return labeled_df
    
    def run_lstm_training(self, df: pd.DataFrame) -> Dict:
        """Run LSTM training step."""
        logger.info("=" * 60)
        logger.info("STEP 3: LSTM TRAINING")
        logger.info("=" * 60)

        sequences_df, feature_names = prepare_lstm_dataset(
            df,
            sequence_length=LSTM_CONFIG["sequence_length"],
            target_horizons=LSTM_CONFIG["target_horizons"],

            # NEW: how many anomalies are needed in the window for a positive label
            min_anomaly_count=LSTM_CONFIG.get("min_anomaly_count", 1),

            stride=LSTM_CONFIG["stride"],
            max_sequences_per_meter=LSTM_CONFIG["max_sequences_per_meter"],
            meters_sample_frac=LSTM_CONFIG["meters_sample_frac"],
            output_format="dataframe",
        )

        logger.info(
            "LSTM dataset built: %d sequences, %d meters, %d features",
            len(sequences_df),
            sequences_df["meter_id"].nunique() if len(sequences_df) > 0 else 0,
            len(feature_names),
        )

        training_pipeline = LSTMAnomalyTrainingPipeline(
            sequence_length=LSTM_CONFIG["sequence_length"],
            target_horizons=LSTM_CONFIG["target_horizons"],
            units=LSTM_CONFIG["lstm_units"],
            dropout=LSTM_CONFIG["dropout_rate"],
            learning_rate=LSTM_CONFIG["learning_rate"],
            batch_size=LSTM_CONFIG["batch_size"],
            epochs=LSTM_CONFIG["epochs"],
            patience=LSTM_CONFIG["patience"],
            validation_split=LSTM_CONFIG["validation_split"],
            output_dir=self.run_dir,
        )

        lstm_results = training_pipeline.run(sequences_df, feature_names)

        self.results["lstm_training"] = {
            "models_trained": len(training_pipeline.models),
            "output_path": training_pipeline.run_dir,
            "metrics": lstm_results,
        }

        return lstm_results

    def run_lstm_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run LSTM prediction step."""
        logger.info("=" * 60)
        logger.info("STEP 4: LSTM PREDICTION")
        logger.info("=" * 60)

        sequences_df, feature_names = prepare_lstm_dataset(
            df,
            sequence_length=LSTM_CONFIG["sequence_length"],
            target_horizons=LSTM_CONFIG["target_horizons"],

            # Use the same label definition for consistency, even though
            # we don't need targets during prediction
            min_anomaly_count=LSTM_CONFIG.get("min_anomaly_count", 1),

            stride=LSTM_CONFIG["stride"],
            max_sequences_per_meter=LSTM_CONFIG["max_sequences_per_meter"],
            meters_sample_frac=LSTM_CONFIG.get(
                "meters_sample_frac_prediction",
                LSTM_CONFIG["meters_sample_frac"],
            ),
            output_format="dataframe",
        )

        logger.info(
            "LSTM prediction dataset: %d sequences for %d meters",
            len(sequences_df),
            sequences_df["meter_id"].nunique() if len(sequences_df) > 0 else 0,
        )

        pred_pipeline = LSTMAnomalyPredictionPipeline(
            models_dir=self.run_dir,
            sequence_length=LSTM_CONFIG["sequence_length"],
            threshold=LSTM_CONFIG["anomaly_threshold"],
            version="latest",
        )

        predictions_df = pred_pipeline.predict(sequences_df)

        self.results["lstm_prediction"] = {
            "total_sequences": len(predictions_df),
            "output_path": os.path.join(
                pred_pipeline.output_dir, "reports", "lstm_predictions.csv"
            ),
        }

        return predictions_df

    def run_pipeline(self, input_file: str, steps: Dict[str, bool] = None):
        """
        Run the complete anomaly detection pipeline.
        
        Parameters:
        -----------
        input_file : str
            Path to input data file
        steps : Dict[str, bool]
            Override which steps to run
            
        Returns:
        --------
        Dict with results from each step
        """
        # Load input data
        logger.info(f"Loading input data from: {input_file}")
        df = load_data(input_file, date_column=COLUMN_CONFIG['date'])
        
        # Validate input
        is_valid, missing_cols = validate_input_data(df, COLUMN_CONFIG['required_columns'])
        if not is_valid:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Determine which steps to run
        if steps is None:
            steps = {
                'feature_engineering': PIPELINE_CONFIG['run_feature_engineering'],
                'kmeans': PIPELINE_CONFIG['run_kmeans'],
                'lstm_training': PIPELINE_CONFIG['run_lstm_training'],
                'lstm_prediction': PIPELINE_CONFIG['run_lstm_prediction']
            }
        
        # Run pipeline steps
        current_df = df
        
        if steps.get('feature_engineering', True):
            current_df, kmeans_feature_columns = self.run_feature_engineering(current_df)
        
        if steps.get('kmeans', True):
            current_df = self.run_kmeans_clustering(current_df, kmeans_feature_columns)
        
        if steps.get('lstm_training', True):
            # Train models for the current run directory
            self.run_lstm_training(current_df)

        if steps.get('lstm_prediction', True):
            # Use models saved under this run's directory (self.run_dir)
            predictions = self.run_lstm_prediction(current_df)

        
        # Generate summary report
        if PIPELINE_CONFIG['generate_reports']:
            summary = generate_summary_report(self.results, self.run_dir)
            logger.info("\n" + summary)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Results saved to: {self.run_dir}")
        logger.info("="*60)
        
        return self.results


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Water Consumption Anomaly Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
        Examples:
            # Run complete pipeline with default settings
            python main.py data/water_consumption.csv
            
            # Run only feature engineering and k-means
            python main.py data/water_consumption.csv --steps feature_engineering kmeans
            
            # Run with custom k-means threshold
            python main.py data/water_consumption.csv --kmeans-threshold 3.0
            
            # Run prediction only with existing models
            python main.py data/labeled_data.parquet --steps lstm_prediction --models-dir models/lstm_run_20240101
            
            # Process specific date range
            python main.py data/water_consumption.csv --start-date 2021-01-01 --end-date 2021-12-31
        '''
    )
    
    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input data file (CSV, Parquet, or Excel)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['feature_engineering', 'kmeans', 'lstm_training', 'lstm_prediction'],
        help='Specify which pipeline steps to run (default: all)'
    )
    
    parser.add_argument(
        '--kmeans-threshold',
        type=float,
        default=2.5,
        help='Standard deviation threshold for k-means anomaly detection (default: 2.5)'
    )
    
    parser.add_argument(
        '--lstm-sequence-length',
        type=int,
        default=90,
        help='Sequence length for LSTM input (default: 90 days)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        help='Directory containing pre-trained models (for prediction only)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for data filtering (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for data filtering (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to custom configuration file (JSON)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Update configurations with command line arguments
    if args.kmeans_threshold:
        KMEANS_CONFIG['std_threshold'] = args.kmeans_threshold
    
    if args.lstm_sequence_length:
        LSTM_CONFIG['sequence_length'] = args.lstm_sequence_length
    
    # Load custom config if provided
    custom_config = {}
    if args.config_file:
        import json
        with open(args.config_file, 'r') as f:
            custom_config = json.load(f)
    
    # Determine which steps to run
    steps_to_run = None
    if args.steps:
        steps_to_run = {step: step in args.steps for step in 
                       ['feature_engineering', 'kmeans', 'lstm_training', 'lstm_prediction']}
    
    # Initialize and run pipeline
    try:
        pipeline = WaterAnomalyDetectionPipeline(config=custom_config)
        
        # Load and filter data if dates provided
        if args.start_date or args.end_date:
            df = load_data(args.input_file, date_column=COLUMN_CONFIG['date'])
            
            if args.start_date:
                df = df[df[COLUMN_CONFIG['date']] >= args.start_date]
            if args.end_date:
                df = df[df[COLUMN_CONFIG['date']] <= args.end_date]
            
            # Save filtered data
            filtered_path = os.path.join(pipeline.run_dir, 'filtered_input.parquet')
            save_results(df, filtered_path)
            input_file = filtered_path
        else:
            input_file = args.input_file
        
        # Run pipeline
        results = pipeline.run_pipeline(input_file, steps=steps_to_run)
        
        logger.info("Pipeline execution successful!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()