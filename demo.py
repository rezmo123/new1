"""
Production-ready TimeGNN and StructuralGNN Demo Script
---------------------------------------------------
Demonstrates the functionality of TimeGNN and StructuralGNN models
for anomaly detection in time series and structural data.
"""

import logging
import argparse
import time
from typing import Optional, Dict, Any
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.time_gnn_model import TimeGNNAnomalyDetector
from core.models.structural_gnn_model import StructuralGNNDetector
from core.data_generator import CloudWatchDataGenerator

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('anomaly_detection_demo.log')
    ]
)
logger = logging.getLogger(__name__)

def run_timegnn_demo(data_gen: CloudWatchDataGenerator, threshold_percentile: float = 95.0) -> Dict[str, Any]:
    """Run TimeGNN demonstration with comprehensive error handling and logging"""
    try:
        logger.info("Starting TimeGNN demonstration...")
        start_time = time.time()

        # Generate data with error handling
        try:
            entities_df = data_gen.generate_entity_metadata()
            relationships_df = data_gen.generate_relationships()
            timeseries_df = data_gen.generate_time_series()
            logger.info("Successfully generated synthetic data")
        except Exception as e:
            logger.error(f"Failed to generate data: {str(e)}")
            raise

        # Initialize and train model
        try:
            n_features = 4  # CPU, Memory, NetworkIn, NetworkOut
            n_categories = len(entities_df['service_type'].unique())

            model = TimeGNNAnomalyDetector(
                input_shape=(5, n_features),
                n_categories=n_categories
            )
            logger.info("Successfully initialized TimeGNN model")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

        # Train model with error handling
        try:
            history = model.train(
                timeseries_df=timeseries_df,
                relationships_df=relationships_df,
                epochs=10,
                threshold_percentile=threshold_percentile
            )
            logger.info("Successfully trained TimeGNN model")
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise

        # Detect anomalies with error handling
        try:
            predictions, patterns = model.predict(timeseries_df, relationships_df)
            logger.info("Successfully completed anomaly detection")
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            raise

        runtime = time.time() - start_time
        results = {
            'model_type': 'TimeGNN',
            'runtime': runtime,
            'n_anomalies': int(predictions.sum()) if predictions is not None else 0,
            'patterns': list(set(patterns[predictions])) if predictions is not None and patterns is not None else [],
            'training_history': history.history if history else None
        }

        logger.info(f"TimeGNN demo completed in {runtime:.2f} seconds")
        logger.info(f"Detected {results['n_anomalies']} anomalies")
        logger.info(f"Pattern types found: {results['patterns']}")

        return results

    except Exception as e:
        logger.error(f"TimeGNN demo failed: {str(e)}")
        return None

def run_structuralgnn_demo(data_gen: CloudWatchDataGenerator, threshold_percentile: float = 95.0) -> Dict[str, Any]:
    """Run StructuralGNN demonstration with comprehensive error handling and logging"""
    try:
        logger.info("Starting StructuralGNN demonstration...")
        start_time = time.time()

        # Generate data with error handling
        try:
            relationships_df = data_gen.generate_relationships()
            logger.info("Successfully generated structural data")
        except Exception as e:
            logger.error(f"Failed to generate data: {str(e)}")
            raise

        # Initialize model with error handling
        try:
            model = StructuralGNNDetector()
            logger.info("Successfully initialized StructuralGNN model")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

        # Train model with error handling
        try:
            history = model.train(
                relationships_df=relationships_df,
                epochs=10,
                threshold_percentile=threshold_percentile
            )
            logger.info("Successfully trained StructuralGNN model")
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise

        # Detect anomalies with error handling
        try:
            anomalies, patterns, scores = model.predict(relationships_df)
            logger.info("Successfully completed anomaly detection")
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            raise

        runtime = time.time() - start_time
        results = {
            'model_type': 'StructuralGNN',
            'runtime': runtime,
            'n_anomalies': int(anomalies.sum()) if anomalies is not None else 0,
            'patterns': list(set(patterns[anomalies])) if anomalies is not None and patterns is not None else [],
            'training_history': history['history'] if history else None,
            'reconstruction_scores': scores.tolist() if scores is not None else None
        }

        logger.info(f"StructuralGNN demo completed in {runtime:.2f} seconds")
        logger.info(f"Detected {results['n_anomalies']} anomalies")
        logger.info(f"Pattern types found: {results['patterns']}")

        return results

    except Exception as e:
        logger.error(f"StructuralGNN demo failed: {str(e)}")
        return None

def compare_models(results: Dict[str, Dict[str, Any]]) -> None:
    """Compare performance metrics between models with detailed logging"""
    logger.info("\nModel Comparison Summary:")
    logger.info("-" * 50)

    metrics_log = []
    for model_type, metrics in results.items():
        if metrics:
            metrics_log.append(f"\n{model_type}:")
            metrics_log.append(f"Runtime: {metrics['runtime']:.2f} seconds")
            metrics_log.append(f"Anomalies detected: {metrics['n_anomalies']}")
            metrics_log.append(f"Pattern types: {metrics['patterns']}")

            if metrics['training_history']:
                final_loss = metrics['training_history'].get('loss', [])[-1]
                metrics_log.append(f"Final training loss: {final_loss:.4f}")
        else:
            metrics_log.append(f"{model_type}: Failed to complete")

    # Log all metrics at once for better log consistency
    logger.info("\n".join(metrics_log))
    logger.info("-" * 50)

def run_demo(model_type: str = 'all', threshold_percentile: float = 95.0) -> bool:
    """Run demonstration of the specified anomaly detection model(s)"""
    try:
        # Initialize data generator with error handling
        logger.info("Initializing data generator...")
        try:
            data_gen = CloudWatchDataGenerator(n_entities=10, n_timestamps=100)
        except Exception as e:
            logger.error(f"Failed to initialize data generator: {str(e)}")
            return False

        results = {}

        if model_type in ['time', 'all']:
            results['TimeGNN'] = run_timegnn_demo(data_gen, threshold_percentile)

        if model_type in ['structural', 'all']:
            results['StructuralGNN'] = run_structuralgnn_demo(data_gen, threshold_percentile)

        if len(results) > 1:
            compare_models(results)

        # Verify all models completed successfully
        success = all(result is not None for result in results.values())
        if success:
            logger.info("Demo completed successfully!")
        else:
            logger.warning("Demo completed with some failures")

        return success

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Production-ready Anomaly Detection Demo')
    parser.add_argument('--model', type=str, default='all',
                      choices=['time', 'structural', 'all'],
                      help='Model to demonstrate (default: all)')
    parser.add_argument('--threshold', type=float, default=95.0,
                      help='Threshold percentile for anomaly detection (default: 95.0)')

    args = parser.parse_args()
    success = run_demo(args.model, args.threshold)
    sys.exit(0 if success else 1)