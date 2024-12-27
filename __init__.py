"""
TimeGNN Core Module
------------------
Core implementation of the Temporal Graph Neural Network (TimeGNN) 
for anomaly detection in time series data.
"""

from .models.time_gnn_model import TimeGNNAnomalyDetector
from .models.structural_gnn_model import StructuralGNNDetector
from .models.hybrid_gnn_model import HybridGNNDetector
from .preprocessors.time_series_preprocessor import TimeSeriesPreprocessor
from .preprocessors.structural_preprocessor import StructuralPreprocessor
from .data_generator import CloudWatchDataGenerator
from .visualization import TimeGNNVisualizer

__version__ = "0.1.0"

__all__ = [
    'TimeGNNAnomalyDetector',
    'StructuralGNNDetector',
    'HybridGNNDetector',
    'TimeSeriesPreprocessor',
    'StructuralPreprocessor',
    'CloudWatchDataGenerator',
    'TimeGNNVisualizer'
]