import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import psutil

logger = logging.getLogger(__name__)

class CloudWatchDataGenerator:
    def __init__(self, n_entities=10, n_timestamps=100):
        """Initialize CloudWatch data generator with specified parameters"""
        self.n_entities = max(1, n_entities)
        self.n_timestamps = max(5, n_timestamps)  # Ensure minimum 5 timestamps for sequences
        self.service_types = ['EC2', 'RDS', 'Lambda']
        self.regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        self.system_metrics = [
            'SystemCPUUtilization', 'SystemMemoryUtilization', 
            'SystemDiskIOPS', 'SystemNetworkThroughput',
            'SystemProcessCount', 'SystemThreadCount',
            'SystemOpenFileDescriptors', 'SystemSwapUsage'
        ]
        self.metric_names = [
            'CPUUtilization', 'MemoryUtilization', 'NetworkIn', 'NetworkOut',
            'DiskReadBytes', 'DiskWriteBytes', 'DiskReadOps', 'DiskWriteOps'
        ]
        logger.info(f"Initialized CloudWatchDataGenerator with {n_entities} entities, {n_timestamps} timestamps")

    def _validate_entity_ids(self, entities):
        """Ensure all entity IDs are consistently formatted"""
        if not all(isinstance(e, str) and e.startswith('entity_') for e in entities):
            raise ValueError("All entity IDs must be strings in the format 'entity_XXX'")

    def generate_entity_metadata(self):
        """Generate entity metadata with service types"""
        try:
            entities = []
            # Generate entities with zero-padded IDs for consistent sorting
            for i in range(self.n_entities):
                entity = {
                    'entity_id': f'entity_{i:03d}',  # Consistent format: entity_000, entity_001, etc.
                    'service_type': np.random.choice(self.service_types),
                    'region': np.random.choice(self.regions),
                    'instance_id': f'i-{np.random.randint(10000, 99999):05d}'
                }
                entities.append(entity)

            entities_df = pd.DataFrame(entities)
            self._validate_entity_ids(entities_df['entity_id'])
            logger.info(f"Generated metadata for {len(entities)} entities")
            return entities_df

        except Exception as e:
            logger.error(f"Error generating entity metadata: {str(e)}")
            raise

    def _generate_system_metrics(self, timestamps):
        """Generate realistic system-level metrics"""
        try:
            system_data = []
            base_patterns = {
                'SystemCPUUtilization': (psutil.cpu_percent() if hasattr(psutil, 'cpu_percent') else 50),
                'SystemMemoryUtilization': (psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 60),
                'SystemDiskIOPS': 1000,
                'SystemNetworkThroughput': 5000,
                'SystemProcessCount': (len(psutil.pids()) if hasattr(psutil, 'pids') else 200),
                'SystemThreadCount': 1000,
                'SystemOpenFileDescriptors': 500,
                'SystemSwapUsage': (psutil.swap_memory().percent if hasattr(psutil, 'swap_memory') else 30)
            }

            time_factor = np.array([t.hour / 24.0 + t.minute / (24.0 * 60.0) for t in timestamps])

            for metric, base_value in base_patterns.items():
                values = np.zeros(len(timestamps))
                for i, _ in enumerate(timestamps):
                    # Add daily seasonality
                    seasonal = base_value + base_value * 0.2 * np.sin(2 * np.pi * time_factor[i])
                    # Add random noise
                    noise = np.random.normal(0, base_value * 0.05)
                    values[i] = max(0, seasonal + noise)

                system_data.append(pd.Series(values, name=metric))

            system_df = pd.concat(system_data, axis=1)
            system_df['timestamp'] = timestamps
            return system_df

        except Exception as e:
            logger.error(f"Error generating system metrics: {str(e)}")
            raise

    def generate_relationships(self):
        """Generate relationships between entities"""
        try:
            # First generate entity metadata to ensure consistent IDs
            entity_df = self.generate_entity_metadata()
            entity_ids = entity_df['entity_id'].tolist()
            edges = []

            # Create relationships ensuring consistent entity IDs
            for i, source_entity in enumerate(entity_ids):
                # Get service type for the source entity
                source_type = entity_df[entity_df['entity_id'] == source_entity]['service_type'].iloc[0]

                # Create 1-3 connections for each entity
                n_connections = np.random.randint(1, min(4, self.n_entities))
                possible_targets = [eid for eid in entity_ids if eid != source_entity]

                if possible_targets:
                    targets = np.random.choice(
                        possible_targets,
                        size=min(n_connections, len(possible_targets)),
                        replace=False
                    )

                    for target_entity in targets:
                        edges.append({
                            'source': source_entity,
                            'target': target_entity,
                            'service_type': source_type,
                            'relationship_type': 'communicates_with'
                        })

            relationships_df = pd.DataFrame(edges)
            logger.info(f"Generated {len(edges)} relationships between entities")

            # Validate the relationships DataFrame
            self._validate_entity_ids(relationships_df['source'])
            self._validate_entity_ids(relationships_df['target'])

            return relationships_df

        except Exception as e:
            logger.error(f"Error generating relationships: {str(e)}")
            raise

    def generate_time_series(self):
        """Generate time series data with explicit anomaly patterns"""
        try:
            # Get entity metadata to ensure consistent IDs
            entities_df = self.generate_entity_metadata()

            # Generate evenly spaced timestamps
            end_time = datetime.now().replace(microsecond=0)
            timestamps = [end_time - timedelta(minutes=x) for x in range(self.n_timestamps)]
            timestamps.reverse()  # Ensure chronological order

            # Generate system-level metrics
            system_metrics_df = self._generate_system_metrics(timestamps)

            data = []
            time_factor = np.array([t.hour / 24.0 + t.minute / (24.0 * 60.0) for t in timestamps])

            # Generate data for each entity
            for _, entity in entities_df.iterrows():
                entity_id = entity['entity_id']
                service_type = entity['service_type']

                # Generate base patterns with daily seasonality
                base_patterns = {
                    'CPUUtilization': 50 + 30 * np.sin(2 * np.pi * time_factor),
                    'MemoryUtilization': 60 + 20 * np.sin(2 * np.pi * time_factor + np.pi/4),
                    'NetworkIn': 1000 + 500 * np.sin(2 * np.pi * time_factor + np.pi/3),
                    'NetworkOut': 800 + 400 * np.sin(2 * np.pi * time_factor + np.pi/6),
                    'DiskReadBytes': 500 + 200 * np.sin(2 * np.pi * time_factor + np.pi/2),
                    'DiskWriteBytes': 400 + 150 * np.sin(2 * np.pi * time_factor + np.pi/3),
                    'DiskReadOps': 100 + 50 * np.sin(2 * np.pi * time_factor + np.pi/4),
                    'DiskWriteOps': 80 + 40 * np.sin(2 * np.pi * time_factor + np.pi/6)
                }

                # Adjust base patterns based on service type
                if service_type == 'RDS':
                    # Database instances typically have higher disk I/O
                    for metric in ['DiskReadBytes', 'DiskWriteBytes', 'DiskReadOps', 'DiskWriteOps']:
                        base_patterns[metric] *= 2.0
                elif service_type == 'Lambda':
                    # Lambda functions typically have lower disk I/O
                    for metric in ['DiskReadBytes', 'DiskWriteBytes', 'DiskReadOps', 'DiskWriteOps']:
                        base_patterns[metric] *= 0.3

                # Add controlled random variation and anomalies
                metric_values = {}
                for metric, pattern in base_patterns.items():
                    # Increase base noise level for more variability
                    noise = np.random.normal(0, pattern.std() * 0.4, self.n_timestamps)
                    values = pattern + noise

                    # Add more random anomalies
                    n_anomalies = np.random.randint(6, 12)
                    anomaly_indices = np.random.choice(
                        range(self.n_timestamps),
                        size=n_anomalies,
                        replace=False
                    )

                    for idx in anomaly_indices:
                        pattern_type = np.random.choice(['spike', 'drop', 'trend', 'seasonal_break'])

                        if pattern_type == 'spike':
                            values[idx] *= np.random.uniform(4.0, 6.0)
                            if idx + 1 < len(values):
                                values[idx + 1] *= np.random.uniform(2.0, 3.0)

                        elif pattern_type == 'drop':
                            values[idx] *= np.random.uniform(0.02, 0.15)
                            if idx + 1 < len(values):
                                values[idx + 1] *= np.random.uniform(0.3, 0.5)

                        elif pattern_type == 'trend':
                            shift_length = min(12, len(values) - idx)
                            trend_factor = np.random.uniform(2.5, 4.0)
                            if np.random.random() < 0.5:
                                values[idx:idx + shift_length] *= np.linspace(1, trend_factor, shift_length)
                            else:
                                values[idx:idx + shift_length] *= np.linspace(trend_factor, 1, shift_length)

                        else:  # seasonal_break
                            break_length = min(8, len(values) - idx)
                            seasonal_shift = pattern.std() * np.random.uniform(2.0, 3.0)
                            values[idx:idx + break_length] += seasonal_shift

                    # Add correlated anomalies
                    if metric in ['CPUUtilization', 'MemoryUtilization']:
                        # Correlate CPU and Memory spikes
                        n_corr_anomalies = np.random.randint(2, 5)
                        for _ in range(n_corr_anomalies):
                            idx = np.random.randint(0, self.n_timestamps)
                            values[idx] *= 5.0

                            if idx + 1 < len(values):
                                cascade_length = min(4, len(values) - idx)
                                cascade_factors = np.linspace(4.0, 1.0, cascade_length)
                                values[idx:idx + cascade_length] *= cascade_factors

                    # Add disk I/O correlation patterns
                    if metric.startswith('Disk'):
                        # Correlate disk read/write operations
                        n_io_anomalies = np.random.randint(2, 4)
                        for _ in range(n_io_anomalies):
                            idx = np.random.randint(0, self.n_timestamps)
                            values[idx] *= np.random.uniform(3.0, 5.0)

                    # Ensure values are within reasonable bounds
                    if metric in ['CPUUtilization', 'MemoryUtilization']:
                        values = np.clip(values, 0, 100)
                    else:  # Network and Disk metrics
                        values = np.clip(values, 0, None)

                    metric_values[metric] = values

                # Create data points for each timestamp
                for t in range(self.n_timestamps):
                    data_point = {
                        'timestamp': timestamps[t],
                        'entity_id': entity_id,
                        'CPUUtilization': metric_values['CPUUtilization'][t],
                        'MemoryUtilization': metric_values['MemoryUtilization'][t],
                        'NetworkIn': metric_values['NetworkIn'][t],
                        'NetworkOut': metric_values['NetworkOut'][t],
                        'DiskReadBytes': metric_values['DiskReadBytes'][t],
                        'DiskWriteBytes': metric_values['DiskWriteBytes'][t],
                        'DiskReadOps': metric_values['DiskReadOps'][t],
                        'DiskWriteOps': metric_values['DiskWriteOps'][t]
                    }
                    # Add system-level metrics
                    for metric in self.system_metrics:
                        data_point[metric] = system_metrics_df.loc[t, metric]
                    data.append(data_point)

            # Create DataFrame and ensure proper sorting
            timeseries_df = pd.DataFrame(data)
            timeseries_df = timeseries_df.sort_values(['timestamp', 'entity_id']).reset_index(drop=True)

            logger.info(f"Generated {len(timeseries_df)} time series records")
            logger.info(f"Time range: {timestamps[0]} to {timestamps[-1]}")
            logger.info("Added enhanced anomaly patterns to the data")

            return timeseries_df

        except Exception as e:
            logger.error(f"Error generating time series data: {str(e)}")
            raise