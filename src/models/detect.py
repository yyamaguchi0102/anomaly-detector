import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging
from datetime import datetime
import time
import argparse
from typing import Dict, Any
from colorama import init, Fore, Style
import statistics

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.stream_simulator import LogStreamSimulator

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.total_logs = 0
        self.flagged_logs = 0
        self.processing_times = []
        
    def start(self):
        """Start the performance monitoring."""
        self.start_time = time.time()
        
    def log_processing(self, is_anomaly: bool, processing_time: float):
        """Log the processing of a single log entry."""
        self.total_logs += 1
        if is_anomaly:
            self.flagged_logs += 1
        self.processing_times.append(processing_time)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the performance metrics."""
        if not self.start_time:
            return {}
            
        elapsed_time = time.time() - self.start_time
        avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'total_logs': self.total_logs,
            'flagged_logs': self.flagged_logs,
            'flag_rate': (self.flagged_logs / self.total_logs * 100) if self.total_logs > 0 else 0,
            'avg_processing_time': avg_processing_time,
            'total_elapsed_time': elapsed_time,
            'logs_per_second': self.total_logs / elapsed_time if elapsed_time > 0 else 0
        }

class AnomalyDetector:
    def __init__(self, model_path: str, threshold: float = None):
        """
        Initialize the anomaly detector.
        
        Args:
            model_path (str): Path to the saved model
            threshold (float, optional): Anomaly score threshold. If None, will use model's contamination
        """
        # Load model and feature engineer
        logger.info(f"Loading model from {model_path}")
        saved_data = joblib.load(model_path)
        self.model = saved_data['model']
        self.feature_engineer = saved_data['feature_engineer']
        self.training_timestamp = saved_data['timestamp']
        
        # Set threshold
        self.threshold = threshold
        
        # Initialize buffer for rolling features
        self.buffer = pd.DataFrame()
        
    def _update_buffer(self, new_data: pd.DataFrame):
        """Update the rolling feature buffer with new data."""
        self.buffer = pd.concat([self.buffer, new_data])
        # Keep only last 24 hours of data
        cutoff_time = new_data['timestamp'].max() - pd.Timedelta(hours=24)
        self.buffer = self.buffer[self.buffer['timestamp'] >= cutoff_time]
        
    def detect(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in a single log entry.
        
        Args:
            log_entry (dict): Single log entry with all required fields
            
        Returns:
            dict: Original log entry with added anomaly score and prediction
        """
        try:
            # Convert single entry to DataFrame
            df = pd.DataFrame([log_entry])
            
            # Update buffer for rolling features
            self._update_buffer(df)
            
            # Transform features
            X = self.feature_engineer.transform(self.buffer)
            
            # Get the last row (current entry)
            X_current = X.iloc[[-1]]
            
            # Calculate anomaly score
            score = self.model.score_samples(X_current)[0]
            anomaly_score = -score  # Convert to positive scale
            
            # Make prediction
            is_anomaly = anomaly_score > self.threshold if self.threshold else self.model.predict(X_current)[0] == -1
            
            # Add results to log entry
            log_entry['anomaly_score'] = anomaly_score
            log_entry['is_anomaly'] = is_anomaly
            log_entry['detection_timestamp'] = datetime.now()
            
            return log_entry
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            raise

def print_log_entry(log_entry: Dict[str, Any]):
    """Print a log entry with color coding."""
    # Determine color based on anomaly status
    color = Fore.RED if log_entry['is_anomaly'] else Fore.GREEN
    status = "ANOMALY" if log_entry['is_anomaly'] else "NORMAL"
    
    # Format the output
    output = (
        f"{color}[{status}] {Style.RESET_ALL}"
        f"Time: {log_entry['timestamp']} | "
        f"User: {log_entry['user_id']} | "
        f"IP: {log_entry['ip_address']} | "
        f"Action: {log_entry['action']} | "
        f"Status: {log_entry['status_code']} | "
        f"Score: {log_entry['anomaly_score']:.3f}"
    )
    
    print(output)

def print_performance_summary(summary: Dict[str, Any]):
    """Print the performance summary."""
    print("\n" + "="*50)
    print(f"{Fore.CYAN}Performance Summary{Style.RESET_ALL}")
    print("="*50)
    print(f"Total Logs Processed: {summary['total_logs']}")
    print(f"Anomalies Flagged: {summary['flagged_logs']} ({summary['flag_rate']:.1f}%)")
    print(f"Average Processing Time: {summary['avg_processing_time']*1000:.2f} ms")
    print(f"Total Elapsed Time: {summary['total_elapsed_time']:.1f} seconds")
    print(f"Processing Rate: {summary['logs_per_second']:.1f} logs/second")
    print("="*50)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time anomaly detection')
    parser.add_argument('--data', default='data/raw/sample_logs.csv',
                      help='Path to the log data file')
    parser.add_argument('--model', default='data/processed/isolation_forest_model.joblib',
                      help='Path to the trained model')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Anomaly score threshold (optional)')
    parser.add_argument('--speed', type=float, default=1.0,
                      help='Simulation speed factor (1.0 = real-time)')
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        stream_simulator = LogStreamSimulator(args.data, args.speed)
        detector = AnomalyDetector(args.model, args.threshold)
        performance_monitor = PerformanceMonitor()
        
        # Start monitoring
        performance_monitor.start()
        
        # Process the stream
        for log_entry in stream_simulator.stream():
            # Measure processing time
            start_time = time.time()
            
            # Detect anomalies
            result = detector.detect(log_entry)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update performance metrics
            performance_monitor.log_processing(result['is_anomaly'], processing_time)
            
            # Print the result
            print_log_entry(result)
            
        # Print performance summary
        print_performance_summary(performance_monitor.get_summary())
        
    except KeyboardInterrupt:
        print("\nStream interrupted by user")
        print_performance_summary(performance_monitor.get_summary())
    except Exception as e:
        logger.error(f"Error in real-time detection: {str(e)}")
        raise

if __name__ == "__main__":
    main() 