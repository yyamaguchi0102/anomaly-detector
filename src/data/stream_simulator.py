import pandas as pd
import time
from datetime import datetime
import logging
from typing import Generator, Dict, Any

logger = logging.getLogger(__name__)

class LogStreamSimulator:
    def __init__(self, data_path: str, speed_factor: float = 1.0):
        """
        Initialize the log stream simulator.
        
        Args:
            data_path (str): Path to the CSV file containing logs
            speed_factor (float): Speed multiplier for simulation (1.0 = real-time)
        """
        self.data_path = data_path
        self.speed_factor = speed_factor
        self.df = None
        self.start_time = None
        self.current_idx = 0
        
    def _load_data(self):
        """Load and prepare the log data."""
        logger.info(f"Loading log data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp')
        
    def _calculate_delay(self, current_timestamp: datetime, next_timestamp: datetime) -> float:
        """Calculate the delay needed between log entries."""
        if self.start_time is None:
            return 0.0
            
        # Calculate real time difference
        real_diff = (next_timestamp - current_timestamp).total_seconds()
        # Adjust for speed factor
        return real_diff / self.speed_factor
        
    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream log entries one at a time.
        
        Yields:
            dict: A single log entry
        """
        if self.df is None:
            self._load_data()
            
        self.start_time = datetime.now()
        last_timestamp = None
        
        for idx, row in self.df.iterrows():
            # Calculate delay if not first entry
            if last_timestamp is not None:
                delay = self._calculate_delay(last_timestamp, row['timestamp'])
                if delay > 0:
                    time.sleep(delay)
            
            # Convert row to dict and yield
            log_entry = row.to_dict()
            last_timestamp = row['timestamp']
            self.current_idx = idx
            
            yield log_entry
            
    def get_progress(self) -> float:
        """Get the current progress of the stream (0.0 to 1.0)."""
        if self.df is None:
            return 0.0
        return (self.current_idx + 1) / len(self.df) 