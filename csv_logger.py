"""
CSV Logger utility for training scripts
Logs training metrics to CSV files for easy analysis
"""
import csv
import os
from datetime import datetime
from typing import Dict, List, Union, Any


class CSVLogger:
    """
    A utility class to log training metrics to CSV files.
    
    Usage:
        logger = CSVLogger(log_dir='logs', filename='training_log.csv')
        logger.log({'epoch': 1, 'train_loss': 0.5, 'val_loss': 0.3})
        logger.close()
    """
    
    def __init__(self, log_dir: str, filename: str = None, fieldnames: List[str] = None):
        """
        Initialize CSV Logger
        
        Args:
            log_dir: Directory to save the CSV log file
            filename: Name of the CSV file (default: training_log_TIMESTAMP.csv)
            fieldnames: List of column names for the CSV file
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.csv"
        
        self.log_file = os.path.join(log_dir, filename)
        self.fieldnames = fieldnames
        self.file_handle = None
        self.writer = None
        self.header_written = False
        
    def log(self, metrics: Dict[str, Any]):
        """
        Log a dictionary of metrics to the CSV file
        
        Args:
            metrics: Dictionary containing metric names and values
        """
        # Open file if not already open
        if self.file_handle is None:
            self.file_handle = open(self.log_file, 'w', newline='', encoding='utf-8')
            
            # Initialize fieldnames if not provided
            if self.fieldnames is None:
                self.fieldnames = list(metrics.keys())
            
            self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
            self.writer.writeheader()
            self.header_written = True
        
        # Write the metrics
        self.writer.writerow(metrics)
        self.file_handle.flush()  # Ensure data is written immediately
        
    def close(self):
        """Close the CSV file"""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
            self.writer = None
            
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        
    def __del__(self):
        """Destructor to ensure file is closed"""
        self.close()
