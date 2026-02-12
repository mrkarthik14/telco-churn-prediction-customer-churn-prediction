"""
Utility Functions
Helper functions used across the project
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict, filepath: str) -> None:
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def create_directory(path: str) -> Path:
    """Create directory if it doesn't exist"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def calculate_clv(monthly_charges: float, avg_tenure_months: float = 36) -> float:
    """
    Calculate Customer Lifetime Value
    
    Args:
        monthly_charges: Monthly charges in dollars
        avg_tenure_months: Average customer tenure in months
        
    Returns:
        Estimated CLV
    """
    return monthly_charges * avg_tenure_months


def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value*100:.1f}%"


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_metrics(metrics: Dict[str, float], filepath: str = None) -> None:
    """
    Log metrics to console and optionally to file
    
    Args:
        metrics: Dictionary of metrics
        filepath: Optional file path to save metrics
    """
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    if filepath:
        save_json(metrics, filepath)


def detect_data_drift(
    train_data: pd.DataFrame,
    new_data: pd.DataFrame,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Detect data drift using Population Stability Index (PSI)
    
    Args:
        train_data: Training dataset
        new_data: New dataset to compare
        threshold: PSI threshold for drift detection
        
    Returns:
        Dictionary with drift analysis
    """
    psi_values = {}
    drift_detected = {}
    
    for column in train_data.columns:
        if train_data[column].dtype in ['int64', 'float64']:
            # Calculate PSI for numerical columns
            psi = calculate_psi(train_data[column], new_data[column])
            psi_values[column] = psi
            drift_detected[column] = psi > threshold
    
    return {
        'psi_values': psi_values,
        'drift_detected': drift_detected,
        'columns_with_drift': [k for k, v in drift_detected.items() if v],
        'drift_percentage': sum(drift_detected.values()) / len(drift_detected) * 100
    }


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    buckets: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI)
    
    Args:
        expected: Expected distribution (training)
        actual: Actual distribution (new data)
        buckets: Number of buckets for binning
        
    Returns:
        PSI value
    """
    # Create bins
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    
    # Calculate percentiles
    expected_percents = np.percentile(expected.dropna(), breakpoints)
    
    # Bin both arrays
    expected_binned = np.digitize(expected, expected_percents)
    actual_binned = np.digitize(actual, expected_percents)
    
    # Calculate distributions
    expected_dist = pd.Series(expected_binned).value_counts(normalize=True).sort_index()
    actual_dist = pd.Series(actual_binned).value_counts(normalize=True).sort_index()
    
    # Align distributions
    all_bins = set(expected_dist.index) | set(actual_dist.index)
    expected_dist = expected_dist.reindex(all_bins, fill_value=0.0001)
    actual_dist = actual_dist.reindex(all_bins, fill_value=0.0001)
    
    # Calculate PSI
    psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
    
    return psi


def print_section_header(title: str, width: int = 70) -> None:
    """Print formatted section header"""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def print_subsection_header(title: str, width: int = 70) -> None:
    """Print formatted subsection header"""
    print("\n" + title)
    print("-"*width)


class PerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, log_file: str = 'models/performance_log.json'):
        self.log_file = log_file
        self.logs = self._load_logs()
    
    def _load_logs(self) -> List[Dict]:
        """Load existing logs"""
        if Path(self.log_file).exists():
            return load_json(self.log_file)
        return []
    
    def log_performance(
        self,
        model_name: str,
        metrics: Dict[str, float],
        data_info: Dict[str, Any] = None
    ) -> None:
        """
        Log model performance
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics
            data_info: Optional data information
        """
        log_entry = {
            'timestamp': get_timestamp(),
            'model_name': model_name,
            'metrics': metrics,
            'data_info': data_info or {}
        }
        
        self.logs.append(log_entry)
        save_json(self.logs, self.log_file)
    
    def check_degradation(
        self,
        model_name: str,
        metric: str = 'f1',
        threshold: float = 0.05
    ) -> bool:
        """
        Check if model performance has degraded
        
        Args:
            model_name: Name of the model
            metric: Metric to check
            threshold: Degradation threshold
            
        Returns:
            True if degradation detected
        """
        model_logs = [
            log for log in self.logs 
            if log['model_name'] == model_name
        ]
        
        if len(model_logs) < 2:
            return False
        
        # Compare latest with baseline (first)
        baseline = model_logs[0]['metrics'][metric]
        latest = model_logs[-1]['metrics'][metric]
        
        degradation = baseline - latest
        
        if degradation > threshold:
            print(f"⚠️  Performance degradation detected!")
            print(f"   Baseline {metric}: {baseline:.4f}")
            print(f"   Latest {metric}: {latest:.4f}")
            print(f"   Degradation: {degradation:.4f}")
            return True
        
        return False


if __name__ == "__main__":
    print("Utility functions module")
    print("Import this module to use helper functions")
