"""
Unit Tests for Data Processing Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import DataProcessor


@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    return pd.DataFrame({
        'customerID': ['C001', 'C002', 'C003'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'tenure': [12, 24, 6],
        'MonthlyCharges': [50.0, 75.0, 60.0],
        'TotalCharges': ['600.0', '1800.0', '360.0'],
        'Churn': ['No', 'Yes', 'No']
    })


@pytest.fixture
def config():
    """Mock configuration"""
    return {
        'data': {
            'raw_path': 'data/raw/test.csv',
            'processed_path': 'data/processed/',
            'test_size': 0.2,
            'random_state': 42,
            'stratify': True
        }
    }


class TestDataProcessor:
    """Test suite for DataProcessor class"""
    
    def test_clean_data_removes_customer_id(self, sample_data, config):
        """Test that customerID column is removed"""
        processor = DataProcessor(config)
        cleaned = processor.clean_data(sample_data)
        assert 'customerID' not in cleaned.columns
    
    def test_clean_data_fixes_total_charges(self, sample_data, config):
        """Test that TotalCharges is converted to numeric"""
        processor = DataProcessor(config)
        cleaned = processor.clean_data(sample_data)
        assert cleaned['TotalCharges'].dtype in ['int64', 'float64']
    
    def test_split_features_target(self, sample_data, config):
        """Test feature-target split"""
        processor = DataProcessor(config)
        cleaned = processor.clean_data(sample_data)
        X, y = processor.split_features_target(cleaned)
        
        assert 'Churn' not in X.columns
        assert len(y) == len(X)
        assert y.dtype in ['int64', 'int32']
    
    def test_get_feature_types(self, sample_data, config):
        """Test feature type categorization"""
        processor = DataProcessor(config)
        cleaned = processor.clean_data(sample_data)
        X, y = processor.split_features_target(cleaned)
        
        feature_types = processor.get_feature_types(X)
        
        assert 'numerical' in feature_types
        assert 'categorical' in feature_types
        assert 'binary' in feature_types
        assert isinstance(feature_types['numerical'], list)
    
    def test_validate_data_checks_required_columns(self, sample_data, config):
        """Test data validation"""
        processor = DataProcessor(config)
        cleaned = processor.clean_data(sample_data)
        
        # Should pass validation
        assert processor.validate_data(cleaned) == True
        
        # Should fail if required column missing
        cleaned_missing = cleaned.drop('tenure', axis=1)
        assert processor.validate_data(cleaned_missing) == False


class TestFeatureEngineering:
    """Test suite for feature engineering (placeholder)"""
    
    def test_placeholder(self):
        """Placeholder test"""
        assert True


class TestModelTraining:
    """Test suite for model training (placeholder)"""
    
    def test_placeholder(self):
        """Placeholder test"""
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
