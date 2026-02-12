"""
Data Processing Module
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Handles all data loading and preprocessing operations"""
    
    def __init__(self, config: dict):
        """
        Initialize DataProcessor
        
        Args:
            config: Configuration dictionary with data processing parameters
        """
        self.config = config
        self.raw_data_path = config['data']['raw_path']
        self.processed_data_path = config['data']['processed_path']
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV
        
        Returns:
            DataFrame with raw data
        """
        print(f"📂 Loading data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        print(f"✓ Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset:
        - Handle missing values
        - Fix data types
        - Remove duplicates
        - Handle inconsistencies
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("\n🧹 Cleaning data...")
        df_clean = df.copy()
        
        # Remove customerID (not a feature)
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
            print("  ✓ Removed customerID column")
        
        # Fix TotalCharges data type issue
        if 'TotalCharges' in df_clean.columns:
            # TotalCharges has spaces for new customers - convert to numeric
            df_clean['TotalCharges'] = pd.to_numeric(
                df_clean['TotalCharges'], errors='coerce'
            )
            
            # Fill missing TotalCharges with 0 (new customers)
            missing_count = df_clean['TotalCharges'].isna().sum()
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
            print(f"  ✓ Fixed TotalCharges: {missing_count} missing values filled")
        
        # Check for duplicates
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            print(f"  ✓ Removed {duplicates} duplicate rows")
        
        # Standardize Yes/No to binary
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
        
        # Convert SeniorCitizen to more readable format
        if 'SeniorCitizen' in df_clean.columns:
            df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(int)
        
        # Handle 'No internet service' and 'No phone service'
        service_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        for col in service_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].replace({
                    'No internet service': 'No',
                    'No phone service': 'No'
                })
        
        print(f"✓ Data cleaned: {len(df_clean):,} rows remaining")
        return df_clean
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        print("\n🔍 Validating data quality...")
        
        # Check required columns
        required_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            print(f"  ❌ Missing required columns: {missing_cols}")
            return False
        
        # Check for excessive missing values
        missing_pct = (df.isna().sum() / len(df) * 100).round(2)
        high_missing = missing_pct[missing_pct > 20]
        if len(high_missing) > 0:
            print(f"  ⚠️  Columns with >20% missing:")
            for col, pct in high_missing.items():
                print(f"     - {col}: {pct}%")
        
        # Check target variable
        if 'Churn' in df.columns:
            churn_dist = df['Churn'].value_counts(normalize=True)
            print(f"\n  Target Distribution:")
            print(f"     - No Churn: {churn_dist.get('No', 0)*100:.1f}%")
            print(f"     - Churn: {churn_dist.get('Yes', 0)*100:.1f}%")
            
            if churn_dist.get('Yes', 0) < 0.05:
                print("  ⚠️  Warning: Severe class imbalance (<5% churn)")
        
        print("✓ Validation complete")
        return True
    
    def split_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split features and target variable
        
        Args:
            df: Full DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        if 'Churn' not in df.columns:
            raise ValueError("Target column 'Churn' not found")
        
        # Convert target to binary
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        X = df.drop('Churn', axis=1)
        
        print(f"\n📊 Features: {X.shape[1]} columns")
        print(f"📊 Target: {len(y)} samples")
        print(f"   - Churn rate: {y.mean()*100:.2f}%")
        
        return X, y
    
    def get_feature_types(self, X: pd.DataFrame) -> dict:
        """
        Categorize features by type
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with lists of column names by type
        """
        numerical_features = X.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        categorical_features = X.select_dtypes(
            include=['object']
        ).columns.tolist()
        
        # Binary features (0/1)
        binary_features = [
            col for col in numerical_features 
            if X[col].nunique() == 2
        ]
        
        # Remove binary from numerical
        numerical_features = [
            col for col in numerical_features 
            if col not in binary_features
        ]
        
        return {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'binary': binary_features
        }
    
    def save_processed_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        suffix: str = 'train'
    ) -> None:
        """
        Save processed data to disk
        
        Args:
            X: Features DataFrame
            y: Target Series
            suffix: File suffix (train/test)
        """
        # Create directory if it doesn't exist
        Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)
        
        # Save features and target
        X_path = f"{self.processed_data_path}X_{suffix}.csv"
        y_path = f"{self.processed_data_path}y_{suffix}.csv"
        
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False, header=True)
        
        print(f"\n💾 Saved processed data:")
        print(f"   - Features: {X_path}")
        print(f"   - Target: {y_path}")
    
    def process_pipeline(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run full processing pipeline
        
        Returns:
            Processed features and target
        """
        print("="*60)
        print("DATA PROCESSING PIPELINE")
        print("="*60)
        
        # Load
        df = self.load_raw_data()
        
        # Clean
        df_clean = self.clean_data(df)
        
        # Validate
        self.validate_data(df_clean)
        
        # Split
        X, y = self.split_features_target(df_clean)
        
        # Get feature types
        feature_types = self.get_feature_types(X)
        print(f"\n📋 Feature Types:")
        print(f"   - Numerical: {len(feature_types['numerical'])}")
        print(f"   - Categorical: {len(feature_types['categorical'])}")
        print(f"   - Binary: {len(feature_types['binary'])}")
        
        print("\n" + "="*60)
        print("✓ PROCESSING COMPLETE")
        print("="*60)
        
        return X, y


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Process data
    processor = DataProcessor(config)
    X, y = processor.process_pipeline()
    
    # Display sample
    print("\nSample Data:")
    print(X.head())
    print(f"\nTarget distribution:\n{y.value_counts()}")
