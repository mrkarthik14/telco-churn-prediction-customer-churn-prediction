"""
Feature Engineering Module
Creates new features and transforms existing ones for better model performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Handles feature creation and transformation"""
    
    def __init__(self, config: dict):
        """
        Initialize FeatureEngineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def create_tenure_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based features
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with new tenure features
        """
        X_new = X.copy()
        
        if 'tenure' in X_new.columns:
            # Tenure buckets (customer lifecycle stages)
            buckets = self.config['features']['tenure_buckets']
            X_new['tenure_bucket'] = pd.cut(
                X_new['tenure'],
                bins=buckets + [np.inf],
                labels=[f'{buckets[i]}-{buckets[i+1]}' for i in range(len(buckets)-1)] + [f'{buckets[-1]}+'],
                include_lowest=True
            )
            
            # Early customer (first year)
            X_new['is_early_customer'] = (X_new['tenure'] <= 12).astype(int)
            
            # Long-term customer (>5 years)
            X_new['is_longterm_customer'] = (X_new['tenure'] >= 60).astype(int)
            
            # Tenure in years
            X_new['tenure_years'] = X_new['tenure'] / 12
            
            print("  ✓ Created tenure features")
        
        return X_new
    
    def create_monetary_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create monetary value features
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with monetary features
        """
        X_new = X.copy()
        
        if 'MonthlyCharges' in X_new.columns and 'tenure' in X_new.columns:
            # Average charges per month of tenure
            X_new['avg_charge_per_tenure'] = (
                X_new['MonthlyCharges'] / (X_new['tenure'] + 1)
            )
            
            # Charge increase rate (if TotalCharges available)
            if 'TotalCharges' in X_new.columns:
                expected_total = X_new['MonthlyCharges'] * X_new['tenure']
                X_new['charge_variance'] = (
                    X_new['TotalCharges'] - expected_total
                ) / (expected_total + 1)
                
                # High/Low spender
                monthly_median = X_new['MonthlyCharges'].median()
                X_new['is_high_spender'] = (
                    X_new['MonthlyCharges'] > monthly_median
                ).astype(int)
            
            print("  ✓ Created monetary features")
        
        return X_new
    
    def create_service_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create service usage features
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with service features
        """
        X_new = X.copy()
        
        # Service columns
        service_cols = [
            'PhoneService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Count total services
        available_services = [col for col in service_cols if col in X_new.columns]
        
        if available_services:
            # Total number of services
            service_df = X_new[available_services].copy()
            # Convert Yes/No to 1/0 if needed
            for col in available_services:
                if service_df[col].dtype == 'object':
                    service_df[col] = service_df[col].map({'Yes': 1, 'No': 0})
            
            X_new['total_services'] = service_df.sum(axis=1)
            
            # Has streaming services
            streaming_cols = [c for c in ['StreamingTV', 'StreamingMovies'] if c in service_df.columns]
            if streaming_cols:
                X_new['has_streaming'] = (service_df[streaming_cols].sum(axis=1) > 0).astype(int)
            
            # Has security services
            security_cols = [c for c in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport'] 
                           if c in service_df.columns]
            if security_cols:
                X_new['has_security'] = (service_df[security_cols].sum(axis=1) > 0).astype(int)
            
            print("  ✓ Created service features")
        
        return X_new
    
    def create_contract_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create contract-related features
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with contract features
        """
        X_new = X.copy()
        
        if 'Contract' in X_new.columns:
            # Is month-to-month (high risk)
            X_new['is_month_to_month'] = (
                X_new['Contract'] == 'Month-to-month'
            ).astype(int)
            
            # Long-term contract
            X_new['has_long_contract'] = (
                X_new['Contract'].isin(['One year', 'Two year'])
            ).astype(int)
        
        if 'PaymentMethod' in X_new.columns:
            # Electronic check (associated with higher churn)
            X_new['pays_electronic_check'] = (
                X_new['PaymentMethod'] == 'Electronic check'
            ).astype(int)
            
            # Auto payment
            X_new['has_auto_payment'] = (
                X_new['PaymentMethod'].str.contains('automatic', case=False, na=False)
            ).astype(int)
        
        print("  ✓ Created contract features")
        return X_new
    
    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with interaction features
        """
        X_new = X.copy()
        
        # Senior citizen interactions
        if 'SeniorCitizen' in X_new.columns and 'MonthlyCharges' in X_new.columns:
            X_new['senior_monthly_charges'] = (
                X_new['SeniorCitizen'] * X_new['MonthlyCharges']
            )
        
        # Contract × Tenure interaction
        if 'is_month_to_month' in X_new.columns and 'tenure' in X_new.columns:
            X_new['mtm_tenure_interaction'] = (
                X_new['is_month_to_month'] * X_new['tenure']
            )
        
        # Internet service × Total services
        if 'InternetService' in X_new.columns and 'total_services' in X_new.columns:
            has_internet = (X_new['InternetService'] != 'No').astype(int)
            X_new['internet_services_interaction'] = (
                has_internet * X_new['total_services']
            )
        
        print("  ✓ Created interaction features")
        return X_new
    
    def encode_categorical_features(
        self, 
        X: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            X: Input features
            fit: Whether to fit encoders (True for train, False for test)
            
        Returns:
            DataFrame with encoded features
        """
        X_new = X.copy()
        
        if fit:
            # Get categorical columns
            categorical_cols = X_new.select_dtypes(include=['object']).columns.tolist()
            
            # Remove tenure_bucket if it exists (we'll one-hot encode it)
            if 'tenure_bucket' in categorical_cols:
                categorical_cols.remove('tenure_bucket')
            
            self.categorical_cols = categorical_cols
        else:
            # Use columns from fit
            categorical_cols = getattr(self, 'categorical_cols', [])
            if not categorical_cols:
                # Fallback if not fitted properly or no cat cols
                categorical_cols = [col for col in X_new.columns if col in self.encoders]
        
        for col in categorical_cols:
            if col in X_new.columns:
                if fit:
                    # Fit and transform
                    le = LabelEncoder()
                    X_new[col] = le.fit_transform(X_new[col].astype(str))
                    self.encoders[col] = le
                else:
                    # Transform only (for test data)
                    if col in self.encoders:
                        # Handle unseen categories
                        le = self.encoders[col]
                        # Use apply to handle single values safely
                        X_new[col] = X_new[col].apply(
                            lambda x: le.transform([str(x)])[0] 
                            if str(x) in le.classes_ else -1
                        )
                    else:
                        print(f"  ⚠️  Warning: No encoder found for {col}")
        
        # One-hot encode tenure_bucket
        if 'tenure_bucket' in X_new.columns:
            X_new = pd.get_dummies(X_new, columns=['tenure_bucket'], prefix='tenure')
        
        print(f"  ✓ Encoded {len(categorical_cols)} categorical features")
        return X_new
    
    def scale_numerical_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            X: Input features
            fit: Whether to fit scalers
            
        Returns:
            DataFrame with scaled features
        """
        X_new = X.copy()
        
        if fit:
            # Get numerical columns (exclude binary)
            numerical_cols = X_new.select_dtypes(include=['int64', 'float64']).columns
            numerical_cols = [
                col for col in numerical_cols 
                if X_new[col].nunique() > 2  # Not binary
            ]
            self.numerical_cols = numerical_cols
            
            scaler = StandardScaler()
            X_new[numerical_cols] = scaler.fit_transform(X_new[numerical_cols])
            self.scalers['standard'] = scaler
        else:
            # Use columns from fit
            numerical_cols = getattr(self, 'numerical_cols', [])
            
            if 'standard' in self.scalers:
                scaler = self.scalers['standard']
                # Ensure columns exist
                valid_cols = [c for c in numerical_cols if c in X_new.columns]
                if valid_cols:
                    X_new[valid_cols] = scaler.transform(X_new[valid_cols])
        
        print(f"  ✓ Scaled {len(numerical_cols)} numerical features")
        return X_new
    
    def engineer_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Run full feature engineering pipeline
        
        Args:
            X: Input features
            fit: Whether to fit transformers
            
        Returns:
            Engineered features
        """
        print("\n🔧 Engineering features...")
        
        # Create new features
        X_eng = self.create_tenure_features(X)
        X_eng = self.create_monetary_features(X_eng)
        X_eng = self.create_service_features(X_eng)
        X_eng = self.create_contract_features(X_eng)
        X_eng = self.create_interaction_features(X_eng)
        
        # Encode categorical
        X_eng = self.encode_categorical_features(X_eng, fit=fit)
        
        # Scale numerical
        X_eng = self.scale_numerical_features(X_eng, fit=fit)
        
        if fit:
            self.feature_names = X_eng.columns.tolist()
        
        print(f"\n✓ Feature engineering complete: {X_eng.shape[1]} features")
        return X_eng
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names after engineering"""
        return self.feature_names


# Example usage
if __name__ == "__main__":
    import yaml
    from data_processing import DataProcessor
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and process data
    processor = DataProcessor(config)
    X, y = processor.process_pipeline()
    
    # Engineer features
    engineer = FeatureEngineer(config)
    X_eng = engineer.engineer_features(X, fit=True)
    
    print("\nEngineered features:")
    print(X_eng.head())
    print(f"\nFeature count: {len(X_eng.columns)}")
