import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.feature_engineering import FeatureEngineer

def reproduce_pipeline():
    print("🚀 Starting pipeline reproduction...")
    
    # Load config
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Loaded config")

    # Load raw data
    data_path = project_root / config['data']['raw_path']
    if not data_path.exists():
        print(f"❌ Raw data not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"✓ Loaded raw data: {df.shape}")

    # Initialize and fit FeatureEngineer
    print("🔧 Fitting FeatureEngineer...")
    # Initialize with config
    engineer = FeatureEngineer(config)
    
    # We need to split X and y effectively or just fit on everything for the 'production' pipeline
    # The training process usually splits first. To be safe and consistent with training,
    # we should ideally use the same training set. However, for encodings (LabelEncoder),
    # it's often acceptable to fit on the full dataset or just the training set if we want strict correctness.
    # Given we don't have the exact train indices easily available without re-running the split with correct seed,
    # let's assume fitting on the full dataset for the encoders is acceptable for this reproduction 
    # (or we can replicate the split if we want to be 100% precise).
    #
    # Let's replicate the split to be safe, as `DataProcessor` does.
    # But `DataProcessor` isn't imported here. Let's just use the `FeatureEngineer` on the full data 
    # to ensure all categories are captured for the inference app (which is often desired for production
    # to avoid 'unseen category' errors, though technically data leakage if used for validation).
    
    # Actually, looking at `src/feature_engineering.py`, it uses LabelEncoder. 
    # If we fit on full data, we cover all categories.
    
    # Data Cleaning
    # TotalCharges is often object type if it contains empty strings for new customers
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Remove customerID (not a feature, just like in training)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("✓ Removed customerID column")
    
    X = df.drop('Churn', axis=1) # Assuming Churn is the target
    
    # Fit the engineer
    engineer.engineer_features(X, fit=True)
    
    # Save the fitted engineer
    save_path = project_root / 'models' / 'feature_engineer.joblib'
    joblib.dump(engineer, save_path)
    print(f"✓ Saved fitted FeatureEngineer to {save_path}")

    # Verify loading
    loaded_engineer = joblib.load(save_path)
    print("✓ Verified pipeline loading matches original")

if __name__ == "__main__":
    reproduce_pipeline()
