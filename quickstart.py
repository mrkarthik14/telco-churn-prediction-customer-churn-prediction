#!/usr/bin/env python3
"""
Quick Start Script
Runs a simplified version of the ML pipeline for quick testing
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   TELCO CUSTOMER CHURN PREDICTION - QUICK START              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")


def check_dependencies():
    """Check if required packages are installed"""
    print("\n[1/5] Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'matplotlib', 'seaborn', 'yaml'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"  ❌ Missing packages: {', '.join(missing)}")
        print(f"\n  Install with: pip install {' '.join(missing)}")
        return False
    
    print("  ✓ All dependencies installed")
    return True


def check_data():
    """Check if dataset exists"""
    print("\n[2/5] Checking for dataset...")
    
    data_path = project_root / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if data_path.exists():
        print(f"  ✓ Dataset found: {data_path}")
        return True
    else:
        print(f"  ❌ Dataset not found at: {data_path}")
        print("\n  To download the dataset:")
        print("    1. Run: python data/download_data.py")
        print("    2. Or download manually from:")
        print("       https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        return False


def run_quick_pipeline():
    """Run a simplified pipeline"""
    print("\n[3/5] Running quick pipeline...")
    
    import yaml
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.data_processing import DataProcessor
    from src.feature_engineering import FeatureEngineer
    from src.models import ModelTrainer
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Disable all but one model for speed
    for model_name in config['models']:
        config['models'][model_name]['enabled'] = False
        config['models'][model_name]['tune'] = False
    
    # Enable only XGBoost
    config['models']['xgboost']['enabled'] = True
    
    print("  Processing data...")
    processor = DataProcessor(config)
    X, y = processor.process_pipeline()
    
    print("  Engineering features...")
    engineer = FeatureEngineer(config)
    X_eng = engineer.engineer_features(X, fit=True)
    
    print("  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("  Training XGBoost model (this may take a minute)...")
    trainer = ModelTrainer(config)
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    return trainer, results


def display_results(trainer, results):
    """Display results summary"""
    print("\n[4/5] Results Summary")
    print("-" * 70)
    
    for model_name, result in results.items():
        metrics = result['test_metrics']
        business = result['business_value']
        
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")
        print(f"\n  Business Value: ${business['net_business_value']:,.0f}")
        print(f"  ROI:            {business['roi_percentage']:.1f}%")


def next_steps():
    """Show next steps"""
    print("\n[5/5] Next Steps")
    print("-" * 70)
    print("""
✅ Quick start complete! Here's what to do next:

📊 EXPLORE IN DEPTH:
  1. Run full pipeline:
     python -m src.main --train --evaluate
  
  2. Open Jupyter notebooks:
     jupyter notebook notebooks/01_EDA.ipynb
  
  3. Generate all visualizations:
     python -m src.main --train --evaluate --shap

🚀 DEPLOY:
  4. Start the API:
     cd api && python app.py
  
  5. Test predictions:
     curl -X POST http://localhost:8000/predict \\
       -H "Content-Type: application/json" \\
       -d @api/sample_request.json

📖 LEARN MORE:
  6. Read executive summary: reports/executive_summary.md
  7. Read model card: reports/model_card.md
  8. Check documentation: README.md

💡 Need help? Check the README.md or open an issue on GitHub.
""")


def main():
    """Main execution"""
    try:
        # Step 1: Check dependencies
        if not check_dependencies():
            return
        
        # Step 2: Check data
        if not check_data():
            return
        
        # Step 3: Run pipeline
        trainer, results = run_quick_pipeline()
        
        # Step 4: Display results
        display_results(trainer, results)
        
        # Step 5: Next steps
        next_steps()
        
        print("\n" + "="*70)
        print("✨ SUCCESS! Quick start completed.")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nFor detailed error information, run:")
        print("  python -m src.main --train --evaluate")
        sys.exit(1)


if __name__ == "__main__":
    main()
