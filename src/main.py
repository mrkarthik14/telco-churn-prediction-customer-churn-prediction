"""
Main Pipeline Script
Runs the complete end-to-end ML pipeline
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict, args: argparse.Namespace):
    """
    Run complete ML pipeline
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("TELCO CUSTOMER CHURN PREDICTION - FULL PIPELINE")
    print("="*70)
    
    # 1. Data Processing
    print("\n[STEP 1/5] DATA PROCESSING")
    print("-" * 70)
    processor = DataProcessor(config)
    X, y = processor.process_pipeline()
    
    # 2. Feature Engineering
    print("\n[STEP 2/5] FEATURE ENGINEERING")
    print("-" * 70)
    engineer = FeatureEngineer(config)
    X_eng = engineer.engineer_features(X, fit=True)
    
    # 3. Train-Test Split
    print("\n[STEP 3/5] TRAIN-TEST SPLIT")
    print("-" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y if config['data']['stratify'] else None
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # 4. Model Training
    if args.train:
        print("\n[STEP 4/5] MODEL TRAINING")
        print("-" * 70)
        trainer = ModelTrainer(config)
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Get best model
        best_name, best_model = trainer.get_best_model('f1')
        print(f"\n🏆 Best Model: {best_name}")
        print(f"   F1 Score: {results[best_name]['test_metrics']['f1']:.4f}")
        
        # Save models
        if args.save_models:
            print("\n💾 Saving models...")
            trainer.save_models('models')
    else:
        print("\n[STEP 4/5] MODEL TRAINING - SKIPPED")
        print("-" * 70)
        print("  Use --train flag to train models")
        return
    
    # 5. Evaluation & Visualization
    if args.evaluate:
        print("\n[STEP 5/5] MODEL EVALUATION")
        print("-" * 70)
        evaluator = ModelEvaluator(config)
        
        # Generate comparison report
        comparison_df = evaluator.generate_comparison_report(results, save=True)
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Detailed evaluation for best model
        print(f"\n📊 Detailed evaluation for {best_name}...")
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Plots
        evaluator.plot_confusion_matrix(y_test, y_pred, best_name, save=True)
        evaluator.plot_roc_curve(y_test, y_pred_proba, best_name, save=True)
        evaluator.plot_precision_recall_curve(y_test, y_pred_proba, best_name, save=True)
        
        # Feature importance
        feature_names = engineer.get_feature_importance_names()
        evaluator.plot_feature_importance(
            best_model, feature_names, best_name, top_n=20, save=True
        )
        
        # Threshold analysis
        threshold_df = evaluator.plot_threshold_analysis(
            y_test, y_pred_proba, best_name, save=True
        )
        
        # SHAP analysis (can be slow for large datasets)
        if args.shap:
            evaluator.plot_shap_summary(best_model, X_test, best_name, save=True)
        
        print("\n✓ Evaluation complete. Plots saved to reports/figures/")
    else:
        print("\n[STEP 5/5] MODEL EVALUATION - SKIPPED")
        print("-" * 70)
        print("  Use --evaluate flag to generate evaluation plots")
    
    # Pipeline complete
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    
    # Next steps
    print("\n📋 NEXT STEPS:")
    print("  1. Review evaluation plots in reports/figures/")
    print("  2. Check model comparison in reports/figures/model_comparison.csv")
    print("  3. Start API server: cd api && python app.py")
    print("  4. Explore notebooks for detailed analysis")
    print("\n💡 TIP: Run with --help to see all options")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Telco Churn Prediction - ML Pipeline'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train models'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Generate evaluation plots'
    )
    
    parser.add_argument(
        '--save-models',
        action='store_true',
        default=True,
        help='Save trained models'
    )
    
    parser.add_argument(
        '--shap',
        action='store_true',
        help='Generate SHAP plots (can be slow)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run pipeline
    run_pipeline(config, args)


if __name__ == "__main__":
    main()
