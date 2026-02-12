"""
Models Module
Handles model training, hyperparameter tuning, and prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any, List
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: dict):
        """
        Initialize ModelTrainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.best_models = {}
        self.results = {}
        
    def get_model(self, model_name: str) -> Any:
        """
        Get model instance based on name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance
        """
        model_config = self.config['models'][model_name]
        params = model_config['params']
        
        if model_name == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'xgboost':
            return XGBClassifier(**params, eval_metric='logloss')
        elif model_name == 'lightgbm':
            return LGBMClassifier(**params, verbose=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_baseline_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str
    ) -> Any:
        """
        Train baseline model without tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            Trained model
        """
        print(f"\n🎯 Training {model_name}...")
        
        model = self.get_model(model_name)
        model.fit(X_train, y_train)
        
        print(f"  ✓ {model_name} trained")
        return model
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str
    ) -> Any:
        """
        Tune hyperparameters using cross-validation
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            Best model after tuning
        """
        print(f"\n🔧 Tuning {model_name} hyperparameters...")
        
        model_config = self.config['models'][model_name]
        tune_config = self.config['tuning']
        
        # Get base model
        base_model = self.get_model(model_name)
        
        # Get param grid
        param_grid = model_config.get('tune_params', {})
        
        if not param_grid:
            print(f"  ⚠️  No tuning params specified for {model_name}")
            return self.train_baseline_model(X_train, y_train, model_name)
        
        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=tune_config['cv_folds'],
            shuffle=True,
            random_state=self.config['data']['random_state']
        )
        
        # Choose search method
        if tune_config['method'] == 'grid_search':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=tune_config['scoring'],
                n_jobs=-1,
                verbose=tune_config['verbose']
            )
        else:  # random_search
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=tune_config['n_iter'],
                cv=cv,
                scoring=tune_config['scoring'],
                n_jobs=-1,
                verbose=tune_config['verbose'],
                random_state=self.config['data']['random_state']
            )
        
        # Fit
        search.fit(X_train, y_train)
        
        print(f"  ✓ Best params: {search.best_params_}")
        print(f"  ✓ Best CV {tune_config['scoring']}: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv_folds: Number of folds
            
        Returns:
            Dictionary of CV scores
        """
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.config['data']['random_state']
        )
        
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_scores = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=metric, n_jobs=-1
            )
            cv_scores[f'{metric}_mean'] = scores.mean()
            cv_scores[f'{metric}_std'] = scores.std()
        
        return cv_scores
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        return metrics
    
    def calculate_business_value(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate business value metrics
        
        Args:
            metrics: Model performance metrics
            
        Returns:
            Business value calculations
        """
        business_config = self.config['evaluation']['business']
        
        # Extract confusion matrix values
        tp = metrics['true_positives']
        fp = metrics['false_positives']
        tn = metrics['true_negatives']
        fn = metrics['false_negatives']
        
        # Calculate business metrics
        clv = business_config['avg_clv']
        campaign_cost = business_config['campaign_cost_per_customer']
        retention_with = business_config['retention_rate_with_intervention']
        retention_without = business_config['retention_rate_without_intervention']
        fp_cost = business_config['false_positive_cost']
        
        # Value from true positives (correctly identified churners we save)
        tp_value = tp * clv * retention_with
        
        # Cost of false positives (wasted campaign spend + annoyance)
        fp_cost_total = fp * (campaign_cost + fp_cost)
        
        # Cost of false negatives (churners we missed)
        fn_cost_total = fn * clv * (1 - retention_without)
        
        # Net value
        net_value = tp_value - fp_cost_total - fn_cost_total
        
        # ROI
        total_campaign_cost = (tp + fp) * campaign_cost
        roi = (net_value / total_campaign_cost * 100) if total_campaign_cost > 0 else 0
        
        return {
            'true_positive_value': tp_value,
            'false_positive_cost': fp_cost_total,
            'false_negative_cost': fn_cost_total,
            'net_business_value': net_value,
            'campaign_cost': total_campaign_cost,
            'roi_percentage': roi,
            'value_per_customer_contacted': net_value / (tp + fp) if (tp + fp) > 0 else 0
        }
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict]:
        """
        Train all enabled models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of results for all models
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        results = {}
        
        for model_name, model_config in self.config['models'].items():
            if not model_config.get('enabled', False):
                print(f"\n⏭️  Skipping {model_name} (disabled)")
                continue
            
            # Train model (with or without tuning)
            if model_config.get('tune', False):
                model = self.tune_hyperparameters(X_train, y_train, model_name)
            else:
                model = self.train_baseline_model(X_train, y_train, model_name)
            
            # Store model
            self.models[model_name] = model
            
            # Cross-validation scores
            print(f"\n📊 Cross-validating {model_name}...")
            cv_scores = self.cross_validate_model(model, X_train, y_train)
            
            # Test set evaluation
            print(f"📊 Evaluating {model_name} on test set...")
            test_metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Business value
            business_value = self.calculate_business_value(test_metrics)
            
            # Combine results
            results[model_name] = {
                'cv_scores': cv_scores,
                'test_metrics': test_metrics,
                'business_value': business_value
            }
            
            # Print summary
            print(f"\n  Results for {model_name}:")
            print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"    Precision: {test_metrics['precision']:.4f}")
            print(f"    Recall:    {test_metrics['recall']:.4f}")
            print(f"    F1 Score:  {test_metrics['f1']:.4f}")
            print(f"    ROC-AUC:   {test_metrics['roc_auc']:.4f}")
            print(f"    Net Value: ${business_value['net_business_value']:,.0f}")
            print(f"    ROI:       {business_value['roi_percentage']:.1f}%")
        
        self.results = results
        
        print("\n" + "="*60)
        print("✓ ALL MODELS TRAINED")
        print("="*60)
        
        return results
    
    def get_best_model(self, metric: str = 'f1') -> Tuple[str, Any]:
        """
        Get best performing model
        
        Args:
            metric: Metric to optimize for
            
        Returns:
            Tuple of (model_name, model)
        """
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_score = -np.inf
        best_model_name = None
        
        for model_name, result in self.results.items():
            score = result['test_metrics'][metric]
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, save_dir: str = 'models') -> None:
        """
        Save all trained models
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_file = save_path / f"{model_name}_model.joblib"
            joblib.dump(model, model_file)
            print(f"  ✓ Saved {model_name} to {model_file}")
        
        # Save results
        results_file = save_path / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  ✓ Saved results to {results_file}")
    
    def load_model(self, model_name: str, load_dir: str = 'models') -> Any:
        """
        Load a saved model
        
        Args:
            model_name: Name of the model
            load_dir: Directory containing saved models
            
        Returns:
            Loaded model
        """
        model_file = Path(load_dir) / f"{model_name}_model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model = joblib.load(model_file)
        print(f"✓ Loaded {model_name} from {model_file}")
        return model


# Example usage
if __name__ == "__main__":
    import yaml
    from data_processing import DataProcessor
    from feature_engineering import FeatureEngineer
    from sklearn.model_selection import train_test_split
    
    # Load config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load and process data
    processor = DataProcessor(config)
    X, y = processor.process_pipeline()
    
    # Engineer features
    engineer = FeatureEngineer(config)
    X_eng = engineer.engineer_features(X, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    # Train models
    trainer = ModelTrainer(config)
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Get best model
    best_name, best_model = trainer.get_best_model('f1')
    print(f"\n🏆 Best model: {best_name}")
    
    # Save models
    trainer.save_models()
