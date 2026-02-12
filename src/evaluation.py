"""
Evaluation Module
Handles model evaluation, visualization, and explainability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import shap
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Handles model evaluation and visualization"""
    
    def __init__(self, config: dict, output_dir: str = 'reports/figures'):
        """
        Initialize ModelEvaluator
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save plots
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save: bool = True
    ) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save: Whether to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save:
            plt.savefig(
                self.output_dir / f'{model_name}_confusion_matrix.png',
                dpi=300, bbox_inches='tight'
            )
        plt.show()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        save: bool = True
    ) -> float:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the plot
            
        Returns:
            AUC score
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(
                self.output_dir / f'{model_name}_roc_curve.png',
                dpi=300, bbox_inches='tight'
            )
        plt.show()
        
        return roc_auc
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        save: bool = True
    ) -> float:
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the plot
            
        Returns:
            Average precision score
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.axhline(y=y_true.mean(), color='red', linestyle='--',
                   label=f'Baseline (churn rate = {y_true.mean():.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(
                self.output_dir / f'{model_name}_pr_curve.png',
                dpi=300, bbox_inches='tight'
            )
        plt.show()
        
        return avg_precision
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str,
        top_n: int = 20,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Plot feature importance
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to show
            save: Whether to save the plot
            
        Returns:
            DataFrame of feature importances
        """
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"⚠️  {model_name} does not support feature importance")
            return None
        
        # Create DataFrame
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top N
        plt.figure(figsize=(10, 8))
        top_features = feature_imp.head(top_n)
        sns.barplot(
            data=top_features,
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        if save:
            plt.savefig(
                self.output_dir / f'{model_name}_feature_importance.png',
                dpi=300, bbox_inches='tight'
            )
        plt.show()
        
        return feature_imp
    
    def plot_shap_summary(
        self,
        model: Any,
        X_test: pd.DataFrame,
        model_name: str,
        save: bool = True
    ) -> None:
        """
        Plot SHAP summary
        
        Args:
            model: Trained model
            X_test: Test features
            model_name: Name of the model
            save: Whether to save the plot
        """
        print(f"\n🔍 Calculating SHAP values for {model_name}...")
        
        try:
            # Create explainer
            if 'tree' in model_name.lower() or 'forest' in model_name.lower() \
               or 'xgb' in model_name.lower() or 'lgbm' in model_name.lower():
                explainer = shap.TreeExplainer(model)
            else:
                # Sample for linear models (faster)
                X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
                explainer = shap.LinearExplainer(model, X_sample)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test)
            
            # Plot
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            shap.summary_plot(
                shap_values, X_test, 
                plot_type="bar", 
                show=False
            )
            plt.title(f'SHAP Feature Importance - {model_name}')
            
            if save:
                plt.savefig(
                    self.output_dir / f'{model_name}_shap_summary.png',
                    dpi=300, bbox_inches='tight'
                )
            plt.show()
            
            print("✓ SHAP analysis complete")
            
        except Exception as e:
            print(f"⚠️  Could not generate SHAP plots: {str(e)}")
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Analyze different probability thresholds
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the plot
            
        Returns:
            DataFrame with threshold analysis
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Business value
            business_config = self.config['evaluation']['business']
            clv = business_config['avg_clv']
            campaign_cost = business_config['campaign_cost_per_customer']
            retention_rate = business_config['retention_rate_with_intervention']
            
            value = (tp * clv * retention_rate) - ((tp + fp) * campaign_cost)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'business_value': value,
                'true_positives': tp,
                'false_positives': fp
            })
        
        results_df = pd.DataFrame(results)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Metrics vs threshold
        ax1.plot(results_df['threshold'], results_df['precision'], 
                label='Precision', marker='o')
        ax1.plot(results_df['threshold'], results_df['recall'], 
                label='Recall', marker='s')
        ax1.plot(results_df['threshold'], results_df['f1'], 
                label='F1 Score', marker='^')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Metrics vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Business value vs threshold
        ax2.plot(results_df['threshold'], results_df['business_value'], 
                color='green', marker='o')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Business Value ($)')
        ax2.set_title('Business Value vs Threshold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Threshold Analysis - {model_name}')
        
        if save:
            plt.savefig(
                self.output_dir / f'{model_name}_threshold_analysis.png',
                dpi=300, bbox_inches='tight'
            )
        plt.show()
        
        # Find optimal threshold
        optimal_idx = results_df['business_value'].idxmax()
        optimal_threshold = results_df.loc[optimal_idx, 'threshold']
        print(f"\n💡 Optimal threshold for {model_name}: {optimal_threshold:.2f}")
        print(f"   Expected business value: ${results_df.loc[optimal_idx, 'business_value']:,.0f}")
        
        return results_df
    
    def generate_comparison_report(
        self,
        results: Dict[str, Dict],
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate comparison report for all models
        
        Args:
            results: Dictionary of results from ModelTrainer
            save: Whether to save the report
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, result in results.items():
            test_metrics = result['test_metrics']
            business = result['business_value']
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1 Score': test_metrics['f1'],
                'ROC-AUC': test_metrics['roc_auc'],
                'Business Value ($)': business['net_business_value'],
                'ROI (%)': business['roi_percentage']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            sns.barplot(
                data=comparison_df,
                x='Model',
                y=metric,
                ax=ax,
                palette='viridis'
            )
            ax.set_title(metric)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylim([0, 1])
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        if save:
            plt.savefig(
                self.output_dir / 'model_comparison.png',
                dpi=300, bbox_inches='tight'
            )
        plt.show()
        
        # Business metrics comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(
            data=comparison_df,
            x='Model',
            y='Business Value ($)',
            palette='RdYlGn'
        )
        plt.title('Net Business Value')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(1, 2, 2)
        sns.barplot(
            data=comparison_df,
            x='Model',
            y='ROI (%)',
            palette='RdYlGn'
        )
        plt.title('Return on Investment')
        plt.xticks(rotation=45, ha='right')
        
        plt.suptitle('Business Metrics Comparison', fontsize=16)
        plt.tight_layout()
        
        if save:
            plt.savefig(
                self.output_dir / 'business_comparison.png',
                dpi=300, bbox_inches='tight'
            )
            comparison_df.to_csv(
                self.output_dir / 'model_comparison.csv',
                index=False
            )
        plt.show()
        
        return comparison_df


# Example usage
if __name__ == "__main__":
    print("Evaluation module loaded successfully")
    print("Import this module to use visualization functions")
