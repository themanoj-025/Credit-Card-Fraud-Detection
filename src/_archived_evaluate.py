"""
Evaluation Module

Comprehensive evaluation with business-relevant metrics:
- Precision-Recall AUC (primary metric for imbalanced data)
- Confusion matrix (counts and dollar values)
- Business cost function for threshold optimization
- Precision-recall tradeoff curves
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    auc,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class FraudEvaluator:
    """
    Evaluate fraud detection models with business-relevant metrics.
    
    Key design decisions:
    - PR-AUC is the primary metric (not ROC-AUC which is misleading on imbalanced data)
    - Threshold is optimized using a business cost function, not default 0.5
    - Confusion matrix is shown in both counts and dollar values
    """
    
    def __init__(
        self,
        avg_fraud_loss: float = 150.0,
        review_cost: float = 5.0,
    ):
        """
        Args:
            avg_fraud_loss: Average dollar loss per missed fraud (false negative)
            review_cost: Cost to manually review a flagged transaction (false positive)
        """
        self.avg_fraud_loss = avg_fraud_loss
        self.review_cost = review_cost
    
    def compute_business_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute the business cost of a set of predictions.
        
        Args:
            y_true: True labels
            y_pred: Binary predictions
            
        Returns:
            Dictionary with cost breakdown
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        fraud_caught = tp * self.avg_fraud_loss          # Money saved by catching fraud
        fraud_missed = fn * self.avg_fraud_loss           # Money lost to missed fraud
        review_costs = (tp + fp) * self.review_cost      # Cost of reviewing flagged transactions
        total_cost = fraud_missed + review_costs - fraud_caught  # Net cost
        
        # Net benefit = what we saved - what we spent
        net_benefit = fraud_caught - review_costs
        
        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'fraud_caught_usd': round(fraud_caught, 2),
            'fraud_missed_usd': round(fraud_missed, 2),
            'review_costs_usd': round(review_costs, 2),
            'net_benefit_usd': round(net_benefit, 2),
            'total_cost_usd': round(total_cost, 2),
        }
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_thresholds: int = 100,
    ) -> Tuple[float, Dict]:
        """
        Find the threshold that minimizes total business cost.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_thresholds: Number of thresholds to evaluate
            
        Returns:
            Tuple of (optimal_threshold, cost_dict)
        """
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        best_cost = float('inf')
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost = self.compute_business_cost(y_true, y_pred)
            
            if cost['total_cost_usd'] < best_cost:
                best_cost = cost['total_cost_usd']
                best_threshold = threshold
                best_metrics = cost
        
        logger.info(f"Optimal threshold: {best_threshold:.4f}")
        logger.info(f"  Net benefit: ${best_metrics['net_benefit_usd']:.2f}")
        logger.info(f"  Total cost: ${best_metrics['total_cost_usd']:.2f}")
        
        return best_threshold, best_metrics
    
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = None,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            threshold: Decision threshold (if None, uses optimal)
            model_name: Name for logging
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Find optimal threshold if not provided
        if threshold is None:
            threshold, opt_cost = self.find_optimal_threshold(y_true, y_proba)
        else:
            opt_cost = self.compute_business_cost(y_true, (y_proba >= threshold).astype(int))
        
        # Binary predictions at chosen threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Core metrics
        pr_auc = average_precision_score(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # PR curve data
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
        pr_curve_auc = auc(recalls, precisions)
        
        results = {
            'model_name': model_name,
            'threshold': round(threshold, 4),
            'pr_auc': round(pr_auc, 4),
            'roc_auc': round(roc_auc, 4),
            'pr_curve_auc': round(pr_curve_auc, 4),
            'f1': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'confusion_matrix': cm.tolist(),
            'business': opt_cost,
            'pr_curve': {
                'precisions': precisions.tolist(),
                'recalls': recalls.tolist(),
                'thresholds': pr_thresholds.tolist(),
            },
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation: {model_name}")
        logger.info(f"{'='*50}")
        logger.info(f"  Threshold:   {threshold:.4f}")
        logger.info(f"  PR-AUC:      {pr_auc:.4f}")
        logger.info(f"  ROC-AUC:     {roc_auc:.4f}")
        logger.info(f"  F1:          {f1:.4f}")
        logger.info(f"  Precision:   {precision:.4f}")
        logger.info(f"  Recall:      {recall:.4f}")
        logger.info(f"  Net benefit: ${opt_cost['net_benefit_usd']:.2f}")
        
        return results
    
    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            y_true: True labels
            predictions: Dict mapping model names to probability arrays
            
        Returns:
            DataFrame with comparison metrics
        """
        rows = []
        for model_name, y_proba in predictions.items():
            result = self.evaluate_model(y_true, y_proba, model_name=model_name)
            rows.append({
                'Model': model_name,
                'Threshold': result['threshold'],
                'PR-AUC': result['pr_auc'],
                'ROC-AUC': result['roc_auc'],
                'F1': result['f1'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'Net Benefit ($)': result['business']['net_benefit_usd'],
                'Fraud Caught ($)': result['business']['fraud_caught_usd'],
                'Missed Fraud ($)': result['business']['fraud_missed_usd'],
            })
        
        comparison_df = pd.DataFrame(rows).sort_values('PR-AUC', ascending=False)
        
        logger.info("\n=== Model Comparison ===")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        return comparison_df
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot precision-recall curves for multiple models.
        
        Args:
            y_true: True labels
            predictions: Dict mapping model names to probability arrays
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PR Curve
        ax1 = axes[0]
        for model_name, y_proba in predictions.items():
            precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            ax1.plot(recalls, precisions, label=f'{model_name} (PR-AUC={pr_auc:.4f})', linewidth=2)
        
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision-Recall Curves', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Add baseline (random classifier)
        baseline = y_true.sum() / len(y_true)
        ax1.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.4f})')
        
        # Confusion Matrices
        ax2 = axes[1]
        best_model = max(predictions.keys(), key=lambda m: average_precision_score(y_true, predictions[m]))
        y_pred = (predictions[best_model] >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        ax2.set_xlabel('Predicted', fontsize=12)
        ax2.set_ylabel('Actual', fontsize=12)
        ax2.set_title(f'Confusion Matrix ({best_model})', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_cost_vs_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "model",
        save_path: str = None,
    ) -> plt.Figure:
        """
        Plot total cost as a function of threshold.
        
        Shows why default threshold of 0.5 is suboptimal.
        """
        thresholds = np.linspace(0.01, 0.99, 200)
        costs = []
        benefits = []
        
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            business = self.compute_business_cost(y_true, y_pred)
            costs.append(business['total_cost_usd'])
            benefits.append(business['net_benefit_usd'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(thresholds, costs, label='Total Cost ($)', linewidth=2, color='red')
        ax.plot(thresholds, benefits, label='Net Benefit ($)', linewidth=2, color='green')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default threshold (0.5)')
        
        # Mark optimal threshold
        optimal_t, _ = self.find_optimal_threshold(y_true, y_proba)
        ax.axvline(x=optimal_t, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Optimal threshold ({optimal_t:.4f})')
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('USD ($)', fontsize=12)
        ax.set_title(f'Cost Analysis vs Threshold — {model_name}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def print_evaluation_summary(results: Dict[str, Any]) -> str:
    """
    Format evaluation results as a readable string.
    """
    lines = [
        f"\n{'='*60}",
        f"  EVALUATION: {results['model_name']}",
        f"{'='*60}",
        f"  Threshold:       {results['threshold']:.4f}",
        f"  PR-AUC:          {results['pr_auc']:.4f}  <-- PRIMARY METRIC",
        f"  ROC-AUC:         {results['roc_auc']:.4f}",
        f"  F1 Score:        {results['f1']:.4f}",
        f"  Precision:       {results['precision']:.4f}",
        f"  Recall:          {results['recall']:.4f}",
        f"{'-'*60}",
        f"  BUSINESS IMPACT:",
        f"  Fraud Caught:    ${results['business']['fraud_caught_usd']:>10,.2f}",
        f"  Fraud Missed:    ${results['business']['fraud_missed_usd']:>10,.2f}",
        f"  Review Costs:    ${results['business']['review_costs_usd']:>10,.2f}",
        f"  --------------------------------------------",
        f"  Net Benefit:     ${results['business']['net_benefit_usd']:>10,.2f}",
        f"{'='*60}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick demo with synthetic data
    np.random.seed(42)
    y_true = np.array([0]*998 + [1]*2)
    y_proba = np.random.rand(1000) * 0.1
    y_proba[998] = 0.9
    y_proba[999] = 0.85
    
    evaluator = FraudEvaluator()
    results = evaluator.evaluate_model(y_true, y_proba, model_name="demo")
    print(print_evaluation_summary(results))
