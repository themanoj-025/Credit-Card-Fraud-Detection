"""
Full Pipeline Runner
Executes: EDA -> Preprocessing -> Modeling -> Evaluation -> Explainability
Saves all results, charts, and metrics for README population.
"""

import sys
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.data_loader import DataLoader, load_data
from src.preprocessing import FraudPreprocessor, Resampler, get_class_weights
from src.train import FraudTrainer, IsolationForestDetector
from src.evaluate import FraudEvaluator, print_evaluation_summary

print("=" * 70)
print("  CREDIT CARD FRAUD DETECTION - FULL PIPELINE")
print("=" * 70)

# STAGE 1: DATA LOADING
print("\n" + "-" * 70)
print("  STAGE 1: Data Loading & Statistics")
print("-" * 70)

loader = DataLoader('Dataset/Dataset/creditcard.csv')
df = loader.load()
stats = loader.get_basic_stats()

print("\nDataset Statistics:")
for k, v in stats.items():
    print("  %s: %s" % (k, v))

# Save stats
os.makedirs('data/processed', exist_ok=True)
with open('data/processed/dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2, default=str)

# STAGE 2: PREPROCESSING
print("\n" + "-" * 70)
print("  STAGE 2: Preprocessing (No Data Leakage)")
print("-" * 70)

preprocessor = FraudPreprocessor(test_size=0.2, random_state=42)
data = preprocessor.full_preprocess(df)

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print("\nTrain set: %d samples (%d fraud, %.4f%%)" % (X_train.shape[0], y_train.sum(), y_train.mean()*100))
print("Test set:  %d samples (%d fraud, %.4f%%)" % (X_test.shape[0], y_test.sum(), y_test.mean()*100))
print("Features:  %s" % list(X_train.columns))

# Save scaler
os.makedirs('models', exist_ok=True)
preprocessor.save_scaler('models/scaler.pkl')

# Save preprocessed data
joblib.dump(X_train, 'data/processed/X_train.pkl')
joblib.dump(X_test, 'data/processed/X_test.pkl')
joblib.dump(y_train, 'data/processed/y_train.pkl')
joblib.dump(y_test, 'data/processed/y_test.pkl')

# STAGE 3: RESAMPLING STRATEGY COMPARISON
print("\n" + "-" * 70)
print("  STAGE 3: Resampling Strategy Comparison")
print("-" * 70)

resampler = Resampler(random_state=42)
strategies = ['none', 'random_under', 'smote', 'adasyn', 'smote_tomek']
resampled = resampler.compare_strategies(X_train, y_train, strategies)

# Save comparison
comparison_rows = []
for strat, (X_r, y_r) in resampled.items():
    comparison_rows.append({
        'Strategy': strat,
        'Total Samples': len(X_r),
        'Fraud Samples': int(y_r.sum()),
        'Fraud Rate': '%.2f%%' % (y_r.mean()*100),
    })
    print("  %-20s: %8d samples, %5d fraud (%.2f%%)" % (strat, len(X_r), int(y_r.sum()), y_r.mean()*100))
    joblib.dump(X_r, 'data/processed/X_train_%s.pkl' % strat)
    joblib.dump(y_r, 'data/processed/y_train_%s.pkl' % strat)

resampling_df = pd.DataFrame(comparison_rows)
resampling_df.to_csv('data/processed/resampling_comparison.csv', index=False)

# STAGE 4: MODEL TRAINING
print("\n" + "-" * 70)
print("  STAGE 4: Model Training")
print("-" * 70)

trainer = FraudTrainer()
t_start = time.time()
models = trainer.train_all(X_train, y_train)
t_train = time.time() - t_start

print("\nTraining completed in %.1fs" % t_train)
for name, model in models.items():
    print("  [OK] %s: %s" % (name, type(model).__name__))

# Train Isolation Forest
iso_detector = IsolationForestDetector(contamination=0.005, n_estimators=200)
iso_detector.fit(X_train, y_train)
print("  [OK] isolation_forest: IsolationForest (unsupervised)")

# Save all models
trainer.save_all_models('models')

# STAGE 5: MODEL EVALUATION
print("\n" + "-" * 70)
print("  STAGE 5: Model Evaluation (PR-AUC & Business Cost)")
print("-" * 70)

evaluator = FraudEvaluator(avg_fraud_loss=150, review_cost=5)

predictions = {}
results = {}

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    predictions[name] = y_proba
    results[name] = evaluator.evaluate_model(y_test, y_proba, model_name=name)
    print(print_evaluation_summary(results[name]))

# Isolation Forest
iso_probas = iso_detector.predict_proba_as_fraud(X_test)
predictions['isolation_forest'] = iso_probas
results['isolation_forest'] = evaluator.evaluate_model(y_test, iso_probas, model_name='isolation_forest')
print(print_evaluation_summary(results['isolation_forest']))

# Model comparison table
comparison = evaluator.compare_models(y_test, predictions)
print("\n=== FINAL MODEL COMPARISON ===")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv('data/processed/model_comparison.csv', index=False)

# STAGE 6: SAVE BEST MODEL & THRESHOLD
print("\n" + "-" * 70)
print("  STAGE 6: Save Best Model")
print("-" * 70)

best_model_name = comparison.iloc[0]['Model']
best_result = results[best_model_name]

# Save best model
trainer.save_model(best_model_name, 'models/best_model.pkl')

# Save threshold
with open('models/threshold.txt', 'w') as f:
    f.write(str(best_result['threshold']))

# Also save as xgboost.pkl for API compatibility
if 'xgboost' in models:
    trainer.save_model('xgboost', 'models/xgboost.pkl')

print("\n  Best model: %s" % best_model_name)
print("  PR-AUC:     %.4f" % best_result['pr_auc'])
print("  Threshold:  %.4f" % best_result['threshold'])
print("  Net Benefit: $%.2f" % best_result['business']['net_benefit_usd'])

# STAGE 7: GENERATE CHARTS
print("\n" + "-" * 70)
print("  STAGE 7: Generate Charts")
print("-" * 70)

# Chart 1: PR Curves
fig = evaluator.plot_precision_recall_curve(y_test, predictions, save_path='data/processed/pr_curves.png')
plt.close()
print("  [OK] PR Curves saved")

# Chart 2: Cost vs Threshold
fig = evaluator.plot_cost_vs_threshold(y_test, predictions[best_model_name],
                                        model_name=best_model_name,
                                        save_path='data/processed/cost_vs_threshold.png')
plt.close()
print("  [OK] Cost vs Threshold saved")

# Chart 3: Class imbalance and feature importance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_counts = df['Class'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Legitimate', 'Fraud'], class_counts.values, color=colors)
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 2000, '{:,}'.format(v), ha='center', fontweight='bold')

# Chart 4: Feature importance
feat_imp = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': models[best_model_name].feature_importances_
}).sort_values('Importance', ascending=True).tail(15)

axes[1].barh(feat_imp['Feature'], feat_imp['Importance'], color='#667eea')
axes[1].set_title('Top 15 Features (%s)' % best_model_name, fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('data/processed/class_distribution_and_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Class distribution & features saved")

# Chart 5: Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics_to_plot = ['PR-AUC', 'F1', 'Net Benefit ($)']
chart_colors = ['#667eea', '#764ba2', '#f093fb', '#38ef7d', '#ff416c']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    vals = comparison[metric].values
    names = comparison['Model'].values
    bars = ax.bar(range(len(names)), vals, color=chart_colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_title(metric, fontsize=13, fontweight='bold')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                '%.2f' % val, ha='center', va='bottom', fontsize=9)

plt.suptitle('Model Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('data/processed/model_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Model comparison chart saved")

# STAGE 8: SAVE FINAL RESULTS
print("\n" + "-" * 70)
print("  STAGE 8: Save Final Results")
print("-" * 70)

final_results = {
    'dataset_stats': stats,
    'best_model': best_model_name,
    'best_threshold': best_result['threshold'],
    'metrics': {
        'pr_auc': best_result['pr_auc'],
        'roc_auc': best_result['roc_auc'],
        'f1': best_result['f1'],
        'precision': best_result['precision'],
        'recall': best_result['recall'],
    },
    'business': best_result['business'],
    'all_models': {}
}

for name, result in results.items():
    final_results['all_models'][name] = {
        'pr_auc': result['pr_auc'],
        'roc_auc': result['roc_auc'],
        'f1': result['f1'],
        'precision': result['precision'],
        'recall': result['recall'],
        'threshold': result['threshold'],
        'net_benefit': result['business']['net_benefit_usd'],
        'fraud_caught': result['business']['fraud_caught_usd'],
        'fraud_missed': result['business']['fraud_missed_usd'],
    }

with open('data/processed/final_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE - RESULTS SUMMARY")
print("=" * 70)
print("\n  Best Model:      %s" % best_model_name)
print("  PR-AUC:          %.4f" % best_result['pr_auc'])
print("  ROC-AUC:         %.4f" % best_result['roc_auc'])
print("  F1 Score:        %.4f" % best_result['f1'])
print("  Precision:       %.4f" % best_result['precision'])
print("  Recall:          %.4f" % best_result['recall'])
print("  Optimal Threshold: %.4f" % best_result['threshold'])
print("\n  Business Impact:")
print("    Fraud Caught:    $%10.2f" % best_result['business']['fraud_caught_usd'])
print("    Fraud Missed:    $%10.2f" % best_result['business']['fraud_missed_usd'])
print("    Review Costs:    $%10.2f" % best_result['business']['review_costs_usd'])
print("    Net Benefit:     $%10.2f" % best_result['business']['net_benefit_usd'])
print("\n  All Results Saved to: data/processed/")
print("  Models Saved to:     models/")
print("=" * 70)
