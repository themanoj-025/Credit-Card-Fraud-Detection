"""
Enhanced Training & Comparison Pipeline
Trains all ML algorithms, compares with detailed charts, saves best model
"""

import sys
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from src.data_loader import DataLoader
from src.preprocessing import FraudPreprocessor
from src.train import FraudTrainer, IsolationForestDetector
from src.evaluate import FraudEvaluator

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("  COMPREHENSIVE ML TRAINING & COMPARISON")
print("=" * 70)

# STAGE 1: Load and preprocess
print("\n[1/6] Loading and preprocessing data...")
loader = DataLoader('Dataset/Dataset/creditcard.csv')
df = loader.load()

preprocessor = FraudPreprocessor(test_size=0.2, random_state=42)
data = preprocessor.full_preprocess(df)
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

print("  Train: %d samples, Test: %d samples" % (len(X_train), len(X_test)))
print("  Fraud in train: %d (%.4f%%)" % (y_train.sum(), y_train.mean()*100))

# STAGE 2: Train all models
print("\n[2/6] Training all ML models...")
trainer = FraudTrainer()
t_start = time.time()
models = trainer.train_all(X_train, y_train)

# Train Isolation Forest
iso_detector = IsolationForestDetector(contamination=0.005, n_estimators=200)
iso_detector.fit(X_train, y_train)
t_total = time.time() - t_start

print("  Training completed in %.1fs" % t_total)
for name in models:
    print("    - %s" % name)
print("    - isolation_forest (unsupervised)")

# STAGE 3: Evaluate all models
print("\n[3/6] Evaluating all models...")
evaluator = FraudEvaluator(avg_fraud_loss=150, review_cost=5)

predictions = {}
results = {}
timing = {}

for name, model in models.items():
    t0 = time.time()
    y_proba = model.predict_proba(X_test)[:, 1]
    timing[name] = time.time() - t0
    predictions[name] = y_proba
    results[name] = evaluator.evaluate_model(y_test, y_proba, model_name=name)

# Isolation Forest
t0 = time.time()
iso_probas = iso_detector.predict_proba_as_fraud(X_test)
timing['isolation_forest'] = time.time() - t0
predictions['isolation_forest'] = iso_probas
results['isolation_forest'] = evaluator.evaluate_model(y_test, iso_probas, model_name='isolation_forest')

# Comparison table
comparison = evaluator.compare_models(y_test, predictions)

# Add training time
timing_rows = []
for name in comparison['Model'].values:
    if name in timing:
        timing_rows.append({'Model': name, 'Inference Time (s)': timing[name]})
timing_df = pd.DataFrame(timing_rows)
comparison = comparison.merge(timing_df, on='Model', how='left')

print("\n=== MODEL COMPARISON ===")
print(comparison.to_string(index=False))

# STAGE 4: Generate comprehensive charts
print("\n[4/6] Generating comparison charts...")
os.makedirs('data/processed', exist_ok=True)

# --- Chart 1: PR Curves ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

ax = axes[0, 0]
for name, y_proba in predictions.items():
    from sklearn.metrics import precision_recall_curve, average_precision_score
    p, r, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    ax.plot(r, p, linewidth=2, label='%s (AP=%.3f)' % (name, ap))
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Chart 2: ROC Curves ---
ax = axes[0, 1]
from sklearn.metrics import roc_curve, roc_auc_score
for name, y_proba in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, linewidth=2, label='%s (AUC=%.3f)' % (name, auc))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Chart 3: PR-AUC Comparison ---
ax = axes[0, 2]
models_list = comparison['Model'].values
pr_aucs = comparison['PR-AUC'].values
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models_list)))
bars = ax.barh(models_list, pr_aucs, color=colors)
ax.set_xlabel('PR-AUC Score', fontsize=11)
ax.set_title('PR-AUC Comparison (Higher is Better)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, pr_aucs):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, '%.4f' % val, va='center', fontsize=10)
ax.set_xlim(0, max(pr_aucs) * 1.15)

# --- Chart 4: Business Impact ---
ax = axes[1, 0]
net_benefits = comparison['Net Benefit ($)'].values
colors_biz = ['#38ef7d' if nb > 0 else '#ff416c' for nb in net_benefits]
bars = ax.barh(models_list, net_benefits, color=colors_biz)
ax.set_xlabel('Net Benefit ($)', fontsize=11)
ax.set_title('Business Impact - Net Benefit', fontsize=13, fontweight='bold')
for bar, val in zip(bars, net_benefits):
    ax.text(val + 100, bar.get_y() + bar.get_height()/2, '$%s' % '{:,.0f}'.format(val), va='center', fontsize=10)

# --- Chart 5: Precision vs Recall ---
ax = axes[1, 1]
precisions = comparison['Precision'].values
recalls = comparison['Recall'].values
scatter = ax.scatter(recalls, precisions, s=200, c=pr_aucs, cmap='RdYlGn', edgecolors='black', zorder=5)
for i, name in enumerate(models_list):
    ax.annotate(name, (recalls[i], precisions[i]), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision vs Recall Trade-off', fontsize=13, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='PR-AUC')
ax.grid(True, alpha=0.3)

# --- Chart 6: F1 Score Comparison ---
ax = axes[1, 2]
f1_scores = comparison['F1'].values
colors_f1 = plt.cm.viridis(np.linspace(0.3, 0.9, len(models_list)))
bars = ax.bar(range(len(models_list)), f1_scores, color=colors_f1)
ax.set_xticks(range(len(models_list)))
ax.set_xticklabels(models_list, rotation=45, ha='right')
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('F1 Score Comparison', fontsize=13, fontweight='bold')
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, '%.4f' % val, ha='center', fontsize=10)

plt.suptitle('Comprehensive ML Model Comparison - Credit Card Fraud Detection', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data/processed/comprehensive_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Comprehensive comparison chart saved")

# --- Chart 7: Confusion Matrices ---
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
from sklearn.metrics import confusion_matrix

for idx, name in enumerate(models_list[:5]):
    ax = axes[idx]
    y_pred = (predictions[name] >= results[name]['threshold']).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    ax.set_title('%s\n(PR-AUC=%.3f)' % (name, results[name]['pr_auc']), fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/processed/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] Confusion matrices saved")

# STAGE 5: Save best model
print("\n[5/6] Saving best model...")
best_model_name = comparison.iloc[0]['Model']
best_result = results[best_model_name]

# Save best model
trainer.save_model(best_model_name, 'models/best_model.pkl')
trainer.save_model(best_model_name, 'models/%s.pkl' % best_model_name)

# Save as xgboost.pkl for API compatibility
if 'xgboost' in models:
    trainer.save_model('xgboost', 'models/xgboost.pkl')

# Save threshold
with open('models/threshold.txt', 'w') as f:
    f.write(str(best_result['threshold']))

# Save all models
trainer.save_all_models('models')
preprocessor.save_scaler('models/scaler.pkl')

print("  Best model: %s" % best_model_name)
print("  Saved to: models/best_model.pkl, models/xgboost.pkl")
print("  Threshold: %.4f" % best_result['threshold'])

# STAGE 6: Summary
print("\n[6/6] Final Summary")
print("=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print("\n  Models Trained: %d" % (len(models) + 1))
print("  Best Model:     %s" % best_model_name)
print("  PR-AUC:         %.4f" % best_result['pr_auc'])
print("  ROC-AUC:        %.4f" % best_result['roc_auc'])
print("  F1:             %.4f" % best_result['f1'])
print("  Precision:      %.4f" % best_result['precision'])
print("  Recall:         %.4f" % best_result['recall'])
print("  Threshold:      %.4f" % best_result['threshold'])
print("\n  Business Impact:")
print("    Fraud Caught:  $%s" % '{:,.2f}'.format(best_result['business']['fraud_caught_usd']))
print("    Fraud Missed:  $%s" % '{:,.2f}'.format(best_result['business']['fraud_missed_usd']))
print("    Review Costs:  $%s" % '{:,.2f}'.format(best_result['business']['review_costs_usd']))
print("    Net Benefit:   $%s" % '{:,.2f}'.format(best_result['business']['net_benefit_usd']))
print("\n  Charts saved to: data/processed/")
print("  Models saved to: models/")
print("=" * 70)

# Save comparison for README
comparison.to_csv('data/processed/model_comparison.csv', index=False)

# Save final results
final_results = {
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
    }
with open('data/processed/final_results.json', 'w') as f:
    json.dump(final_results, f, indent=2, default=str)
