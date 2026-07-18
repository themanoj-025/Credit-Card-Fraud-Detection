"""
FraudLens — Full Pipeline Runner

Executes the complete ML pipeline:
    Data Loading → Preprocessing → Model Training → Evaluation → Explainability

Usage:
    python run_pipeline.py
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fraudshield.config import (
    AVG_FRAUD_LOSS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REVIEW_COST,
)
from src.fraudshield.data.loaders import DataLoader
from src.fraudshield.data.preprocessing import FraudPreprocessor, Resampler
from src.fraudshield.evaluation.business_cost import BusinessCostCalculator
from src.fraudshield.evaluation.metrics import FraudEvaluator, print_evaluation_summary
from src.fraudshield.models.anomaly import IsolationForestDetector
from src.fraudshield.models.model_selection import ModelSelector
from src.fraudshield.models.train import FraudTrainer

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 70)
print("  FRAUDLENS — Full Pipeline Runner")
print("=" * 70)

# ─── STAGE 1: Data Loading ───────────────────────────────────────────────
print("\n" + "-" * 70)
print("  STAGE 1: Data Loading & Statistics")
print("-" * 70)

loader = DataLoader()
try:
    df = loader.load()
except FileNotFoundError as e:
    print(f"\n  ❌ {e}")
    print("  Download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  Place it at: data/raw/creditcard.csv")
    sys.exit(1)

stats = loader.get_basic_stats()
print("\nDataset Statistics:")
for k, v in stats.items():
    print(f"  {k}: {v}")

with open(PROCESSED_DATA_DIR / "dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2, default=str)

# ─── STAGE 2: Preprocessing ──────────────────────────────────────────────
print("\n" + "-" * 70)
print("  STAGE 2: Preprocessing (No Data Leakage)")
print("-" * 70)

preprocessor = FraudPreprocessor(test_size=0.2, random_state=42)
data = preprocessor.full_preprocess(df)

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

print(f"\n  Train set: {X_train.shape[0]} samples ({y_train.sum()} fraud, {y_train.mean()*100:.4f}%)")
print(f"  Test set:  {X_test.shape[0]} samples ({y_test.sum()} fraud, {y_test.mean()*100:.4f}%)")
print(f"  Features:  {list(X_train.columns)}")

preprocessor.save_scaler(str(MODELS_DIR / "scaler.pkl"))
joblib.dump(X_train, PROCESSED_DATA_DIR / "X_train.pkl")
joblib.dump(X_test, PROCESSED_DATA_DIR / "X_test.pkl")
joblib.dump(y_train, PROCESSED_DATA_DIR / "y_train.pkl")
joblib.dump(y_test, PROCESSED_DATA_DIR / "y_test.pkl")

# ─── STAGE 3: Resampling Comparison ──────────────────────────────────────
print("\n" + "-" * 70)
print("  STAGE 3: Resampling Strategy Comparison")
print("-" * 70)

resampler = Resampler(random_state=42)
strategies = ["none", "random_under", "smote", "adasyn", "smote_tomek"]
resampled = resampler.compare_strategies(X_train, y_train, strategies)

comparison_rows = []
for strat, (X_r, y_r) in resampled.items():
    comparison_rows.append({
        "Strategy": strat,
        "Total Samples": len(X_r),
        "Fraud Samples": int(y_r.sum()),
        "Fraud Rate": f"{y_r.mean()*100:.2f}%",
    })
    print(f"  {strat:<20}: {len(X_r):>8} samples, {int(y_r.sum()):>5} fraud ({y_r.mean()*100:.2f}%)")
    joblib.dump(X_r, PROCESSED_DATA_DIR / f"X_train_{strat}.pkl")
    joblib.dump(y_r, PROCESSED_DATA_DIR / f"y_train_{strat}.pkl")

pd.DataFrame(comparison_rows).to_csv(PROCESSED_DATA_DIR / "resampling_comparison.csv", index=False)

# ─── STAGE 4: Model Training ─────────────────────────────────────────────
print("\n" + "-" * 70)
print("  STAGE 4: Model Training")
print("-" * 70)

trainer = FraudTrainer()
t_start = time.time()
models = trainer.train_all(X_train, y_train)
t_train = time.time() - t_start

print(f"\n  Training completed in {t_train:.1f}s")

# Train Isolation Forest (unsupervised)
iso_detector = IsolationForestDetector(contamination=0.005, n_estimators=200)
iso_detector.fit(X_train, y_train)
print("  [OK] isolation_forest: trained (unsupervised)")

# Save all models
trainer.save_all_models(str(MODELS_DIR))
joblib.dump(iso_detector.model, MODELS_DIR / "anomaly_detector.pkl")
print(f"\n  Models saved to: {MODELS_DIR}")

# ─── STAGE 5: Evaluation ─────────────────────────────────────────────────
print("\n" + "-" * 70)
print("  STAGE 5: Model Evaluation")
print("-" * 70)

evaluator = FraudEvaluator(avg_fraud_loss=AVG_FRAUD_LOSS, review_cost=REVIEW_COST)
cost_calc = BusinessCostCalculator(avg_fraud_loss=AVG_FRAUD_LOSS, review_cost=REVIEW_COST)

predictions = {}
results = {}
thresholds = {}
business_costs = {}

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    predictions[name] = y_proba

    # Find optimal threshold
    threshold, biz_cost = cost_calc.find_optimal_threshold(y_test, y_proba)
    thresholds[name] = threshold
    business_costs[name] = biz_cost

    result = evaluator.evaluate_model(y_test, y_proba, threshold=threshold, 
                                       model_name=name, business_cost=biz_cost)
    results[name] = result
    print(print_evaluation_summary(result))

# Isolation Forest
iso_probas = iso_detector.predict_proba_as_fraud(X_test)
predictions["Isolation Forest"] = iso_probas
thresh_if, biz_if = cost_calc.find_optimal_threshold(y_test, iso_probas)
thresholds["Isolation Forest"] = thresh_if
business_costs["Isolation Forest"] = biz_if
rf_result = evaluator.evaluate_model(y_test, iso_probas, threshold=thresh_if,
                                       model_name="Isolation Forest", business_cost=biz_if)
results["Isolation Forest"] = rf_result
print(print_evaluation_summary(rf_result))

# Comparison table
comparison = evaluator.compare_models(y_test, predictions, thresholds, business_costs)
print("\n=== FINAL MODEL COMPARISON ===")
print(comparison.to_string(index=False))

# Save comparison
comparison_path = PROCESSED_DATA_DIR / "model_comparison.csv"
comparison.to_csv(comparison_path, index=False)
print(f"  Comparison saved to: {comparison_path}")

# ─── STAGE 6: Auto-Select Best Model ──────────────────────────────────────
print("\n" + "-" * 70)
print("  STAGE 6: Auto-Select Best Model")
print("-" * 70)

selector = ModelSelector(metric="PR-AUC")
all_trained = {**models, "Isolation Forest": iso_detector.model}
selection = selector.select(comparison, all_trained)

selector.save_best_model(str(MODELS_DIR / "best_fraud_model.pkl"))
print(selector.get_selection_summary())

# Save optimal threshold
best_threshold = thresholds.get(selection["best_model_name"], 0.5)
with open(MODELS_DIR / "threshold.txt", "w") as f:
    f.write(str(best_threshold))

# ─── STAGE 7: Generate Charts ────────────────────────────────────────────
print("\n" + "-" * 70)
print("  STAGE 7: Generate Charts")
print("-" * 70)

charts_dir = PROCESSED_DATA_DIR

# Chart 1: PR Curves
fig = evaluator.plot_precision_recall_curve(y_test, predictions, 
                                              save_path=str(charts_dir / "pr_curves.png"))
plt.close()
print("  [OK] PR Curves saved")

# Chart 2: Cost vs Threshold (for best model)
best_name = selection["best_model_name"]
fig = cost_calc.plot_cost_vs_threshold(y_test, predictions[best_name],
                                         model_name=best_name,
                                         save_path=str(charts_dir / "cost_vs_threshold.png"))
plt.close()
print("  [OK] Cost vs Threshold saved")

# Chart 3: Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics_to_plot = ["PR-AUC", "F1", "Net Benefit ($)"]
chart_colors = ["#667eea", "#764ba2", "#f093fb", "#38ef7d", "#ff416c"]

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    vals = comparison[metric].values
    names = comparison["Model"].values
    bars = ax.bar(range(len(names)), vals, color=chart_colors[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Model Comparison", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(str(charts_dir / "model_comparison_chart.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  [OK] Model comparison chart saved")

# ─── SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE")
print("=" * 70)
print(f"\n  Best Model:      {selection['best_model_name']}")
print(f"  PR-AUC:          {selection['metric_value']:.4f}")
print(f"  Threshold:       {best_threshold:.4f}")
business = business_costs.get(selection["best_model_name"], {})
if business:
    print(f"\n  Business Impact:")
    print(f"    Fraud Caught:    ${business.get('fraud_caught_usd', 0):,.2f}")
    print(f"    Fraud Missed:    ${business.get('fraud_missed_usd', 0):,.2f}")
    print(f"    Review Costs:    ${business.get('review_costs_usd', 0):,.2f}")
    print(f"    Net Benefit:     ${business.get('net_benefit_usd', 0):,.2f}")
print(f"\n  Results saved to: {PROCESSED_DATA_DIR}")
print(f"  Models saved to:  {MODELS_DIR}")
print("=" * 70)
