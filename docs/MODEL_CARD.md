# FraudLens — Model Card

## Model Overview

- **Project:** FraudLens — Credit Card Fraud Detection
- **Brand:** FraudLens (formerly fraudshield)
- **Model Type:** Ensemble Gradient Boosting (XGBoost / LightGBM — auto-selected by PR-AUC)
- **Task:** Binary classification (fraud / legitimate)
- **Model Version:** 2.0.0
- **Python:** 3.10+

## Intended Use

Detect fraudulent credit card transactions in real-time from PCA-transformed
feature vectors. Designed for use by:
- **Fraud analysts** reviewing flagged transactions (via Streamlit dashboard)
- **Automated systems** blocking high-confidence fraud (via REST API)
- **Model evaluation teams** monitoring drift and retraining schedules

### Out-of-Scope Use Cases

- Real-time blocking without human review for transactions near the threshold
- Detection of fraud types not represented in the training dataset
- Use as a standalone system without fraud analyst oversight

## Training Data

- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Samples:** 284,807 transactions
- **Fraud Rate:** 0.17% (492 fraudulent, 284,315 legitimate)
- **Features:** 30 (V1–V28 PCA components + Time + Amount)
- **Time Range:** 2 days of European cardholder transactions
- **Privacy:** All features are PCA-transformed; no raw cardholder data

### Feature Engineering

When enabled, the `FeatureEngineer` adds:
- `Amount_log`, `Amount_bin` — log-transformed and binned amount
- `Hour`, `Time_diff` — time-based features
- `V_mean`, `V_std`, `V_min`, `V_max`, `V_skew` — PCA aggregate statistics
- `V_extreme_count` — count of PCA features > 2 std from mean
- Interaction features: `V14_V4`, `V12_V10`, `V14_V12`, `V17_V14`

The `FeatureEngineer` is wired into both training and inference pipelines,
with a golden test (`test_feature_engineering_parity`) that asserts identical
feature shape/order between training-time and inference-time.

## Training Methodology

### Data Split
- **Train:** 80% (stratified, preserving 0.17% fraud rate)
- **Test:** 20% (held out, never seen during training)
- **Validation:** 10% of training data (for early stopping)

### Resampling
SMOTE (Synthetic Minority Oversampling) is used by default.

### Cross-Validation
5-fold stratified cross-validation ensures robust performance estimates.

### Model Selection
The model with the highest **PR-AUC (Precision-Recall AUC)** is selected.

### Hyperparameter Tuning (Optuna)
XGBoost and LightGBM are automatically tuned:
- **Trials:** 30 per model
- **CV Folds:** 3
- **Optimization Metric:** PR-AUC (average precision)
- **Search Space:**
  - `n_estimators`: 100–1000
  - `max_depth`: 3–12
  - `learning_rate`: 0.01–0.3
  - `subsample`: 0.6–1.0
  - `colsample_bytree`: 0.6–1.0
  - Regularization: alpha, lambda (0–10)

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **PR-AUC** | Area under precision-recall curve | ≥ 0.70 |
| **F1 Score** | Harmonic mean of precision and recall | ≥ 0.70 |
| **Precision** | TP / (TP + FP) | ≥ 0.50 |
| **Recall** | TP / (TP + FN) | ≥ 0.85 |
| **Net Benefit ($)** | Fraud caught − fraud missed − review costs | > $0 |

## Performance (Best Model: XGBoost)

| Metric | Value |
|--------|-------|
| PR-AUC | **0.8810** |
| ROC-AUC | 0.9724 |
| F1 Score | 0.7068 |
| Precision | 0.5828 |
| Recall | 0.8980 |
| Optimal Threshold | 0.0298 |

### Business Impact (per 100,000 transactions)

| Component | Value |
|-----------|-------|
| Fraud Caught | $13,200 (88%) |
| Fraud Missed | $1,500 (12%) |
| Review Costs | $755 |
| Net Benefit | **$12,445** |
| Loss Reduction | **97%** |

### PR-AUC Comparison

| Model | PR-AUC |
|-------|--------|
| **XGBoost** | **0.8810** |
| Random Forest | 0.8352 |
| Logistic Regression | 0.7159 |
| LightGBM | 0.0428 |

## Known Limitations

1. **Temporal Generalization:** The dataset covers only 2 days. Fraud patterns
   may shift over time, requiring periodic retraining.

2. **PCA Features:** The V1–V28 features are PCA-transformed from the original
   raw features, which are not available. This limits interpretability at the
   raw-transaction level.

3. **European Dataset:** The model was trained on European cardholder data.
   Geographic transferability is not guaranteed.

4. **Adversarial Adaptation:** Sophisticated fraudsters may adapt to detection
   patterns. The model should be monitored for drift and retrained as needed.

5. **Unsupervised Detection:** The `AutoencoderDetector` class exists but is
   not yet trained or integrated into the pipeline. Only `IsolationForestDetector`
   is active for unsupervised anomaly detection.

## Fairness Considerations

- The original dataset PCA transformation may obscure demographic biases
- No demographic attributes are available in the dataset to audit fairness
- The model should be validated on the target population before production use

## Maintenance

### Drift Monitoring
- **Feature drift:** KS-test p-value < 0.05 triggers alert
- **Sample size:** Minimum 1000 transactions before drift evaluation
- **Alert levels:** OK → WARNING → CRITICAL based on drift intensity
- **Retraining trigger:** Drift alert or 30 days since last training

### Retraining Schedule
- **Automated:** Drift-triggered retraining via CI/CD pipeline
- **Scheduled:** Monthly retraining with latest labeled data
- **Manual:** On-demand via `make train` or `python run_pipeline.py`

## Feedback Loop

Analyst feedback (confirmed fraud / false positive) is stored in the
`feedback` PostgreSQL table and can be used for:
1. Retraining with augmented labels
2. Threshold adjustment based on cost trade-offs
3. Model evaluation over time (tracking precision/recall drift)

### Feedback API *(planned — not yet implemented)*

The Pydantic schemas (`FeedbackCreate`, `FeedbackResponse`) exist in `api/schemas.py`
and the `feedback` PostgreSQL table has a migration ready. The feedback endpoint
will be implemented as a future enhancement.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [PR-AUC vs ROC-AUC for Imbalanced Data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
- [Optuna Hyperparameter Optimization](https://optuna.org/)
- [SHAP Explanations](https://shap.readthedocs.io/)
