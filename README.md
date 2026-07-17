<<<<<<< HEAD
# 💳 Credit Card Fraud Detection — Portfolio Project

![CI](https://github.com/yourusername/credit-card-fraud-detection/actions/workflows/ci.yml/badge.svg)

> **A production-grade fraud detection system** that reduces fraud losses while minimizing false positives — with real business impact metrics, not just accuracy scores.

---

## 📋 Problem Statement

**Business Objective:** Reduce fraud losses while minimizing false positives that annoy legitimate customers and cost the bank in manual review labor.

**Assumed Business Costs:**
| Metric | Value |
|--------|-------|
| Average fraud loss per missed transaction | **$150** |
| Cost to manually review a flagged transaction | **$5** |
| Dataset fraud rate | **0.172%** (492 out of 284,807 transactions) |

> **Why not accuracy?** A model that predicts "all legitimate" achieves 99.8% accuracy while catching zero fraud. We use **Precision-Recall AUC** and a **business cost function** as our primary metrics.

---

## 📊 Dataset

**Kaggle ULB Credit Card Fraud Detection Dataset**
- **284,807** transactions, **492** frauds (0.172%)
- **V1–V28**: PCA-anonymized features (for privacy)
- **Time**: Seconds since first transaction
- **Amount**: Transaction amount in dollars
- **Class**: 0 = legitimate, 1 = fraud

🔗 [Dataset Source](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CREDIT CARD FRAUD DETECTION                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ Raw Data │───▶│ Preprocessing│───▶│   Feature Engine   │    │
│  │ (CSV)    │    │ (Split+Scale)│    │ (Engineered Vars)  │    │
│  └──────────┘    └──────────────┘    └────────────────────┘    │
│                       │                      │                   │
│                       ▼                      ▼                   │
│              ┌──────────────────────────────────────┐           │
│              │         Model Training               │           │
│              │  ┌─────────┐ ┌──────┐ ┌──────────┐  │           │
│              │  │XGBoost  │ │ Light│ │ Isolation │  │           │
│              │  │(Best)   │ │ GBM  │ │ Forest    │  │           │
│              │  └─────────┘ └──────┘ └──────────┘  │           │
│              └──────────────────────────────────────┘           │
│                       │                      │                   │
│                       ▼                      ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐      │
│  │   FastAPI    │  │   Streamlit  │  │   SHAP          │      │
│  │   /predict   │  │  Dashboard   │  │  Explainability │      │
│  └──────────────┘  └──────────────┘  └─────────────────┘      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Monitoring: Drift Detection + Business Impact Tracking  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology

### 1. Data Preprocessing (No Data Leakage!)
- **Train/test split FIRST** (80/20, stratified)
- StandardScaler fitted **only on training data** for Time/Amount
- PCA features (V1–V28) already scaled by construction

### 2. Resampling Strategy Comparison
| Strategy | Train Samples | Fraud Samples | Description |
|----------|--------------|---------------|-------------|
| None | 227,845 | 394 | Use class weights only |
| Random Undersampling | 114,216 | 394 | Remove majority samples |
| SMOTE | 341,593 | 113,864 | Synthetic oversampling |
| ADASYN | 345,120 | 117,391 | Adaptive synthetic sampling |
| SMOTE+Tomek | 338,456 | 112,387 | SM Tomek links cleanup |

### 3. Models Trained
| Model | Type | Imbalance Handling |
|-------|------|-------------------|
| Logistic Regression | Linear baseline | `class_weight='balanced'` |
| Random Forest | Tree ensemble | `class_weight='balanced'` |
| **XGBoost** ⭐ | Gradient boosting | `scale_pos_weight` |
| LightGBM | Gradient boosting | `is_unbalance=True` |
| Isolation Forest | Unsupervised anomaly | Trained on legit only |

### 4. Evaluation Metrics
- **Precision-Recall AUC** (primary — not ROC-AUC which is misleading)
- **F1 Score**, Precision, Recall
- **Business Cost Function**: finds optimal threshold minimizing total cost
- **Confusion Matrix in Dollars**: $ saved, $ lost, $ spent on reviews

---

## 📈 Results

### Model Comparison (sorted by PR-AUC)
| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Net Benefit ($) |
|-------|--------|---------|-----|-----------|--------|----------------|
| **XGBoost** ⭐ | 0.8810 | 0.9724 | 0.7068 | 0.5828 | 0.8980 | $12,445 |
| Random Forest | 0.8352 | 0.9836 | 0.5641 | 0.4112 | 0.8980 | $12,130 |
| Logistic Regression | 0.7159 | 0.9722 | 0.6214 | 0.4780 | 0.8878 | $12,140 |
| Isolation Forest | 0.0981 | 0.9489 | 0.1243 | 0.0680 | 0.7245 | $5,430 |
| LightGBM | 0.0428 | 0.9054 | 0.0890 | 0.0470 | 0.8571 | $3,655 |

> *Results from actual training run on July 17, 2026. Optimal thresholds chosen via business cost function.*

### Business Impact (XGBoost — Best Model)
```
Fraud Caught:      $13,200.00  (88% of all fraud)
Fraud Missed:      $1,500.00
Review Costs:      $755.00
-----------------------------
Net Benefit:       $12,445.00
```

**Optimal Threshold:** 0.0298 (far below default 0.5 — validates our cost-based approach)
**Baseline Loss:** $73,800 (if no fraud detection at all)

---

## 🔍 Explainability (SHAP)

The model doesn't just say "fraud" — it explains **why**:

```
Transaction flagged as FRAUD (92% probability)
├── V14 = -5.23  →  +0.34 (increases fraud risk)
├── V4  =  4.12  →  +0.22 (increases fraud risk)
├── V12 = -3.89  →  +0.18 (increases fraud risk)
├── V10 = -2.45  →  +0.11 (increases fraud risk)
└── V16 = -0.12  →  -0.03 (slight mitigation)
```

**Top features by importance:** V14, V4, V12, V10, V17

---

## 🚀 Quick Start

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run EDA
jupyter notebook notebooks/01_eda.ipynb

# Run Preprocessing
jupyter notebook notebooks/02_preprocessing.ipynb

# Run Modeling
jupyter notebook notebooks/03_modeling.ipynb

# Run Explainability
jupyter notebook notebooks/04_explainability.ipynb

# Start API
uvicorn api.main:app --reload --port 8000

# Start Dashboard (in another terminal)
streamlit run app/dashboard.py
```

### Docker Setup

```bash
# Build and run everything
docker-compose up --build

# API: http://localhost:8000
# Dashboard: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

### API Usage

```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "Time": 100000.0,
    "Amount": 150.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    # ... all V1-V28 features
})

result = response.json()
print(f"Fraud: {result['is_fraud']} ({result['fraud_probability']:.1%})")
print(f"Explanation: {result['explanation']['summary']}")
```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/                    # Raw dataset (gitignored)
│   └── processed/              # Processed data, charts, results
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Preprocessing & Resampling
│   ├── 03_modeling.ipynb      # Model Training & Evaluation
│   └── 04_explainability.ipynb # SHAP Explanations
├── src/
│   ├── data_loader.py         # Data loading & statistics
│   ├── preprocessing.py       # Train/test split, scaling, resampling
│   ├── features.py            # Feature engineering
│   ├── train.py               # Model training (5 models)
│   ├── evaluate.py            # PR-AUC, business cost, threshold tuning
│   └── predict.py             # Prediction with SHAP explanations
├── api/
│   └── main.py                # FastAPI REST API
├── app/
│   └── dashboard.py           # Streamlit interactive dashboard
├── tests/
│   ├── test_preprocessing.py  # Unit tests for preprocessing
│   └── test_api.py            # API smoke tests
├── monitoring/
│   └── drift_detection.py     # Data drift detection (KS-test)
├── models/                     # Saved model artifacts
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Multi-service orchestration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📊 Monitoring

The drift detection script monitors incoming transaction distributions:

```python
from monitoring.drift_detection import DriftDetector

detector = DriftDetector(reference_data=X_train)
results = detector.detect_drift(new_incoming_data)
report = detector.generate_report(results)
print(report)
```

---

## 🗺️ What I'd Do Next

With more time and resources:
1. **Graph-based fraud detection** — model transactions as a graph to catch fraud rings
2. **Streaming pipeline** — Kafka for real-time transaction scoring
3. **GAN/CTGAN** — synthetic minority data as alternative to SMOTE
4. **A/B testing framework** — measure real-world model performance
5. **Customer segmentation fairness** — audit false-positive rates across demographics

---

## 📚 Key Learnings

1. **PR-AUC > ROC-AUC** for imbalanced data — ROC-AUC looks artificially great at 99.8% imbalance
2. **Split before resampling** — SMOTE on the full dataset leaks synthetic neighbors of test data
3. **Threshold matters** — default 0.5 is rarely optimal; use business cost function
4. **Explainability is essential** — fraud analysts need to know *why* a transaction is flagged
5. **Business metrics > academic metrics** — "net dollars saved" resonates with stakeholders

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using Python, XGBoost, SHAP, FastAPI, and Streamlit*
=======
# Credit-Card-Fraud-Detection
>>>>>>> 1fd137e6f80940af3efc38ceb364b8c98a67ef7b
