"""
Streamlit Dashboard for Credit Card Fraud Detection

Features:
- Live-simulated transaction feed
- Fraud detection with real-time flagging
- Model confidence visualization
- Business impact metrics ($ saved vs $ lost)
- SHAP explanation display
"""

import os
import sys
import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import FraudPredictor

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="💳 Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown(
    """
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 2.2em;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9em;
        opacity: 0.85;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .safe-alert {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    .stMetric > div {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Load Models ────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model and predictor."""
    model_path = "models/xgboost.pkl"
    scaler_path = "models/scaler.pkl"
    threshold_path = "models/threshold.txt"

    if not os.path.exists(model_path):
        return None

    predictor = FraudPredictor(
        feature_names=[f"V{i}" for i in range(1, 29)] + ["Time", "Amount"],
        threshold=0.5,
    )
    predictor.load_model(model_path)

    if os.path.exists(scaler_path):
        predictor.load_scaler(scaler_path)

    if os.path.exists(threshold_path):
        predictor.threshold = float(threshold_path.read_text().strip())

    return predictor


# ─── Simulated Transaction Generator ────────────────────────
def generate_transaction(is_fraud: bool = None) -> dict:
    """Generate a simulated transaction (legitimate or fraudulent)."""
    if is_fraud is None:
        is_fraud = np.random.random() < 0.002  # ~0.2% fraud rate

    transaction = {}

    if is_fraud:
        # Fraudulent patterns: extreme values in key PCA features
        transaction["V1"] = np.random.normal(-4, 2)
        transaction["V2"] = np.random.normal(3, 2)
        transaction["V3"] = np.random.normal(-3, 3)
        transaction["V4"] = np.random.normal(4, 2)
        transaction["V5"] = np.random.normal(-2, 3)
        transaction["V6"] = np.random.normal(-1, 2)
        transaction["V7"] = np.random.normal(3, 2)
        transaction["V8"] = np.random.normal(0, 2)
        transaction["V9"] = np.random.normal(1, 2)
        transaction["V10"] = np.random.normal(-3, 2)
        transaction["V11"] = np.random.normal(3, 2)
        transaction["V12"] = np.random.normal(-4, 2)
        transaction["V13"] = np.random.normal(0, 1)
        transaction["V14"] = np.random.normal(-5, 2)
        transaction["V15"] = np.random.normal(0, 1)
        transaction["V16"] = np.random.normal(-3, 2)
        transaction["V17"] = np.random.normal(-4, 2)
        transaction["V18"] = np.random.normal(-2, 2)
        transaction["V19"] = np.random.normal(1, 1)
        transaction["V20"] = np.random.normal(1, 1)
        transaction["V21"] = np.random.normal(1, 1)
        transaction["V22"] = np.random.normal(0, 1)
        transaction["V23"] = np.random.normal(0, 1)
        transaction["V24"] = np.random.normal(0, 1)
        transaction["V25"] = np.random.normal(0, 1)
        transaction["V26"] = np.random.normal(0, 1)
        transaction["V27"] = np.random.normal(0, 0.5)
        transaction["V28"] = np.random.normal(0, 0.5)
        transaction["Time"] = np.random.uniform(0, 172800)
        transaction["Amount"] = np.random.uniform(50, 5000)
    else:
        # Legitimate patterns: near-zero PCA values
        for i in range(1, 29):
            transaction[f"V{i}"] = np.random.normal(0, 1)
        transaction["Time"] = np.random.uniform(0, 172800)
        transaction["Amount"] = np.random.exponential(50)

    transaction["true_class"] = int(is_fraud)
    return transaction


# ─── Initialize Session State ───────────────────────────────
if "transactions" not in st.session_state:
    st.session_state.transactions = []
if "total_caught" not in st.session_state:
    st.session_state.total_caught = 0
if "total_missed" not in st.session_state:
    st.session_state.total_missed = 0
if "total_review_cost" not in st.session_state:
    st.session_state.total_review_cost = 0
if "total_saved" not in st.session_state:
    st.session_state.total_saved = 0


# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-card-back-side.png", width=80)
    st.title("⚙️ Controls")

    st.markdown("---")
    simulation_speed = st.slider("Simulation Speed", 1, 10, 3)
    fraud_rate = st.slider("Fraud Rate (%)", 0.1, 5.0, 0.5, 0.1) / 100
    n_transactions = st.number_input("Transactions per batch", 1, 50, 5)

    st.markdown("---")
    if st.button("🚀 Run Simulation Batch", type="primary"):
        st.session_state.run_batch = True
        st.session_state.batch_size = n_transactions
        st.session_state.fraud_rate = fraud_rate

    if st.button("🔄 Reset Dashboard"):
        st.session_state.transactions = []
        st.session_state.total_caught = 0
        st.session_state.total_missed = 0
        st.session_state.total_review_cost = 0
        st.session_state.total_saved = 0
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    predictor = load_model()
    if predictor:
        st.success("✅ Model loaded")
        st.info(f"Threshold: {predictor.threshold:.4f}")
    else:
        st.warning("⚠️ No model found. Run training first.")


# ─── Main Dashboard ─────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown(
    "*Real-time transaction monitoring with AI-powered fraud detection and SHAP explainability*"
)

# Top-level metrics
col1, col2, col3, col4, col5 = st.columns(5)

total_txns = len(st.session_state.transactions)
fraud_count = sum(1 for t in st.session_state.transactions if t.get("true_class") == 1)
flagged_count = sum(
    1 for t in st.session_state.transactions if t.get("is_fraud", False)
)

with col1:
    st.metric("Total Transactions", f"{total_txns:,}")
with col2:
    st.metric("Actual Fraud", f"{fraud_count}", delta=None, delta_color="inverse")
with col3:
    st.metric("Flagged as Fraud", f"{flagged_count}")
with col4:
    st.metric("💰 Saved", f"${st.session_state.total_saved:,.2f}", delta_color="normal")
with col5:
    st.metric(
        "💸 Review Costs",
        f"${st.session_state.total_review_cost:,.2f}",
        delta_color="inverse",
    )

st.markdown("---")

# Run simulation batch
if st.session_state.get("run_batch", False):
    st.session_state.run_batch = False
    fr = st.session_state.get("fraud_rate", 0.005)

    progress = st.progress(0)
    status = st.empty()

    for i in range(st.session_state.get("batch_size", 5)):
        progress.progress((i + 1) / st.session_state.get("batch_size", 5))

        is_fraud = np.random.random() < fr
        txn = generate_transaction(is_fraud)

        if predictor:
            txn_for_pred = {k: v for k, v in txn.items() if k != "true_class"}
            result = predictor.predict_single(txn_for_pred, return_shap=False)
            txn["fraud_probability"] = result["fraud_probability"]
            txn["is_fraud"] = result["is_fraud"]
            txn["decision"] = result["decision"]
        else:
            txn["fraud_probability"] = 0.0
            txn["is_fraud"] = False
            txn["decision"] = "UNKNOWN"

        # Update business metrics
        if txn["true_class"] == 1:
            if txn["is_fraud"]:
                st.session_state.total_saved += 150.0  # Fraud caught
                st.session_state.total_caught += 1
            else:
                st.session_state.total_missed += 1  # Fraud missed
        if txn["is_fraud"]:
            st.session_state.total_review_cost += 5.0  # Review cost

        st.session_state.transactions.append(txn)
        # Keep only last 500 transactions to prevent memory growth
        if len(st.session_state.transactions) > 500:
            st.session_state.transactions = st.session_state.transactions[-500:]
        status.text(
            f"Processed transaction {i+1}/{st.session_state.get('batch_size', 5)}..."
        )
        time.sleep(0.1 / simulation_speed)

    progress.empty()
    status.empty()
    st.rerun()

# ─── Transaction Feed ───────────────────────────────────────
if st.session_state.transactions:
    st.markdown("## 📋 Recent Transactions")

    # Show last 20 transactions
    recent = st.session_state.transactions[-20:][::-1]

    for i, txn in enumerate(recent):
        is_fraud = txn.get("is_fraud", False)
        prob = txn.get("fraud_probability", 0)

        if is_fraud:
            icon = "🔴"
            badge = f"FRAUD ({prob:.1%})"
        else:
            icon = "🟢"
            badge = f"OK ({prob:.1%})"

        with st.expander(
            f"{icon} Transaction #{total_txns - i} — {badge} — ${txn.get('Amount', 0):.2f}",
            expanded=is_fraud,
        ):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**Amount:** ${txn.get('Amount', 0):.2f}")
                st.write(f"**Time:** {txn.get('Time', 0):.0f}s")
                st.write(
                    f"**True Class:** {'Fraud' if txn.get('true_class') else 'Legitimate'}"
                )
            with c2:
                st.write(f"**Predicted:** {txn.get('decision', 'N/A')}")
                st.write(f"**Confidence:** {prob:.2%}")
            with c3:
                if is_fraud and txn.get("true_class") == 1:
                    st.success("✅ Correctly caught fraud — $150 saved!")
                elif is_fraud and not txn.get("true_class"):
                    st.warning("⚠️ False positive — $5 review cost")
                elif not is_fraud and txn.get("true_class"):
                    st.error("❌ Missed fraud — $150 lost!")
                else:
                    st.info("✅ Correctly approved")

    # ─── Charts ─────────────────────────────────────────────
    st.markdown("## 📊 Analytics")

    col_a, col_b = st.columns(2)

    with col_a:
        # Probability distribution
        probs = [t.get("fraud_probability", 0) for t in st.session_state.transactions]
        fig_prob = px.histogram(
            x=probs,
            nbins=50,
            title="Fraud Probability Distribution",
            labels={"x": "Fraud Probability", "y": "Count"},
            color_discrete_sequence=["#667eea"],
        )
        fig_prob.add_vline(
            x=predictor.threshold if predictor else 0.5,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_b:
        # Cumulative business impact
        cumulative_saved = []
        cumulative_lost = []
        cumulative_cost = []
        saved_total = 0
        lost_total = 0
        cost_total = 0

        for t in st.session_state.transactions:
            if t.get("true_class") == 1 and t.get("is_fraud"):
                saved_total += 150
            elif t.get("true_class") == 1 and not t.get("is_fraud"):
                lost_total += 150
            if t.get("is_fraud"):
                cost_total += 5
            cumulative_saved.append(saved_total)
            cumulative_lost.append(lost_total)
            cumulative_cost.append(cost_total)

        fig_impact = go.Figure()
        fig_impact.add_trace(
            go.Scatter(
                y=cumulative_saved,
                name="Fraud Caught ($)",
                fill="tozeroy",
                line=dict(color="#38ef7d"),
            )
        )
        fig_impact.add_trace(
            go.Scatter(
                y=cumulative_lost,
                name="Fraud Missed ($)",
                fill="tozeroy",
                line=dict(color="#ff416c"),
            )
        )
        fig_impact.add_trace(
            go.Scatter(
                y=cumulative_cost,
                name="Review Costs ($)",
                fill="tozeroy",
                line=dict(color="#ffa726"),
            )
        )
        fig_impact.update_layout(
            title="Cumulative Business Impact",
            xaxis_title="Transactions",
            yaxis_title="USD ($)",
        )
        st.plotly_chart(fig_impact, use_container_width=True)

    # ─── Business Impact Summary ────────────────────────────
    st.markdown("## 💰 Business Impact Summary")

    net_benefit = st.session_state.total_saved - st.session_state.total_review_cost

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        st.metric(
            "Fraud Caught",
            f"{st.session_state.total_caught}",
            f"${st.session_state.total_saved:,.0f} saved",
        )
    with bc2:
        st.metric(
            "Fraud Missed",
            f"{st.session_state.total_missed}",
            f"${st.session_state.total_missed * 150:,.0f} lost",
        )
    with bc3:
        flagged_total = sum(
            1 for t in st.session_state.transactions if t.get("is_fraud")
        )
        st.metric(
            "Reviews Conducted",
            f"{flagged_total}",
            f"${st.session_state.total_review_cost:,.0f} cost",
        )
    with bc4:
        st.metric("Net Benefit", f"${net_benefit:,.2f}", delta_color="normal")

else:
    # No transactions yet
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; padding: 60px 20px;">
        <h2>👋 Welcome to the Fraud Detection Dashboard</h2>
        <p style="font-size: 1.2em; color: #666;">
            Click <strong>"Run Simulation Batch"</strong> in the sidebar to start detecting fraud.
        </p>
        <p style="color: #999;">
            The dashboard simulates real-time credit card transactions and uses an XGBoost model
            with SHAP explainability to flag fraudulent activity.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ─── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Built with ❤️ using Streamlit, XGBoost, and SHAP | "
    "Credit Card Fraud Detection Portfolio Project*"
)
