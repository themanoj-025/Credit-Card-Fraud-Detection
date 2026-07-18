"""
Case Investigator Page

Select any flagged transaction and see:
- SHAP force plot
- LLM plain-English narrative
- 3 similar historical cases with their outcomes

This page is the explainability showcase — make it the most polished screen.
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from app.components.metric_cards import metric_card
from src.fraudshield.config import API_URL


def show() -> None:
    """Render the Case Investigator page."""
    st.markdown(
        "<h1>🔍 Case Investigator</h1>"
        "<p style='color: #a0a0a0; margin-top: -12px;'>"
        "Deep-dive into flagged transactions with SHAP, LLM narratives, "
        "and similar case retrieval</p>",
        unsafe_allow_html=True,
    )

    # ─── Transaction Selection ─────────────────────────────────────────
    st.markdown("<h3>Select a Transaction</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        # Simulate a few recent transactions
        if "recent_cases" not in st.session_state:
            st.session_state.recent_cases = _generate_sample_cases()

        selected_tx = st.selectbox(
            "Choose a recent flagged transaction:",
            options=st.session_state.recent_cases,
            format_func=lambda x: f"Transaction #{x['id']} — ${x['amount']:,.2f} ({x['probability']:.1%})",
        )

    with col2:
        if st.button("🔍 Analyze", use_container_width=True):
            st.session_state.analyze_tx = selected_tx

    # ─── Analysis Results ─────────────────────────────────────────────
    if "analyze_tx" in st.session_state:
        tx = st.session_state.analyze_tx

        st.markdown("---")

        # Top-level verdict
        is_fraud = tx.get("is_fraud", True)
        verdict_color = "#ff6b6b" if is_fraud else "#38ef7d"
        verdict_text = "FRAUD FLAGGED" if is_fraud else "CLEARED"

        st.markdown(f"""
        <div style="
            background: #1a1a2e;
            border: 1px solid {verdict_color};
            border-left: 4px solid {verdict_color};
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: {verdict_color}; font-size: 18px; font-weight: 700;">
                        {verdict_text}
                    </span>
                    <span style="color: #a0a0a0; margin-left: 12px;">
                        Probability: {tx['probability']:.1%} | 
                        Amount: ${tx['amount']:,.2f} |
                        Account: {tx.get('account', 'N/A')}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─── Three-Column Layout: SHAP | Narrative | Similar Cases ─────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3>📊 SHAP Analysis</h3>", unsafe_allow_html=True)
            shap_values = tx.get("shap_values", [])
            if shap_values:
                shap_df = pd.DataFrame(shap_values)
                fig = px.bar(
                    shap_df,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color="impact",
                    color_discrete_map={
                        "increases": "#ff6b6b",
                        "decreases": "#38ef7d",
                    },
                    title="Feature Contributions",
                    labels={"shap_value": "SHAP Value", "feature": ""},
                )
                fig.update_layout(
                    plot_bgcolor="#1a1a2e",
                    paper_bgcolor="#0f0f1a",
                    font_color="#e0e0e0",
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Feature values table
            st.markdown("<h4>Feature Values</h4>", unsafe_allow_html=True)
            features_df = pd.DataFrame([
                {"Feature": f"V{i}", "Value": tx.get(f"V{i}", 0)}
                for i in range(1, 15)
            ])
            st.dataframe(features_df, use_container_width=True, hide_index=True)

        with col2:
            # LLM Narrative
            st.markdown("<h3>📝 Analyst Narrative</h3>", unsafe_allow_html=True)
            narrative = tx.get("narrative", "")
            if narrative:
                st.markdown(f"""
                <div style="
                    background: #1a1a2e;
                    border: 1px solid #2a2a3e;
                    border-radius: 8px;
                    padding: 16px;
                    min-height: 120px;
                ">
                    <p style="color: #e0e0e0; font-size: 14px; line-height: 1.6;">
                        {narrative}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("LLM narrative unavailable. Set ANTHROPIC_API_KEY to enable.")

            # Similar Cases (RAG)
            st.markdown("<h3>🔗 Similar Historical Cases</h3>", unsafe_allow_html=True)
            similar = tx.get("similar_cases", [])
            for case in similar:
                outcome = case.get("actual_outcome", "unknown")
                color = "#3a1a1a" if outcome == "confirmed_fraud" else "#1a3a2a"
                label = "🔴 Confirmed Fraud" if outcome == "confirmed_fraud" else "🟢 False Positive"
                st.markdown(f"""
                <div style="
                    background: {color};
                    border: 1px solid {'#ff6b6b33' if outcome == 'confirmed_fraud' else '#38ef7d33'};
                    border-radius: 8px;
                    padding: 10px 12px;
                    margin: 4px 0;
                ">
                    <span style="color: #e0e0e0; font-weight: 600;">{label}</span>
                    <span style="color: #a0a0a0; margin-left: 8px;">
                        Similarity: {case.get('similarity_score', 0):.2%}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            if not similar:
                st.caption("No similar cases found.")

        # Business Impact Summary
        st.markdown("---")
        st.markdown("<h3>💰 Business Impact</h3>", unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            metric_card("Potential Loss",
                         f"${tx.get('amount', 0) * 1.5:,.0f}",
                         icon="⚠️", color="#ff6b6b")
        with b2:
            metric_card("Review Cost",
                         "$5.00", icon="💰", color="#f1c40f")
        with b3:
            metric_card("Similar Cases",
                         str(len(tx.get("similar_cases", []))),
                         icon="🔗", color="#667eea")
        with b4:
            metric_card("Action",
                         "Manual Review" if is_fraud else "Auto-Approve",
                         icon="🎯", color="#38ef7d")


def _generate_sample_cases() -> List[Dict[str, Any]]:
    """Generate sample cases for demo purposes."""
    return [
        {
            "id": 4821,
            "amount": 2980.50,
            "probability": 0.94,
            "is_fraud": True,
            "account": "****4532",
            "narrative": (
                "This transaction was flagged due to an unusual combination of V14 and V4, "
                "consistent with patterns seen in card-not-present fraud. The amount ($2,980) "
                "is above the customer's typical spending profile, and the transaction occurred "
                "at 3:14 AM local time, which is outside their usual activity window."
            ),
            "shap_values": [
                {"feature": "V14", "shap_value": 0.34, "impact": "increases"},
                {"feature": "V4", "shap_value": 0.22, "impact": "increases"},
                {"feature": "V12", "shap_value": 0.18, "impact": "increases"},
                {"feature": "V10", "shap_value": 0.11, "impact": "increases"},
                {"feature": "V17", "shap_value": -0.03, "impact": "decreases"},
            ],
            "similar_cases": [
                {"actual_outcome": "confirmed_fraud", "similarity_score": 0.87},
                {"actual_outcome": "confirmed_fraud", "similarity_score": 0.76},
                {"actual_outcome": "false_positive", "similarity_score": 0.62},
            ],
        },
        {
            "id": 4819,
            "amount": 152.30,
            "probability": 0.12,
            "is_fraud": False,
            "account": "****7891",
            "narrative": (
                "This transaction appears legitimate. The amount ($152.30) is consistent "
                "with the customer's typical spending patterns, and all feature values fall "
                "within normal ranges. No fraud indicators detected."
            ),
            "shap_values": [
                {"feature": "V14", "shap_value": -0.02, "impact": "decreases"},
                {"feature": "V4", "shap_value": 0.01, "impact": "increases"},
                {"feature": "V12", "shap_value": -0.01, "impact": "decreases"},
            ],
            "similar_cases": [
                {"actual_outcome": "false_positive", "similarity_score": 0.91},
            ],
        },
        {
            "id": 4817,
            "amount": 7500.00,
            "probability": 0.87,
            "is_fraud": True,
            "account": "****1209",
            "narrative": (
                "High-value transaction flagged at elevated confidence. The amount ($7,500) "
                "is significantly above normal patterns. Strong V14 and V4 signals suggest "
                "this matches a known synthetic identity fraud pattern observed in recent weeks."
            ),
            "shap_values": [
                {"feature": "V14", "shap_value": 0.41, "impact": "increases"},
                {"feature": "V4", "shap_value": 0.31, "impact": "increases"},
                {"feature": "Amount", "shap_value": 0.25, "impact": "increases"},
                {"feature": "V12", "shap_value": 0.15, "impact": "increases"},
            ],
            "similar_cases": [
                {"actual_outcome": "confirmed_fraud", "similarity_score": 0.94},
                {"actual_outcome": "confirmed_fraud", "similarity_score": 0.82},
                {"actual_outcome": "confirmed_fraud", "similarity_score": 0.71},
            ],
        },
    ]
