"""Streamlit interface for the student-risk advisory API."""

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
FEATURE_INPUTS = [
    ("attendance_pct", "Attendance (%)", 80.0, 1.0),
    ("attendance_trend_3w", "Attendance trend (percentage points)", 0.0, 0.5),
    ("assignment_completion_rate", "Assignment completion (0-1)", 0.6, 0.01),
    ("late_submission_rate", "Late submission rate (0-1)", 0.25, 0.01),
    ("prev_score_1", "Previous score 1", 65.0, 1.0),
    ("prev_score_2", "Previous score 2", 65.0, 1.0),
    ("prev_score_3", "Previous score 3", 65.0, 1.0),
    ("days_since_last_submission", "Days since last submission", 5.0, 1.0),
]

st.set_page_config(page_title="Student Risk Advisory", page_icon="📘", layout="wide")
st.title("Student Performance & At-Risk Advisory")
st.caption(
    "A statistical estimate to support teacher/counsellor check-ins. "
    "It does not replace professional judgement."
)

with st.sidebar:
    st.header("Week-8 student information")
    st.caption("Optional fields can be marked missing. Out-of-range values are clipped by the model.")
    with st.form("student_features"):
        entered: dict[str, float | None] = {}
        for feature, label, default, step in FEATURE_INPUTS:
            missing = st.checkbox(f"{label} is missing", key=f"{feature}_missing")
            value = st.number_input(
                label,
                value=float(default),
                step=float(step),
                format="%.2f",
                key=f"{feature}_value",
                disabled=missing,
            )
            entered[feature] = None if missing else float(value)
        assess = st.form_submit_button("Assess risk", type="primary")

tabs = st.tabs(["Student assessment", "Model information"])

with tabs[0]:
    st.subheader("Assessment")
    if assess:
        st.session_state.pop("risk_result", None)
        try:
            response = requests.post(
                f"{API_URL}/predict", json=entered, timeout=30
            )
            response.raise_for_status()
            st.session_state["risk_result"] = response.json()
        except requests.RequestException as exc:
            st.error(
                f"Could not reach the prediction API at {API_URL}. "
                "Start it with `uvicorn api.main:app --reload` and try again. "
                f"Details: {exc}"
            )

    result: dict[str, Any] | None = st.session_state.get("risk_result")
    if result:
        badge_colors = {"Low": "#2E7D32", "Medium": "#B26A00", "High": "#B3261E"}
        band = result["risk_band"]
        color = badge_colors.get(band, "#555555")
        left, middle, right = st.columns(3)
        left.markdown(
            f'<span style="background-color:{color};color:white;padding:0.35em 0.8em;'
            f'border-radius:0.6em;font-weight:700;">{band} risk</span>',
            unsafe_allow_html=True,
        )
        middle.metric("Estimated risk probability", f"{result['risk_probability']:.1%}")
        right.metric("At/above review threshold", "Yes" if result["flagged"] else "No")
        if result["uncertainty"] == "high":
            st.warning(
                f"Uncertainty: high. {result['confidence_note']}",
                icon="⚠️",
            )
        else:
            st.caption(
                f"Uncertainty: normal. {result['confidence_note']}"
            )

        st.markdown("#### Main signals")
        signals = result.get("top_signals", [])
        if signals:
            chart_data = pd.DataFrame(signals).sort_values("impact")
            colors = [
                "#C0392B" if effect == "increases risk" else "#2874A6"
                for effect in chart_data["effect"]
            ]
            fig, ax = plt.subplots(figsize=(8, 2.8))
            ax.barh(chart_data["feature"], chart_data["impact"], color=colors)
            ax.set(
                title="Change in risk probability when a feature is median-replaced",
                xlabel="Absolute probability change",
                ylabel="Week-8 feature",
            )
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            for signal in result["top_signals"]:
                st.write(f"- {signal['message']} (impact: {signal['impact']:.3f})")
        else:
            st.info("No feature-level signals were returned.")

        for warning in result.get("warnings", []):
            st.warning(warning)
        st.info(result["human_review_note"])
    elif not assess:
        st.info("Enter week-8 information in the sidebar and choose **Assess risk**.")

with tabs[1]:
    st.subheader("Model information")
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=10)
        response.raise_for_status()
        model_info = response.json()
        st.write(f"**Model:** {model_info['model_name']}")
        st.write(f"**Trained at:** {model_info.get('trained_at') or 'Not recorded'}")
        st.write(f"**Decision threshold:** {model_info['threshold']:.3f}")
        st.write(f"**Raw features:** {', '.join(model_info['features'])}")

        metrics = model_info.get("test_metrics", {})
        metric_columns = st.columns(5)
        for column, metric_name in zip(
            metric_columns, ("recall", "precision", "f1", "roc_auc", "pr_auc")
        ):
            if metric_name in metrics:
                column.metric(metric_name.replace("_", " ").title(), f"{metrics[metric_name]:.3f}")

        confusion_path = ROOT / "reports" / "figures" / "confusion_matrix_test.png"
        importance_path = ROOT / "reports" / "figures" / "permutation_importance.png"
        if confusion_path.is_file():
            st.image(str(confusion_path), caption="Held-out test confusion matrix")
        else:
            st.info("Confusion matrix image is not available in reports/figures.")
        if importance_path.is_file():
            st.image(str(importance_path), caption="Permutation importance, computed on validation data")
        else:
            st.info(
                "Validation permutation-importance plot is not available. "
                "Create it with `python -m src.make_global_importance`."
            )
    except requests.RequestException as exc:
        st.error(
            f"Could not load model information from {API_URL}. "
            "Start the API with `uvicorn api.main:app --reload`. "
            f"Details: {exc}"
        )

    st.markdown("#### Limitations and responsible use")
    st.markdown(
        "- **Synthetic data:** Evaluation and examples use synthetic data and may not reflect real students or schools.\n"
        "- **Proxy risk:** Attendance and other educational indicators may reflect circumstances not represented in the data; interpret them carefully.\n"
        "- **Advisory only:** Never automate grading, punishment, or admission decisions with this estimate.\n"
        "- **Human in the loop:** A teacher or counsellor must review individual context before any supportive action.\n"
        "- **Narrow target:** The estimate concerns only the defined week-10 score threshold."
    )
