"""Streamlit interface for the student-risk advisory API."""

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from app.dashboard_utils import (
    HUMAN_REVIEW_NOTE,
    RAW_FEATURES,
    bounded_slider_value,
    parse_class_csv,
    template_csv,
)

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
        if assess:
            st.session_state["assessed_inputs"] = entered.copy()

tabs = st.tabs(
    ["Student assessment", "Class overview", "What-if", "Model information"]
)

with tabs[0]:
    st.subheader("Assessment")
    if assess:
        st.session_state.pop("risk_result", None)
        st.session_state.pop("what_if_result", None)
        for key in ("what_if_completion", "what_if_attendance", "what_if_late_rate"):
            st.session_state.pop(key, None)
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
            st.info(
                result.get("signal_note")
                or "No single input moved the estimate noticeably."
            )

        for warning in result.get("warnings", []):
            st.warning(warning)
        st.info(result["human_review_note"])
    elif not assess:
        st.info("Enter week-8 information in the sidebar and choose **Assess risk**.")

with tabs[1]:
    st.subheader("Class overview")
    st.caption(
        "Upload up to 100 rows. Only the eight model features are sent to the API; "
        "any additional CSV columns remain local."
    )
    st.download_button(
        "Download CSV template",
        data=template_csv(),
        file_name="student_features_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader(
        "Upload class features CSV",
        type=["csv"],
        key="class_csv_upload",
    )
    if uploaded_file is not None:
        try:
            class_features = parse_class_csv(uploaded_file.getvalue())
            st.caption(f"Validated {len(class_features)} student rows.")
            if st.button("Assess class", type="primary", key="assess_class"):
                st.session_state.pop("class_results", None)
                safe_features = class_features.astype(object).where(
                    pd.notna(class_features), None
                )
                records = safe_features.to_dict(orient="records")
                try:
                    response = requests.post(
                        f"{API_URL}/predict-batch",
                        json={"records": records},
                        timeout=60,
                    )
                    response.raise_for_status()
                    predictions = response.json()["predictions"]
                    if len(predictions) != len(class_features):
                        raise ValueError(
                            "The API returned a different number of predictions "
                            "than uploaded rows."
                        )
                    output = class_features.copy()
                    output.insert(0, "row_number", range(1, len(output) + 1))
                    output["risk_probability"] = [
                        prediction["risk_probability"] for prediction in predictions
                    ]
                    output["risk_band"] = [
                        prediction["risk_band"] for prediction in predictions
                    ]
                    output["flagged"] = [
                        prediction["flagged"] for prediction in predictions
                    ]
                    output["uncertainty"] = [
                        prediction["uncertainty"] for prediction in predictions
                    ]
                    st.session_state["class_results"] = output
                    st.session_state["class_review_note"] = predictions[0][
                        "human_review_note"
                    ]
                except (requests.RequestException, KeyError, ValueError) as exc:
                    st.error(
                        f"Could not get valid batch predictions from {API_URL}. "
                        "Check that the API is running, then try again. "
                        f"Details: {exc}"
                    )
        except ValueError as exc:
            st.error(str(exc))

    class_results: pd.DataFrame | None = st.session_state.get("class_results")
    if class_results is not None:
        sorted_results = class_results.sort_values(
            "risk_probability", ascending=False
        ).reset_index(drop=True)
        counts = sorted_results["risk_band"].value_counts()
        count_columns = st.columns(3)
        for column, band in zip(count_columns, ("Low", "Medium", "High")):
            column.metric(f"{band} risk", int(counts.get(band, 0)))
        band_colors = {
            "Low": "background-color: #d9ead3; color: #1b5e20",
            "Medium": "background-color: #fff2cc; color: #7f6000",
            "High": "background-color: #f4cccc; color: #990000",
        }
        styled_results = sorted_results.style.map(
            lambda value: band_colors.get(value, ""),
            subset=["risk_band"],
        )
        st.dataframe(styled_results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download prediction results CSV",
            data=sorted_results.to_csv(index=False).encode("utf-8"),
            file_name="student_risk_predictions.csv",
            mime="text/csv",
        )
        st.info(st.session_state.get("class_review_note", HUMAN_REVIEW_NOTE))

with tabs[2]:
    st.subheader("What-if scenario")
    assessed_inputs: dict[str, float | None] | None = st.session_state.get(
        "assessed_inputs"
    )
    original_result: dict[str, Any] | None = st.session_state.get("risk_result")
    if assessed_inputs is None or original_result is None:
        st.info("Assess a student in the sidebar first to create a what-if baseline.")
    else:
        with st.form("what_if_scenario"):
            completion = st.slider(
                "Assignment completion",
                min_value=0.0,
                max_value=1.0,
                value=bounded_slider_value(
                    assessed_inputs["assignment_completion_rate"], 0.6, 0.0, 1.0
                ),
                step=0.01,
                key="what_if_completion",
            )
            attendance = st.slider(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=bounded_slider_value(
                    assessed_inputs["attendance_pct"], 80.0, 0.0, 100.0
                ),
                step=1.0,
                key="what_if_attendance",
            )
            late_rate = st.slider(
                "Late-submission rate",
                min_value=0.0,
                max_value=1.0,
                value=bounded_slider_value(
                    assessed_inputs["late_submission_rate"], 0.25, 0.0, 1.0
                ),
                step=0.01,
                key="what_if_late_rate",
            )
            compare_scenario = st.form_submit_button("Compare scenario")

        if compare_scenario:
            scenario = assessed_inputs.copy()
            scenario.update(
                {
                    "assignment_completion_rate": completion,
                    "attendance_pct": attendance,
                    "late_submission_rate": late_rate,
                }
            )
            try:
                response = requests.post(
                    f"{API_URL}/predict", json=scenario, timeout=30
                )
                response.raise_for_status()
                st.session_state["what_if_result"] = response.json()
            except requests.RequestException as exc:
                st.error(
                    f"Could not reach the prediction API at {API_URL}. "
                    "Start it with `uvicorn api.main:app --reload` and try again. "
                    f"Details: {exc}"
                )

        scenario_result: dict[str, Any] | None = st.session_state.get(
            "what_if_result"
        )
        if scenario_result:
            old_probability = float(original_result["risk_probability"])
            new_probability = float(scenario_result["risk_probability"])
            old_column, new_column = st.columns(2)
            old_column.metric("Current estimate", f"{old_probability:.1%}")
            new_column.metric(
                "What-if estimate",
                f"{new_probability:.1%}",
                delta=f"{new_probability - old_probability:+.1%}",
            )
        st.info(
            "This comparison is a model-based statistical association, not a "
            "guaranteed causal effect. Changing these values in the model does "
            "not establish what would happen if a student's circumstances changed."
        )

with tabs[3]:
    st.subheader("Model information")
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=10)
        response.raise_for_status()
        model_info = response.json()
        st.write(f"**Model:** {model_info['model_name']}")
        st.write(f"**Trained at:** {model_info.get('trained_at') or 'Not recorded'}")
        st.write(f"**Decision threshold:** {model_info['threshold']:.3f}")
        st.caption(
            "Risk bands: "
            + "; ".join(
                f"{band}: {definition}"
                for band, definition in model_info["risk_band_definitions"].items()
            )
            + f". Predictions are flagged for review at or above the decision "
            f"threshold ({model_info['threshold']:.1%}), including Medium-band "
            "predictions when the threshold is below 50%."
        )
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
