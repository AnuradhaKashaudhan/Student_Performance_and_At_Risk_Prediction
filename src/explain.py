"""Model-agnostic explanations and inference helpers for the saved pipeline."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.preprocess import FEATURES

HUMAN_REVIEW_NOTE = (
    "Advisory only. This is a statistical estimate, not a judgement about the "
    "student. A teacher or counsellor must review before any action. Never use "
    "for grading, punishment or admission decisions."
)
ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

FEATURE_LABELS = {
    "attendance_pct": "Attendance",
    "attendance_trend_3w": "Three-week attendance trend",
    "assignment_completion_rate": "Assignment completion",
    "late_submission_rate": "Late submission rate",
    "prev_score_1": "Previous score 1",
    "prev_score_2": "Previous score 2",
    "prev_score_3": "Previous score 3",
    "days_since_last_submission": "Days since last submission",
}
RATE_FEATURES = {"assignment_completion_rate", "late_submission_rate"}


def load_artifact(
    model_path: str | Path, meta_path: str | Path
) -> tuple[Any, dict[str, Any]]:
    """Load the trained sklearn pipeline and its JSON metadata.

    Both paths must point to artifacts produced by ``python -m src.train``.
    """
    model_file, metadata_file = Path(model_path), Path(meta_path)
    missing = [str(path) for path in (model_file, metadata_file) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s): {', '.join(missing)}. "
            "Run `python -m src.train` from the student-risk project directory."
        )
    try:
        pipeline = joblib.load(model_file)
    except (OSError, EOFError, ValueError, ImportError) as exc:
        raise RuntimeError(
            f"Could not load model artifact at {model_file}: {exc}. "
            "Run `python -m src.train` to recreate it."
        ) from exc
    try:
        with metadata_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read model metadata at {metadata_file}: {exc}") from exc

    required_metadata = {
        "model_name",
        "features",
        "threshold",
        "band_cutoffs",
        "training_medians",
        "valid_ranges",
        "test_metrics",
    }
    missing_metadata = required_metadata.difference(metadata)
    if missing_metadata:
        raise ValueError(
            f"Model metadata is missing required fields: {sorted(missing_metadata)}. "
            "Run `python -m src.train` to recreate it."
        )
    if metadata["features"] != FEATURES:
        raise ValueError("Artifact feature list does not match the project's raw features.")
    return pipeline, metadata


def _is_missing(value: Any) -> bool:
    """Return whether a raw numeric value is null or NaN."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _display_value(feature: str, value: Any) -> str:
    """Format feature values in concise, student-friendly language."""
    if _is_missing(value):
        return "missing"
    number = float(value)
    if feature == "attendance_pct":
        return f"{number:.0f}%"
    if feature in RATE_FEATURES:
        return f"{number:.0%}"
    if feature == "attendance_trend_3w":
        return f"{number:+.1f} percentage points"
    if feature.startswith("prev_score_"):
        return f"{number:.1f} points"
    if feature == "days_since_last_submission":
        return f"{number:g} days"
    return f"{number:.1f}"


def _signal_message(
    feature: str, value: Any, effect: str, median: float
) -> str:
    """Explain a feature effect using its value relative to the training median."""
    label = FEATURE_LABELS[feature]
    if _is_missing(value):
        return (
            f"{label} is missing (median-imputed); this is associated with "
            f"{effect} in this prediction."
        )
    number = float(value)
    comparison = (
        "below"
        if number < median
        else "above"
        if number > median
        else "at"
    )
    return (
        f"{label} of {_display_value(feature, value)} is {comparison} the typical level "
        f"and {effect}."
    )


def predict_one(
    pipeline: Any, meta: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Predict one student's risk and explain its strongest raw-feature effects.

    Each local effect is the original model probability minus the probability
    after replacing one raw feature with its training median. This is an
    interpretable perturbation, not a causal explanation.
    """
    features = meta["features"]
    unexpected = set(record).difference(features)
    if unexpected:
        raise ValueError(f"Unexpected input feature(s): {sorted(unexpected)}")
    raw_record = {feature: record.get(feature) for feature in features}
    for feature, value in raw_record.items():
        if _is_missing(value):
            continue
        try:
            finite = np.isfinite(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be a finite number or null.") from exc
        if not finite:
            raise ValueError(f"{feature} must be a finite number or null.")

    counterfactual_rows = [dict(raw_record)]
    for feature in features:
        counterfactual = dict(raw_record)
        counterfactual[feature] = float(meta["training_medians"][feature])
        counterfactual_rows.append(counterfactual)
    inference_frame = pd.DataFrame(counterfactual_rows, columns=features)
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("The loaded model does not expose a positive risk class (1).")
    probabilities = pipeline.predict_proba(inference_frame)[:, classes.index(1)]
    probability = float(probabilities[0])
    threshold = float(meta["threshold"])
    band_cutoffs = meta["band_cutoffs"]
    low_max = float(band_cutoffs["low_max"])
    medium_max = float(band_cutoffs["medium_max"])
    if probability < low_max:
        risk_band = "Low"
    elif probability < medium_max:
        risk_band = "Medium"
    else:
        risk_band = "High"

    warnings: list[str] = []
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor")
    cleaner = getattr(preprocessor, "named_steps", {}).get("cleaner")
    clipped_features = getattr(cleaner, "clipping_rules_", {})
    for feature, value in raw_record.items():
        if _is_missing(value):
            warnings.append(f"{feature} missing: median imputed, prediction less certain")
            continue
        bounds = meta["valid_ranges"].get(feature, {})
        minimum, maximum = bounds.get("min"), bounds.get("max")
        number = float(value)
        if (minimum is not None and number < minimum) or (
            maximum is not None and number > maximum
        ):
            lower_text = str(minimum) if minimum is not None else "-infinity"
            upper_text = str(maximum) if maximum is not None else "infinity"
            action = (
                "clipped"
                if feature in clipped_features
                else "not clipped by preprocessing"
            )
            warnings.append(
                f"{feature}={number:g} is outside {lower_text}-{upper_text}: {action}"
            )
    if abs(probability - threshold) <= 0.07:
        warnings.append("borderline case: higher uncertainty")

    medians = meta["training_medians"]
    signals: list[tuple[float, dict[str, Any]]] = []
    for feature, counterfactual_probability in zip(features, probabilities[1:]):
        delta = probability - float(counterfactual_probability)
        effect = "increases risk" if delta > 0 else "reduces risk"
        signals.append(
            (
                abs(delta),
                {
                    "feature": feature,
                    "value": raw_record[feature],
                    "effect": effect,
                    "impact": round(abs(delta), 3),
                    "message": _signal_message(
                        feature, raw_record[feature], effect, float(medians[feature])
                    ),
                },
            )
        )
    signals.sort(key=lambda pair: pair[0], reverse=True)

    return {
        "risk_probability": round(probability, 3),
        "risk_band": risk_band,
        "flagged": probability >= threshold,
        "top_signals": [signal for _, signal in signals[:3]],
        "warnings": warnings,
        "human_review_note": HUMAN_REVIEW_NOTE,
        "model_name": meta["model_name"],
        "threshold": threshold,
    }


def global_importance(
    pipeline: Any,
    meta: dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Compute and save permutation importance using validation data only."""
    features = meta["features"]
    missing = set(features).difference(X_val.columns)
    if missing:
        raise ValueError(f"Validation data is missing features: {sorted(missing)}")
    validation_features = X_val.loc[:, features]
    importance = permutation_importance(
        pipeline,
        validation_features,
        y_val,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )
    results = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "global_importance.csv", index=False)
    import matplotlib.pyplot as plt

    chart = results.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        chart["feature"],
        chart["importance_mean"],
        xerr=chart["importance_std"],
        color="#4C78A8",
        alpha=0.9,
    )
    ax.set(
        title="Validation permutation importance (average precision)",
        xlabel="Decrease in average precision after permutation",
        ylabel="Raw feature",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return results
