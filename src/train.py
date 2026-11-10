"""Train, tune, evaluate, and save the student at-risk advisory model."""

from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluate import (
    evaluate,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
    select_threshold,
)
from src.preprocess import (
    FEATURES,
    FORBIDDEN,
    Cleaner,
    build_preprocessor,
    assert_no_forbidden,
)
from src.split import chronological_split, load_data

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


class RuleBaseline(ClassifierMixin, BaseEstimator):
    """A transparent baseline that flags a low newest historical score.

    Missing ``prev_score_3`` values do not satisfy the rule and are assigned a
    risk probability of zero; this keeps the baseline deterministic and easy to
    explain, while the learned models impute missing values from training data.
    """

    def fit(self, X: pd.DataFrame, y: object) -> "RuleBaseline":
        """Record the two binary classes to satisfy the sklearn classifier API."""
        if "prev_score_3" not in X.columns:
            raise ValueError("Rule baseline requires the raw prev_score_3 feature.")
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return deterministic zero/one probabilities from the score rule."""
        if "prev_score_3" not in X.columns:
            raise ValueError("Rule baseline requires the raw prev_score_3 feature.")
        score = pd.to_numeric(X["prev_score_3"], errors="coerce")
        risk = (score < 50).fillna(False).to_numpy(dtype=int)
        return np.column_stack((1 - risk, risk)).astype(float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the rule's binary decision."""
        return self.predict_proba(X)[:, 1].astype(int)


def _make_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Put preprocessing and a classifier in one raw-feature pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def _build_models() -> tuple[dict[str, BaseEstimator], bool]:
    """Create seeded baselines and learned classifiers.

    Older sklearn releases did not expose ``class_weight`` on histogram
    gradient boosting. In that case callers supply balanced sample weights to
    each fit; current releases use the native class-weight option.
    """
    supports_class_weight = (
        "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters
    )
    histogram_parameters: dict[str, Any] = {"random_state": SEED}
    if supports_class_weight:
        histogram_parameters["class_weight"] = "balanced"

    return (
        {
            "dummy_stratified": DummyClassifier(
                strategy="stratified", random_state=SEED
            ),
            "rule_prev_score_3_lt_50": RuleBaseline(),
            "logistic_regression": _make_pipeline(
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=SEED
                )
            ),
            "random_forest": _make_pipeline(
                RandomForestClassifier(
                    class_weight="balanced", random_state=SEED, n_jobs=-1
                )
            ),
            "hist_gradient_boosting": _make_pipeline(
                HistGradientBoostingClassifier(**histogram_parameters)
            ),
        },
        supports_class_weight,
    )


def _fit_model(
    model: BaseEstimator,
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    supports_hgb_class_weight: bool,
) -> BaseEstimator:
    """Fit one model, applying balanced sample weights for older HGB versions."""
    if name == "hist_gradient_boosting" and not supports_hgb_class_weight:
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, classifier__sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def _get_search_space(name: str) -> dict[str, list[Any]]:
    """Return small, explicit randomized-search grids for a learned model."""
    if name == "logistic_regression":
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__solver": ["lbfgs", "liblinear"],
        }
    if name == "random_forest":
        return {
            "classifier__n_estimators": [100, 160, 240],
            "classifier__max_depth": [None, 5, 10, 18],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.8],
        }
    if name == "hist_gradient_boosting":
        return {
            "classifier__learning_rate": [0.03, 0.06, 0.1],
            "classifier__max_iter": [100, 150, 200],
            "classifier__max_leaf_nodes": [7, 15, 31],
            "classifier__l2_regularization": [0.0, 0.5, 1.0],
        }
    raise ValueError(f"No tuning search space for model: {name}")


def _positive_probability(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """Return probability for class 1, independent of class-column ordering."""
    class_index = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(X)[:, class_index]


def _comparison_row(
    name: str, metrics: dict[str, object], threshold: float
) -> dict[str, object]:
    """Flatten one model's metrics for tables and CSV reports."""
    return {
        "model_name": name,
        "threshold": threshold,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "brier_score": metrics["brier_score"],
        "tn": metrics["confusion_matrix"][0],
        "fp": metrics["confusion_matrix"][1],
        "fn": metrics["confusion_matrix"][2],
        "tp": metrics["confusion_matrix"][3],
    }


def _write_error_analysis(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    """Save the most confident test errors and print three descriptive facts."""
    predictions = (probabilities >= threshold).astype(int)
    error_rows = test.loc[:, ["student_id", "term", *FEATURES, "next_score", "at_risk"]].copy()
    error_rows["predicted_probability"] = probabilities
    error_rows["predicted_at_risk"] = predictions

    false_positives = error_rows.loc[
        (error_rows["at_risk"] == 0) & (error_rows["predicted_at_risk"] == 1)
    ].sort_values("predicted_probability", ascending=False)
    false_negatives = error_rows.loc[
        (error_rows["at_risk"] == 1) & (error_rows["predicted_at_risk"] == 0)
    ].sort_values("predicted_probability", ascending=False)
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    false_positives.head(10).to_csv(reports / "false_positives.csv", index=False)
    false_negatives.head(10).to_csv(reports / "false_negatives.csv", index=False)

    print("\nError analysis (top false positives are highest-confidence; "
          "top false negatives are closest to the threshold):")
    print(
        "1. False positives:",
        f"{len(false_positives)} total;",
        f"median predicted probability={false_positives['predicted_probability'].median():.3f}"
        if not false_positives.empty
        else "no false positives.",
    )
    print(
        "2. False negatives:",
        f"{len(false_negatives)} total;",
        f"median predicted probability={false_negatives['predicted_probability'].median():.3f}; "
        f"median prev_score_3={false_negatives['prev_score_3'].median():.1f}"
        if not false_negatives.empty
        else "no false negatives.",
    )
    error_mask = (error_rows["at_risk"] != error_rows["predicted_at_risk"])
    error_count = int(error_mask.sum())
    missing_error_rate = (
        float(error_rows.loc[error_mask, "prev_score_3"].isna().mean())
        if error_count
        else 0.0
    )
    missing_test_rate = float(test["prev_score_3"].isna().mean())
    print(
        "3. Newest prior score is missing in "
        f"{missing_error_rate:.1%} of errors (versus {missing_test_rate:.1%} "
        "of all test students); inspect missing-history cases when reviewing "
        "individual recommendations."
    )


def _band_cutoffs(threshold: float) -> tuple[dict[str, float], str]:
    """Build ordered low/medium/high probability cutoffs."""
    if threshold >= 0.30:
        return (
            {"low_max": 0.30, "medium_max": threshold},
            "Low < 0.30; Medium 0.30 to threshold; High >= threshold.",
        )
    # Keep three ordered, non-overlapping bands when the selected high cutoff
    # falls below the usual 0.30 low-risk boundary.
    low_max = threshold / 2
    return (
        {"low_max": low_max, "medium_max": threshold},
        "The selected threshold is below 0.30, so low_max is set to half "
        "the threshold; this preserves ordered Low, Medium, and High bands.",
    )


def train() -> None:
    """Run the chronological training workflow and write all requested reports."""
    np.random.seed(SEED)
    data_path = ROOT / "data" / "raw" / "students.csv"
    data = load_data(data_path)
    required = {*FEATURES, "student_id", "term", "next_score", "at_risk"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    assert_no_forbidden(FEATURES)
    unexpected_features = set(FEATURES).intersection(FORBIDDEN)
    if unexpected_features:
        raise AssertionError(f"Forbidden columns selected as features: {unexpected_features}")

    train_data, validation_data, test_data = chronological_split(data)
    X_train, y_train = train_data[FEATURES], train_data["at_risk"].astype(int)
    X_val, y_val = validation_data[FEATURES], validation_data["at_risk"].astype(int)
    X_test, y_test = test_data[FEATURES], test_data["at_risk"].astype(int)

    models, supports_hgb_class_weight = _build_models()
    validation_rows: list[dict[str, object]] = []
    for name, model in models.items():
        fitted = _fit_model(
            model, name, X_train, y_train, supports_hgb_class_weight
        )
        models[name] = fitted
        probabilities = _positive_probability(fitted, X_val)
        validation_metrics = evaluate(y_val, probabilities, threshold=0.5)
        validation_rows.append(_comparison_row(name, validation_metrics, 0.5))

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    validation_comparison = pd.DataFrame(validation_rows)
    validation_comparison.to_csv(reports / "model_comparison_val.csv", index=False)
    print("Validation metrics at threshold 0.5 (fitted on train terms only):")
    print(validation_comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if float(validation_comparison["roc_auc"].max()) > 0.98:
        print(
            "\n"
            "!" * 72
            + "\nWARNING: validation ROC-AUC exceeds 0.98. Investigate possible "
            "target leakage or an overly easy split.\n"
            + "!" * 72
        )

    non_baseline_names = [
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]
    initial_ap = {
        row["model_name"]: float(row["pr_auc"])
        for row in validation_rows
        if row["model_name"] in non_baseline_names
    }
    tune_names = sorted(non_baseline_names, key=initial_ap.get, reverse=True)[:2]
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tuned_models: dict[str, BaseEstimator] = {}
    tuned_rows: list[dict[str, object]] = []
    fit_params: dict[str, object] = {}

    for name in tune_names:
        if name == "hist_gradient_boosting" and not supports_hgb_class_weight:
            fit_params = {
                "classifier__sample_weight": compute_sample_weight(
                    class_weight="balanced", y=y_train
                )
            }
        else:
            fit_params = {}
        search = RandomizedSearchCV(
            estimator=_make_pipeline(clone(models[name].named_steps["classifier"])),
            param_distributions=_get_search_space(name),
            n_iter=10,
            scoring="average_precision",
            cv=cross_validation,
            random_state=SEED,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train, **fit_params)
        tuned_models[name] = search.best_estimator_
        probabilities = _positive_probability(search.best_estimator_, X_val)
        metrics = evaluate(y_val, probabilities, threshold=0.5)
        tuned_rows.append(_comparison_row(name, metrics, 0.5))
        print(f"\nTuned {name}: best CV average precision={search.best_score_:.3f}")
        print(f"Best parameters: {search.best_params_}")

    tuned_comparison = pd.DataFrame(tuned_rows)
    print("\nTuned candidate validation metrics at threshold 0.5:")
    print(tuned_comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    final_name = str(tuned_comparison.sort_values("pr_auc", ascending=False).iloc[0]["model_name"])

    # Refit the selected full pipeline on training terms only, not validation.
    final_model = clone(tuned_models[final_name])
    final_fit_params: dict[str, object] = {}
    if final_name == "hist_gradient_boosting" and not supports_hgb_class_weight:
        final_fit_params["classifier__sample_weight"] = compute_sample_weight(
            class_weight="balanced", y=y_train
        )
    final_model.fit(X_train, y_train, **final_fit_params)

    # Fit the probability mapping through five-fold CV on training rows only.
    # ensemble=False then refits the base pipeline on all training rows.
    calibrated_model = CalibratedClassifierCV(
        estimator=clone(final_model),
        method="sigmoid",
        cv=5,
        ensemble=False,
    )
    calibrated_model.fit(X_train, y_train)

    uncalibrated_val_probabilities = _positive_probability(final_model, X_val)
    calibrated_val_probabilities = _positive_probability(calibrated_model, X_val)
    calibration_comparison = pd.DataFrame(
        [
            {
                "model_name": f"{final_name}_uncalibrated",
                "brier_score": evaluate(
                    y_val, uncalibrated_val_probabilities, threshold=0.5
                )["brier_score"],
                "roc_auc": evaluate(
                    y_val, uncalibrated_val_probabilities, threshold=0.5
                )["roc_auc"],
            },
            {
                "model_name": f"{final_name}_sigmoid_calibrated",
                "brier_score": evaluate(
                    y_val, calibrated_val_probabilities, threshold=0.5
                )["brier_score"],
                "roc_auc": evaluate(
                    y_val, calibrated_val_probabilities, threshold=0.5
                )["roc_auc"],
            },
        ]
    )
    calibration_comparison.to_csv(
        reports / "calibration_comparison_val.csv", index=False
    )
    plot_calibration_curve(
        y_val,
        uncalibrated_val_probabilities,
        calibrated_val_probabilities,
        final_name,
    )
    print("\nValidation probability calibration comparison (lower Brier is better):")
    print(calibration_comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Calibrated probabilities change the operating point, so select a fresh
    # threshold from validation predictions after calibration.
    final_threshold = select_threshold(
        y_val, calibrated_val_probabilities, min_recall=0.80
    )
    calibrated_validation_metrics = evaluate(
        y_val, calibrated_val_probabilities, final_threshold
    )

    # Evaluate each selected test candidate exactly once, after all choices are fixed.
    test_probabilities: dict[str, np.ndarray] = {
        final_name: _positive_probability(calibrated_model, X_test)
    }
    thresholds = {final_name: final_threshold}
    for baseline_name in ("dummy_stratified", "rule_prev_score_3_lt_50"):
        test_probabilities[baseline_name] = _positive_probability(
            models[baseline_name], X_test
        )
        thresholds[baseline_name] = 0.5

    test_rows: list[dict[str, object]] = []
    test_metrics: dict[str, dict[str, object]] = {}
    for name, probabilities in test_probabilities.items():
        metrics = evaluate(y_test, probabilities, thresholds[name])
        test_metrics[name] = metrics
        test_rows.append(_comparison_row(name, metrics, thresholds[name]))
    test_comparison = pd.DataFrame(test_rows)
    test_comparison.to_csv(reports / "model_comparison_test.csv", index=False)
    with (reports / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

    plot_confusion_matrix(
        y_test, test_probabilities[final_name], final_threshold, final_name
    )
    plot_pr_curve(y_test, test_probabilities[final_name], final_name)
    plot_roc_curve(y_test, test_probabilities[final_name], final_name)
    plot_model_comparison(test_comparison, "model_comparison_test.png")
    _write_error_analysis(test_data, test_probabilities[final_name], final_threshold)

    cleaned_train = Cleaner().fit_transform(X_train)
    medians = cleaned_train.median().to_dict()
    low_max, band_note = _band_cutoffs(final_threshold)
    valid_ranges = {
        "attendance_pct": {"min": 0, "max": 100},
        "attendance_trend_3w": {"min": -100, "max": 100},
        "assignment_completion_rate": {"min": 0, "max": 1},
        "late_submission_rate": {"min": 0, "max": 1},
        "prev_score_1": {"min": 0, "max": 100},
        "prev_score_2": {"min": 0, "max": 100},
        "prev_score_3": {"min": 0, "max": 100},
        "days_since_last_submission": {"min": 0, "max": None},
    }
    metadata = {
        "model_name": final_name,
        "calibration": {
            "method": "sigmoid",
            "cv": 5,
            "fitted_on": "training terms 1 and 2 only",
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "features": FEATURES,
        "threshold": final_threshold,
        "band_cutoffs": {
            **low_max,
        },
        "band_cutoff_note": band_note,
        "training_medians": {key: float(value) for key, value in medians.items()},
        "valid_ranges": valid_ranges,
        "test_metrics": test_metrics[final_name],
        "split_description": {
            "train": "terms 1 and 2",
            "validation": "term 3",
            "test": "term 4",
            "final_refit": "training terms 1 and 2 only; validation not merged",
        },
        "sklearn_version": sklearn.__version__,
    }
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, model_dir / "model.joblib")
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nFinal held-out test summary:")
    print(test_comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(
        f"\nSelected model: {final_name}; validation threshold={final_threshold:.3f}; "
        f"validation recall={calibrated_validation_metrics['recall']:.3f}"
    )
    print("Saved models/model.joblib and models/metadata.json")
    print("Saved reports/test_metrics.json, comparison CSVs, error analyses, and figures.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train()
