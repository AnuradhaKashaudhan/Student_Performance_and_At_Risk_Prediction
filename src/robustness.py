"""Evaluate model sensitivity to feature noise and missingness on test data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

from src.explain import load_artifact
from src.preprocess import FEATURES
from src.split import chronological_split, load_data

ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
NOISE_FRACTIONS = (0.05, 0.10)
MISSINGNESS_RATES = (0.05, 0.15)
REPORT_COLUMNS = [
    "scenario",
    "noise_fraction_of_train_std",
    "random_missing_rate",
    "actual_missing_fraction",
    "recall",
    "recall_change",
    "precision",
    "precision_change",
    "f1",
    "f1_change",
]


def _validate_test_data(
    test_data: pd.DataFrame,
    training_stds: Mapping[str, float],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Validate the raw test frame, labels, and training-only noise scales."""
    required = {"student_id", "term", "at_risk", *FEATURES}
    missing_columns = required.difference(test_data.columns)
    if missing_columns:
        raise ValueError(f"Test data is missing columns: {sorted(missing_columns)}")
    if test_data["term"].isna().any() or set(test_data["term"].unique()) != {4}:
        raise ValueError("Robustness analysis must use test term 4 only.")
    if test_data["student_id"].isna().any() or test_data["student_id"].duplicated().any():
        raise ValueError("Test student_id values must be present and unique.")
    missing_scales = set(FEATURES).difference(training_stds)
    if missing_scales:
        raise ValueError(f"Training standard deviations are missing: {sorted(missing_scales)}")
    scales = np.asarray([training_stds[feature] for feature in FEATURES], dtype=float)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Training standard deviations must be finite and positive.")

    labels = test_data["at_risk"].to_numpy(dtype=int)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Test labels must be binary and match the test rows.")
    if not (labels == 0).any() or not (labels == 1).any():
        raise ValueError("Test data must include both classes.")
    return test_data.loc[:, FEATURES].copy(), labels


def _classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute recall, precision, and F1 using one fixed operating threshold."""
    predictions = probabilities >= threshold
    return {
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def analyze_robustness(
    test_data: pd.DataFrame,
    pipeline: Any,
    threshold: float,
    training_stds: Mapping[str, float],
    *,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Measure test metric changes under Gaussian noise and random missingness.

    Gaussian noise is added independently to each observed raw feature with a
    per-feature scale equal to 5% or 10% of its training-term standard
    deviation. For missingness experiments, the stated rate is applied to
    observed cells only; naturally missing test values remain missing. Each
    scenario is independently seeded and measured at the original validation-
    selected threshold. The test labels are used only for evaluation.
    """
    features, labels = _validate_test_data(test_data, training_stds)
    if not 0 <= threshold <= 1:
        raise ValueError("The fixed decision threshold must be between 0 and 1.")
    scales = np.asarray([training_stds[feature] for feature in FEATURES], dtype=float)

    baseline_probabilities = _positive_probability(pipeline, features)
    baseline_metrics = _classification_metrics(labels, baseline_probabilities, threshold)
    original = features.to_numpy(dtype=float)
    original_missing_fraction = float(np.isnan(original).mean())
    scenarios: list[dict[str, Any]] = [
        {
            "scenario": "baseline",
            "noise_fraction_of_train_std": 0.0,
            "random_missing_rate": 0.0,
            "X": features,
        }
    ]

    for noise_fraction in NOISE_FRACTIONS:
        rng = np.random.default_rng(seed + int(noise_fraction * 1000))
        observed = np.isfinite(original)
        noise = rng.normal(
            loc=0.0, scale=noise_fraction * scales, size=original.shape
        )
        perturbed = original.copy()
        perturbed[observed] += noise[observed]
        scenarios.append(
            {
                "scenario": f"gaussian_noise_{int(noise_fraction * 100)}pct",
                "noise_fraction_of_train_std": noise_fraction,
                "random_missing_rate": 0.0,
                "X": pd.DataFrame(perturbed, columns=FEATURES, index=features.index),
            }
        )

    for missing_rate in MISSINGNESS_RATES:
        rng = np.random.default_rng(seed + int(missing_rate * 1000) + 10_000)
        perturbed = original.copy()
        mask = (rng.random(size=original.shape) < missing_rate) & np.isfinite(original)
        perturbed[mask] = np.nan
        scenarios.append(
            {
                "scenario": f"random_missingness_{int(missing_rate * 100)}pct",
                "noise_fraction_of_train_std": 0.0,
                "random_missing_rate": missing_rate,
                "X": pd.DataFrame(perturbed, columns=FEATURES, index=features.index),
            }
        )

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_features = scenario["X"]
        probabilities = _positive_probability(pipeline, scenario_features)
        metrics = _classification_metrics(labels, probabilities, threshold)
        actual_missing_fraction = float(
            scenario_features.isna().to_numpy().mean()
        )
        row = {key: value for key, value in scenario.items() if key != "X"}
        row["actual_missing_fraction"] = actual_missing_fraction
        for metric, value in metrics.items():
            row[metric] = value
            row[f"{metric}_change"] = value - baseline_metrics[metric]
        rows.append(row)

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def _positive_probability(pipeline: Any, features: pd.DataFrame) -> np.ndarray:
    """Get calibrated positive-class probabilities from the fitted artifact."""
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not expose a positive risk class.")
    return np.asarray(pipeline.predict_proba(features)[:, classes.index(1)], dtype=float)


def _plot_robustness(report: pd.DataFrame, output_path: Path) -> None:
    """Plot recall, precision, and F1 by test perturbation scenario."""
    metrics = ["recall", "precision", "f1"]
    chart = report.set_index("scenario")[metrics]
    figure, axis = plt.subplots(figsize=(10, 5))
    chart.plot(kind="bar", ax=axis, color=["#E45756", "#4C78A8", "#54A24B"])
    axis.set(
        title="Held-out test robustness to noise and missing features",
        xlabel="Test scenario",
        ylabel="Score at the fixed decision threshold",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_robustness(
    test_data: pd.DataFrame,
    pipeline: Any,
    threshold: float,
    training_stds: Mapping[str, float],
    *,
    output_csv: str | Path | None = None,
    figure_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run all robustness scenarios and save the CSV and comparison figure."""
    report = analyze_robustness(
        test_data, pipeline, threshold, training_stds
    )
    csv_path = Path(output_csv) if output_csv else ROOT / "reports" / "robustness.csv"
    image_path = (
        Path(figure_path)
        if figure_path
        else ROOT / "reports" / "figures" / "robustness.png"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_path, index=False, float_format="%.6f")
    _plot_robustness(report, image_path)
    print("Robustness metrics on held-out test data, fixed threshold:")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved {csv_path} and {image_path}")
    return report


def main() -> None:
    """Load the artifact and run robustness analysis on chronological test data."""
    _, _, test_data = chronological_split(
        load_data(ROOT / "data" / "raw" / "students.csv")
    )
    pipeline, metadata = load_artifact(
        ROOT / "models" / "model.joblib", ROOT / "models" / "metadata.json"
    )
    features = metadata["features"]
    if features != FEATURES:
        raise ValueError("The artifact feature list does not match project features.")
    run_robustness(
        test_data,
        pipeline,
        float(metadata["threshold"]),
        metadata["training_stds"],
    )


if __name__ == "__main__":
    main()
