"""Metrics, threshold selection, and plots for at-risk classifier evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

FIGURE_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def evaluate(y_true: object, y_proba: object, threshold: float) -> dict[str, object]:
    """Calculate threshold metrics and rank-based metrics for the positive class."""
    truth = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_proba, dtype=float)
    if truth.ndim != 1 or probabilities.ndim != 1 or len(truth) != len(probabilities):
        raise ValueError("y_true and y_proba must be one-dimensional arrays of equal length.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities must be finite values in [0, 1].")

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "pr_auc": float(average_precision_score(truth, probabilities)),
        "brier_score": float(brier_score_loss(truth, probabilities)),
        "confusion_matrix": (int(tn), int(fp), int(fn), int(tp)),
        "threshold": float(threshold),
    }


def select_threshold(
    y_val: object, proba_val: object, min_recall: float = 0.80
) -> float:
    """Choose the most precise validation threshold that meets the recall floor.

    Recall is prioritized because missing a struggling student can delay useful
    support, while an unnecessary check-in has a lower expected cost. Among
    thresholds meeting the requested recall, a higher precision reduces
    unnecessary check-ins; ties choose the higher threshold.
    """
    truth = np.asarray(y_val, dtype=int)
    probabilities = np.asarray(proba_val, dtype=float)
    if not 0 < min_recall <= 1:
        raise ValueError("min_recall must be in (0, 1].")
    if len(truth) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Validation labels and finite probabilities must have equal length.")
    if not np.isin(truth, [0, 1]).all() or not (truth == 1).any():
        raise ValueError("Validation data must contain at least one positive label.")

    precision, recall, thresholds = precision_recall_curve(truth, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= min_recall)
    if eligible.size == 0:
        raise ValueError(f"No validation threshold meets recall >= {min_recall:.2f}.")
    best_precision = precision[eligible].max()
    best_candidates = eligible[precision[eligible] == best_precision]
    best_index = best_candidates[np.argmax(thresholds[best_candidates])]
    return float(thresholds[best_index])


def _save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a finished figure in the shared report directory."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / filename
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_confusion_matrix(
    y_true: object, y_proba: object, threshold: float, model_name: str
) -> Path:
    """Save a labeled confusion-matrix heatmap for one model."""
    predictions = (np.asarray(y_proba) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        title=f"{model_name}: test confusion matrix (threshold={threshold:.3f})",
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Not at risk", "At risk"],
        yticklabels=["Not at risk", "At risk"],
    )
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Students")
    return _save_figure(fig, "confusion_matrix_test.png")


def plot_pr_curve(y_true: object, y_proba: object, model_name: str) -> Path:
    """Save the precision-recall curve and its average-precision score."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set(
        title=f"{model_name}: test precision-recall curve",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    return _save_figure(fig, "precision_recall_curve_test.png")


def plot_roc_curve(y_true: object, y_proba: object, model_name: str) -> Path:
    """Save the ROC curve and its area under the curve."""
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set(
        title=f"{model_name}: test ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    return _save_figure(fig, "roc_curve_test.png")


def plot_calibration_curve(
    y_true: object,
    uncalibrated_proba: object,
    calibrated_proba: object,
    model_name: str,
) -> Path:
    """Compare validation reliability curves before and after sigmoid calibration."""
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for probabilities, label, color in (
        (uncalibrated_proba, "Uncalibrated", "#E45756"),
        (calibrated_proba, "Sigmoid calibrated", "#4C78A8"),
    ):
        observed, predicted = calibration_curve(
            y_true, probabilities, n_bins=10, strategy="quantile"
        )
        brier = brier_score_loss(y_true, probabilities)
        ax.plot(
            predicted,
            observed,
            marker="o",
            label=f"{label} (Brier={brier:.3f})",
            color=color,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set(
        title=f"{model_name}: validation reliability curve",
        xlabel="Mean predicted probability",
        ylabel="Observed fraction of at-risk students",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    return _save_figure(fig, "calibration_curve.png")


def plot_model_comparison(comparison: pd.DataFrame, filename: str) -> Path:
    """Save a comparison chart for recall, precision, F1, ROC-AUC, and PR-AUC."""
    metrics = ["recall", "precision", "f1", "roc_auc", "pr_auc"]
    missing = set(["model_name", "role", *metrics]).difference(comparison.columns)
    if missing:
        raise ValueError(f"Comparison table is missing columns: {sorted(missing)}")
    chart = comparison.set_index("model_name")[metrics]
    chart.index = [
        f"{row.model_name}\n({row.role.replace('_', ' ')})"
        for row in comparison.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    chart.plot(kind="bar", ax=ax)
    ax.set(
        title="Held-out test comparison (selection fixed on validation)",
        xlabel="Model and reporting role",
        ylabel="Score",
        ylim=(0, 1),
    )
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, filename)
