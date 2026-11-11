"""Test-set ranking capacity and threshold-cost analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.explain import load_artifact
from src.preprocess import FEATURES
from src.split import chronological_split, load_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPACITIES = (25, 50, 100, 150)
DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))
ANALYSIS_COLUMNS = [
    "analysis",
    "k",
    "threshold",
    "precision_at_k",
    "recall_at_k",
    "lift_over_random",
    "tp",
    "fp",
    "fn",
    "tn",
    "total_cost",
    "cost_false_negative",
    "cost_false_positive",
    "is_chosen_threshold",
]


def _validate_test_inputs(
    test_data: pd.DataFrame, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate labels, IDs, and probabilities for held-out term data."""
    required = {"student_id", "term", "at_risk"}
    missing = required.difference(test_data.columns)
    if missing:
        raise ValueError(f"Test data is missing columns: {sorted(missing)}")
    if set(test_data["term"].dropna().unique()) != {4}:
        raise ValueError("Capacity analysis must use test term 4 only.")
    if test_data["student_id"].isna().any() or test_data["student_id"].duplicated().any():
        raise ValueError("Test student_id values must be present and unique.")

    labels = test_data["at_risk"].to_numpy(dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Test labels must be a one-dimensional binary array.")
    if probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("Provide one predicted probability for each test row.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Test probabilities must be finite values between 0 and 1.")
    if labels.sum() == 0:
        raise ValueError("The test set must include at least one at-risk student.")
    return labels, probabilities


def analyze_capacity(
    test_data: pd.DataFrame,
    probabilities: np.ndarray,
    chosen_threshold: float,
    *,
    capacities: tuple[int, ...] = DEFAULT_CAPACITIES,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    cost_false_negative: float = 5,
    cost_false_positive: float = 1,
) -> pd.DataFrame:
    """Calculate top-K capacity metrics and threshold costs on test predictions.

    Top-K ties are broken by student ID to make the ranking deterministic.
    Lift compares precision at K with the positive-class prevalence in the
    complete held-out test set. Threshold costs are an explicit weighted error
    total, not a financial estimate.
    """
    labels, probabilities = _validate_test_inputs(test_data, probabilities)
    if not capacities or any(not isinstance(k, (int, np.integer)) or k <= 0 for k in capacities):
        raise ValueError("Capacities must be a non-empty sequence of positive integers.")
    if len(set(capacities)) != len(capacities):
        raise ValueError("Capacities must not contain duplicates.")
    if not 0 <= chosen_threshold <= 1:
        raise ValueError("The chosen threshold must be between 0 and 1.")
    if not np.isfinite([cost_false_negative, cost_false_positive]).all() or (
        cost_false_negative < 0 or cost_false_positive < 0
    ):
        raise ValueError("Error costs must be finite and non-negative.")
    threshold_values = np.asarray(thresholds, dtype=float)
    if threshold_values.ndim != 1 or not len(threshold_values):
        raise ValueError("Provide a non-empty one-dimensional threshold sequence.")
    if not np.isfinite(threshold_values).all() or (
        (threshold_values < 0) | (threshold_values > 1)
    ).any():
        raise ValueError("Thresholds must be finite values between 0 and 1.")
    if len(np.unique(threshold_values)) != len(threshold_values):
        raise ValueError("Threshold values must be unique.")

    # Ranking is performed over only the held-out test rows. Student ID is a
    # deterministic tie-breaker, not an input to the risk model.
    order = np.lexsort(
        (
            test_data["student_id"].astype(str).to_numpy(),
            -probabilities,
        )
    )
    prevalence = float(labels.mean())
    rows: list[dict[str, object]] = []
    for capacity in capacities:
        selected_count = min(int(capacity), len(test_data))
        hits = int(labels[order[:selected_count]].sum())
        precision_at_k = hits / selected_count
        rows.append(
            {
                "analysis": "top_k",
                "k": int(capacity),
                "threshold": np.nan,
                "precision_at_k": precision_at_k,
                "recall_at_k": hits / int(labels.sum()),
                "lift_over_random": precision_at_k / prevalence,
                "tp": np.nan,
                "fp": np.nan,
                "fn": np.nan,
                "tn": np.nan,
                "total_cost": np.nan,
                "cost_false_negative": cost_false_negative,
                "cost_false_positive": cost_false_positive,
                "is_chosen_threshold": False,
            }
        )

    cost_thresholds = np.unique(
        np.append(threshold_values, float(chosen_threshold))
    )
    for threshold in np.sort(cost_thresholds):
        predictions = probabilities >= threshold
        tp = int(((labels == 1) & predictions).sum())
        fp = int(((labels == 0) & predictions).sum())
        fn = int(((labels == 1) & ~predictions).sum())
        tn = int(((labels == 0) & ~predictions).sum())
        total_cost = cost_false_negative * fn + cost_false_positive * fp
        rows.append(
            {
                "analysis": "threshold_cost",
                "k": np.nan,
                "threshold": float(threshold),
                "precision_at_k": np.nan,
                "recall_at_k": np.nan,
                "lift_over_random": np.nan,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "total_cost": float(total_cost),
                "cost_false_negative": cost_false_negative,
                "cost_false_positive": cost_false_positive,
                "is_chosen_threshold": bool(
                    np.isclose(threshold, chosen_threshold, rtol=0, atol=1e-12)
                ),
            }
        )
    return pd.DataFrame(rows, columns=ANALYSIS_COLUMNS)


def _plot_capacity(result: pd.DataFrame, output_path: Path) -> None:
    """Plot precision/recall at K and lift over random for capacity budgets."""
    top_k = result.loc[result["analysis"] == "top_k"].sort_values("k")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        top_k["k"], top_k["precision_at_k"], marker="o",
        label="Precision@K", color="#4C78A8",
    )
    axis.plot(
        top_k["k"], top_k["recall_at_k"], marker="o",
        label="Recall@K", color="#E45756",
    )
    axis.set(
        title="Held-out test capacity: quality of top-risk referrals",
        xlabel="Teacher check-in capacity (K students)",
        ylabel="Precision / recall",
        ylim=(0, 1),
    )
    lift_axis = axis.twinx()
    lift_axis.plot(
        top_k["k"], top_k["lift_over_random"], marker="s", linestyle="--",
        label="Lift over random", color="#54A24B",
    )
    lift_axis.set_ylabel("Lift over random")
    handles, labels = axis.get_legend_handles_labels()
    lift_handles, lift_labels = lift_axis.get_legend_handles_labels()
    axis.legend(handles + lift_handles, labels + lift_labels, loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_cost(
    result: pd.DataFrame, chosen_threshold: float, output_path: Path
) -> None:
    """Plot weighted test errors over the threshold grid and mark the choice."""
    costs = result.loc[result["analysis"] == "threshold_cost"].sort_values(
        "threshold"
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        costs["threshold"], costs["total_cost"], marker="o",
        label="Weighted test error cost", color="#4C78A8",
    )
    chosen = costs.loc[costs["is_chosen_threshold"]]
    if not chosen.empty:
        chosen_cost = float(chosen["total_cost"].iloc[0])
        axis.axvline(
            chosen_threshold, linestyle="--", color="#E45756",
            label=f"Chosen threshold = {chosen_threshold:.3f}",
        )
        axis.scatter(
            [chosen_threshold], [chosen_cost], color="#E45756", zorder=3,
            label=f"Chosen threshold cost = {chosen_cost:.0f}",
        )
    axis.set(
        title="Held-out test weighted error cost by threshold",
        xlabel="Risk probability threshold",
        ylabel="Cost units (5 × FN + 1 × FP)",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_capacity_analysis(
    test_data: pd.DataFrame,
    probabilities: np.ndarray,
    chosen_threshold: float,
    *,
    capacities: tuple[int, ...] = DEFAULT_CAPACITIES,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    cost_false_negative: float = 5,
    cost_false_positive: float = 1,
    output_csv: str | Path | None = None,
    figure_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compute, save, and print held-out capacity and threshold-cost analyses."""
    result = analyze_capacity(
        test_data,
        probabilities,
        chosen_threshold,
        capacities=capacities,
        thresholds=thresholds,
        cost_false_negative=cost_false_negative,
        cost_false_positive=cost_false_positive,
    )
    report_dir = Path(output_csv).parent if output_csv else ROOT / "reports"
    figures_dir = Path(figure_dir) if figure_dir else report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_csv) if output_csv else report_dir / "capacity_analysis.csv"
    capacity_figure = figures_dir / "capacity_metrics.png"
    cost_figure = figures_dir / "cost_vs_threshold.png"
    result.to_csv(csv_path, index=False, float_format="%.6f")
    _plot_capacity(result, capacity_figure)
    _plot_cost(result, chosen_threshold, cost_figure)

    print("Held-out test capacity metrics:")
    print(
        result.loc[result["analysis"] == "top_k", [
            "k", "precision_at_k", "recall_at_k", "lift_over_random"
        ]].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nHeld-out test threshold costs:")
    print(
        result.loc[result["analysis"] == "threshold_cost", [
            "threshold", "fp", "fn", "total_cost", "is_chosen_threshold"
        ]].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"Saved {csv_path}, {capacity_figure}, and {cost_figure}")
    return result


def main() -> None:
    """Load the artifact and run analysis on the held-out term-4 predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost-false-negative", type=float, default=5,
        help="Relative cost of a missed at-risk student (default: 5).",
    )
    parser.add_argument(
        "--cost-false-positive", type=float, default=1,
        help="Relative cost of an unnecessary check-in (default: 1).",
    )
    args = parser.parse_args()

    _, _, test_data = chronological_split(
        load_data(ROOT / "data" / "raw" / "students.csv")
    )
    pipeline, metadata = load_artifact(
        ROOT / "models" / "model.joblib", ROOT / "models" / "metadata.json"
    )
    features = metadata["features"]
    if features != FEATURES:
        raise ValueError("The artifact feature list does not match the project features.")
    probabilities = pipeline.predict_proba(test_data.loc[:, features])[
        :, list(pipeline.classes_).index(1)
    ]
    run_capacity_analysis(
        test_data,
        probabilities,
        float(metadata["threshold"]),
        cost_false_negative=args.cost_false_negative,
        cost_false_positive=args.cost_false_positive,
    )


if __name__ == "__main__":
    main()
