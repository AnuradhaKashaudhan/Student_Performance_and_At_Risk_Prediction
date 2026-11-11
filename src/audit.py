"""Evaluate test-set error rates across separately stored audit groups.

Audit-group labels are used only after model inference. They are never joined
to training, validation, or the feature frame passed to the saved model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.explain import load_artifact
from src.preprocess import FEATURES, assert_no_forbidden
from src.split import chronological_split, load_data

ROOT = Path(__file__).resolve().parents[1]
AUDIT_GROUP_COLUMNS = ("gender", "school_support_level")
METRICS = (
    "recall",
    "precision",
    "false_positive_rate",
    "false_negative_rate",
    "flag_rate",
)
AUDIT_COLUMNS = [
    "group_attribute",
    "group",
    "n",
    *METRICS,
    "largest_gap_metric",
    "largest_gap",
]


def _rate(numerator: int, denominator: int) -> float:
    """Return a rate or NaN when the group has no relevant cases."""
    return float(numerator / denominator) if denominator else float("nan")


def run_fairness_audit(
    test_data: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    audit_groups: pd.DataFrame,
    output_csv: str | Path | None = None,
    figure_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute group metrics by joining audit labels to held-out test results.

    The input ``test_data`` supplies only test IDs and outcomes to the audit.
    Audit labels are joined after predictions have already been produced; they
    are not model features and are not used in fitting or threshold selection.
    The largest gap is the maximum absolute difference in any reported rate
    across groups of the same audit attribute.
    """
    assert_no_forbidden(FEATURES)
    if set(AUDIT_GROUP_COLUMNS).intersection(FEATURES):
        raise AssertionError("Audit-group columns must never be model features.")

    required_test_columns = {"student_id", "term", "at_risk"}
    missing_test = required_test_columns.difference(test_data.columns)
    if missing_test:
        raise ValueError(f"Test data is missing columns: {sorted(missing_test)}")
    if set(test_data["term"].dropna().unique()) != {4}:
        raise ValueError("Fairness metrics must be computed on test term 4 only.")
    required_audit_columns = {"student_id", *AUDIT_GROUP_COLUMNS}
    missing_audit = required_audit_columns.difference(audit_groups.columns)
    if missing_audit:
        raise ValueError(f"Audit groups are missing columns: {sorted(missing_audit)}")
    if test_data["student_id"].isna().any() or test_data["student_id"].duplicated().any():
        raise ValueError("Test student_id values must be present and unique.")
    if audit_groups["student_id"].isna().any() or audit_groups["student_id"].duplicated().any():
        raise ValueError("Audit student_id values must be present and unique.")

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(test_data):
        raise ValueError("Provide exactly one predicted probability per test row.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Test probabilities must be finite values between 0 and 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("The decision threshold must be between 0 and 1.")

    test_predictions = test_data.loc[:, ["student_id", "at_risk"]].copy()
    test_predictions["predicted_at_risk"] = (probabilities >= threshold).astype(int)
    # Filter audit metadata to test IDs before the join, preventing any training
    # or validation rows from entering this evaluation table.
    test_groups = audit_groups.loc[
        audit_groups["student_id"].isin(test_predictions["student_id"]),
        ["student_id", *AUDIT_GROUP_COLUMNS],
    ]
    joined = test_predictions.merge(
        test_groups, on="student_id", how="left", validate="one_to_one"
    )
    if joined[list(AUDIT_GROUP_COLUMNS)].isna().any().any():
        missing_ids = joined.loc[
            joined[list(AUDIT_GROUP_COLUMNS)].isna().any(axis=1), "student_id"
        ].tolist()
        raise ValueError(
            "Audit groups are missing for test student IDs: "
            f"{missing_ids[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for attribute in AUDIT_GROUP_COLUMNS:
        attribute_rows: list[dict[str, Any]] = []
        for group, subset in joined.groupby(attribute, sort=True, observed=True):
            actual = subset["at_risk"].to_numpy(dtype=int)
            predicted = subset["predicted_at_risk"].to_numpy(dtype=int)
            tp = int(((actual == 1) & (predicted == 1)).sum())
            fp = int(((actual == 0) & (predicted == 1)).sum())
            fn = int(((actual == 1) & (predicted == 0)).sum())
            tn = int(((actual == 0) & (predicted == 0)).sum())
            attribute_rows.append(
                {
                    "group_attribute": attribute,
                    "group": str(group),
                    "n": int(len(subset)),
                    "recall": _rate(tp, tp + fn),
                    "precision": _rate(tp, tp + fp),
                    "false_positive_rate": _rate(fp, fp + tn),
                    "false_negative_rate": _rate(fn, fn + tp),
                    "flag_rate": float(predicted.mean()),
                }
            )

        metric_gaps = {
            metric: float(values.max() - values.min())
            for metric in METRICS
            if (values := pd.Series(
                [row[metric] for row in attribute_rows], dtype=float
            )).notna().sum() >= 2
        }
        if not attribute_rows:
            continue
        if metric_gaps:
            largest_metric = max(metric_gaps, key=metric_gaps.get)
            largest_gap = metric_gaps[largest_metric]
        else:
            largest_metric, largest_gap = "unavailable", float("nan")
        for row in attribute_rows:
            row["largest_gap_metric"] = largest_metric
            row["largest_gap"] = largest_gap
            rows.append(row)

    result = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if result.empty:
        raise ValueError("No audit groups were available for the test set.")

    csv_path = Path(output_csv) if output_csv else ROOT / "reports" / "fairness_audit.csv"
    plot_path = (
        Path(figure_path)
        if figure_path
        else ROOT / "reports" / "figures" / "fairness_audit.png"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False, float_format="%.6f")
    _plot_audit(result, plot_path)

    print("Held-out test fairness audit (descriptive; synthetic groups):")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    for attribute, attribute_rows in result.groupby("group_attribute", sort=False):
        summary = attribute_rows.iloc[0]
        print(
            f"Largest {attribute} gap: {summary['largest_gap_metric']} = "
            f"{summary['largest_gap']:.3f}"
        )
    print(f"Saved {csv_path} and {plot_path}")
    return result


def _plot_audit(result: pd.DataFrame, output_path: Path) -> None:
    """Save grouped test-set rates in one panel per audit attribute."""
    attributes = result["group_attribute"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        1, len(attributes), figsize=(7 * len(attributes), 5), squeeze=False
    )
    colors = ["#2f6f9f", "#e69f00", "#009e73", "#cc79a7", "#666666"]
    for axis, attribute in zip(axes[0], attributes):
        subset = result.loc[result["group_attribute"] == attribute]
        positions = np.arange(len(subset), dtype=float)
        width = 0.8 / len(METRICS)
        for metric_index, metric in enumerate(METRICS):
            axis.bar(
                positions - 0.4 + width * (metric_index + 0.5),
                subset[metric],
                width,
                label=metric.replace("_", " "),
                color=colors[metric_index],
            )
        axis.set_xticks(positions, subset["group"].tolist())
        axis.set_ylim(0, 1)
        axis.set_ylabel("Rate")
        axis.set_title(f"Test-set rates by {attribute.replace('_', ' ')}")
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        "Synthetic-group audit on held-out test data\n"
        "Descriptive diagnostic only; not evidence of real-world fairness"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load the artifact and audit only predictions for chronological test term."""
    raw_dir = ROOT / "data" / "raw"
    data_path = raw_dir / "students.csv"
    groups_path = raw_dir / "audit_groups.csv"
    if not groups_path.is_file():
        raise FileNotFoundError(
            f"Missing {groups_path}. Run `python -m src.generate_data --seed 42`."
        )

    _, _, test_data = chronological_split(load_data(data_path))
    pipeline, metadata = load_artifact(
        ROOT / "models" / "model.joblib", ROOT / "models" / "metadata.json"
    )
    features = metadata["features"]
    if features != FEATURES:
        raise ValueError("The artifact feature list does not match the audit workflow.")
    # Keep the protected/audit fields out of this raw model input frame.
    test_probabilities = pipeline.predict_proba(test_data.loc[:, features])[
        :, list(pipeline.classes_).index(1)
    ]
    audit_groups = pd.read_csv(groups_path)
    run_fairness_audit(
        test_data,
        test_probabilities,
        float(metadata["threshold"]),
        audit_groups,
    )


if __name__ == "__main__":
    main()
