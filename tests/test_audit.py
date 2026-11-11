from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.audit import AUDIT_GROUP_COLUMNS, METRICS, run_fairness_audit
from src.explain import load_artifact
from src.generate_data import generate_datasets
from src.preprocess import FEATURES, assert_no_forbidden
from src.split import chronological_split

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_group_audit_runs_only_on_test_and_keeps_groups_out_of_features(
    tmp_path: Path,
) -> None:
    students, audit_groups = generate_datasets(seed=42)
    _, _, test_data = chronological_split(students)
    pipeline, metadata = load_artifact(
        ROOT / "models" / "model.joblib", ROOT / "models" / "metadata.json"
    )

    assert not set(AUDIT_GROUP_COLUMNS).intersection(students.columns)
    assert not set(AUDIT_GROUP_COLUMNS).intersection(FEATURES)
    assert_no_forbidden(FEATURES)
    probabilities = pipeline.predict_proba(test_data[FEATURES])[
        :, list(pipeline.classes_).index(1)
    ]

    report = run_fairness_audit(
        test_data,
        probabilities,
        metadata["threshold"],
        audit_groups,
        output_csv=tmp_path / "fairness_audit.csv",
        figure_path=tmp_path / "fairness_audit.png",
    )

    assert set(report["group_attribute"]) == set(AUDIT_GROUP_COLUMNS)
    assert set(report.columns).issuperset(
        {"group_attribute", "group", "n", *METRICS, "largest_gap"}
    )
    for _, groups in report.groupby("group_attribute"):
        assert groups["n"].sum() == len(test_data)
        assert np.isfinite(groups[list(METRICS)].to_numpy()).all()
        assert groups["largest_gap"].nunique() == 1
        metric_gaps = groups[list(METRICS)].max() - groups[list(METRICS)].min()
        assert groups["largest_gap"].iloc[0] == pytest.approx(metric_gaps.max())
        assert groups["largest_gap_metric"].iloc[0] == metric_gaps.idxmax()
    assert (tmp_path / "fairness_audit.csv").is_file()
    assert (tmp_path / "fairness_audit.png").is_file()


def test_fairness_audit_rejects_non_test_terms() -> None:
    test_data = pd.DataFrame(
        {"student_id": ["T3-0001"], "term": [3], "at_risk": [1]}
    )
    audit_groups = pd.DataFrame(
        {"student_id": ["T3-0001"], "gender": ["F"], "school_support_level": ["low"]}
    )

    with pytest.raises(ValueError, match="test term 4 only"):
        run_fairness_audit(test_data, np.array([0.8]), 0.5, audit_groups)
