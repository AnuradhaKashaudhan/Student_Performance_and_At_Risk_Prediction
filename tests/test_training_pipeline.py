from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate, select_threshold
from src.preprocess import FEATURES, Cleaner, assert_no_forbidden, build_preprocessor
from src.split import chronological_split, load_data
from src.train import RuleBaseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_data_removes_exact_duplicate_rows(tmp_path: Path) -> None:
    source = pd.DataFrame(
        {"student_id": ["T1-0001", "T1-0001"], "term": [1, 1], "at_risk": [0, 0]}
    )
    path = tmp_path / "duplicate.csv"
    source.to_csv(path, index=False)

    loaded = load_data(path)

    assert len(loaded) == 1


def test_chronological_split_uses_whole_terms() -> None:
    data = load_data(PROJECT_ROOT / "data" / "raw" / "students.csv")

    train, validation, test = chronological_split(data)

    assert set(train["term"]) == {1, 2}
    assert set(validation["term"]) == {3}
    assert set(test["term"]) == {4}
    assert set(train["student_id"]).isdisjoint(validation["student_id"])
    assert set(validation["student_id"]).isdisjoint(test["student_id"])


def test_preprocessor_clips_and_imputes_raw_features() -> None:
    data = load_data(PROJECT_ROOT / "data" / "raw" / "students.csv")
    raw_features = data.loc[:20, FEATURES].copy()
    raw_features.loc[raw_features.index[0], "attendance_pct"] = 105
    raw_features.loc[raw_features.index[0], "prev_score_2"] = 105
    raw_features.loc[raw_features.index[0], "days_since_last_submission"] = -3

    cleaner = Cleaner().fit(raw_features)
    cleaned = cleaner.transform(raw_features)
    transformed = build_preprocessor().fit_transform(raw_features)

    assert cleaned.loc[raw_features.index[0], "attendance_pct"] == 100
    assert cleaned.loc[raw_features.index[0], "prev_score_2"] == 100
    assert cleaned.loc[raw_features.index[0], "days_since_last_submission"] == 0
    assert np.isfinite(transformed).all()


def test_forbidden_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="student_id"):
        assert_no_forbidden([*FEATURES, "student_id"])


def test_rule_baseline_and_highest_precision_threshold() -> None:
    raw = pd.DataFrame({"prev_score_3": [40, 50, np.nan]})
    baseline = RuleBaseline().fit(raw, [1, 0, 0])
    assert baseline.predict(raw).tolist() == [1, 0, 0]

    threshold = select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], min_recall=0.5)
    metrics = evaluate([0, 1, 1, 0], [0.1, 0.8, 0.4, 0.7], threshold=0.5)

    assert threshold == 0.9
    assert metrics["confusion_matrix"] == (1, 1, 1, 1)
