from pathlib import Path

import pytest

from src.preprocess import FEATURES, assert_no_forbidden
from src.split import chronological_split, load_data

ROOT = Path(__file__).resolve().parents[1]


def test_chronological_split_has_no_student_or_term_leakage() -> None:
    data = load_data(ROOT / "data" / "raw" / "students.csv")
    train, validation, test = chronological_split(data)

    assert set(train["student_id"]).isdisjoint(validation["student_id"])
    assert set(train["student_id"]).isdisjoint(test["student_id"])
    assert set(validation["student_id"]).isdisjoint(test["student_id"])
    assert train["term"].max() < validation["term"].min() < test["term"].min()
    assert len(train) + len(validation) + len(test) == len(data)


@pytest.mark.parametrize("forbidden", ["next_score", "student_id", "term"])
def test_forbidden_features_raise(forbidden: str) -> None:
    with pytest.raises(ValueError, match=forbidden):
        assert_no_forbidden([*FEATURES, forbidden])
