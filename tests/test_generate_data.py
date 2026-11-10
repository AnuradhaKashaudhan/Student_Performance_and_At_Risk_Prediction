import pandas as pd

from src.generate_data import OUTPUT_COLUMNS, generate

PROTECTED_COLUMNS = {
    "gender",
    "religion",
    "caste",
    "income",
    "disability",
    "location",
}


def test_same_seed_generates_identical_dataframes() -> None:
    pd.testing.assert_frame_equal(generate(seed=42), generate(seed=42))


def test_different_seed_generates_different_data() -> None:
    assert not generate(seed=42).equals(generate(seed=43))


def test_at_risk_rate_is_within_requested_range() -> None:
    data = generate(seed=42)
    assert 0.15 <= data["at_risk"].mean() <= 0.28


def test_expected_columns_and_no_protected_columns() -> None:
    data = generate(seed=42)

    assert data.columns.tolist() == OUTPUT_COLUMNS
    assert not PROTECTED_COLUMNS.intersection(data.columns)
