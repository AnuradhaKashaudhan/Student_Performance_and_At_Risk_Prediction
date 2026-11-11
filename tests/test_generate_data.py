import pandas as pd

from src.generate_data import OUTPUT_COLUMNS, generate, generate_datasets

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


def test_audit_groups_are_generated_separately() -> None:
    students, audit_groups = generate_datasets(seed=42)
    repeated_students, repeated_audit_groups = generate_datasets(seed=42)

    assert audit_groups.columns.tolist() == [
        "student_id",
        "gender",
        "school_support_level",
    ]
    pd.testing.assert_frame_equal(students, repeated_students)
    pd.testing.assert_frame_equal(audit_groups, repeated_audit_groups)
    assert audit_groups["student_id"].tolist() == students["student_id"].tolist()
    assert set(audit_groups["gender"]) <= {"M", "F", "Other"}
    assert set(audit_groups["school_support_level"]) <= {"low", "medium", "high"}
    assert not set(audit_groups.columns[1:]).intersection(students.columns)
