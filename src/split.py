"""Load data and make chronological, term-level train/validation/test splits."""

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV and remove exact repeated rows before splitting."""
    data = pd.read_csv(path)
    return data.drop_duplicates().reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split terms 1-2 for training, 3 for validation, and 4 for testing."""
    required = {"student_id", "term"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required split columns: {sorted(missing)}")

    train = df.loc[df["term"].isin([1, 2])].copy()
    validation = df.loc[df["term"] == 3].copy()
    test = df.loc[df["term"] == 4].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Terms 1, 2, 3, and 4 are required for chronological splitting.")

    splits = {"train": train, "validation": validation, "test": test}
    for left_name, right_name in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        left, right = splits[left_name], splits[right_name]
        assert set(left["student_id"]).isdisjoint(right["student_id"]), (
            f"Student IDs overlap between {left_name} and {right_name}."
        )
        assert set(left["term"]).isdisjoint(right["term"]), (
            f"Terms overlap between {left_name} and {right_name}."
        )

    assert train["term"].max() < validation["term"].min() < test["term"].min()
    return train, validation, test
