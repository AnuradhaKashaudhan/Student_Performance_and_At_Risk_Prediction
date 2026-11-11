from pathlib import Path

import pandas as pd
import pytest

from src.explain import load_artifact
from src.robustness import analyze_robustness, run_robustness
from src.split import chronological_split, load_data

ROOT = Path(__file__).resolve().parents[1]


def _test_sample():
    _, _, test_data = chronological_split(
        load_data(ROOT / "data" / "raw" / "students.csv")
    )
    pipeline, metadata = load_artifact(
        ROOT / "models" / "model.joblib", ROOT / "models" / "metadata.json"
    )
    return test_data.iloc[:120], pipeline, metadata


def test_robustness_has_five_reproducible_scenarios() -> None:
    test_data, pipeline, metadata = _test_sample()

    first = analyze_robustness(
        test_data,
        pipeline,
        metadata["threshold"],
        metadata["training_stds"],
    )
    repeated = analyze_robustness(
        test_data,
        pipeline,
        metadata["threshold"],
        metadata["training_stds"],
    )

    pd.testing.assert_frame_equal(first, repeated)
    assert first["scenario"].tolist() == [
        "baseline",
        "gaussian_noise_5pct",
        "gaussian_noise_10pct",
        "random_missingness_5pct",
        "random_missingness_15pct",
    ]
    assert first.loc[0, ["recall_change", "precision_change", "f1_change"]].eq(0).all()
    assert first.loc[3, "actual_missing_fraction"] > first.loc[0, "actual_missing_fraction"]
    assert first.loc[4, "actual_missing_fraction"] > first.loc[3, "actual_missing_fraction"]
    assert first[["recall", "precision", "f1"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()


def test_robustness_rejects_non_test_terms() -> None:
    test_data, pipeline, metadata = _test_sample()
    test_data.loc[test_data.index[0], "term"] = 3

    with pytest.raises(ValueError, match="test term 4 only"):
        analyze_robustness(
            test_data,
            pipeline,
            metadata["threshold"],
            metadata["training_stds"],
        )


def test_robustness_writes_csv_and_figure(tmp_path: Path) -> None:
    test_data, pipeline, metadata = _test_sample()

    report = run_robustness(
        test_data,
        pipeline,
        metadata["threshold"],
        metadata["training_stds"],
        output_csv=tmp_path / "robustness.csv",
        figure_path=tmp_path / "robustness.png",
    )

    assert len(report) == 5
    assert (tmp_path / "robustness.csv").is_file()
    assert (tmp_path / "robustness.png").is_file()
