import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.preprocess import assert_no_forbidden
from src.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_and_metadata_are_valid() -> None:
    pipeline = joblib.load(ROOT / "models" / "model.joblib")
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))

    assert hasattr(pipeline, "predict_proba")
    assert isinstance(metadata["threshold"], float)
    assert metadata["features"]
    assert_no_forbidden(metadata["features"])
    assert isinstance(pipeline, CalibratedClassifierCV)
    assert metadata["calibration"]["method"] == "sigmoid"
    assert metadata["calibration"]["cv"] == 5
    assert set(metadata["training_means"]) == set(metadata["features"])
    assert set(metadata["training_stds"]) == set(metadata["features"])
    assert all(value > 0 for value in metadata["training_stds"].values())
    assert "brier_score" in metadata["test_metrics"]
    assert "brier_score" in evaluate([0, 1], [0.2, 0.8], 0.5)


def test_validation_calibration_comparison_is_saved() -> None:
    comparison = pd.read_csv(ROOT / "reports" / "calibration_comparison_val.csv")
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    model_name = metadata["model_name"]

    assert set(comparison["model_name"]) == {
        f"{model_name}_uncalibrated",
        f"{model_name}_sigmoid_calibrated",
    }
    assert comparison["brier_score"].notna().all()
    uncalibrated = comparison.loc[
        comparison["model_name"].str.endswith("_uncalibrated"), "brier_score"
    ].item()
    calibrated = comparison.loc[
        comparison["model_name"].str.endswith("_sigmoid_calibrated"), "brier_score"
    ].item()
    assert calibrated < uncalibrated
    assert (ROOT / "reports" / "figures" / "calibration_curve.png").is_file()


def test_test_comparison_includes_selected_baselines_and_reporting_candidates() -> None:
    comparison = pd.read_csv(ROOT / "reports" / "model_comparison_test.csv")
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))

    assert set(comparison["role"]) == {
        "selected",
        "baseline",
        "reported_after_selection",
    }
    assert len(comparison) == 5
    assert comparison.loc[comparison["role"] == "selected", "model_name"].tolist() == [
        metadata["model_name"]
    ]
    assert set(comparison.loc[comparison["role"] == "baseline", "model_name"]) == {
        "dummy_stratified",
        "rule_prev_score_3_lt_50",
    }
    assert set(
        comparison.loc[
            comparison["role"] == "reported_after_selection", "model_name"
        ]
    ) == {"random_forest", "hist_gradient_boosting"}
    assert comparison.loc[comparison["role"] == "selected", "threshold"].item() == (
        metadata["threshold"]
    )
    assert (ROOT / "reports" / "figures" / "model_comparison_test.png").is_file()
