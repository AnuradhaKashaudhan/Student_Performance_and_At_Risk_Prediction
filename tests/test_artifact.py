import json
from pathlib import Path

import joblib

from src.preprocess import assert_no_forbidden

ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_and_metadata_are_valid() -> None:
    pipeline = joblib.load(ROOT / "models" / "model.joblib")
    metadata = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))

    assert hasattr(pipeline, "predict_proba")
    assert isinstance(metadata["threshold"], float)
    assert metadata["features"]
    assert_no_forbidden(metadata["features"])
