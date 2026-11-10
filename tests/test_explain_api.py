from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.explain import load_artifact, predict_one
from src.preprocess import FEATURES

ROOT = Path(__file__).resolve().parents[1]

HIGH_RISK_STUDENT = {
    "attendance_pct": 55,
    "attendance_trend_3w": -8,
    "assignment_completion_rate": 0.3,
    "late_submission_rate": 0.6,
    "prev_score_1": 52,
    "prev_score_2": 47,
    "prev_score_3": 41,
    "days_since_last_submission": 12,
}
LOW_RISK_STUDENT = {
    "attendance_pct": 98,
    "attendance_trend_3w": 5,
    "assignment_completion_rate": 0.95,
    "late_submission_rate": 0.02,
    "prev_score_1": 91,
    "prev_score_2": 93,
    "prev_score_3": 96,
    "days_since_last_submission": 0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_explanation_uses_saved_model_and_medians() -> None:
    pipeline, metadata = load_artifact(
        ROOT / "models" / "model.joblib",
        ROOT / "models" / "metadata.json",
    )

    result = predict_one(pipeline, metadata, HIGH_RISK_STUDENT)

    assert 0 <= result["risk_probability"] <= 1
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert result["flagged"] == (
        result["risk_probability"] >= metadata["threshold"]
    )
    assert len(result["top_signals"]) == 3
    assert all("message" in signal and signal["impact"] >= 0 for signal in result["top_signals"])


def test_explanation_warns_and_clips_outliers_and_missing_values() -> None:
    pipeline, metadata = load_artifact(
        ROOT / "models" / "model.joblib",
        ROOT / "models" / "metadata.json",
    )
    record = dict(HIGH_RISK_STUDENT, attendance_pct=130, prev_score_3=None)

    result = predict_one(pipeline, metadata, record)

    assert any("attendance_pct=130" in warning and "clipped" in warning for warning in result["warnings"])
    assert any("prev_score_3 missing" in warning for warning in result["warnings"])


def test_api_serves_model_and_predictions(client: TestClient) -> None:
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/model-info").status_code == 200

    high = client.post("/predict", json=HIGH_RISK_STUDENT)
    low = client.post("/predict", json=LOW_RISK_STUDENT)

    assert high.status_code == low.status_code == 200
    assert high.json()["risk_probability"] > low.json()["risk_probability"]
    assert "human_review_note" in high.json()


def test_api_accepts_outliers_and_missing_features(client: TestClient) -> None:
    record = dict(HIGH_RISK_STUDENT, attendance_pct=130, prev_score_3=None)

    response = client.post("/predict", json=record)

    assert response.status_code == 200
    assert any("clipped" in warning for warning in response.json()["warnings"])
    assert any("prev_score_3 missing" in warning for warning in response.json()["warnings"])


def test_api_batch_limit_and_unknown_features(client: TestClient) -> None:
    batch = client.post("/predict-batch", json={"records": [HIGH_RISK_STUDENT, LOW_RISK_STUDENT]})
    too_many = client.post("/predict-batch", json={"records": [HIGH_RISK_STUDENT] * 101})
    extra_feature = client.post("/predict", json={**HIGH_RISK_STUDENT, "term": 4})

    assert batch.status_code == 200
    assert len(batch.json()["predictions"]) == 2
    assert too_many.status_code == 422
    assert extra_feature.status_code == 422


def test_api_feature_schema_matches_raw_features() -> None:
    schema = app.openapi()
    properties = schema["components"]["schemas"]["StudentFeatures"]["properties"]

    assert set(properties) == set(FEATURES)
    assert all(
        properties[name]["anyOf"][0]["type"] == "number"
        and properties[name]["anyOf"][1]["type"] == "null"
        for name in FEATURES
    )
    assert properties["attendance_pct"]["maximum"] == 100
