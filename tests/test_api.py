import json
from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from api.main import app
from src.preprocess import FEATURES

ROOT = Path(__file__).resolve().parents[1]
HIGH_RISK = {
    "attendance_pct": 55,
    "attendance_trend_3w": -8,
    "assignment_completion_rate": 0.3,
    "late_submission_rate": 0.6,
    "prev_score_1": 52,
    "prev_score_2": 47,
    "prev_score_3": 41,
    "days_since_last_submission": 12,
}
LOW_RISK = {
    "attendance_pct": 98,
    "attendance_trend_3w": 5,
    "assignment_completion_rate": 0.95,
    "late_submission_rate": 0.02,
    "prev_score_1": 91,
    "prev_score_2": 93,
    "prev_score_3": 96,
    "days_since_last_submission": 0,
}


def test_health_and_prediction_schema() -> None:
    pipeline = joblib.load(ROOT / "models" / "model.joblib")
    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=HIGH_RISK)

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert prediction.status_code == 200
    result = prediction.json()
    assert 0 <= result["risk_probability"] <= 1
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert 0 <= len(result["top_signals"]) <= 3
    assert all(signal["impact"] >= 0.01 for signal in result["top_signals"])
    assert result["signal_note"] is None or isinstance(result["signal_note"], str)
    assert result["human_review_note"]
    assert result["uncertainty"] in {"normal", "high"}
    assert result["confidence_note"]
    signal_schema = app.openapi()["components"]["schemas"]["PredictionResponse"][
        "properties"
    ]["top_signals"]
    assert signal_schema["minItems"] == 0
    assert signal_schema["maxItems"] == 3
    # Local explanations need counterfactual predictions, so independently
    # reproduce only the primary model probability from the persisted pipeline.
    import pandas as pd

    raw = pd.DataFrame([{name: HIGH_RISK[name] for name in FEATURES}])
    model_probability = float(pipeline.predict_proba(raw)[0, list(pipeline.classes_).index(1)])
    assert result["risk_probability"] == round(model_probability, 3)


def test_missing_feature_is_reported_as_warning() -> None:
    record = dict(HIGH_RISK, prev_score_3=None)
    with TestClient(app) as client:
        response = client.post("/predict", json=record)

    assert response.status_code == 200
    assert any("prev_score_3 missing" in warning for warning in response.json()["warnings"])
    assert response.json()["uncertainty"] in {"normal", "high"}
    assert response.json()["confidence_note"]


def test_drifted_feature_is_reported_as_api_warning() -> None:
    metadata = json.loads(
        (ROOT / "models" / "metadata.json").read_text(encoding="utf-8")
    )
    feature = "attendance_trend_3w"
    record = dict(HIGH_RISK)
    record[feature] = metadata["training_means"][feature] + (
        3.1 * metadata["training_stds"][feature]
    )

    with TestClient(app) as client:
        response = client.post("/predict", json=record)

    assert response.status_code == 200
    assert any(
        feature in warning and "possible input drift" in warning
        for warning in response.json()["warnings"]
    )


def test_different_students_get_model_derived_probabilities() -> None:
    import pandas as pd

    pipeline = joblib.load(ROOT / "models" / "model.joblib")

    def model_probability(record: dict[str, int | float]) -> float:
        row = pd.DataFrame([{feature: record[feature] for feature in FEATURES}])
        return float(pipeline.predict_proba(row)[0, list(pipeline.classes_).index(1)])

    with TestClient(app) as client:
        high = client.post("/predict", json=HIGH_RISK).json()
        low = client.post("/predict", json=LOW_RISK).json()

    assert high["risk_probability"] != low["risk_probability"]
    assert high["risk_probability"] == round(model_probability(HIGH_RISK), 3)
    assert low["risk_probability"] == round(model_probability(LOW_RISK), 3)


def test_batch_prediction_returns_both_results() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict-batch",
            json={"records": [HIGH_RISK, LOW_RISK]},
        )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
    assert response.json()["predictions"][0]["risk_probability"] > response.json()["predictions"][1]["risk_probability"]




def test_batch_prediction_accepts_20_rows_and_uses_model_probabilities() -> None:
    import pandas as pd

    pipeline = joblib.load(ROOT / "models" / "model.joblib")
    records = [HIGH_RISK if index % 2 == 0 else LOW_RISK for index in range(20)]
    raw_features = pd.DataFrame(
        [{feature: record[feature] for feature in FEATURES} for record in records]
    )
    expected_probabilities = pipeline.predict_proba(raw_features)[
        :, list(pipeline.classes_).index(1)
    ]

    with TestClient(app) as client:
        response = client.post("/predict-batch", json={"records": records})

    assert response.status_code == 200
    results = response.json()["predictions"]
    assert len(results) == 20
    assert [row["risk_probability"] for row in results] == [
        round(float(probability), 3) for probability in expected_probabilities
    ]
