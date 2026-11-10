"""FastAPI service for advisory student-risk predictions."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.explain import load_artifact, predict_one
from src.preprocess import FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"


class StudentFeatures(BaseModel):
    """Raw week-8 features for one student; null values are median-imputed."""

    model_config = ConfigDict(extra="forbid")

    attendance_pct: Annotated[
        float | None,
        Field(
            default=None,
            description="Attendance percentage. Outliers are accepted, clipped to 0-100, and warned about.",
            json_schema_extra={"minimum": 0, "maximum": 100},
        ),
    ]
    attendance_trend_3w: Annotated[
        float | None,
        Field(
            default=None,
            description="Attendance change over the latest three weeks in percentage points.",
            json_schema_extra={"minimum": -100, "maximum": 100},
        ),
    ]
    assignment_completion_rate: Annotated[
        float | None,
        Field(
            default=None,
            description="Fraction of assignments completed. Outliers are accepted, clipped to 0-1, and warned about.",
            json_schema_extra={"minimum": 0, "maximum": 1},
        ),
    ]
    late_submission_rate: Annotated[
        float | None,
        Field(
            default=None,
            description="Fraction of assignments submitted late. Outliers are accepted, clipped to 0-1, and warned about.",
            json_schema_extra={"minimum": 0, "maximum": 1},
        ),
    ]
    prev_score_1: Annotated[
        float | None,
        Field(
            default=None,
            description="Oldest previous assessment score. Outliers are accepted, clipped to 0-100, and warned about.",
            json_schema_extra={"minimum": 0, "maximum": 100},
        ),
    ]
    prev_score_2: Annotated[
        float | None,
        Field(
            default=None,
            description="Middle previous assessment score. Outliers are accepted, clipped to 0-100, and warned about.",
            json_schema_extra={"minimum": 0, "maximum": 100},
        ),
    ]
    prev_score_3: Annotated[
        float | None,
        Field(
            default=None,
            description="Newest previous assessment score. Outliers are accepted, clipped to 0-100, and warned about.",
            json_schema_extra={"minimum": 0, "maximum": 100},
        ),
    ]
    days_since_last_submission: Annotated[
        float | None,
        Field(
            default=None,
            description="Days since the latest submission. Negative values are accepted, clipped to 0, and warned about.",
            json_schema_extra={"minimum": 0},
        ),
    ]

    @field_validator(*FEATURES, mode="before")
    @classmethod
    def require_finite_numbers(cls, value: Any) -> Any:
        """Allow missing values but reject NaN/infinity from non-JSON clients."""
        if value is None:
            return None
        try:
            if not np.isfinite(float(value)):
                raise ValueError("Feature values must be finite numbers or null.")
        except (TypeError, ValueError) as exc:
            raise ValueError("Feature values must be finite numbers or null.") from exc
        return value


class BatchPredictionRequest(BaseModel):
    """A bounded list of raw feature records for batch inference."""

    records: Annotated[
        list[StudentFeatures],
        Field(min_length=1, max_length=100, description="Between 1 and 100 records."),
    ]


class RiskSignal(BaseModel):
    """One local feature perturbation that influenced a prediction."""

    feature: str
    value: float | None
    effect: Literal["increases risk", "reduces risk"]
    impact: float
    message: str


class PredictionResponse(BaseModel):
    """Schema for a calibrated risk estimate and its uncertainty notice."""

    risk_probability: Annotated[float, Field(ge=0, le=1)]
    risk_band: Literal["Low", "Medium", "High"]
    flagged: bool
    uncertainty: Literal["normal", "high"]
    confidence_note: str
    top_signals: Annotated[list[RiskSignal], Field(max_length=3)]
    warnings: list[str]
    human_review_note: str
    model_name: str
    threshold: Annotated[float, Field(ge=0, le=1)]


class BatchPredictionResponse(BaseModel):
    """Schema for a batch of artifact-backed predictions."""

    predictions: list[PredictionResponse]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and metadata once when the service starts."""
    app.state.pipeline = None
    app.state.metadata = None
    app.state.artifact_error = None
    try:
        app.state.pipeline, app.state.metadata = load_artifact(
            MODEL_PATH, METADATA_PATH
        )
    except FileNotFoundError as exc:
        app.state.artifact_error = str(exc)
    yield


app = FastAPI(
    title="Student Risk Advisory API",
    description=(
        "Provides advisory estimates of whether a week-10 assessment score may "
        "fall below 50, using only week-8 student features. Human review is required."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_artifact(request: Request) -> tuple[Any, dict[str, Any]]:
    """Return startup-loaded artifacts or a helpful service-unavailable error."""
    pipeline = getattr(request.app.state, "pipeline", None)
    metadata = getattr(request.app.state, "metadata", None)
    if pipeline is None or metadata is None:
        detail = getattr(request.app.state, "artifact_error", None) or (
            "The trained model is unavailable. Run `python -m src.train` from "
            "the student-risk project directory, then restart the API."
        )
        raise HTTPException(status_code=503, detail=detail)
    return pipeline, metadata


@app.get("/health", summary="Check API and model availability")
def health(request: Request) -> dict[str, Any]:
    """Report whether the API process started and the model was loaded."""
    loaded = getattr(request.app.state, "pipeline", None) is not None
    return {
        "status": "ok" if loaded else "model_unavailable",
        "model_loaded": loaded,
    }


@app.get("/model-info", summary="Get model details and responsible-use limitations")
def model_info(request: Request) -> dict[str, Any]:
    """Return the loaded model's metadata and documented limitations."""
    _, metadata = _require_artifact(request)
    return {
        "model_name": metadata["model_name"],
        "trained_at": metadata.get("trained_at"),
        "features": metadata["features"],
        "threshold": metadata["threshold"],
        "band_cutoffs": metadata["band_cutoffs"],
        "test_metrics": metadata["test_metrics"],
        "limitations": [
            "The training and evaluation data are synthetic and may not represent real students.",
            "Attendance and other educational indicators may act as proxies for circumstances the data does not measure.",
            "This estimate is advisory and must not automate grading, punishment, or admission decisions.",
            "A teacher or counsellor must review context before any supportive action.",
            "The model predicts only the defined week-10 assessment threshold and is not a diagnosis.",
        ],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Estimate risk for one student",
)
def predict(student: StudentFeatures, request: Request) -> dict[str, Any]:
    """Return an estimated risk, local feature signals, and review warnings."""
    pipeline, metadata = _require_artifact(request)
    try:
        return predict_one(pipeline, metadata, student.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    summary="Estimate risk for up to 100 students",
)
def predict_batch(
    batch: BatchPredictionRequest, request: Request
) -> dict[str, list[dict[str, Any]]]:
    """Return one independent advisory prediction for each supplied record."""
    pipeline, metadata = _require_artifact(request)
    try:
        predictions = [
            predict_one(pipeline, metadata, student.model_dump())
            for student in batch.records
        ]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"predictions": predictions}
