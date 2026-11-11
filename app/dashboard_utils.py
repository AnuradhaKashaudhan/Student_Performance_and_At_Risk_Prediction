"""Small, testable helpers for the Streamlit dashboard."""

from io import BytesIO

import pandas as pd

RAW_FEATURES = [
    "attendance_pct",
    "attendance_trend_3w",
    "assignment_completion_rate",
    "late_submission_rate",
    "prev_score_1",
    "prev_score_2",
    "prev_score_3",
    "days_since_last_submission",
]
HUMAN_REVIEW_NOTE = (
    "Advisory only. This is a statistical estimate, not a judgement about the "
    "student. A teacher or counsellor must review before any action. Never use "
    "for grading, punishment or admission decisions."
)


def parse_class_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Validate an uploaded CSV and return only the eight raw model features.

    Extra columns are allowed for caller-side row identifiers, but are never
    sent to the API. Blank feature cells remain missing for pipeline imputation.
    """
    try:
        uploaded = pd.read_csv(BytesIO(csv_bytes))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Could not read the uploaded CSV: {exc}") from exc
    if uploaded.empty:
        raise ValueError("The uploaded CSV must contain at least one student row.")
    if len(uploaded) > 100:
        raise ValueError("The batch API accepts at most 100 student rows per upload.")
    missing_columns = set(RAW_FEATURES).difference(uploaded.columns)
    if missing_columns:
        raise ValueError(
            "The CSV is missing required feature columns: "
            f"{sorted(missing_columns)}"
        )

    features = uploaded.loc[:, RAW_FEATURES].copy()
    for column in RAW_FEATURES:
        original = features[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(
                f"Column '{column}' contains non-numeric values at CSV rows "
                f"{(invalid[invalid].index + 2).tolist()[:5]}."
            )
        features[column] = numeric
    return features


def template_csv() -> bytes:
    """Return a downloadable CSV with the exact accepted raw feature headers."""
    examples = pd.DataFrame(
        [
            [55, -8, 0.3, 0.6, 52, 47, 41, 12],
            [98, 5, 0.95, 0.02, 91, 93, 96, 0],
        ],
        columns=RAW_FEATURES,
    )
    return examples.to_csv(index=False).encode("utf-8")


def bounded_slider_value(
    value: float | None, default: float, minimum: float, maximum: float
) -> float:
    """Keep a what-if slider's initial value inside its display range."""
    candidate = default if value is None else float(value)
    return min(max(candidate, minimum), maximum)
