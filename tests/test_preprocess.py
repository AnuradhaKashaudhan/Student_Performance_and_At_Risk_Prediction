import numpy as np
import pandas as pd

from src.preprocess import (
    ENGINEERED_FEATURES,
    FEATURES,
    Cleaner,
    FeatureEngineer,
    build_preprocessor,
)


def _raw_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "attendance_pct": 130,
                "attendance_trend_3w": -8,
                "assignment_completion_rate": 1.2,
                "late_submission_rate": -0.1,
                "prev_score_1": 52,
                "prev_score_2": 105,
                "prev_score_3": np.nan,
                "days_since_last_submission": -3,
            }
        ],
        columns=FEATURES,
    )


def test_cleaner_clips_out_of_range_values() -> None:
    raw = _raw_row()
    cleaned = Cleaner().fit_transform(raw)

    assert cleaned.loc[0, "attendance_pct"] == 100
    assert cleaned.loc[0, "assignment_completion_rate"] == 1
    assert cleaned.loc[0, "late_submission_rate"] == 0
    assert cleaned.loc[0, "prev_score_2"] == 100
    assert cleaned.loc[0, "days_since_last_submission"] == 0


def test_feature_engineer_adds_columns_and_handles_missing_scores() -> None:
    engineered = FeatureEngineer().fit_transform(_raw_row())

    assert set(ENGINEERED_FEATURES).issubset(engineered.columns)
    assert engineered.loc[0, "score_trend"] != engineered.loc[0, "score_trend"]
    assert engineered.loc[0, "score_mean"] == 78.5
    assert engineered.loc[0, "low_engagement_flag"] == 0


def test_full_preprocessor_imputes_missing_values() -> None:
    raw = pd.concat(
        [
            _raw_row(),
            pd.DataFrame(
                [
                    {
                        "attendance_pct": 80,
                        "attendance_trend_3w": 0,
                        "assignment_completion_rate": 0.6,
                        "late_submission_rate": 0.2,
                        "prev_score_1": 60,
                        "prev_score_2": 65,
                        "prev_score_3": 70,
                        "days_since_last_submission": 4,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    transformed = build_preprocessor().fit_transform(raw)

    assert transformed.shape[0] == 2
    assert np.isfinite(transformed).all()
