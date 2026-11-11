"""Training-only preprocessing for the week-8 student-risk features."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "attendance_pct",
    "attendance_trend_3w",
    "assignment_completion_rate",
    "late_submission_rate",
    "prev_score_1",
    "prev_score_2",
    "prev_score_3",
    "days_since_last_submission",
]
FORBIDDEN = [
    "student_id",
    "term",
    "next_score",
    "at_risk",
    "gender",
    "religion",
    "caste",
    "income",
    "disability",
    "location",
    "school_support_level",
]
CLIPPING_RULES = {
    "attendance_pct": (0.0, 100.0),
    "prev_score_1": (0.0, 100.0),
    "prev_score_2": (0.0, 100.0),
    "prev_score_3": (0.0, 100.0),
    "assignment_completion_rate": (0.0, 1.0),
    "late_submission_rate": (0.0, 1.0),
    "days_since_last_submission": (0.0, None),
}
ENGINEERED_FEATURES = [
    "score_trend",
    "score_mean",
    "score_drop_last",
    "low_engagement_flag",
]


def assert_no_forbidden(columns: Sequence[str]) -> None:
    """Reject identifiers, outcomes, and excluded sensitive fields as inputs."""
    forbidden = set(columns).intersection(FORBIDDEN)
    if forbidden:
        raise ValueError(f"Forbidden model feature(s): {sorted(forbidden)}")


def _require_frame(frame: pd.DataFrame, expected: Sequence[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Preprocessing expects a pandas DataFrame of raw features.")
    missing = set(expected).difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required feature(s): {sorted(missing)}")


class Cleaner(BaseEstimator, TransformerMixin):
    """Clip known raw-feature outliers to valid values before imputation."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "Cleaner":
        """Validate columns and record the clipping rules for inspection."""
        assert_no_forbidden(X.columns)
        _require_frame(X, FEATURES)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.clipping_rules_ = dict(CLIPPING_RULES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a clipped copy, leaving caller data unchanged."""
        _require_frame(X, FEATURES)
        cleaned = X.loc[:, FEATURES].copy()
        for column, (lower, upper) in self.clipping_rules_.items():
            cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
        return cleaned

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Expose the unchanged raw feature names to sklearn tools."""
        return np.asarray(FEATURES, dtype=object)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add simple historical-score and engagement indicators."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureEngineer":
        """Validate the raw feature columns before building derived features."""
        _require_frame(X, FEATURES)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add features while leaving missing numeric values for the imputer."""
        _require_frame(X, FEATURES)
        engineered = X.loc[:, FEATURES].copy()
        score_columns = ["prev_score_1", "prev_score_2", "prev_score_3"]
        engineered["score_trend"] = engineered["prev_score_3"] - engineered["prev_score_1"]
        engineered["score_mean"] = engineered[score_columns].mean(axis=1)
        engineered["score_drop_last"] = (
            engineered["prev_score_2"] - engineered["prev_score_3"]
        )
        # Comparisons with NaN are false; derived numeric NaNs are imputed next.
        engineered["low_engagement_flag"] = (
            (engineered["attendance_pct"] < 75)
            & (engineered["assignment_completion_rate"] < 0.6)
        ).astype(float)
        return engineered

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return raw and derived names in the transformer's output order."""
        return np.asarray(FEATURES + ENGINEERED_FEATURES, dtype=object)


def build_preprocessor() -> Pipeline:
    """Build a pipeline whose learned imputation/scaling fits training rows only."""
    assert_no_forbidden(FEATURES)
    return Pipeline(
        steps=[
            ("cleaner", Cleaner()),
            ("feature_engineer", FeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
