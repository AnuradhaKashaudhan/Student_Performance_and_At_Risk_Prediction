"""Generate a reproducible, synthetic student checkpoint dataset.

The dataset is designed for an educational-support decision, not to represent
real students or to make decisions about them. Its explicit feature list avoids
protected characteristics and obvious demographic proxies.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "attendance_pct",
    "attendance_trend_3w",
    "assignment_completion_rate",
    "late_submission_rate",
    "prev_score_1",
    "prev_score_2",
    "prev_score_3",
    "days_since_last_submission",
]
OUTPUT_COLUMNS = [
    "student_id",
    "term",
    *FEATURE_COLUMNS[:4],
    *FEATURE_COLUMNS[4:7],
    FEATURE_COLUMNS[7],
    "next_score",
    "at_risk",
]


def _calibrate_intercept(base_score: np.ndarray, target_rate: float = 0.215) -> float:
    """Find an intercept yielding a central at-risk rate near 21.5%.

    The midpoint between adjacent ordered base scores gives a deterministic
    boundary with margin from the strict 50-point cutoff. Calibrating from the
    actual sample is preferable to a hard-coded intercept because its score
    distribution changes with the sample size, terms, and seed.
    """
    ordered_scores = np.sort(base_score)
    desired_count = int(round(target_rate * len(ordered_scores)))
    if not 0 < desired_count < len(ordered_scores):
        raise ValueError("The sample is too small to calibrate the at-risk rate.")
    cutoff = (ordered_scores[desired_count - 1] + ordered_scores[desired_count]) / 2
    return 50 - cutoff


def generate(n_per_term: int = 600, n_terms: int = 4, seed: int = 42) -> pd.DataFrame:
    """Return a seeded synthetic dataset with one record per student.

    Ability and engagement are standard-normal latent traits with correlation
    0.4. They are used only to construct observable educational indicators and
    assessment scores, and are deliberately never included in the output.
    Attendance is based on eight noisy weekly measurements, making the
    three-week trend a genuine late-versus-earlier change rather than a random
    column. Assignment completion and late submission rates use bounded logistic
    transforms so their values remain in [0, 1].

    The three historical scores combine ability, independent measurement noise,
    a small term drift, and a slight chronological offset. The next score follows
    the requested equation; its intercept is calibrated per generated dataset so
    the at-risk prevalence is realistic and asserted below. Scores are clipped to
    [0, 100] before defining the target.

    Missingness is injected at exactly 5% of rows for the newest prior score and
    checkpoint attendance. Three separate 1%-scale anomalies are injected:
    attendance above 100, negative days since submission, and a historical score
    above 100. The score anomaly is placed in ``prev_score_2`` because
    ``next_score`` is explicitly clipped to 100. Finally, about 1% of rows copy
    another same-term feature/label profile but retain a fresh ID. This allows
    duplicate-profile checks while preserving the explicit unique-ID and fixed
    row-count requirements.
    """
    if n_per_term < 2:
        raise ValueError("n_per_term must be at least 2 to inject duplicate profiles.")
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1.")

    rng = np.random.default_rng(seed)
    records: list[dict[str, int | float | str]] = []
    latent_ability: list[np.ndarray] = []
    latent_engagement: list[np.ndarray] = []

    for term in range(1, n_terms + 1):
        ability = rng.normal(0, 1, n_per_term)
        engagement = 0.4 * ability + np.sqrt(1 - 0.4**2) * rng.normal(
            0, 1, n_per_term
        )
        latent_ability.append(ability)
        latent_engagement.append(engagement)

        # A per-student slope plus weekly noise makes attendance trends variable.
        weeks = np.arange(1, 9, dtype=float)
        attendance_slope = rng.normal(0, 0.9, n_per_term)
        weekly_attendance = (
            80
            + 7 * engagement[:, None]
            + attendance_slope[:, None] * (weeks[None, :] - 4.5)
            + rng.normal(0, 5, (n_per_term, 8))
        )
        weekly_attendance = np.clip(weekly_attendance, 60, 100)
        attendance_pct = weekly_attendance.mean(axis=1)
        attendance_trend = (
            weekly_attendance[:, -3:].mean(axis=1)
            - weekly_attendance[:, :5].mean(axis=1)
        )

        completion_logit = 0.2 + 1.0 * engagement + 0.2 * ability
        assignment_completion = 1 / (1 + np.exp(-completion_logit))
        late_logit = -1.0 - 0.9 * engagement + rng.normal(0, 0.35, n_per_term)
        late_submission = 1 / (1 + np.exp(-late_logit))
        days_since_submission = np.maximum(
            0, np.rint(6 - 2.2 * engagement + rng.normal(0, 3, n_per_term))
        )

        term_drift = 0.6 * (term - 1)
        historical_scores = (
            63
            + 11 * ability[:, None]
            + term_drift
            + np.array([-1.5, -0.5, 0.0])[None, :]
            + rng.normal(0, 9, (n_per_term, 3))
        )
        historical_scores = np.clip(historical_scores, 0, 100)
        next_score_base = (
            12 * ability
            + 6 * engagement
            + 0.3 * (historical_scores[:, 2] - historical_scores[:, 0])
            + rng.normal(0, 10, n_per_term)
        )

        for index in range(n_per_term):
            records.append(
                {
                    "student_id": f"T{term}-{index + 1:04d}",
                    "term": term,
                    "attendance_pct": float(attendance_pct[index]),
                    "attendance_trend_3w": float(attendance_trend[index]),
                    "assignment_completion_rate": float(assignment_completion[index]),
                    "late_submission_rate": float(late_submission[index]),
                    "prev_score_1": float(historical_scores[index, 0]),
                    "prev_score_2": float(historical_scores[index, 1]),
                    "prev_score_3": float(historical_scores[index, 2]),
                    "days_since_last_submission": int(days_since_submission[index]),
                    "_next_score_base": float(next_score_base[index]),
                }
            )

    frame = pd.DataFrame.from_records(records)
    intercept = _calibrate_intercept(frame["_next_score_base"].to_numpy())
    frame["next_score"] = np.clip(intercept + frame["_next_score_base"], 0, 100)
    frame["at_risk"] = (frame["next_score"] < 50).astype(int)
    frame = frame.drop(columns="_next_score_base")

    # Inject one anomaly per issue type, at 1% of the full dataset.
    outlier_count = max(1, round(len(frame) * 0.01))
    attendance_indices = rng.choice(frame.index, outlier_count, replace=False)
    days_indices = rng.choice(frame.index, outlier_count, replace=False)
    score_indices = rng.choice(frame.index, outlier_count, replace=False)
    frame.loc[attendance_indices, "attendance_pct"] = 105.0
    frame.loc[days_indices, "days_since_last_submission"] = -3
    frame.loc[score_indices, "prev_score_2"] = 105.0

    # Missingness is injected before profile copying so the duplicates retain
    # exactly matching missing-value patterns and remain detectable.
    missing_count = max(1, round(len(frame) * 0.05))
    for column in ("prev_score_3", "attendance_pct"):
        missing_indices = rng.choice(frame.index, missing_count, replace=False)
        frame.loc[missing_indices, column] = np.nan

    # Copy complete feature/label profiles within a term, with distinct IDs.
    duplicate_count = 0
    copy_columns = [column for column in OUTPUT_COLUMNS if column != "student_id"]
    for term in range(1, n_terms + 1):
        term_indices = frame.index[frame["term"] == term].to_numpy()
        permutation = rng.permutation(term_indices)
        count = round(n_per_term * 0.01)
        donors, recipients = permutation[:count], permutation[count : 2 * count]
        original_profiles = frame.loc[donors, copy_columns].to_numpy(copy=True)
        frame.loc[recipients, copy_columns] = original_profiles
        duplicate_count += len(recipients)

    frame = frame.loc[:, OUTPUT_COLUMNS]
    at_risk_rate = float(frame["at_risk"].mean())
    LOGGER.info(
        "Generated %d rows; duplicate feature/label profiles: %d; "
        "missing values: %s; outliers: attendance_pct>100=%d, "
        "days_since_last_submission<0=%d, prev_score_2>100=%d; "
        "at-risk rate: %.2f%%",
        len(frame),
        frame.duplicated(subset=copy_columns, keep="first").sum(),
        frame[["prev_score_3", "attendance_pct"]].isna().sum().to_dict(),
        int((frame["attendance_pct"] > 100).sum()),
        int((frame["days_since_last_submission"] < 0).sum()),
        int((frame["prev_score_2"] > 100).sum()),
        100 * at_risk_rate,
    )
    assert frame["student_id"].is_unique, "Student IDs must be unique."
    assert 0.18 <= at_risk_rate <= 0.25, (
        f"At-risk rate {at_risk_rate:.3f} is outside 18%-25%."
    )
    return frame


def main() -> None:
    """Parse generator options and write the standard raw CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--n-per-term", type=int, default=600, help="Students per term (default: 600)."
    )
    parser.add_argument(
        "--n-terms", type=int, default=4, help="Number of terms (default: 4)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "students.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate(args.n_per_term, args.n_terms, args.seed)
    dataset.to_csv(output_path, index=False, float_format="%.10f")
    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
