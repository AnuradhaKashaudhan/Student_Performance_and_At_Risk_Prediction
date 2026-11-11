import pandas as pd
import pytest

from app.dashboard_utils import (
    RAW_FEATURES,
    bounded_slider_value,
    parse_class_csv,
    template_csv,
)


def test_template_and_csv_parser_use_only_raw_features() -> None:
    template = template_csv()
    parsed = parse_class_csv(template)

    assert parsed.columns.tolist() == RAW_FEATURES
    assert len(parsed) == 2
    assert "student_id" not in parsed.columns

    with_extra_column = template.replace(
        b"attendance_pct,", b"external_id,attendance_pct,"
    ).replace(b"55,-8", b"sample-1,55,-8").replace(
        b"98,5", b"sample-2,98,5"
    )
    parsed_with_extra = parse_class_csv(with_extra_column)
    assert parsed_with_extra.columns.tolist() == RAW_FEATURES


def test_csv_parser_allows_missing_values_and_rejects_bad_columns_or_values() -> None:
    missing_value = (
        b"attendance_pct,attendance_trend_3w,assignment_completion_rate,"
        b"late_submission_rate,prev_score_1,prev_score_2,prev_score_3,"
        b"days_since_last_submission\n"
        b"80,,0.5,0.2,60,61,62,4\n"
    )
    parsed = parse_class_csv(missing_value)
    assert pd.isna(parsed.loc[0, "attendance_trend_3w"])

    with pytest.raises(ValueError, match="missing required feature columns"):
        parse_class_csv(b"attendance_pct,term\n80,4\n")
    with pytest.raises(ValueError, match="non-numeric values"):
        parse_class_csv(template_csv().replace(b"55,-8", b"bad,-8"))


def test_csv_parser_enforces_batch_limit_and_slider_bounds() -> None:
    template = template_csv().decode("utf-8").splitlines()
    header, *rows = template
    row = rows[0]
    large_csv = (header + "\n" + "\n".join([row] * 101)).encode("utf-8")

    with pytest.raises(ValueError, match="at most 100"):
        parse_class_csv(large_csv)
    assert bounded_slider_value(130, 80, 0, 100) == 100
    assert bounded_slider_value(None, 0.6, 0, 1) == 0.6
