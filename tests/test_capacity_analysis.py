import numpy as np
import pandas as pd
import pytest

from src.capacity_analysis import analyze_capacity, run_capacity_analysis


def _sample_test_data() -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.DataFrame(
        {
            "student_id": [f"T4-{index:04d}" for index in range(1, 9)],
            "term": [4] * 8,
            "at_risk": [1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    return data, probabilities


def test_capacity_and_threshold_cost_metrics_are_calculated() -> None:
    data, probabilities = _sample_test_data()

    result = analyze_capacity(
        data,
        probabilities,
        chosen_threshold=0.6,
        capacities=(2, 4),
        thresholds=(0.5, 0.8),
        cost_false_negative=5,
        cost_false_positive=1,
    )
    top_k = result.loc[result["analysis"] == "top_k"].set_index("k")
    costs = result.loc[result["analysis"] == "threshold_cost"].set_index("threshold")

    assert top_k.loc[2, "precision_at_k"] == pytest.approx(1.0)
    assert top_k.loc[2, "recall_at_k"] == pytest.approx(2 / 3)
    assert top_k.loc[2, "lift_over_random"] == pytest.approx(1 / (3 / 8))
    assert costs.loc[0.6, "tp"] == 3
    assert costs.loc[0.6, "fp"] == 1
    assert costs.loc[0.6, "fn"] == 0
    assert costs.loc[0.6, "tn"] == 4
    assert costs.loc[0.6, "total_cost"] == pytest.approx(1)
    assert costs.loc[0.6, "is_chosen_threshold"]


def test_capacity_analysis_saves_csv_and_both_figures(tmp_path) -> None:
    data, probabilities = _sample_test_data()

    result = run_capacity_analysis(
        data,
        probabilities,
        chosen_threshold=0.6,
        capacities=(2,),
        thresholds=(0.5, 0.8),
        output_csv=tmp_path / "capacity_analysis.csv",
        figure_dir=tmp_path / "figures",
    )

    assert len(result) == 4
    assert (tmp_path / "capacity_analysis.csv").is_file()
    assert (tmp_path / "figures" / "capacity_metrics.png").is_file()
    assert (tmp_path / "figures" / "cost_vs_threshold.png").is_file()


def test_capacity_analysis_rejects_non_test_data() -> None:
    data, probabilities = _sample_test_data()
    data["term"] = 3

    with pytest.raises(ValueError, match="test term 4 only"):
        analyze_capacity(data, probabilities, chosen_threshold=0.6)
