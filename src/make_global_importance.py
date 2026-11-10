"""Create global permutation-importance reports from the validation term only."""

from pathlib import Path

from src.explain import global_importance, load_artifact
from src.preprocess import FEATURES
from src.split import chronological_split, load_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Load the artifact and compute permutation importance on term 3."""
    pipeline, metadata = load_artifact(
        ROOT / "models" / "model.joblib",
        ROOT / "models" / "metadata.json",
    )
    data = load_data(ROOT / "data" / "raw" / "students.csv")
    _, validation, _ = chronological_split(data)
    importance = global_importance(
        pipeline,
        metadata,
        validation[FEATURES],
        validation["at_risk"],
    )
    print(importance.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Saved reports/global_importance.csv and reports/figures/permutation_importance.png")


if __name__ == "__main__":
    main()
