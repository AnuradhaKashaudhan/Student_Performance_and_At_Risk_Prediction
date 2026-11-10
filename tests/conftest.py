"""Shared fixtures and automatic artifact setup for a fresh checkout."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model() -> None:
    """Train artifacts once when tests are run in a fresh clone."""
    if MODEL_PATH.is_file() and (ROOT / "models" / "metadata.json").is_file():
        return
    result = subprocess.run(
        [sys.executable, "-m", "src.train"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "Could not train the model required by the test suite.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
