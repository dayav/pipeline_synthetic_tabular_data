#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$(mktemp -d -t pipeline_tabular_pipeline_run_XXXX)}"

cleanup() {
  if [[ -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
  fi
}
trap cleanup EXIT

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
# Install locked deps first, then the editable project.
python -m pip install -r "$ROOT_DIR/requirements-pipeline_run.txt"
python -m pip install -e "$ROOT_DIR"

python - <<'PY'
import numpy as np
import pandas as pd

from data_synthesizer.pipeline.pipepline_builder import PipelineBuilder


class BootstrapModel:
    """Simple baseline generator: bootstrap samples from the training data."""

    def __init__(self, random_state: int = 0) -> None:
        self._rng = np.random.default_rng(random_state)
        self._data: pd.DataFrame | None = None

    def fit(self, data: pd.DataFrame) -> "BootstrapModel":
        self._data = data.reset_index(drop=True)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("Model has not been fit")
        idx = self._rng.integers(0, len(self._data), size=n)
        return self._data.iloc[idx].reset_index(drop=True)


rows = 30
rng = np.random.default_rng(42)

df = pd.DataFrame(
    {
        "city": rng.choice(["a", "b", "c"], size=rows),
        "segment": rng.choice(["low", "mid", "high"], size=rows),
        "age": rng.integers(18, 70, size=rows),
        "income": rng.normal(50000, 12000, size=rows).round(2),
    }
)

train = df.sample(frac=0.7, random_state=1).reset_index(drop=True)
test = df.drop(train.index).reset_index(drop=True)

categorical = ["city", "segment"]
numerical = ["age", "income"]

builder = PipelineBuilder(
    train_data=train,
    categorical_features=categorical,
    numerical_features=numerical,
    model=BootstrapModel(random_state=7),
)

builder.add_generation_task()
builder.add_ressemblance_evaluation_task(test)
builder.build()
results = builder.run()

assert "generation_results" in results
assert "resemblance_evaluation_results" in results

print("pipeline_run smoke test ok")
PY

echo "Pipeline_run smoke test venv (removed on exit): $VENV_DIR"
