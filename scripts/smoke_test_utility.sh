#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$(mktemp -d -t pipeline_tabular_utility_XXXX)}"

cleanup() {
  if [[ -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
  fi
}
trap cleanup EXIT

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 is required for this smoke test (requirements-pipeline_run.txt targets Python 3.11)." >&2
  exit 1
fi
python3.11 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements-pipeline_run.txt"
python -m pip install -e "$ROOT_DIR"

python - <<'PY'
import numpy as np
import pandas as pd

from data_evaluator import ClassifierType
from data_synthesizer.pipeline.pipepline_builder import PipelineBuilder


rng = np.random.default_rng(123)
rows = 500

categorical = ["cities", "segment"]
numerical = ["age", "income"]
ordered_cols = categorical + numerical + ["target"]

train_df = pd.DataFrame(
    {
        "cities": rng.choice(["au", "ba", "cr"], size=rows),
        "segment": rng.choice(["low", "mid", "high"], size=rows),
        "age": rng.integers(18, 70, size=rows),
        "income": rng.normal(50000, 12000, size=rows).round(2),
    }
)

# Binary target: higher income -> 1.
train_df["target"] = (train_df["income"] > 52000).astype(int)
train_df = train_df[ordered_cols]

test_df = train_df.sample(n=200, random_state=42).reset_index(drop=True)

synth_df = train_df.copy()
synth_df["cities"] = rng.choice(["au", "ba", "cr"], size=rows)
synth_df["segment"] = rng.choice(["low", "mid", "high"], size=rows)
synth_df["income"] = (synth_df["income"] + rng.normal(0, 2500, size=rows)).round(2)
synth_df["target"] = (synth_df["income"] > 52000).astype(int)
synth_df = synth_df[ordered_cols]

builder = PipelineBuilder(
    train_data=train_df,
    categorical_features=categorical,
    numerical_features=numerical,
)
builder.add_utility_evaluation_task(
    test_data=test_df,
    classifiers=[ClassifierType.XGBOOST],
    synth_data=synth_df,
)
builder.build()
results = builder.run()

assert "utility_evaluation_results" in results
print("Utility evaluation (pipeline) smoke test ok")
PY

echo "Utility smoke test venv (removed on exit): $VENV_DIR"
