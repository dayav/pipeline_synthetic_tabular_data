#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$(mktemp -d -t pipeline_tabular_privacy_XXXX)}"

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

from data_synthesizer.pipeline.evaluation_task import PrivacyEvaluationTask
from data_synthesizer.pipeline.pipeline_results import PipelineResults, GenerationResults

rng = np.random.default_rng(123)
rows = 400

train_df = pd.DataFrame(
    {
        "cities": rng.choice(["au", "ba", "cr"], size=rows),
        "segment": rng.choice(["low", "mid", "high"], size=rows),
        "age": rng.integers(18, 70, size=rows),
        "income": rng.normal(50000, 12000, size=rows).round(2),
    }
)

test_df = train_df.sample(n=150, random_state=42).reset_index(drop=True)

synth_df = train_df.copy()
synth_df["cities"] = rng.choice(["au", "ba", "cr"], size=rows)
synth_df["segment"] = rng.choice(["low", "mid", "high"], size=rows)
synth_df["income"] = (synth_df["income"] + rng.normal(0, 2500, size=rows)).round(2)

categorical = ["cities", "segment"]
numerical = ["age", "income"]

quasi_identifier_features = ["cities", "age"]
non_quasi_identifier_features = ["segment", "income"]

results = PipelineResults(
    generation_results=GenerationResults(
        synthetic_data=synth_df,
        generator_model=None,
        mode_collapse_corrected=False,
    )
)

task = PrivacyEvaluationTask(
    train_data=train_df,
    test_data=test_df,
    cat_features=categorical,
    num_features=numerical,
    quasi_identifier_features=quasi_identifier_features,
    non_quasi_identifier_features=non_quasi_identifier_features,
    synth_data=synth_df,
)

task.process(results)

assert results.privacy_evaluation_results is not None
print("Privacy evaluation smoke test ok")
PY

echo "Privacy smoke test venv (removed on exit): $VENV_DIR"
