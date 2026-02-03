#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$(mktemp -d -t pipeline_tabular_smoke_XXXX)}"

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
python -m pip install -r "$ROOT_DIR/requirements.in"
python -m pip install -e "$ROOT_DIR"

python - <<'PY'
import pandas as pd

from data_loader.data_loader import DataLoader
from data_synthesizer.pipeline.pipeline_results import GenerationResults, PipelineResults

df = pd.DataFrame({"city": pd.Series(["a", "b"], dtype="category"), "age": [10, 20]})
loaded = DataLoader(dataset=df).get_dataframe(categorical_cols=["city"])

results = PipelineResults(
    generation_results=GenerationResults(synthetic_data=loaded)
)

assert results.generation_results.synthetic_data.shape == (2, 2)
print("smoke test ok")
PY

echo "Smoke test venv: $VENV_DIR"
