#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$(mktemp -d -t pipeline_tabular_sdvtvae_XXXX)}"

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

from sdv.metadata import SingleTableMetadata

from data_loader.data_loader import DataLoader
from data_synthesizer.pipeline.pipepline_builder import PipelineBuilder
from data_synthesizer.sdv import SDVTVAE_


rng = np.random.default_rng(123)
rows = 2000

raw_df = pd.DataFrame(
    {
        "cities": rng.choice(["a", "b", "c"], size=rows),
        "segment": rng.choice(["low", "mid", "high"], size=rows),
        "age": rng.integers(18, 70, size=rows),
        "income": rng.normal(50000, 12000, size=rows).round(2),
    }
)

categorical = ["cities", "segment"]
numerical = ["age", "income"]

processed_df = DataLoader(dataset=raw_df).get_dataframe(
    categorical_cols=categorical,
    category_type=str,
)


metadata = SingleTableMetadata()
metadata.detect_from_dataframe(processed_df)


model = SDVTVAE_(
    metadata,
    processed_df,
    enforce_min_max_values=True,
    enforce_rounding=True,
    embedding_dim=16,
    compress_dims=(32, 32),
    decompress_dims=(32, 32),
    l2scale=1e-6,
    batch_size=50,
    epochs=1,
    loss_factor=2,
    cuda=False,
)

builder = PipelineBuilder(
    train_data=processed_df,
    categorical_features=categorical,
    numerical_features=numerical,
    model=model,
)

builder.add_generation_task()
builder.build()
results = builder.run()

assert "generation_results" in results
synthetic = results["generation_results"]["synthetic_data"]
assert list(synthetic.columns) == list(processed_df.columns)
assert len(synthetic) > 0

print("SDVTVAE_ pipeline smoke test ok")
PY

echo "SDVTVAE_ smoke test venv (removed on exit): $VENV_DIR"
