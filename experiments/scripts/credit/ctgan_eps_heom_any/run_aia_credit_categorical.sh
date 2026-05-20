#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.pipeline/bin/python}"
TAUS=("0.01" "0.05")

CATEGORICAL_SENSITIVE_COLS=(
  "PAY_0"
  "PAY_2"
  "PAY_3"
  "PAY_4"
  "PAY_5"
  "PAY_6"
)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN explicitly, e.g.:" >&2
  echo "  PYTHON_BIN=.pipeline/bin/python $0" >&2
  exit 1
fi

resolve_data_file() {
  local filename="$1"
  local candidate_a="${PROJECT_ROOT}/pipeline_tabular_data_results/data/${filename}"
  local candidate_b="${PROJECT_ROOT}/data/${filename}"
  if [[ -f "${candidate_a}" ]]; then
    echo "${candidate_a}"
    return 0
  fi
  if [[ -f "${candidate_b}" ]]; then
    echo "${candidate_b}"
    return 0
  fi
  return 1
}

has_column() {
  local csv_path="$1"
  local col="$2"
  "${PYTHON_BIN}" - "${csv_path}" "${col}" <<'PY'
import csv
import sys

csv_path, target = sys.argv[1], sys.argv[2]
with open(csv_path, newline="") as f:
    header = next(csv.reader(f), [])
sys.exit(0 if target in header else 1)
PY
}

run_for_config() {
  local tau="$1"
  local sensitive_col="$2"

  echo "[run] credit_ctgan.py --aia-tau ${tau} --aia-sensitive-col ${sensitive_col} --aia-sensitive-mode classification"

  (
    cd "${SCRIPT_DIR}"
    "${PYTHON_BIN}" "credit_ctgan.py" \
    --aia-tau "${tau}" \
    --aia-sensitive-col "${sensitive_col}" \
    --aia-sensitive-mode classification \
    --aia-attacker-families logreg,mlp_svd \
    --aia-retrain-every 500 \
    --aia-max-swaps 20000 \
    --aia-random-state 0
  )
}

TRAIN_CSV="$(resolve_data_file "credit_card_Train.csv" || true)"
if [[ -z "${TRAIN_CSV}" ]]; then
  echo "[warn] Could not find credit_card_Train.csv; running without pre-checking sensitive columns."
fi

for sensitive_col in "${CATEGORICAL_SENSITIVE_COLS[@]}"; do
  if [[ -n "${TRAIN_CSV}" ]] && ! has_column "${TRAIN_CSV}" "${sensitive_col}"; then
    echo "[skip] sensitive column '${sensitive_col}' not found in ${TRAIN_CSV}"
    continue
  fi
  for tau in "${TAUS[@]}"; do
    run_for_config "${tau}" "${sensitive_col}"
  done
done
