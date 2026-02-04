# Reproduce the Paper

This guide outlines how to reproduce the experiments in **"Generating High-quality Privacy-preserving Synthetic Data"** using the code and results in this repository.

> **Note**
> The heavy artifacts live in the `pipeline_tabular_data_results/` submodule (Git LFS). Avoid large downloads by default and selectively fetch what you need.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Data Access](#data-access)
- [Smoke Tests](#smoke-tests)
- [Baseline Runs](#baseline-runs)
- [Mode-Patching](#mode-patching)
- [HEOM-kNN εANY Filtering](#heom-knn-εany-filtering)
- [Paper Mapping](#paper-mapping)

## Prerequisites
- **Python**: 3.9+ for core; 3.11 for the `pipeline_run` lockfile.
- **OS**: Linux is the reference platform.
- **GPU**: optional but recommended for training.

Install dependencies (minimal + editable install):
```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.in
python -m pip install -e .
```

Optional extras for full pipeline + notebooks:
```bash
python -m pip install -e ".[pipeline_run]"
```

Full setup details: `docs/SETUP.md`.

## Data Access
Pre-split train/test CSVs are included in the results submodule under `pipeline_tabular_data_results/data/`.

If you want scripts to work without editing paths, create the following symlinks from the repo root:
```bash
ln -s pipeline_tabular_data_results/data data
ln -s pipeline_tabular_data_results experiments/pipeline_tabular_data_results
```

Selective LFS pull (example):
```bash
git -C pipeline_tabular_data_results lfs pull --include "data/**"
```

> **TODO**
> No automated downloader is provided for the raw UCI/Kaggle sources. If you need to rebuild from scratch, download from the original sources and place CSVs into `data/` or `pipeline_tabular_data_results/data/` with the expected filenames.

## Smoke Tests
Core smoke test (no large downloads required):
```bash
./scripts/smoke_test.sh
```

Pipeline smoke test (requires `pipeline_run` deps):
```bash
./scripts/smoke_test_pipeline_run.sh
```

If a script is not executable:
```bash
chmod +x ./scripts/smoke_test.sh
```

## Baseline Runs
Example baseline runs on the Credit dataset (CTGAN + TVAE):
```bash
python experiments/scripts/credit/ctgan_eps_heom_any/credit_ctgan.py --baseline
python experiments/scripts/credit/tvae_eps_heom_any/credit_tvae.py --baseline
```

Outputs are written into the corresponding results folders (for example `baseline/credit_ctgan_baseline/`).

## Mode-Patching
Mode-patching is triggered by the `--mode-collapse` flag and uses the layer-frozen fine-tuning loop.

Example:
```bash
python experiments/scripts/credit/ctgan_eps_heom_any/credit_ctgan.py --mode-collapse
```

## HEOM-kNN εANY Filtering
The εANY rejection-with-replacement filter is invoked with `--epsilon-heom-knn-any` (τANY threshold).

Single run example:
```bash
python experiments/scripts/credit/ctgan_eps_heom_any/credit_ctgan.py --epsilon-heom-knn-any 0.2
```

Sweep example (edit `eps_values` as needed):
```bash
python experiments/scripts/credit/ctgan_eps_heom_any/driver.py
```

## Paper Mapping
Use this section to map paper figures/tables to commands and outputs. Update the figure/table numbers once finalized.

| Paper Ref | Entry Point | Output Path | Notes |
|---|---|---|---|
| Fig. 4 (TODO) | `experiments/pipeline_experiment/epsilon_comparison_univariate_resemblance_evaluation.ipynb` | `pipeline_tabular_data_results/epsilon_comparison_heom_any/` | Univariate JS/Cohen's d sweeps. |
| Fig. 5 (TODO) | `experiments/pipeline_experiment/epsilon_comparison_multivariate_resemblance_evaluation.ipynb` | `pipeline_tabular_data_results/epsilon_comparison_heom_any/` | Pearson/Cramer's V/η^2 matrices. |
| Fig. 6 (TODO) | `experiments/pipeline_experiment/epsilon_comparison_utility_evaluation.ipynb` | `pipeline_tabular_data_results/epsilon_utility_comparison_heom_any/` | TSTR/TRTR utility curves. |
| Table I (TODO) | `experiments/scripts/optuna/*_optimize.py` | `pipeline_tabular_data_results/optuna/` | Hyperparameter search (Optuna TPE, 50 trials). |
| Table II/III (TODO) | `experiments/scripts/*/disclosure_protection/*.py` | `pipeline_tabular_data_results/epsilon_disclosure_protection_comparison_heom_any/` | CAP/DCR summaries. |
| Table VI (TODO) | `experiments/scripts/*/ctgan_eps_heom_any*/driver.py` | `pipeline_tabular_data_results/epsilon_comparison_heom_any/` | εANY sweep results. |

> **Note**
> If a notebook or script is missing, add a TODO entry with the required command and expected output path.
