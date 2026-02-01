# Pipeline Results Storage

This project ships convenience helpers to save and reload complete pipeline results
(generation + evaluation) via `save_all_results` and `load_all_results`.

## Usage

```python
from data_synthesizer.pipeline.storages.evaluation_results_storage import (
    load_all_results,
    save_all_results,
)

# Save results into a folder
save_all_results(results, "results/credit_ctgan_baseline")

# Load them back later
restored = load_all_results("results/credit_ctgan_baseline")
```

You can choose a storage backend for the synthetic data payload:

```python
from data_synthesizer.pipeline.storages.storage_backends import list_backends

print(list_backends())  # ("hdf5", "parquet", "csv")

save_all_results(results, "results/run_parquet", generation_backend="parquet")
restored = load_all_results("results/run_parquet")
```

## Folder layout

`save_all_results(results, "results/my_run")` produces the following structure:

```
results/my_run/
├── my_run_generation.<ext>
├── my_run_generation_mode_collapse_corrected.h5
├── my_run_generation_model.pkl
├── my_run_evaluation.h5
└── my_run_manifest.json
```

- `my_run_generation.<ext>` stores the synthetic data table using the selected
  storage backend (`.h5` for HDF5, `.parquet` for Parquet, `.csv` for CSV).
- `my_run_generation_mode_collapse_corrected.h5` stores the boolean flag for
  mode collapse correction.
- `my_run_generation_model.pkl` is optional and only written if a generator model
  is included in `generation_results`.
- `my_run_evaluation.h5` stores evaluation results (utility, privacy, resemblance,
  anonymeter) when present.
- `my_run_manifest.json` records which files and backend were used.

## Security note

**Loading pickled models can execute arbitrary code.** Only load `.pkl` generator
models from trusted sources. If you do not trust the provenance of saved results,
remove the pickle file before calling `load_all_results`.