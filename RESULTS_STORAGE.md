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

> **Note**
> The current implementation writes generation payloads as HDF5 only. Alternate
> storage backends are not implemented yet.

## Folder layout

`save_all_results(results, "results/my_run")` produces the following structure:

```
results/my_run/
├── my_run_generation.h5
├── my_run_generation_mode_collapse_corrected.h5
├── my_run_generation_model.pkl
├── my_run_evaluation.h5
```

- `my_run_generation.<ext>` stores the synthetic data table using HDF5 (`.h5`).
- `my_run_generation_mode_collapse_corrected.h5` stores the boolean flag for
  mode collapse correction.
- `my_run_generation_model.pkl` is optional and only written if a generator model
  is included in `generation_results`.
- `my_run_evaluation.h5` stores evaluation results (utility, privacy, resemblance,
  anonymeter) when present.
  
> **TODO**
> If you need alternative backends (Parquet/CSV), add them in
> `data_synthesizer/pipeline/storages/evaluation_results_storage.py`.

## Security note

**Loading pickled models can execute arbitrary code.** Only load `.pkl` generator
models from trusted sources. If you do not trust the provenance of saved results,
remove the pickle file before calling `load_all_results`.
