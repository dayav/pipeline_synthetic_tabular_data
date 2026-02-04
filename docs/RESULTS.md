# Results, Artifacts, and Git LFS

This repository keeps heavy experiment outputs in the `pipeline_tabular_data_results/` Git submodule (Git LFS). The main repo stays lightweight; you selectively pull only what you need.

> **Note**
> Avoid large downloads by default. Use `GIT_LFS_SKIP_SMUDGE=1` when cloning and when updating submodules.

## Where Results Live
- `pipeline_tabular_data_results/`: pinned results submodule (Hugging Face dataset repo).
- `RESULTS_STORAGE.md`: storage format produced by `save_all_results` / `load_all_results`.

Top-level folders in the results submodule include:
- `data/`: train/test CSVs and dataset artifacts.
- `baseline/`: baseline generator outputs.
- `epsilon_comparison_heom_any*`: εANY (HEOM-kNN) sweep results.
- `epsilon_disclosure_protection_comparison_heom_any*`: disclosure protection outputs.
- `utility_evaluation_pipeline/`, `privacy_evaluation_pipeline*`: evaluation pipeline outputs.
- `mode_collapse_correction/`: mode-patching outputs.
- `optuna/`, `generator_optimization/`, `extra/`: optimization studies and plots.

## Initialize the Submodule (No LFS Downloads)
```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
```

## Detect LFS Pointer Files
```bash
head -n 3 pipeline_tabular_data_results/data/adult_train.csv
```
If the file starts with `version https://git-lfs.github.com/spec/v1`, it is still a pointer.

## List LFS-Tracked Files
```bash
git -C pipeline_tabular_data_results lfs ls-files
```

## Selective LFS Pull
```bash
git -C pipeline_tabular_data_results lfs pull --include "data/**"
```

## Symlink Helpers
Some scripts expect results or data in alternate locations.

From the repo root:
```bash
ln -s pipeline_tabular_data_results experiments/pipeline_tabular_data_results
ln -s pipeline_tabular_data_results/data data
```

> **Note**
> The `data/` symlink is needed because many scripts load `data/*.csv` from the repo root.

## More Details
- Full setup and troubleshooting: `docs/SETUP.md`
- Storage format: `RESULTS_STORAGE.md`
