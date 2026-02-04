# Datasets

This repository evaluates synthetic tabular data generation on three public benchmarks: Credit Card Clients (UCI), Cardiovascular Disease (Kaggle), and Adult Census Income (UCI). Pre-split train/test CSVs are stored in the results submodule at `pipeline_tabular_data_results/data/`.

> **Note**
> If you plan to regenerate the data from source, you must comply with each dataset's license/terms. No automated download script is included.

## Credit Card Clients (UCI)
- **Source**: UCI ML Repository (Default of Credit Card Clients).
- **Task**: Binary classification (`default.payment.next.month`).
- **Features**: 23 total features (14 numeric, 9 categorical/ordinal).
- **Files in submodule**:
  - `pipeline_tabular_data_results/data/credit_card_Train.csv`
  - `pipeline_tabular_data_results/data/credit_card_Test.csv`
- **Preprocessing in repo**:
  - `DataLoader.get_dataframe(..., drop_identation=True)` drops the first column (index/id) when scripts pass `drop_identation=True`.
  - Categorical columns are cast to `str` or `category` depending on the script.

## Cardiovascular Disease (Kaggle)
- **Source**: Kaggle (Cardiovascular Disease dataset).
- **Task**: Binary classification (`cardio`).
- **Features**: 11 total features (5 numeric, 6 categorical/ordinal).
- **Files in submodule**:
  - `pipeline_tabular_data_results/data/cardio_train.csv`
  - `pipeline_tabular_data_results/data/cardio_test.csv`
- **Preprocessing in repo**:
  - Categories are cast by `DataLoader.get_dataframe` based on dataset-specific categorical lists.

## Adult Census Income (UCI)
- **Source**: UCI ML Repository (Adult / Census Income).
- **Task**: Binary classification (`income` / `<=50K` vs `>50K`).
- **Features**: 14 total features (6 numeric, 8 categorical).
- **Files in submodule**:
  - `pipeline_tabular_data_results/data/adult_train.csv`
  - `pipeline_tabular_data_results/data/adult_test.csv`
- **Preprocessing in repo**:
  - Categories are cast to `category` or `str` depending on the script.
  - `DataLoader.align_categoricals` can be used to align category ordering between datasets.

## Regenerating from Source (TODO)
No automated downloader or preprocessing pipeline is included.

If you need to rebuild the datasets from scratch:
- Download from the original sources.
- Apply train/test splits consistent with the paper.
- Save CSVs into `pipeline_tabular_data_results/data/` or into a `data/` folder at the repo root.

## Licensing and Terms
- UCI datasets have citation and usage terms defined by UCI.
- Kaggle datasets are governed by Kaggle's dataset license and terms of use.

Always review the source terms before redistributing derived artifacts.
