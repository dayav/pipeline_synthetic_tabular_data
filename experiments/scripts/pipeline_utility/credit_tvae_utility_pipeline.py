#!/usr/bin/env python3

"""
Run utility evaluation for the credit TVAE baseline and all epsilon variants
using the pipeline tasks + storage helpers.
Results are written under `experiments/pipeline_tabular_data_results/utility_evaluation_pipeline/<run_name>`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from data_evaluator import ClassifierType
from data_loader import DataLoader
from data_synthesizer.pipeline.pipepline_builder import PipelineBuilder
from data_synthesizer.pipeline.pipeline_results import GenerationResults
from data_synthesizer.pipeline.storages.evaluation_results_storage import (
    load_all_results,
    save_all_results,
)

# ────────────────────────────────────────────────────────────────
# Dataset-specific configuration
# ────────────────────────────────────────────────────────────────
NUM_COLS_CREDIT = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
CAT_COLS_CREDIT = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "default.payment.next.month",
]

CLASSIFIERS = [
    ClassifierType.CART,
    ClassifierType.KNN,
    ClassifierType.LDA,
    ClassifierType.NB,
    ClassifierType.LR,
    ClassifierType.RANDOM_FOREST,
    ClassifierType.SVM,
    ClassifierType.XGBOOST,
]

# Resolve project root (…/PipelineTabulareDataGeneration) from this file
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Training/test CSVs live under experiments/pipeline_tabular_data_results/data
DATA_ROOT = PROJECT_ROOT / "experiments" / "pipeline_tabular_data_results" / "data"
BASE_RESULTS_DIR = PROJECT_ROOT / "experiments" / "pipeline_tabular_data_results"
OUTPUT_ROOT = BASE_RESULTS_DIR / "utility_evaluation_pipeline"


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def load_synthetic_df(folder: Path):
    """Load synthetic data from a pipeline results folder."""
    res = load_all_results(folder)
    if "generation_results" not in res or "synthetic_data" not in res["generation_results"]:
        raise FileNotFoundError(f"No synthetic data found in {folder}")
    return res["generation_results"]["synthetic_data"]


def run_utility_eval(
    label: str,
    synth_df,
    train_df,
    test_df,
) -> None:
    """Run utility eval for one synthetic dataset and persist results."""
    builder = PipelineBuilder(train_df, CAT_COLS_CREDIT, NUM_COLS_CREDIT)
    builder.add_utility_evaluation_task(
        test_data=test_df,
        classifiers=CLASSIFIERS,
        synth_data=synth_df,
    )
    builder.build()
    results = builder.run()

    # Persist the synthetic data alongside the evaluation for easy reloads
    results["generation_results"] = GenerationResults(
        synthetic_data=synth_df,
        generator_model=None,
        mode_collapse_corrected=False,
    )

    out_folder = OUTPUT_ROOT / label
    save_all_results(results, out_folder)
    print(f"[ok] saved utility results for {label} to {out_folder}")


def synth_runs() -> Iterable[Tuple[str, Path]]:
    """Yield (label, folder) pairs for baseline + epsilon runs (TVAE)."""
    epsilons = ["0.005", "0.01", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4"]
    # baseline
    yield "credit_tvae_baseline", BASE_RESULTS_DIR / "baseline" / "credit_tvae_baseline"
    # epsilon variants
    for eps in epsilons:
        yield f"credit_tvae_eps_{eps}", BASE_RESULTS_DIR / "epsilon_comparison_heom_any" / f"credit_tvae_eps_{eps}"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_df = DataLoader(str(DATA_ROOT / "credit_card_Train.csv")).get_dataframe(
        CAT_COLS_CREDIT, drop_identation=True
    )
    test_df_raw = DataLoader(str(DATA_ROOT / "credit_card_Test.csv")).get_dataframe(
        CAT_COLS_CREDIT, drop_identation=True
    )
    test_df = DataLoader.align_categoricals(test_df_raw, train_df, CAT_COLS_CREDIT)

    for label, folder in synth_runs():
        synth_df_raw = load_synthetic_df(folder)
        synth_df = DataLoader.align_categoricals(synth_df_raw, train_df, CAT_COLS_CREDIT)
        run_utility_eval(label, synth_df, train_df, test_df)


if __name__ == "__main__":
    main()
