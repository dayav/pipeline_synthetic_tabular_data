#!/usr/bin/env python3

"""
Run privacy evaluation for the cardio CTGAN baseline and all epsilon variants
using the pipeline tasks + storage helpers.
Results are written under `experiments/pipeline_tabular_data_results/privacy_evaluation_pipeline/<run_name>`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

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
NUM_COLS_CARDIO = ["age", "height", "weight", "ap_hi", "ap_lo"]
CAT_COLS_CARDIO = [
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
]

QAI_COLUMNS = ["age", "gender", "height", "weight"]
RISK_COLUMNS = [
    "ap_lo",
    "ap_hi",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
]

# Resolve project root (…/PipelineTabulareDataGeneration) from this file
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Training/test CSVs live under experiments/pipeline_tabular_data_results/data
DATA_ROOT = PROJECT_ROOT / "experiments" / "pipeline_tabular_data_results" / "data"
BASE_RESULTS_DIR = PROJECT_ROOT / "experiments" / "pipeline_tabular_data_results"
OUTPUT_ROOT = BASE_RESULTS_DIR / "privacy_evaluation_pipeline"


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def load_synthetic_df(folder: Path):
    """Load synthetic data from a pipeline results folder."""
    res = load_all_results(folder)
    if "generation_results" not in res or "synthetic_data" not in res["generation_results"]:
        raise FileNotFoundError(f"No synthetic data found in {folder}")
    return res["generation_results"]["synthetic_data"]


def run_privacy_eval(
    label: str,
    synth_df,
    train_df,
    test_df,
) -> None:
    """Run privacy eval for one synthetic dataset and persist results."""
    builder = PipelineBuilder(train_df, CAT_COLS_CARDIO, NUM_COLS_CARDIO)
    builder.add_privacy_evaluation_task(
        test_data=test_df,
        quasi_identifier_features=QAI_COLUMNS,
        non_quasi_identifier_features=RISK_COLUMNS,
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
    print(f"[ok] saved privacy results for {label} to {out_folder}")


def synth_runs() -> Iterable[Tuple[str, Path]]:
    """Yield (label, folder) pairs for baseline + epsilon runs (cardio CTGAN)."""
    # epsilons = ["0.005", "0.01", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4"]
    epsilons = ["0.4"]
    # baseline
    # yield "cardio_ctgan_baseline", BASE_RESULTS_DIR / "baseline" / "cardio_ctgan_baseline"
    # epsilon variants
    for eps in epsilons:
        yield f"cardio_ctgan_eps_{eps}", BASE_RESULTS_DIR / "epsilon_comparison_heom_any" / f"cardio_ctgan_eps_{eps}"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_df = DataLoader(str(DATA_ROOT / "cardio_train.csv")).get_dataframe(CAT_COLS_CARDIO)
    test_df_raw = DataLoader(str(DATA_ROOT / "cardio_test.csv")).get_dataframe(CAT_COLS_CARDIO)
    test_df = DataLoader.align_categoricals(test_df_raw, train_df, CAT_COLS_CARDIO)

    for label, folder in synth_runs():
        synth_df_raw = load_synthetic_df(folder)
        synth_df = DataLoader.align_categoricals(synth_df_raw, train_df, CAT_COLS_CARDIO)
        run_privacy_eval(label, synth_df, train_df, test_df)


if __name__ == "__main__":
    main()
