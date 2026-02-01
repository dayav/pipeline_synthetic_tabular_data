#!/usr/bin/env python3
"""
Run the synthetic-data pipeline for the Cardio Card dataset.

Usage examples
--------------
Baseline training:
    python cardio_tvae.py --baseline

Mode-collapse correction:
    python cardio_tvae.py --mode-collapse

Mode-collapse + ε-rejection:
    python cardio_tvae.py --mode-collapse --epsilon 0.15

ε-rejection only (no fine-tuning):
    python cardio_tvae.py --epsilon 0.15

ε-rejection with heom (no fine-tuning):
    python cardio_tvae.py --epsilon-heom 0.15

ε-rejection with heom any (no fine-tuning):
    python cardio_tvae.py --epsilon-heom-knn-any 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from sdv.metadata import SingleTableMetadata

from data_loader import DataLoader
from data_evaluator import ClassifierType
from data_synthesizer.pipeline import PipelineBuilder
from data_synthesizer.pipeline.storages.evaluation_results_storage import (
    load_all_results,
    save_all_results,
)
from data_synthesizer.sdv import SDVTVAE_

# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific configuration
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _make_metadata(df_train) -> SingleTableMetadata:
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df_train)
    return meta


def _new_tvae(metadata: SingleTableMetadata, df_train):
    """Return a freshly initialised TVAE with your preferred hyper-parameters."""
    return SDVTVAE_( 
        metadata, 
        df_train, 
        enforce_min_max_values=True, 
        enforce_rounding=True, 
        locales=None,
        embedding_dim=512, 
        generator_dim=[256, 256], 
        discriminator_dim=[256, 256],
        generator_lr=0.00017403142604971523, 
        generator_decay=1e-6, 
        discriminator_lr=0.00037251120555768314,
        discriminator_decay=1e-6,
        batch_size=100, 
        discriminator_steps=1,
        log_frequency=True, 
        verbose=False, 
        epochs=1000, 
        pac=10, 
        cuda=True)


def _output_folder(
    *, baseline: bool, mode_collapse: bool, epsilon: Optional[float], epsilon_heom: Optional[float], epsilon_heom_knn_any: Optional[float], epsilon_heom_knn_any_weighted: Optional[float] = None
) -> Path:
    """Resolve the correct results directory from the chosen flags."""
    root = Path("../..")
    if baseline:
        return root / "baseline" / "cardio_tvae_baseline"
    if mode_collapse and epsilon is not None:
        return (
            root
            / "mode_collapse_correction_epsilon"
            / f"cardio_tvae_mode_collapse_corrected_eps_{epsilon}"
        )
    if mode_collapse:
        return root / "mode_collapse_correction" / "cardio_tvae_mode_collapse_corrected"
    if epsilon is not None:
        return root / "epsilon_comparison" / f"cardio_tvae_eps_{epsilon}"
    if epsilon_heom is not None:
        return root / "epsilon_comparison_heom" / f"cardio_tvae_eps_{epsilon_heom}"
    if epsilon_heom_knn_any is not None:
        return root / "epsilon_comparison_heom_any" / f"cardio_tvae_eps_{epsilon_heom_knn_any}"
    if epsilon_heom_knn_any_weighted is not None:
        return root / "epsilon_comparison_heom_any_weighted_5" / f"cardio_tvae_eps_{epsilon_heom_knn_any_weighted}"  
    raise ValueError("No scenario selected: choose --baseline, --mode-collapse or --epsilon")


# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    *, baseline: bool = False, mode_collapse: bool = False, epsilon: Optional[float] = None, epsilon_heom: Optional[float] = None, epsilon_heom_knn_any: Optional[float] = None,
    epsilon_heom_knn_any_weighted: Optional[float] = None
) -> None:
    # 1. Load data
    df_train = DataLoader("../../../../data/cardio_train.csv").get_dataframe(
        CAT_COLS_CARDIO, category_type=str, sep=","
    )
    df_test = DataLoader("../../../../data/cardio_test.csv").get_dataframe(
        CAT_COLS_CARDIO, category_type=str, sep=","
    )

    # 2. Prepare (or load) the model + synthetic data
    if baseline:
        print("🔧 Creating a new TVAE baseline model...")
        model = _new_tvae(_make_metadata(df_train), df_train)
        synth_data = None
    else:
        print("📦 Loading baseline model & synthetic data…")
        baseline_results = load_all_results("../../baseline/cardio_tvae_baseline")
        model = baseline_results["generation_results"]["generator_model"]
        synth_data = baseline_results["generation_results"]["synthetic_data"]

    # 3. Build the pipeline
    print("🏗️ Building pipeline...")
    builder = PipelineBuilder(df_train, CAT_COLS_CARDIO, NUM_COLS_CARDIO, model)

    if baseline:
        builder.add_generation_task()
    if mode_collapse:
        builder.add_fine_tuning_generation_task(synth_data)
    if epsilon is not None:
        builder.add_sampling_and_reject_task(epsilon, synth_data=synth_data)
    if epsilon_heom is not None:
        builder.add_sampling_and_reject_task(epsilon_heom, synth_data=synth_data, dissimilarity_type="heom")
    if epsilon_heom_knn_any is not None:
        builder.add_sampling_and_reject_task(epsilon_heom_knn_any, synth_data=synth_data, dissimilarity_type="heom-knn-any")
    if epsilon_heom_knn_any_weighted is not None:
        builder.add_sampling_and_reject_task(epsilon_heom_knn_any_weighted, synth_data=synth_data, dissimilarity_type="heom-knn-any-weighted-5")

    builder.add_ressemblance_evaluation_task(df_test)
    # builder.add_utility_evaluation_task(df_test, CLASSIFIERS)
    # builder.add_privacy_evaluation_task(df_test, QAI_COLUMNS, RISK_COLUMNS)
    # builder.add_privacy_anonymeter_evaluation_task(
    #     df_test, first_half=QAI_COLUMNS, second_half=RISK_COLUMNS, runs_count = 10
    # )

    builder.build()

    # 4. Run & save
    print("🚀 Running pipeline…")
    results = builder.run()

    folder = _output_folder(baseline=baseline, mode_collapse=mode_collapse, epsilon=epsilon, epsilon_heom=epsilon_heom, epsilon_heom_knn_any=epsilon_heom_knn_any, epsilon_heom_knn_any_weighted= epsilon_heom_knn_any_weighted)
    print(f"💾 Saving results → {folder.resolve()}")
    save_all_results(results, folder)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cardio TVAE pipeline runner")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--baseline", action="store_true", help="Run plain baseline training")
    mode.add_argument(
        "--mode-collapse",
        action="store_true",
        help="Enable fine-tuning for mode-collapse correction",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        metavar="FLOAT",
        help="Threshold for ε-based rejection sampling (enables that step)",
    )

    parser.add_argument(
        "--epsilon-heom",
        type=float,
        metavar="FLOAT",
        help="Threshold for ε-based rejection sampling (enables that step)",
    )

    parser.add_argument(
        "--epsilon-heom-knn-any",
        type=float,
        metavar="FLOAT",
        help="Threshold for ε-based rejection sampling (enables that step)",
    )

    parser.add_argument(
        "--epsilon-heom-knn-any-weighted",
        type=float,
        metavar="FLOAT",
        help="Threshold for ε-based rejection sampling (enables that step)",
    )       
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(baseline=args.baseline, mode_collapse=args.mode_collapse, epsilon=args.epsilon, epsilon_heom=args.epsilon_heom, epsilon_heom_knn_any=args.epsilon_heom_knn_any,  epsilon_heom_knn_any_weighted=args.epsilon_heom_knn_any_weighted)


if __name__ == "__main__":
    main()

