#!/usr/bin/env python3
"""
Run the synthetic-data pipeline for the Cardio Card dataset.

Usage examples
--------------
Baseline training:
    python cardio_ctgan.py --baseline

Mode-collapse correction:
    python cardio_ctgan.py --mode-collapse

Mode-collapse + ε-rejection:
    python cardio_ctgan.py --mode-collapse --epsilon 0.15

ε-rejection only (no fine-tuning):
    python cardio_ctgan.py --epsilon 0.15

ε-rejection with heom (no fine-tuning):
    python cardio_ctgan.py --epsilon-heom 0.15

ε-rejection with heom any (no fine-tuning):
    python cardio_ctgan.py --epsilon-heom-knn-any 0.15

Attack-aware AIA rejection sampling:
    python cardio_ctgan.py --aia-tau 0.05 --aia-sensitive-col cardio
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
from data_synthesizer.sdv import SDVCTGAN_

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
RESULTS_ROOT = PROJECT_ROOT / "pipeline_tabular_data_results"

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
def _resolve_data_file(filename: str) -> Path:
    candidates = [
        RESULTS_ROOT / "data" / filename,
        PROJECT_ROOT / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find `{filename}`. Tried: {', '.join(str(p) for p in candidates)}"
    )


def _make_metadata(df_train) -> SingleTableMetadata:
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df_train)
    return meta


def _new_ctgan(metadata: SingleTableMetadata, df_train):
    """Return a freshly initialised CTGAN with your preferred hyper-parameters."""
    return SDVCTGAN_( 
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
    *,
    baseline: bool,
    mode_collapse: bool,
    epsilon: Optional[float],
    epsilon_heom: Optional[float],
    epsilon_heom_knn_any: Optional[float],
    aia_tau: Optional[float] = None,
    aia_sensitive_col: Optional[str] = None,
) -> Path:
    """Resolve the correct results directory from the chosen flags."""
    root = RESULTS_ROOT
    if baseline:
        return root / "baseline" / "cardio_ctgan_baseline"
    if mode_collapse and epsilon is not None:
        return (
            root
            / "mode_collapse_correction_epsilon"
            / f"cardio_ctgan_mode_collapse_corrected_eps_{epsilon}"
        )
    if mode_collapse:
        return root / "mode_collapse_correction" / "cardio_ctgan_mode_collapse_corrected"
    if epsilon is not None:
        return root / "epsilon_comparison" / f"cardio_ctgan_eps_{epsilon}"
    if epsilon_heom is not None:
        return root / "epsilon_comparison_heom" / f"cardio_ctgan_eps_{epsilon_heom}"
    if epsilon_heom_knn_any is not None:
        return root / "epsilon_comparison_heom_any" / f"cardio_ctgan_eps_{epsilon_heom_knn_any}"
    if aia_tau is not None:
        safe_sensitive_col = (aia_sensitive_col or "unknown").replace("/", "_").replace(" ", "_")
        return root / "aia" / f"cardio_ctgan_aia_tau_{aia_tau}_sensitive_{safe_sensitive_col}"
    raise ValueError("No scenario selected: choose --baseline, --mode-collapse, --epsilon*, or --aia-tau")


# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    *,
    baseline: bool = False,
    mode_collapse: bool = False,
    epsilon: Optional[float] = None,
    epsilon_heom: Optional[float] = None,
    epsilon_heom_knn_any: Optional[float] = None,
    aia_tau: Optional[float] = None,
    aia_sensitive_col: str = "cardio",
    aia_z_cols: Optional[str] = None,
    aia_sensitive_bins: Optional[int] = None,
    aia_sensitive_mode: str = "auto",
    aia_regression_unique_threshold: int = 50,
    aia_regression_target_transform: str = "auto",
    aia_linear_regressor_kind: str = "sgd",
    aia_regression_normalize: bool = True,
    aia_attacker_families: str = "logreg,mlp_svd",
    aia_retrain_every: int = 500,
    aia_max_swaps: int = 20000,
    aia_random_state: int = 0,
) -> None:
    # 1. Load data
    train_csv = _resolve_data_file("cardio_train.csv")
    test_csv = _resolve_data_file("cardio_test.csv")
    df_train = DataLoader(str(train_csv)).get_dataframe(
        CAT_COLS_CARDIO, category_type=str, sep=","
    )
    df_test = DataLoader(str(test_csv)).get_dataframe(
        CAT_COLS_CARDIO, category_type=str, sep=","
    )

    # 2. Prepare (or load) the model + synthetic data
    if baseline:
        print("🔧 Creating a new CTGAN baseline model...")
        model = _new_ctgan(_make_metadata(df_train), df_train)
        synth_data = None
    else:
        print("📦 Loading baseline model & synthetic data…")
        baseline_results = load_all_results(str(RESULTS_ROOT / "baseline" / "cardio_ctgan_baseline"))
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
    if aia_tau is not None:
        if aia_sensitive_col not in df_train.columns:
            raise ValueError(
                f"`aia_sensitive_col={aia_sensitive_col}` is not a column in the training data."
            )
        z_cols = [c.strip() for c in (aia_z_cols or "").split(",") if c.strip()]
        unknown_z_cols = [c for c in z_cols if c not in df_train.columns]
        if unknown_z_cols:
            raise ValueError(f"`aia_z_cols` contains unknown columns: {unknown_z_cols}")

        excluded = {aia_sensitive_col}
        excluded.update(z_cols)
        x_cols = [c for c in df_train.columns if c not in excluded]
        attacker_families = [a.strip() for a in aia_attacker_families.split(",") if a.strip()]

        aia_parameters = {
            "real_val_df": df_test,
            "sensitive_col": aia_sensitive_col,
            "X_cols": x_cols,
            "Z_cols": z_cols if len(z_cols) > 0 else None,
            "num_cols": NUM_COLS_CARDIO,
            "cat_cols": CAT_COLS_CARDIO,
            "n_samples": len(df_train),
            "sensitive_n_bins": aia_sensitive_bins,
            "sensitive_mode": aia_sensitive_mode,
            "regression_unique_threshold": aia_regression_unique_threshold,
            "regression_target_transform": aia_regression_target_transform,
            "linear_regressor_kind": aia_linear_regressor_kind,
            "regression_normalize": aia_regression_normalize,
            "attacker_families": attacker_families,
            "retrain_every": aia_retrain_every,
            "max_swaps": aia_max_swaps,
            "random_state": aia_random_state,
            "verbose": True,
        }
        builder.add_sampling_and_reject_task(
            aia_tau,
            synth_data=synth_data,
            dissimilarity_type="aia",
            aia_parameters=aia_parameters,
        )

    builder.add_ressemblance_evaluation_task(df_test)
    builder.add_utility_evaluation_task(df_test, CLASSIFIERS)
    builder.add_privacy_evaluation_task(df_test, QAI_COLUMNS, RISK_COLUMNS)
    # builder.add_privacy_anonymeter_evaluation_task(
    #     df_test, first_half=QAI_COLUMNS, second_half=RISK_COLUMNS, runs_count = 10
    # )

    builder.build()

    # 4. Run & save
    print("🚀 Running pipeline…")
    results = builder.run()

    folder = _output_folder(
        baseline=baseline,
        mode_collapse=mode_collapse,
        epsilon=epsilon,
        epsilon_heom=epsilon_heom,
        epsilon_heom_knn_any=epsilon_heom_knn_any,
        aia_tau=aia_tau,
        aia_sensitive_col=aia_sensitive_col,
    )
    print(f"💾 Saving results → {folder.resolve()}")
    save_all_results(results, folder)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cardio CTGAN pipeline runner")
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
        "--aia-tau",
        type=float,
        metavar="FLOAT",
        help="Target threshold for AIA rejection-with-replacement sampling",
    )
    parser.add_argument(
        "--aia-sensitive-col",
        type=str,
        default="cardio",
        help="Sensitive attribute to protect in AIA sampling",
    )
    parser.add_argument(
        "--aia-z-cols",
        type=str,
        default=None,
        help="Optional comma-separated auxiliary columns (Z) for AIA scoring",
    )
    parser.add_argument(
        "--aia-sensitive-bins",
        type=int,
        default=None,
        help="Quantile bins for numeric sensitive attribute (ignored for categorical)",
    )
    parser.add_argument(
        "--aia-sensitive-mode",
        type=str,
        choices=["auto", "classification", "regression"],
        default="auto",
        help="How to treat sensitive attribute in AIA sampler",
    )
    parser.add_argument(
        "--aia-regression-unique-threshold",
        type=int,
        default=50,
        help="In auto mode, numeric sensitive with unique values above this uses regression objective",
    )
    parser.add_argument(
        "--aia-regression-target-transform",
        type=str,
        choices=["auto", "none", "log1p_clip0"],
        default="auto",
        help="Target transform in regression mode",
    )
    parser.add_argument(
        "--aia-linear-regressor-kind",
        type=str,
        choices=["sgd", "ridge"],
        default="sgd",
        help="Linear attacker baseline used in regression mode",
    )
    parser.add_argument(
        "--aia-disable-regression-normalize",
        action="store_true",
        help="Disable regression risk normalization by Var(A_val)",
    )
    parser.add_argument(
        "--aia-attacker-families",
        type=str,
        default="logreg,mlp_svd",
        help="Comma-separated attacker families (e.g. logreg,mlp_svd)",
    )
    parser.add_argument(
        "--aia-retrain-every",
        type=int,
        default=500,
        help="AIA retrain period B (accepted replacements)",
    )
    parser.add_argument(
        "--aia-max-swaps",
        type=int,
        default=20000,
        help="AIA maximum number of replacement attempts",
    )
    parser.add_argument(
        "--aia-random-state",
        type=int,
        default=0,
        help="Random state for AIA sampler",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        baseline=args.baseline,
        mode_collapse=args.mode_collapse,
        epsilon=args.epsilon,
        epsilon_heom=args.epsilon_heom,
        epsilon_heom_knn_any=args.epsilon_heom_knn_any,
        aia_tau=args.aia_tau,
        aia_sensitive_col=args.aia_sensitive_col,
        aia_z_cols=args.aia_z_cols,
        aia_sensitive_bins=args.aia_sensitive_bins,
        aia_sensitive_mode=args.aia_sensitive_mode,
        aia_regression_unique_threshold=args.aia_regression_unique_threshold,
        aia_regression_target_transform=args.aia_regression_target_transform,
        aia_linear_regressor_kind=args.aia_linear_regressor_kind,
        aia_regression_normalize=not args.aia_disable_regression_normalize,
        aia_attacker_families=args.aia_attacker_families,
        aia_retrain_every=args.aia_retrain_every,
        aia_max_swaps=args.aia_max_swaps,
        aia_random_state=args.aia_random_state,
    )


if __name__ == "__main__":
    main()
