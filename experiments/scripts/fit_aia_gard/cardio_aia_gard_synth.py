#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_loader import DataLoader
from data_synthesizer import sampling_reject_epsilon_tabnet_strong
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results
from privacy_sampling.embeddings import make_tabnet_embedder


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


# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../data/cardio_train.csv").get_dataframe(
        CAT_COLS_CARDIO)
    
    cardio_ctgan  = load_all_results('../baseline/cardio_ctgan_baseline')

    model = cardio_ctgan["generation_results"]["generator_model"]
    guards_path = "cardio_aia_guards.pkl"
    with open(guards_path, 'rb') as file:
        guards = pickle.load(file)

    embed_fn = make_tabnet_embedder(df_train,
                                    NUM_COLS_CARDIO,
                                    CAT_COLS_CARDIO)   

    cardio_synth, diag = sampling_reject_epsilon_tabnet_strong(
        model, df_train, min_eps=0.05, embed_fn=embed_fn, n_samples=len(df_train),guards=guards,
        apply_epsilon=False,
        apply_guard=True, guard_stage='prefilter',  trace_guard = True,
        explain_guard = True,debug_guard_table=True,
        return_diag=True
    )

    cardio_synth.to_csv("cardio_aiagards.csv", index=False)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

