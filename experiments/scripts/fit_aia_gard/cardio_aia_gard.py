#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_loader import DataLoader
from data_synthesizer.fit_aia_gard import fit_aia_guard_strong


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

    guards = fit_aia_guard_strong(df_train, RISK_COLUMNS, NUM_COLS_CARDIO, CAT_COLS_CARDIO)
    
    guards = fit_aia_guard_strong(
        real_df=df_train,
        sens_cols=RISK_COLUMNS,
        num_cols=NUM_COLS_CARDIO,
        cat_cols=CAT_COLS_CARDIO,
        clf_n_iter=12,           # tiny randomized search per model
        train_row_cap=None,     # keep guard training fast
        q_clf=0.90,
        q_low=0.10, q_high=0.90,
        tau_reg_width_frac=0.20,
    )
    
    with open("cardio_aia_guards.pkl", "wb") as f:
        pickle.dump(guards, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

