#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_loader import DataLoader
from data_synthesizer.fit_aia_gard import fit_aia_guard_strong


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific configuration
# ──────────────────────────────────────────────────────────────────────────────
NUM_COLS_ADULT= ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
CAT_COLS_ADULT = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','income']

QAI_COLUMNS = ['education','education-num','marital-status','occupation','relationship','sex', 'native-country']
RISK_COLUMNS = ['capital-gain','capital-loss','hours-per-week','native-country','income', 'race']



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../data/adult_train.csv").get_dataframe(
        CAT_COLS_ADULT)


    guards = fit_aia_guard_strong(
        real_df=df_train,
        sens_cols=RISK_COLUMNS,
        num_cols=NUM_COLS_ADULT,
        cat_cols=CAT_COLS_ADULT,
        clf_n_iter=12,           # tiny randomized search per model
        train_row_cap=None,     # keep guard training fast
        q_clf=0.90,
        q_low=0.10, q_high=0.90,
        tau_reg_width_frac=0.20,
    )

    with open("adult_aia_guards.pkl", "wb") as f:
        pickle.dump(guards, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

