#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_loader import DataLoader
from data_synthesizer.fit_aia_gard import fit_aia_guard_strong


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific configuration
# ──────────────────────────────────────────────────────────────────────────────
NUM_COLS_CREDIT= [
    'LIMIT_BAL', 
    'AGE', 
    'BILL_AMT1', 
    'BILL_AMT2', 
    'BILL_AMT3',
    'BILL_AMT4',
    'BILL_AMT5', 
    'BILL_AMT6', 
    'PAY_AMT1', 
    'PAY_AMT2', 
    'PAY_AMT3',
    'PAY_AMT4', 
    'PAY_AMT5', 
    'PAY_AMT6'
    ]
CAT_COLS_CREDIT = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'default.payment.next.month']

QAI_COLUMNS = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE']
RISK_COLUMNS = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month']



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../data/credit_card_Train.csv").get_dataframe(
        CAT_COLS_CREDIT,drop_identation=True
    )

    guards = fit_aia_guard_strong(
        real_df=df_train,
        sens_cols=RISK_COLUMNS,
        num_cols=NUM_COLS_CREDIT,
        cat_cols=CAT_COLS_CREDIT,
        clf_n_iter=12,           # tiny randomized search per model
        train_row_cap=None,     # keep guard training fast
        q_clf=0.90,
        q_low=0.10, q_high=0.90,
        tau_reg_width_frac=0.20,
    )    

    with open("credit_aia_guards.pkl", "wb") as f:
        pickle.dump(guards, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

