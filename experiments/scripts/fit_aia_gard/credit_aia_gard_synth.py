#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_loader import DataLoader
from data_synthesizer import sampling_reject_epsilon_tabnet_strong
from data_synthesizer.fit_aia_gard import fit_aia_guard_strong
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results
from privacy_sampling.embeddings import make_tabnet_embedder


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

    credit_ctgan  = load_all_results('../baseline/credit_ctgan_baseline')

    model = credit_ctgan["generation_results"]["generator_model"]

    guards_path = "credit_aia_guards.pkl"
    with open(guards_path, 'rb') as file:
        guards = pickle.load(file)

    embed_fn = make_tabnet_embedder(df_train,
                                    NUM_COLS_CREDIT,
                                    CAT_COLS_CREDIT)   

    credit_synth, diag = sampling_reject_epsilon_tabnet_strong(
        model, df_train, min_eps=0.05, embed_fn=embed_fn, n_samples=len(df_train),guards=guards,
        apply_epsilon=False,
        apply_guard=True, guard_stage='prefilter',  trace_guard = True,
        explain_guard = True,debug_guard_table=True,
        return_diag=True
    )

    credit_synth.to_csv("credit_aiagards.csv", index=False)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

