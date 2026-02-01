#!/usr/bin/env python3

from __future__ import annotations
import pickle

import pandas as pd

from data_loader import DataLoader
from data_synthesizer.fit_aia_gard import fit_aia_guard_strong
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results

from sdmetrics.single_table import DisclosureProtection


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

QAI_COLUMNS = ['SEX','EDUCATION','MARRIAGE','AGE']
RISK_COLUMNS = ['LIMIT_BAL','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month']



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../../data/credit_card_Train.csv").get_dataframe(
        CAT_COLS_CREDIT,drop_identation=True
    )
    df_test = DataLoader("../../../../data/credit_card_Test.csv").get_dataframe(
    CAT_COLS_CREDIT,drop_identation=True
    )

    '../../data/credit_card_Train.csv'
    credit_ctgan  = load_all_results('../../../results/baseline/credit_ctgan_baseline')
    credit_tvae  = load_all_results('../../../results/baseline/credit_tvae_baseline')



    credit_ctgan  = load_all_results('../../../results/baseline/credit_ctgan_baseline')
    credit_tvae  = load_all_results('../../../results/baseline/credit_tvae_baseline')
    credit_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.01')
    credit_ctgan_heom_any_weighted_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted/credit_ctgan_eps_0.01')
    credit_ctgan_heom_any_weighted_5_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted_5/credit_ctgan_eps_0.01')

    credit_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.01')
    credit_tvae_heom_any_weighted_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted/credit_tvae_eps_0.01')
    credit_tvae_heom_any_weighted_5_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted_5/credit_tvae_eps_0.01')

    dict_results_credit_ctgan  = {
        # 'credit CTGAN' : credit_ctgan,
        'credit CTGAN no-weighted' : credit_ctgan_heom_any_eps_001,
        'credit CTGAN weighted' : credit_ctgan_heom_any_weighted_eps_001,    
        'credit CTGAN weighted 5' : credit_ctgan_heom_any_weighted_5_eps_001,    
    }

    dict_results_credit_tvae  = {
        # 'credit TVAE' : credit_tvae,
        'credit TVAE no-weighted' : credit_tvae_heom_any_eps_001,
        'credit TVAE weighted' : credit_tvae_heom_any_weighted_eps_001,    
        'credit TVAE weighted 5' : credit_tvae_heom_any_weighted_5_eps_001,    

    }


    print("======CTGAN=======")
    for sensitive_feat in RISK_COLUMNS :
        scores = {}
        print(f"============={sensitive_feat}================")
        for key , results in dict_results_credit_ctgan.items() :

            print(f"============={sensitive_feat} - {key}===============")

            synth = results['generation_results']['synthetic_data']
            synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)


            def _decategorize(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                for col in out.select_dtypes(include=["category"]).columns:
                    out[col] = out[col].astype(object)
                return out

            real_dp = _decategorize(df_train)
            synth_dp = _decategorize(synth)

            # Make sure lists are lists
            known = list(QAI_COLUMNS)

            scores[key] = DisclosureProtection.compute_breakdown(
                real_data=real_dp,
                synthetic_data=synth_dp,
                known_column_names= known,
                sensitive_column_names=[sensitive_feat],
                continuous_column_names=NUM_COLS_CREDIT
            )

        with open(f"../../../results/epsilon_disclosure_protection_comparison_heom_any_weighted/credit_ctgan_disclosure_protection_{sensitive_feat}.pkl", "wb") as f:
            pickle.dump(scores, f)

    print("======TVAE=======")

    for sensitive_feat in RISK_COLUMNS :
        scores = {}

        print(f"============={sensitive_feat}================")

        for key , results in dict_results_credit_tvae.items() :

            print(f"============={sensitive_feat} - {key}===============")

            synth = results['generation_results']['synthetic_data']
            synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)


            def _decategorize(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                for col in out.select_dtypes(include=["category"]).columns:
                    out[col] = out[col].astype(object)
                return out

            real_dp = _decategorize(df_train)
            synth_dp = _decategorize(synth)

            # Make sure lists are lists
            known = list(QAI_COLUMNS)

            scores[key] = DisclosureProtection.compute_breakdown(
                real_data=real_dp,
                synthetic_data=synth_dp,
                known_column_names= known,
                sensitive_column_names=[sensitive_feat],
                continuous_column_names=NUM_COLS_CREDIT
            )

        with open(f"../../../results/epsilon_disclosure_protection_comparison_heom_any_weighted/credit_tvae_disclosure_protection_{sensitive_feat}.pkl", "wb") as f:
            pickle.dump(scores, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
