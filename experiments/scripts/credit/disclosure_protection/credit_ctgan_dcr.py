#!/usr/bin/env python3

from __future__ import annotations
import pickle

import pandas as pd

from data_loader import DataLoader
from data_synthesizer.fit_aia_gard import fit_aia_guard_strong
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results

from sdmetrics.single_table import DisclosureProtection
from sdmetrics.single_table import DCROverfittingProtection
from sdv.metadata import SingleTableMetadata


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
    credit_ctgan_heom_any_eps_0005  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.005')
    credit_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.01')
    credit_ctgan_heom_any_eps_005  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.05')
    credit_ctgan_heom_any_eps_01  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.1')
    credit_ctgan_heom_any_eps_015  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.15')
    credit_ctgan_heom_any_eps_02  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.2')
    credit_ctgan_heom_any_eps_025  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.25')
    credit_ctgan_heom_any_eps_03  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.3')
    credit_ctgan_heom_any_eps_035  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.35')
    credit_ctgan_heom_any_eps_04  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.4')

    credit_tvae_heom_any_eps_0005  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.005')
    credit_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.01')
    credit_tvae_heom_any_eps_005  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.05')
    credit_tvae_heom_any_eps_01  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.1')
    credit_tvae_heom_any_eps_015  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.15')
    credit_tvae_heom_any_eps_02  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.2')
    credit_tvae_heom_any_eps_025  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.25')
    credit_tvae_heom_any_eps_03  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.3')
    credit_tvae_heom_any_eps_035  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.35')
    credit_tvae_heom_any_eps_04  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.4')

    dict_results_credit_ctgan_heom_any  = {
        'none' : credit_ctgan,
        '04' : credit_ctgan_heom_any_eps_04,
        '035' : credit_ctgan_heom_any_eps_035,
        '03' : credit_ctgan_heom_any_eps_03,
        '025' : credit_ctgan_heom_any_eps_025,
        '02' : credit_ctgan_heom_any_eps_02,
        '015' : credit_ctgan_heom_any_eps_015,
        '01' : credit_ctgan_heom_any_eps_01,
        '005' : credit_ctgan_heom_any_eps_005,
        '001' : credit_ctgan_heom_any_eps_001,
        '0005' : credit_ctgan_heom_any_eps_0005,    
    }

    dict_results_credit_tvae_heom_any  = {
        'none' : credit_tvae,
        '04' : credit_tvae_heom_any_eps_04,
        '0.35' : credit_tvae_heom_any_eps_035,
        '03' :credit_tvae_heom_any_eps_03,
        '025' :credit_tvae_heom_any_eps_025,
        '02' :credit_tvae_heom_any_eps_02,
        '015' :credit_tvae_heom_any_eps_015,
        '01' :credit_tvae_heom_any_eps_01,
        '005' :credit_tvae_heom_any_eps_005,
        '001' :credit_tvae_heom_any_eps_001,
        '0005' :credit_tvae_heom_any_eps_0005
    }

    print("======CTGAN=======")
    scores = {}
    for key , results in dict_results_credit_ctgan_heom_any.items() :


        synth = results['generation_results']['synthetic_data']
        synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)


        def _decategorize(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col in out.select_dtypes(include=["category"]).columns:
                out[col] = out[col].astype(object)
            return out

        real_dp = _decategorize(df_train)
        synth_dp = _decategorize(synth)
        valid_dp = _decategorize(df_test)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df_train)

        # Make sure lists are lists
        known = list(QAI_COLUMNS)

        scores[key] = DCROverfittingProtection.compute_breakdown(
            real_training_data=real_dp,
            synthetic_data=synth_dp,
            real_validation_data=valid_dp,
            metadata=metadata.to_dict()
        )

        print(f"{key} : {scores[key]}")

    with open(f"../../../results/epsilon_disclosure_protection_comparison_heom_any/credit_ctgan_dcr_protection.pkl", "wb") as f:
        pickle.dump(scores, f)

    print("======TVAE=======")

    scores = {}
    for key , results in dict_results_credit_tvae_heom_any.items() :


        synth = results['generation_results']['synthetic_data']
        synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)


        def _decategorize(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col in out.select_dtypes(include=["category"]).columns:
                out[col] = out[col].astype(object)
            return out

        real_dp = _decategorize(df_train)
        synth_dp = _decategorize(synth)
        valid_dp = _decategorize(df_test)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df_train)

        # Make sure lists are lists
        known = list(QAI_COLUMNS)

        scores[key] = DCROverfittingProtection.compute_breakdown(
            real_training_data=real_dp,
            synthetic_data=synth_dp,
            real_validation_data=valid_dp,
            metadata=metadata.to_dict()
        )

        print(f"{key} : {scores[key]}")

    with open(f"../../../results/epsilon_disclosure_protection_comparison_heom_any/credit_tvae_dcr_protection.pkl", "wb") as f:
        pickle.dump(scores, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
