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
    df_train = DataLoader("../../../../data/cardio_train.csv").get_dataframe(
        CAT_COLS_CARDIO)
    df_test = DataLoader("../../../../data/cardio_test.csv").get_dataframe(
    CAT_COLS_CARDIO)


    cardio_ctgan  = load_all_results('../../../results/baseline/cardio_ctgan_baseline')
    cardio_tvae  = load_all_results('../../../results/baseline/cardio_tvae_baseline')
    cardio_ctgan_heom_any_eps_0005  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.005')
    cardio_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.01')
    cardio_ctgan_heom_any_eps_005  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.05')
    cardio_ctgan_heom_any_eps_01  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.1')
    cardio_ctgan_heom_any_eps_015  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.15')
    cardio_ctgan_heom_any_eps_02  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.2')
    cardio_ctgan_heom_any_eps_025  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.25')
    cardio_ctgan_heom_any_eps_03  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.3')
    cardio_ctgan_heom_any_eps_035  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.35')
    cardio_ctgan_heom_any_eps_04  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_ctgan_eps_0.4')

    cardio_tvae_heom_any_eps_0005  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.005')
    cardio_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.01')
    cardio_tvae_heom_any_eps_005  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.05')
    cardio_tvae_heom_any_eps_01  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.1')
    cardio_tvae_heom_any_eps_015  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.15')
    cardio_tvae_heom_any_eps_02  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.2')
    cardio_tvae_heom_any_eps_025  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.25')
    cardio_tvae_heom_any_eps_03  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.3')
    cardio_tvae_heom_any_eps_035  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.35')
    cardio_tvae_heom_any_eps_04  = load_all_results('../../../results/epsilon_comparison_heom_any/cardio_tvae_eps_0.4')

    dict_results_cardio_ctgan_heom_any  = {
        'none' : cardio_ctgan,
        '04' : cardio_ctgan_heom_any_eps_04,
        '035' : cardio_ctgan_heom_any_eps_035,
        '03' : cardio_ctgan_heom_any_eps_03,
        '025' : cardio_ctgan_heom_any_eps_025,
        '02' : cardio_ctgan_heom_any_eps_02,
        '015' : cardio_ctgan_heom_any_eps_015,
        '01' : cardio_ctgan_heom_any_eps_01,
        '005' : cardio_ctgan_heom_any_eps_005,
        '001' : cardio_ctgan_heom_any_eps_001,
        '0005' : cardio_ctgan_heom_any_eps_0005,    
    }

    dict_results_cardio_tvae_heom_any  = {
        'none' : cardio_tvae,
        '04' : cardio_tvae_heom_any_eps_04,
        '035' : cardio_tvae_heom_any_eps_035,
        '03' :cardio_tvae_heom_any_eps_03,
        '025' :cardio_tvae_heom_any_eps_025,
        '02' :cardio_tvae_heom_any_eps_02,
        '015' :cardio_tvae_heom_any_eps_015,
        '01' :cardio_tvae_heom_any_eps_01,
        '005' :cardio_tvae_heom_any_eps_005,
        '001' :cardio_tvae_heom_any_eps_001,
        '0005' :cardio_tvae_heom_any_eps_0005
    }


    print("======CTGAN=======")
    for sensitive_feat in RISK_COLUMNS :
        scores = {}
        print(f"============={sensitive_feat}================")
        for key , results in dict_results_cardio_ctgan_heom_any.items() :

            print(f"============={sensitive_feat} - {key}===============")

            synth = results['generation_results']['synthetic_data']
            synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CARDIO)


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
                continuous_column_names=NUM_COLS_CARDIO
            )

        with open(f"../../../results/epsilon_disclosure_protection_comparison_heom_any/cardio_ctgan_disclosure_protection_{sensitive_feat}.pkl", "wb") as f:
            pickle.dump(scores, f)

    print("======TVAE=======")

    for sensitive_feat in RISK_COLUMNS :
        scores = {}

        print(f"============={sensitive_feat}================")

        for key , results in dict_results_cardio_tvae_heom_any.items() :

            print(f"============={sensitive_feat} - {key}===============")

            synth = results['generation_results']['synthetic_data']
            synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CARDIO)


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
                continuous_column_names=NUM_COLS_CARDIO
            )

        with open(f"../../../results/epsilon_disclosure_protection_comparison_heom_any/cardio_tvae_disclosure_protection_{sensitive_feat}.pkl", "wb") as f:
            pickle.dump(scores, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
