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
NUM_COLS_ADULT= ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
CAT_COLS_ADULT = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','income']

QAI_COLUMNS = ['education','education-num','marital-status','occupation','relationship','sex', 'native-country']
RISK_COLUMNS = ['capital-gain','capital-loss','hours-per-week','native-country','income', 'race']



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../../data/adult_train.csv").get_dataframe(
        CAT_COLS_ADULT)
    df_test = DataLoader("../../../../data/adult_test.csv").get_dataframe(
    CAT_COLS_ADULT)


    adult_ctgan  = load_all_results('../../../results/baseline/adult_ctgan_baseline')
    adult_tvae  = load_all_results('../../../results/baseline/adult_tvae_baseline')
    adult_ctgan_heom_any_eps_0005  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.005')
    adult_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.01')
    adult_ctgan_heom_any_eps_005  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.05')
    adult_ctgan_heom_any_eps_01  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.1')
    adult_ctgan_heom_any_eps_015  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.15')
    adult_ctgan_heom_any_eps_02  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.2')
    adult_ctgan_heom_any_eps_025  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.25')
    adult_ctgan_heom_any_eps_03  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.3')
    adult_ctgan_heom_any_eps_035  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.35')
    adult_ctgan_heom_any_eps_04  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_ctgan_eps_0.4')

    adult_tvae_heom_any_eps_0005  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.005')
    adult_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.01')
    adult_tvae_heom_any_eps_005  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.05')
    adult_tvae_heom_any_eps_01  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.1')
    adult_tvae_heom_any_eps_015  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.15')
    adult_tvae_heom_any_eps_02  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.2')
    adult_tvae_heom_any_eps_025  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.25')
    adult_tvae_heom_any_eps_03  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.3')
    adult_tvae_heom_any_eps_035  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.35')
    adult_tvae_heom_any_eps_04  = load_all_results('../../../results/epsilon_comparison_heom_any/adult_tvae_eps_0.4')

    dict_results_adult_ctgan_heom_any  = {
        'none' : adult_ctgan,
        '04' : adult_ctgan_heom_any_eps_04,
        '035' : adult_ctgan_heom_any_eps_035,
        '03' : adult_ctgan_heom_any_eps_03,
        '025' : adult_ctgan_heom_any_eps_025,
        '02' : adult_ctgan_heom_any_eps_02,
        '015' : adult_ctgan_heom_any_eps_015,
        '01' : adult_ctgan_heom_any_eps_01,
        '005' : adult_ctgan_heom_any_eps_005,
        '001' : adult_ctgan_heom_any_eps_001,
        '0005' : adult_ctgan_heom_any_eps_0005,    
    }

    dict_results_adult_tvae_heom_any  = {
        'none' : adult_tvae,
        '04' : adult_tvae_heom_any_eps_04,
        '035' : adult_tvae_heom_any_eps_035,
        '03' :adult_tvae_heom_any_eps_03,
        '025' :adult_tvae_heom_any_eps_025,
        '02' :adult_tvae_heom_any_eps_02,
        '015' :adult_tvae_heom_any_eps_015,
        '01' :adult_tvae_heom_any_eps_01,
        '005' :adult_tvae_heom_any_eps_005,
        '001' :adult_tvae_heom_any_eps_001,
        '0005' :adult_tvae_heom_any_eps_0005
    }

    print("======CTGAN=======")
    for sensitive_feat in RISK_COLUMNS :
        scores = {}
        print(f"============={sensitive_feat}================")
        for key , results in dict_results_adult_ctgan_heom_any.items() :

            print(f"============={sensitive_feat} - {key}===============")

            synth = results['generation_results']['synthetic_data']
            synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_ADULT)


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
                continuous_column_names=NUM_COLS_ADULT
            )

        with open(f"../../../results/epsilon_disclosure_protection_comparison_heom_any/adult_ctgan_disclosure_protection_{sensitive_feat}.pkl", "wb") as f:
            pickle.dump(scores, f)

    print("======TVAE=======")

    for sensitive_feat in RISK_COLUMNS :
        scores = {}

        print(f"============={sensitive_feat}================")

        for key , results in dict_results_adult_tvae_heom_any.items() :

            print(f"============={sensitive_feat} - {key}===============")

            synth = results['generation_results']['synthetic_data']
            synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_ADULT)


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
                continuous_column_names=NUM_COLS_ADULT
            )

        with open(f"../../../results/epsilon_disclosure_protection_comparison_heom_any/adult_tvae_disclosure_protection_{sensitive_feat}.pkl", "wb") as f:
            pickle.dump(scores, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
