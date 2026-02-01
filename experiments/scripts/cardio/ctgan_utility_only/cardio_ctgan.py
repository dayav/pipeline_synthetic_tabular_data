#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_evaluator.utility_evaluator_new import ClassifierType, UtilityEvaluation
from data_loader import DataLoader
from data_synthesizer.fit_aia_gard import fit_aia_guard_strong
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results


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

    classifiers = [
        ClassifierType.CART,
        ClassifierType.KNN, 
        ClassifierType.LDA, 
        ClassifierType.LR, 
        ClassifierType.NB, 
        ClassifierType.RANDOM_FOREST,
        ClassifierType.SVM,
        ClassifierType.XGBOOST]

    synth = cardio_ctgan['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CARDIO)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_real_test_real()

    with open("../../../results/epsilon_utility_comparison_heom_any/cardio_ctgan_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open("../../../results/epsilon_utility_comparison_heom_any/cardio_ctgan_trtr.pkl", "wb") as f:
        pickle.dump(utility_results, f)

    
    synth = cardio_tvae['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CARDIO)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_real_test_real()

    with open("../../../results/epsilon_utility_comparison_heom_any/cardio_tvae_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open("../../../results/epsilon_utility_comparison_heom_any/cardio_tvae_trtr.pkl", "wb") as f:
        pickle.dump(utility_results, f)

    for key, results in dict_results_cardio_ctgan_heom_any.items() :
        synth = results['generation_results']['synthetic_data']
        synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CARDIO)
        utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
        utility_results = utility_evaluator.train_synthetic_test_real()

        with open(f"../../../results/epsilon_utility_comparison_heom_any/cardio_ctgan_eps_{key}_utility_evaluator.pkl", "wb") as f:
            pickle.dump(utility_evaluator, f)
        with open(f"../../../results/epsilon_utility_comparison_heom_any/cardio_ctgan_eps_{key}_tstr.pkl", "wb") as f:
            pickle.dump(utility_results, f)

    for key, results in dict_results_cardio_tvae_heom_any.items() :
        synth = results['generation_results']['synthetic_data']
        synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CARDIO)
        utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
        utility_results = utility_evaluator.train_synthetic_test_real()

        with open(f"../../../results/epsilon_utility_comparison_heom_any/cardio_tvae_eps_{key}_utility_evaluator.pkl", "wb") as f:
            pickle.dump(utility_evaluator, f)
        with open(f"../../../results/epsilon_utility_comparison_heom_any/cardio_tvae_eps_{key}_tstr.pkl", "wb") as f:
            pickle.dump(utility_results, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

