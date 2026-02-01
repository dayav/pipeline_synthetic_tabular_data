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
    df_train = DataLoader("../../../../data/credit_card_Train.csv").get_dataframe(
        CAT_COLS_CREDIT,drop_identation=True
    )
    df_test = DataLoader("../../../../data/credit_card_Test.csv").get_dataframe(
    CAT_COLS_CREDIT,drop_identation=True
    )

    # credit_ctgan  = load_all_results('../../../results/baseline/credit_ctgan_baseline')
    # credit_tvae  = load_all_results('../../../results/baseline/credit_tvae_baseline')



    # credit_ctgan  = load_all_results('../../../results/baseline/credit_ctgan_baseline')
    # credit_tvae  = load_all_results('../../../results/baseline/credit_tvae_baseline')
    # credit_ctgan_heom_any_eps_0005  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.005')
    # credit_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.01')
    # credit_ctgan_heom_any_eps_005  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.05')
    # credit_ctgan_heom_any_eps_01  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.1')
    # credit_ctgan_heom_any_eps_015  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.15')
    # credit_ctgan_heom_any_eps_02  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.2')
    # credit_ctgan_heom_any_eps_025  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.25')
    # credit_ctgan_heom_any_eps_03  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.3')
    # credit_ctgan_heom_any_eps_035  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.35')
    # credit_ctgan_heom_any_eps_04  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.4')

    # credit_tvae_heom_any_eps_0005  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.005')
    # credit_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.01')
    # credit_tvae_heom_any_eps_005  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.05')
    # credit_tvae_heom_any_eps_01  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.1')
    # credit_tvae_heom_any_eps_015  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.15')
    # credit_tvae_heom_any_eps_02  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.2')
    # credit_tvae_heom_any_eps_025  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.25')
    # credit_tvae_heom_any_eps_03  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.3')
    credit_tvae_heom_any_eps_035  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.35')
    # credit_tvae_heom_any_eps_04  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.4')

    # dict_results_credit_ctgan_heom_any  = {
    #     'none' : credit_ctgan,
    #     '04' : credit_ctgan_heom_any_eps_04,
    #     '035' : credit_ctgan_heom_any_eps_035,
    #     '03' : credit_ctgan_heom_any_eps_03,
    #     '025' : credit_ctgan_heom_any_eps_025,
    #     '02' : credit_ctgan_heom_any_eps_02,
    #     '015' : credit_ctgan_heom_any_eps_015,
    #     '01' : credit_ctgan_heom_any_eps_01,
    #     '005' : credit_ctgan_heom_any_eps_005,
    #     '001' : credit_ctgan_heom_any_eps_001,
    #     '0005' : credit_ctgan_heom_any_eps_0005,    
    # }

    dict_results_credit_tvae_heom_any  = {
        # 'none' : credit_tvae,
        # '04' : credit_tvae_heom_any_eps_04,
        '0.35' : credit_tvae_heom_any_eps_035,
        # '03' :credit_tvae_heom_any_eps_03,
        # '025' :credit_tvae_heom_any_eps_025,
        # '02' :credit_tvae_heom_any_eps_02,
        # '015' :credit_tvae_heom_any_eps_015,
        # '01' :credit_tvae_heom_any_eps_01,
        # '005' :credit_tvae_heom_any_eps_005,
        # '001' :credit_tvae_heom_any_eps_001,
        # '0005' :credit_tvae_heom_any_eps_0005
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

    # synth = credit_ctgan['generation_results']['synthetic_data']
    # synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)
    # utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    # utility_results = utility_evaluator.train_real_test_real()

    # with open("../../../results/epsilon_utility_comparison_heom_any/credit_ctgan_utility_evaluator.pkl", "wb") as f:
    #     pickle.dump(utility_evaluator, f)
    # with open("../../../results/epsilon_utility_comparison_heom_any/credit_ctgan_trtr.pkl", "wb") as f:
    #     pickle.dump(utility_results, f)

    
    # synth = credit_tvae['generation_results']['synthetic_data']
    # synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)
    # utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    # utility_results = utility_evaluator.train_real_test_real()

    # with open("../../../results/epsilon_utility_comparison_heom_any/credit_tvae_utility_evaluator.pkl", "wb") as f:
    #     pickle.dump(utility_evaluator, f)
    # with open("../../../results/epsilon_utility_comparison_heom_any/credit_tvae_trtr.pkl", "wb") as f:
    #     pickle.dump(utility_results, f)

    # for key, results in dict_results_credit_ctgan_heom_any.items() :
    #     synth = results['generation_results']['synthetic_data']
    #     synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)
    #     utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    #     utility_results = utility_evaluator.train_synthetic_test_real()

    #     with open(f"../../../results/epsilon_utility_comparison_heom_any/credit_ctgan_eps_{key}_utility_evaluator.pkl", "wb") as f:
    #         pickle.dump(utility_evaluator, f)
    #     with open(f"../../../results/epsilon_utility_comparison_heom_any/credit_ctgan_eps_{key}_tstr.pkl", "wb") as f:
    #         pickle.dump(utility_results, f)

    for key, results in dict_results_credit_tvae_heom_any.items() :
        synth = results['generation_results']['synthetic_data']
        synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)
        utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
        utility_results = utility_evaluator.train_synthetic_test_real()

        with open(f"../../../results/epsilon_utility_comparison_heom_any/credit_tvae_eps_{key}_utility_evaluator.pkl", "wb") as f:
            pickle.dump(utility_evaluator, f)
        with open(f"../../../results/epsilon_utility_comparison_heom_any/credit_tvae_eps_{key}_tstr.pkl", "wb") as f:
            pickle.dump(utility_results, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

