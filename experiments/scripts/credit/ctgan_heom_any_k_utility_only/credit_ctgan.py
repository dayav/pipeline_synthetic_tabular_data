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


    credit_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_ctgan_eps_0.01')

    credit_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any/credit_tvae_eps_0.01')




    classifiers = [
        ClassifierType.CART,
        ClassifierType.KNN, 
        ClassifierType.LDA, 
        ClassifierType.LR, 
        ClassifierType.NB, 
        ClassifierType.RANDOM_FOREST,
        ClassifierType.SVM,
        ClassifierType.XGBOOST]

    credit_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted/credit_ctgan_eps_0.01')
    synth = credit_ctgan_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_synthetic_test_real()

    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted/credit_ctgan_eps_001_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted/credit_ctgan_eps_001_tstr.pkl", "wb") as f:
        pickle.dump(utility_results, f)

    credit_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted/credit_tvae_eps_0.01')
    synth = credit_tvae_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_synthetic_test_real()

    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted/credit_tvae_eps_001_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted/credit_tvae_eps_001_tstr.pkl", "wb") as f:
        pickle.dump(utility_results, f)

    credit_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted_5/credit_ctgan_eps_0.01')
    synth = credit_ctgan_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_synthetic_test_real()

    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted_5/credit_ctgan_eps_001_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted_5/credit_ctgan_eps_001_tstr.pkl", "wb") as f:
        pickle.dump(utility_results, f)

    credit_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted_5/credit_tvae_eps_0.01')
    synth = credit_tvae_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_synthetic_test_real()

    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted_5/credit_tvae_eps_001_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted_5/credit_tvae_eps_001_tstr.pkl", "wb") as f:
        pickle.dump(utility_results, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

