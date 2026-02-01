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

    
    classifiers = [
        ClassifierType.CART,
        ClassifierType.KNN, 
        ClassifierType.LDA, 
        ClassifierType.LR, 
        ClassifierType.NB, 
        ClassifierType.RANDOM_FOREST,
        ClassifierType.SVM,
        ClassifierType.XGBOOST]


    adult_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted/adult_ctgan_eps_0.01')
    synth = adult_ctgan_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_ADULT)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_synthetic_test_real()

    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted/adult_ctgan_eps_001_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted/adult_ctgan_eps_001_tstr.pkl", "wb") as f:
        pickle.dump(utility_results, f)

    adult_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted/adult_tvae_eps_0.01')
    synth = adult_tvae_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_ADULT)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_synthetic_test_real()

    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted/adult_tvae_eps_001_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted/adult_tvae_eps_001_tstr.pkl", "wb") as f:
        pickle.dump(utility_results, f)

    adult_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted_5/adult_ctgan_eps_0.01')
    synth = adult_ctgan_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_ADULT)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_synthetic_test_real()

    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted_5/adult_ctgan_eps_001_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted_5/adult_ctgan_eps_001_tstr.pkl", "wb") as f:
        pickle.dump(utility_results, f)

    adult_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted_5/adult_tvae_eps_0.01')
    synth = adult_tvae_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_ADULT)
    utility_evaluator = UtilityEvaluation(df_train, synth, df_test, classifiers)
    utility_results = utility_evaluator.train_synthetic_test_real()

    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted_5/adult_tvae_eps_001_utility_evaluator.pkl", "wb") as f:
        pickle.dump(utility_evaluator, f)
    with open(f"../../../results/epsilon_utility_comparison_heom_any_weighted_5/adult_tvae_eps_001_tstr.pkl", "wb") as f:
        pickle.dump(utility_results, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

