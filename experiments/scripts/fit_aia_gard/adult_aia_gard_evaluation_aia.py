#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_evaluator.privacy_evaluation.privacy_evaluator import PrivacyEvaluator
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
    df_train = DataLoader("../../../data/adult_train.csv").get_dataframe(
        CAT_COLS_ADULT)
    df_test = DataLoader("../../../data/adult_test.csv").get_dataframe(
        CAT_COLS_ADULT)
    df_synth_aia = DataLoader("adult_aiagards.csv").get_dataframe(
        CAT_COLS_ADULT)
    
    adult_ctgan  = load_all_results('../baseline/adult_ctgan_baseline')
    synth = adult_ctgan["generation_results"]["synthetic_data"]
    df_synth = DataLoader(dataset=synth).get_dataframe(
        CAT_COLS_ADULT)
    
    
    df_synth = DataLoader.align_categoricals(df_synth, df_train, CAT_COLS_ADULT)
    df_test = DataLoader.align_categoricals(df_test, df_train, CAT_COLS_ADULT)
    df_synth_aia = DataLoader.align_categoricals(df_synth_aia, df_train, CAT_COLS_ADULT)

    privacy_evaluator_worst_synth = PrivacyEvaluator(df_train, df_synth, df_test, None, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_real_adult_worst = privacy_evaluator_worst_synth.evaluate_attribute_real_prediction()
    _, _, _,_, predictions_dict_synth_adult_worst = privacy_evaluator_worst_synth.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value

    privacy_evaluator_worst_synth_aia = PrivacyEvaluator(df_train, df_synth, df_synth_aia, None, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_adult_aia_worst = privacy_evaluator_worst_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value

    privacy_evaluator_qai_synth = PrivacyEvaluator(df_train, df_synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_real_adult_qai = privacy_evaluator_qai_synth.evaluate_attribute_real_prediction()
    _, _, _,_, predictions_dict_synth_adult_qai = privacy_evaluator_qai_synth.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value

    privacy_evaluator_qai_synth_aia = PrivacyEvaluator(df_train, df_synth_aia, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_adult_aia_qai = privacy_evaluator_qai_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value


    with open("predictions_dict_real_adult_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_real_adult_worst, f)

    with open("predictions_dict_synth_adult_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_adult_worst, f)

    with open("predictions_dict_synth_adult_aia_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_adult_aia_worst, f)

    with open("predictions_dict_real_adult_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_real_adult_qai, f)
        
    with open("predictions_dict_synth_adult_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_adult_qai, f)

    with open("predictions_dict_synth_adult_aia_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_adult_aia_qai, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

