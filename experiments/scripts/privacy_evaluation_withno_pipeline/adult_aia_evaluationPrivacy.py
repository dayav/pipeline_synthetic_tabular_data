#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_evaluator.privacy_evaluation.privacy_evaluator import PrivacyEvaluator
from data_loader import DataLoader
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific configuration
# ──────────────────────────────────────────────────────────────────────────────
NUM_COLS_ADULT= ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
CAT_COLS_ADULT = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','income']

QAI_COLUMNS = ['education','education-num','marital-status','occupation','relationship','sex', 'native-country']
RISK_COLUMNS = ['capital-gain','capital-loss','hours-per-week','native-country','income', 'race']


def load_dataset_from_results(result_path) :
    results  = load_all_results(result_path)
    synth = results["generation_results"]["synthetic_data"]
    return DataLoader(dataset=synth).get_dataframe(
        CAT_COLS_ADULT)

def evaluate_synthetic_prediction_worst(df_train, synth, df_test) :
    privacy_evaluator_worst_synth_aia = PrivacyEvaluator(df_train, synth, df_test, None, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_adult = privacy_evaluator_worst_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value
    return predictions_dict_synth_adult

def evaluate_synthetic_prediction(df_train, synth, df_test) :
    privacy_evaluator_worst_synth_aia = PrivacyEvaluator(df_train, synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_adult = privacy_evaluator_worst_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value
    return predictions_dict_synth_adult



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../data/adult_train.csv").get_dataframe(
        CAT_COLS_ADULT)
    
    df_test = DataLoader("../../../data/adult_test.csv").get_dataframe(
        CAT_COLS_ADULT)
    
    df_synth = load_dataset_from_results('../baseline/adult_ctgan_baseline')
    
    # adult_ctgan_aia_1  = load_dataset_from_results('../aia/adult_ctgan_aia_1')
    # # adult_ctgan_aia_no_target  = load_dataset_from_results('../aia/adult_ctgan_aia_no_target')
    # adult_ctgan_aia_no_target_teacher_quota_1 = load_dataset_from_results('../aia/adult_ctgan_aia_teacher_quota_1')
    # adult_ctgan_aia_no_target_teacher_quota_2 = load_dataset_from_results('../aia/adult_ctgan_aia_teacher_quota_2')
    adult_ctgan_aia_no_target_teacher_quota_4 = load_dataset_from_results('../aia/adult_ctgan_aia_teacher_quota_4')

    df_synth = DataLoader.align_categoricals(df_synth, df_train, CAT_COLS_ADULT)
    df_test = DataLoader.align_categoricals(df_test, df_train, CAT_COLS_ADULT)

    # adult_ctgan_aia_1 = DataLoader.align_categoricals(adult_ctgan_aia_1, df_train, CAT_COLS_ADULT)
    # adult_ctgan_aia_no_target = DataLoader.align_categoricals(adult_ctgan_aia_no_target, df_train, CAT_COLS_ADULT)
    # adult_ctgan_aia_no_target_teacher_quota_1 = DataLoader.align_categoricals(adult_ctgan_aia_no_target_teacher_quota_1, df_train, CAT_COLS_ADULT)
    # adult_ctgan_aia_no_target_teacher_quota_2 = DataLoader.align_categoricals(adult_ctgan_aia_no_target_teacher_quota_2, df_train, CAT_COLS_ADULT)
    adult_ctgan_aia_no_target_teacher_quota_4 = DataLoader.align_categoricals(adult_ctgan_aia_no_target_teacher_quota_4, df_train, CAT_COLS_ADULT)


    # predictions_synth_adult_ctgan_aia_worst_1 = evaluate_synthetic_prediction_worst(df_train, adult_ctgan_aia_1, df_test)
    # with open("predictions_synth_adult_ctgan_aia_worst_1.pkl", "wb") as f:
    #     pickle.dump(predictions_synth_adult_ctgan_aia_worst_1, f)

    # predictions_synth_adult_ctgan_aia_qai_1 = evaluate_synthetic_prediction(df_train, adult_ctgan_aia_1, df_test)
    # with open("predictions_synth_adult_ctgan_aia_qai_1.pkl", "wb") as f:
    #     pickle.dump(predictions_synth_adult_ctgan_aia_qai_1, f)

    # predictions_synth_adult_ctgan_aia_no_target_worst_1 = evaluate_synthetic_prediction_worst(df_train, adult_ctgan_aia_no_target, df_test)
    # with open("predictions_synth_adult_ctgan_aia_no_target_worst_1.pkl", "wb") as f:
    #     pickle.dump(predictions_synth_adult_ctgan_aia_no_target_worst_1, f)

    # predictions_synth_adult_ctgan_aia_qai_1 = evaluate_synthetic_prediction(df_train, adult_ctgan_aia_no_target, df_test)
    # with open("predictions_synth_adult_ctgan_aia_qai_1.pkl", "wb") as f:
    #     pickle.dump(predictions_synth_adult_ctgan_aia_qai_1, f)

    # predictions_synth_adult_ctgan_aia_no_target_teacher_quota_worst_1 = evaluate_synthetic_prediction_worst(df_train, adult_ctgan_aia_no_target_teacher_quota_1, df_test)
    # with open("predictions_synth_adult_ctgan_aia_no_target_teacher_quota_worst_1.pkl", "wb") as f:
    #     pickle.dump(predictions_synth_adult_ctgan_aia_no_target_teacher_quota_worst_1, f)

    # predictions_synth_adult_ctgan_aia_no_target_teacher_quota_aia_qai_1 = evaluate_synthetic_prediction(df_train, adult_ctgan_aia_no_target_teacher_quota_1, df_test)
    # with open("predictions_synth_adult_ctgan_aia_no_target_teacher_quota_aia_qai_1.pkl", "wb") as f:
    #     pickle.dump(predictions_synth_adult_ctgan_aia_no_target_teacher_quota_aia_qai_1, f)

    # predictions_synth_adult_ctgan_aia_no_target_teacher_quota_worst_2 = evaluate_synthetic_prediction_worst(df_train, adult_ctgan_aia_no_target_teacher_quota_2, df_test)
    # with open("predictions_synth_adult_ctgan_aia_no_target_teacher_quota_worst_2_last.pkl", "wb") as f:
    #     pickle.dump(predictions_synth_adult_ctgan_aia_no_target_teacher_quota_worst_2, f)

    # predictions_synth_adult_ctgan_aia_no_target_teacher_quota_aia_qai_2 = evaluate_synthetic_prediction(df_train, adult_ctgan_aia_no_target_teacher_quota_2, df_test)
    # with open("predictions_synth_adult_ctgan_aia_no_target_teacher_quota_aia_qai_2_last.pkl", "wb") as f:
    #     pickle.dump(predictions_synth_adult_ctgan_aia_no_target_teacher_quota_aia_qai_2, f)    

    predictions_synth_adult_ctgan_aia_no_target_teacher_quota_worst_4 = evaluate_synthetic_prediction_worst(df_train, adult_ctgan_aia_no_target_teacher_quota_4, df_test)
    with open("predictions_synth_adult_ctgan_aia_no_target_teacher_quota_worst_4.pkl", "wb") as f:
        pickle.dump(predictions_synth_adult_ctgan_aia_no_target_teacher_quota_worst_4, f)

    predictions_synth_adult_ctgan_aia_no_target_teacher_quota_aia_qai_4 = evaluate_synthetic_prediction(df_train, adult_ctgan_aia_no_target_teacher_quota_4, df_test)
    with open("predictions_synth_adult_ctgan_aia_no_target_teacher_quota_aia_qai_4.pkl", "wb") as f:
        pickle.dump(predictions_synth_adult_ctgan_aia_no_target_teacher_quota_aia_qai_4, f)   


def main() -> None:
    run()


if __name__ == "__main__":
    main()

