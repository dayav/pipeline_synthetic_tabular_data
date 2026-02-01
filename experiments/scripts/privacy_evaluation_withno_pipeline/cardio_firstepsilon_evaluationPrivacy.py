#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_evaluator.privacy_evaluation.privacy_evaluator import PrivacyEvaluator
from data_loader import DataLoader
from data_synthesizer import sampling_reject_epsilon_tabnet_strong
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results
from privacy_sampling.embeddings import make_tabnet_embedder


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

def load_dataset_from_results(result_path) :
    results  = load_all_results(result_path)
    synth = results["generation_results"]["synthetic_data"]
    return DataLoader(dataset=synth).get_dataframe(
        CAT_COLS_CARDIO)

def evaluate_synthetic_prediction_worst(df_train, synth, df_test) :
    privacy_evaluator_worst_synth_aia = PrivacyEvaluator(df_train, synth, df_test, None, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_cardio = privacy_evaluator_worst_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value
    return predictions_dict_synth_cardio

def evaluate_synthetic_prediction(df_train, synth, df_test) :
    privacy_evaluator_worst_synth_aia = PrivacyEvaluator(df_train, synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_cardio = privacy_evaluator_worst_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value
    return predictions_dict_synth_cardio



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../data/cardio_train.csv").get_dataframe(
        CAT_COLS_CARDIO)
    
    df_test = DataLoader("../../../data/cardio_test.csv").get_dataframe(
        CAT_COLS_CARDIO)
    
    df_synth = load_dataset_from_results('../baseline/cardio_ctgan_baseline')
    
    cardio_ctgan_eps_001  = load_dataset_from_results('../epsilon_comparison/cardio_ctgan_eps_0.01')
    cardio_ctgan_eps_005  = load_dataset_from_results('../epsilon_comparison/cardio_ctgan_eps_0.05')
    cardio_ctgan_eps_01  = load_dataset_from_results('../epsilon_comparison/cardio_ctgan_eps_0.1')
    cardio_ctgan_eps_015  = load_dataset_from_results('../epsilon_comparison/cardio_ctgan_eps_0.15')
    cardio_ctgan_eps_02  = load_dataset_from_results('../epsilon_comparison/cardio_ctgan_eps_0.2')
    cardio_ctgan_eps_025  = load_dataset_from_results('../epsilon_comparison/cardio_ctgan_eps_0.25')
    cardio_ctgan_eps_03  = load_dataset_from_results('../epsilon_comparison/cardio_ctgan_eps_0.3')
    cardio_ctgan_eps_035  = load_dataset_from_results('../epsilon_comparison/cardio_ctgan_eps_0.35')
    
    
    df_synth = DataLoader.align_categoricals(df_synth, df_train, CAT_COLS_CARDIO)
    df_test = DataLoader.align_categoricals(df_test, df_train, CAT_COLS_CARDIO)
    cardio_ctgan_eps_001 = DataLoader.align_categoricals(cardio_ctgan_eps_001, df_train, CAT_COLS_CARDIO)
    cardio_ctgan_eps_005 = DataLoader.align_categoricals(cardio_ctgan_eps_005, df_train, CAT_COLS_CARDIO)
    cardio_ctgan_eps_01 = DataLoader.align_categoricals(cardio_ctgan_eps_01, df_train, CAT_COLS_CARDIO)
    cardio_ctgan_eps_015 = DataLoader.align_categoricals(cardio_ctgan_eps_015, df_train, CAT_COLS_CARDIO)
    cardio_ctgan_eps_02 = DataLoader.align_categoricals(cardio_ctgan_eps_02, df_train, CAT_COLS_CARDIO)
    cardio_ctgan_eps_025 = DataLoader.align_categoricals(cardio_ctgan_eps_025, df_train, CAT_COLS_CARDIO)
    cardio_ctgan_eps_03 = DataLoader.align_categoricals(cardio_ctgan_eps_03, df_train, CAT_COLS_CARDIO)
    cardio_ctgan_eps_035 = DataLoader.align_categoricals(cardio_ctgan_eps_035, df_train, CAT_COLS_CARDIO)

    privacy_evaluator_worst_synth = PrivacyEvaluator(df_train, df_synth, df_test, None, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_real_cardio_worst = privacy_evaluator_worst_synth.evaluate_attribute_real_prediction()

    with open("predictions_dict_real_cardio_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_real_cardio_worst, f)

    _, _, _,_, predictions_dict_synth_cardio_worst = privacy_evaluator_worst_synth.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value
    with open("predictions_dict_synth_cardio_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_cardio_worst, f)


    privacy_evaluator_qai_synth = PrivacyEvaluator(df_train, df_synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_real_cardio_qai = privacy_evaluator_qai_synth.evaluate_attribute_real_prediction()
    with open("predictions_dict_real_cardio_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_real_cardio_qai, f)

    _, _, _,_, predictions_dict_synth_cardio_qai = privacy_evaluator_qai_synth.evaluate_attribute_synthetic_prediction()
    with open("predictions_dict_synth_cardio_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_cardio_qai, f)

    predictions_synth_cardio_cardio_ctgan_eps_001_worst = evaluate_synthetic_prediction_worst(df_train, cardio_ctgan_eps_001, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_001_worst.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_001_worst, f)

    predictions_synth_cardio_cardio_ctgan_eps_001 = evaluate_synthetic_prediction(df_train, cardio_ctgan_eps_001, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_001.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_001, f)

    predictions_synth_cardio_cardio_ctgan_eps_005_worst = evaluate_synthetic_prediction_worst(df_train, cardio_ctgan_eps_005, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_005_worst.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_005_worst, f)

    predictions_synth_cardio_cardio_ctgan_eps_005 = evaluate_synthetic_prediction(df_train, cardio_ctgan_eps_005, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_005.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_005, f)

    predictions_synth_cardio_cardio_ctgan_eps_01_worst = evaluate_synthetic_prediction_worst(df_train, cardio_ctgan_eps_01, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_01_worst.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_01_worst, f)

    predictions_synth_cardio_cardio_ctgan_eps_01 = evaluate_synthetic_prediction(df_train, cardio_ctgan_eps_01, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_01.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_01, f)

    predictions_synth_cardio_cardio_ctgan_eps_015_worst = evaluate_synthetic_prediction_worst(df_train, cardio_ctgan_eps_015, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_015_worst.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_015_worst, f)
    
    predictions_synth_cardio_cardio_ctgan_eps_015 = evaluate_synthetic_prediction(df_train, cardio_ctgan_eps_015, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_015.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_015, f)

    predictions_synth_cardio_cardio_ctgan_eps_02_worst = evaluate_synthetic_prediction_worst(df_train, cardio_ctgan_eps_02, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_02_worst.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_02_worst, f)

    predictions_synth_cardio_cardio_ctgan_eps_02 = evaluate_synthetic_prediction(df_train, cardio_ctgan_eps_02, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_02.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_02, f)

    predictions_synth_cardio_cardio_ctgan_eps_025_worst = evaluate_synthetic_prediction_worst(df_train, cardio_ctgan_eps_025, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_025_worst.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_025_worst, f)

    predictions_synth_cardio_cardio_ctgan_eps_025 = evaluate_synthetic_prediction(df_train, cardio_ctgan_eps_025, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_025.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_025, f) 

    predictions_synth_cardio_cardio_ctgan_eps_03_worst = evaluate_synthetic_prediction_worst(df_train, cardio_ctgan_eps_03, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_03_worst.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_03_worst, f)

    predictions_synth_cardio_cardio_ctgan_eps_03 = evaluate_synthetic_prediction(df_train, cardio_ctgan_eps_03, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_03.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_03, f)

    predictions_synth_cardio_cardio_ctgan_eps_035_worst = evaluate_synthetic_prediction_worst(df_train, cardio_ctgan_eps_035, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_035_worst.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_035_worst, f)

    predictions_synth_cardio_cardio_ctgan_eps_035 = evaluate_synthetic_prediction(df_train, cardio_ctgan_eps_035, df_test)
    with open("predictions_synth_cardio_cardio_ctgan_eps_035.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_cardio_ctgan_eps_035, f)                 
    

def main() -> None:
    run()


if __name__ == "__main__":
    main()

