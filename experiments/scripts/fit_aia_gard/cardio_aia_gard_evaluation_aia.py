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


# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../data/cardio_train.csv").get_dataframe(
        CAT_COLS_CARDIO)
    
    df_test = DataLoader("../../../data/cardio_test.csv").get_dataframe(
        CAT_COLS_CARDIO)
    df_synth_aia = DataLoader("cardio_aiagards.csv").get_dataframe(
        CAT_COLS_CARDIO)
    
    cardio_ctgan  = load_all_results('../baseline/cardio_ctgan_baseline')
    synth = cardio_ctgan["generation_results"]["synthetic_data"]
    df_synth = DataLoader(dataset=synth).get_dataframe(
        CAT_COLS_CARDIO)
    
    
    df_synth = DataLoader.align_categoricals(df_synth, df_train, CAT_COLS_CARDIO)
    df_test = DataLoader.align_categoricals(df_test, df_train, CAT_COLS_CARDIO)
    df_synth_aia = DataLoader.align_categoricals(df_synth_aia, df_train, CAT_COLS_CARDIO)

    privacy_evaluator_worst_synth = PrivacyEvaluator(df_train, df_synth, df_test, None, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_real_cardio_worst = privacy_evaluator_worst_synth.evaluate_attribute_real_prediction()
    _, _, _,_, predictions_dict_synth_cardio_worst = privacy_evaluator_worst_synth.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value

    privacy_evaluator_worst_synth_aia = PrivacyEvaluator(df_train, df_synth_aia, df_test, None, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_cardio_aia_worst = privacy_evaluator_worst_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value

    privacy_evaluator_qai_synth = PrivacyEvaluator(df_train, df_synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_real_cardio_qai = privacy_evaluator_qai_synth.evaluate_attribute_real_prediction()
    _, _, _,_, predictions_dict_synth_cardio_qai = privacy_evaluator_qai_synth.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value

    privacy_evaluator_qai_synth_aia = PrivacyEvaluator(df_train, df_synth_aia, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_cardio_aia_qai = privacy_evaluator_qai_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value


    with open("predictions_dict_real_cardio_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_real_cardio_worst, f)

    with open("predictions_dict_synth_cardio_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_cardio_worst, f)

    with open("predictions_dict_synth_cardio_aia_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_cardio_aia_worst, f)

    with open("predictions_dict_real_cardio_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_real_cardio_qai, f)
        
    with open("predictions_dict_synth_cardio_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_cardio_qai, f)

    with open("predictions_dict_synth_cardio_aia_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_cardio_aia_qai, f)

def main() -> None:
    run()


if __name__ == "__main__":
    main()

