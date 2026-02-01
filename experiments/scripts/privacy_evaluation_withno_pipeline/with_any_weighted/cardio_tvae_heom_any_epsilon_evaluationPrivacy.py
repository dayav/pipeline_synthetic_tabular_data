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

def evaluate_synthetic_prediction(df_train, synth, df_test, eps) :
    privacy_evaluator__synth_aia = PrivacyEvaluator(df_train, synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_cardio = privacy_evaluator__synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value
    with open(f"cardio/cardio_tvae_attributes_synthetic_models_{eps}.pkl", "wb") as f:
        pickle.dump(privacy_evaluator__synth_aia.attributes_synthetic_models, f)
    return predictions_dict_synth_cardio



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../../data/cardio_train.csv").get_dataframe(
        CAT_COLS_CARDIO)
    
    df_test = DataLoader("../../../../data/cardio_test.csv").get_dataframe(
        CAT_COLS_CARDIO)
    
    cardio_tvae_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted/cardio_tvae_eps_0.01')
    synth = cardio_tvae_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CARDIO)

    predictions_synth_cardio_heom_any_cardio_tvae_eps_001 = evaluate_synthetic_prediction(df_train, synth, df_test, '001')
    with open("cardio/predictions_synth_cardio_heom_any_cardio_tvae_eps_001.pkl", "wb") as f:
        pickle.dump(predictions_synth_cardio_heom_any_cardio_tvae_eps_001, f)
    

def main() -> None:
    run()


if __name__ == "__main__":
    main()

