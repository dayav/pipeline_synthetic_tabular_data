#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_evaluator.privacy_evaluation.privacy_evaluator import PrivacyEvaluator
from data_loader import DataLoader
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

QAI_COLUMNS = ['SEX','EDUCATION','MARRIAGE','AGE']
RISK_COLUMNS = ['LIMIT_BAL', 'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month']


def load_dataset_from_results(result_path) :
    results  = load_all_results(result_path)
    synth = results["generation_results"]["synthetic_data"]
    return DataLoader(dataset=synth).get_dataframe(
        CAT_COLS_CREDIT)


def evaluate_synthetic_prediction(df_train, synth, df_test, eps) :
    privacy_evaluator__synth_aia = PrivacyEvaluator(df_train, synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_credit = privacy_evaluator__synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value
    with open(f"credit/credit_ctgan_attributes_synthetic_models_{eps}.pkl", "wb") as f:
        pickle.dump(privacy_evaluator__synth_aia.attributes_synthetic_models, f)
    return predictions_dict_synth_credit



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../../data/credit_card_Train.csv").get_dataframe(
        CAT_COLS_CREDIT,drop_identation=True)
    
    df_test = DataLoader("../../../../data/credit_card_Test.csv").get_dataframe(
        CAT_COLS_CREDIT,drop_identation=True)
    
    credit_ctgan_heom_any_eps_001  = load_all_results('../../../results/epsilon_comparison_heom_any_weighted_5/credit_ctgan_eps_0.01')
    synth = credit_ctgan_heom_any_eps_001['generation_results']['synthetic_data']
    synth = DataLoader.align_categoricals(synth, df_train, CAT_COLS_CREDIT)

    predictions_synth_credit_heom_any_credit_ctgan_eps_001 = evaluate_synthetic_prediction(df_train, synth, df_test, '001')
    with open("credit/predictions_synth_credit_heom_any_credit_ctgan_eps_001.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_ctgan_eps_001, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

