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


def evaluate_synthetic_prediction(df_train, synth, df_test) :
    privacy_evaluator__synth_aia = PrivacyEvaluator(df_train, synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_credit = privacy_evaluator__synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value
    return predictions_dict_synth_credit



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../data/credit_card_Train.csv").get_dataframe(
        CAT_COLS_CREDIT,drop_identation=True)
    
    df_test = DataLoader("../../../data/credit_card_Test.csv").get_dataframe(
        CAT_COLS_CREDIT,drop_identation=True)
    
    df_synth = load_dataset_from_results('../baseline/credit_tvae_baseline')
    
    credit_tvae_eps_0005  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.005')
    credit_tvae_eps_001  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.01')
    credit_tvae_eps_005  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.05')
    credit_tvae_eps_01  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.1')
    credit_tvae_eps_015  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.15')
    credit_tvae_eps_02  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.2')
    credit_tvae_eps_025  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.25')
    credit_tvae_eps_03  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.3')
    # credit_tvae_eps_035  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.35')
    credit_tvae_eps_04  = load_dataset_from_results('../epsilon_comparison_heom_any/credit_tvae_eps_0.4')
    
    
    df_synth = DataLoader.align_categoricals(df_synth, df_train, CAT_COLS_CREDIT)
    df_test = DataLoader.align_categoricals(df_test, df_train, CAT_COLS_CREDIT)
    credit_tvae_eps_0005 = DataLoader.align_categoricals(credit_tvae_eps_0005, df_train, CAT_COLS_CREDIT)
    credit_tvae_eps_001 = DataLoader.align_categoricals(credit_tvae_eps_001, df_train, CAT_COLS_CREDIT)
    credit_tvae_eps_005 = DataLoader.align_categoricals(credit_tvae_eps_005, df_train, CAT_COLS_CREDIT)
    credit_tvae_eps_01 = DataLoader.align_categoricals(credit_tvae_eps_01, df_train, CAT_COLS_CREDIT)
    credit_tvae_eps_015 = DataLoader.align_categoricals(credit_tvae_eps_015, df_train, CAT_COLS_CREDIT)
    credit_tvae_eps_02 = DataLoader.align_categoricals(credit_tvae_eps_02, df_train, CAT_COLS_CREDIT)
    credit_tvae_eps_025 = DataLoader.align_categoricals(credit_tvae_eps_025, df_train, CAT_COLS_CREDIT)
    credit_tvae_eps_03 = DataLoader.align_categoricals(credit_tvae_eps_03, df_train, CAT_COLS_CREDIT)
    # credit_tvae_eps_035 = DataLoader.align_categoricals(credit_tvae_eps_035, df_train, CAT_COLS_CREDIT)
    credit_tvae_eps_04 = DataLoader.align_categoricals(credit_tvae_eps_04, df_train, CAT_COLS_CREDIT)


    privacy_evaluator_qai_synth = PrivacyEvaluator(df_train, df_synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_real_credit_heom_any_qai = privacy_evaluator_qai_synth.evaluate_attribute_real_prediction()
    with open("predictions_dict_real_credit_heom_any_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_real_credit_heom_any_qai, f)

    _, _, _,_, predictions_dict_synth_credit_heom_any_qai = privacy_evaluator_qai_synth.evaluate_attribute_synthetic_prediction()
    with open("predictions_dict_synth_credit_heom_any_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_credit_heom_any_qai, f)


    predictions_synth_credit_heom_any_credit_tvae_eps_0005 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_0005, df_test)
    with open("predictions_synth_credit_heom_any_credit_tvae_eps_0005.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_0005, f)


    predictions_synth_credit_heom_any_credit_tvae_eps_001 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_001, df_test)
    with open("predictions_synth_credit_heom_any_credit_tvae_eps_001.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_001, f)


    predictions_synth_credit_heom_any_credit_tvae_eps_005 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_005, df_test)
    with open("predictions_synth_credit_heom_any_credit_tvae_eps_005.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_005, f)


    predictions_synth_credit_heom_any_credit_tvae_eps_01 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_01, df_test)
    with open("predictions_synth_credit_heom_any_credit_tvae_eps_01.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_01, f)

    
    predictions_synth_credit_heom_any_credit_tvae_eps_015 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_015, df_test)
    with open("predictions_synth_credit_heom_any_credit_tvae_eps_015.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_015, f)

    predictions_synth_credit_heom_any_credit_tvae_eps_02 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_02, df_test)
    with open("predictions_synth_credit_heom_any_credit_tvae_eps_02.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_02, f)

    predictions_synth_credit_heom_any_credit_tvae_eps_025 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_025, df_test)
    with open("predictions_synth_credit_heom_any_credit_tvae_eps_025.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_025, f) 


    predictions_synth_credit_heom_any_credit_tvae_eps_03 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_03, df_test)
    with open("predictions_synth_credit_heom_any_credit_tvae_eps_03.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_03, f)       

    
    # predictions_synth_credit_heom_any_credit_tvae_eps_035 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_035, df_test)
    # with open("predictions_synth_credit_heom_any_credit_tvae_eps_035.pkl", "wb") as f:
    #     pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_035, f) 


    predictions_synth_credit_heom_any_credit_tvae_eps_04 = evaluate_synthetic_prediction(df_train, credit_tvae_eps_04, df_test)
    with open("predictions_synth_credit_heom_any_credit_tvae_eps_04.pkl", "wb") as f:
        pickle.dump(predictions_synth_credit_heom_any_credit_tvae_eps_04, f)          
    

def main() -> None:
    run()


if __name__ == "__main__":
    main()

