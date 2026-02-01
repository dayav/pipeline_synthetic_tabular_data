#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_evaluator.privacy_evaluation.privacy_evaluator import PrivacyEvaluator
from data_loader import DataLoader
from data_synthesizer import sampling_reject_epsilon_tabnet_strong
from data_synthesizer.fit_aia_gard import fit_aia_guard_strong
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results
from privacy_sampling.embeddings import make_tabnet_embedder


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
    df_train = DataLoader("../../../data/credit_card_Train.csv").get_dataframe(
        CAT_COLS_CREDIT,drop_identation=True
    )
    df_test = DataLoader("../../../data/credit_card_Test.csv").get_dataframe(
        CAT_COLS_CREDIT,drop_identation=True
    )
    df_synth_aia = DataLoader("credit_aiagards.csv").get_dataframe(
        CAT_COLS_CREDIT)
    
    credit_ctgan  = load_all_results('../baseline/credit_ctgan_baseline')
    synth = credit_ctgan["generation_results"]["synthetic_data"]
    df_synth = DataLoader(dataset=synth).get_dataframe(
        CAT_COLS_CREDIT)
    
    
    df_synth = DataLoader.align_categoricals(df_synth, df_train, CAT_COLS_CREDIT)
    df_test = DataLoader.align_categoricals(df_test, df_train, CAT_COLS_CREDIT)
    df_synth_aia = DataLoader.align_categoricals(df_synth_aia, df_train, CAT_COLS_CREDIT)

    privacy_evaluator_worst_synth = PrivacyEvaluator(df_train, df_synth, df_test, None, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_real_credit_worst = privacy_evaluator_worst_synth.evaluate_attribute_real_prediction()
    _, _, _,_, predictions_dict_synth_credit_worst = privacy_evaluator_worst_synth.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value

    privacy_evaluator_worst_synth_aia = PrivacyEvaluator(df_train, df_synth_aia, df_test, None, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_credit_aia_worst = privacy_evaluator_worst_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value

    privacy_evaluator_qai_synth = PrivacyEvaluator(df_train, df_synth, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_real_credit_qai = privacy_evaluator_qai_synth.evaluate_attribute_real_prediction()
    _, _, _,_, predictions_dict_synth_credit_qai = privacy_evaluator_qai_synth.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value

    privacy_evaluator_qai_synth_aia = PrivacyEvaluator(df_train, df_synth_aia, df_test, QAI_COLUMNS, RISK_COLUMNS, "none")
    _, _, _,_, predictions_dict_synth_credit_aia_qai = privacy_evaluator_qai_synth_aia.evaluate_attribute_synthetic_prediction() #after test be sure to return the only rigth value


    with open("predictions_dict_real_credit_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_real_credit_worst, f)

    with open("predictions_dict_synth_credit_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_credit_worst, f)

    with open("predictions_dict_synth_credit_aia_worst.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_credit_aia_worst, f)

    with open("predictions_dict_real_credit_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_real_credit_qai, f)
        
    with open("predictions_dict_synth_credit_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_credit_qai, f)

    with open("predictions_dict_synth_credit_aia_qai.pkl", "wb") as f:
        pickle.dump(predictions_dict_synth_credit_aia_qai, f)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

