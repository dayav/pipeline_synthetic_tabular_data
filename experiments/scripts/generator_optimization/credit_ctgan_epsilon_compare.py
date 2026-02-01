#!/usr/bin/env python3

from data_loader import DataLoader
from data_synthesizer.pipeline.pipeline_results import PipelineResults
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results
from data_synthesizer.sdv import SDVCTGAN_, SDVTVAE_
from data_evaluator import ClassifierType
from data_synthesizer.pipeline import PipelineBuilder, save_all_results
from sdv.metadata import SingleTableMetadata
import pickle




import argparse



def main(eps):

    cat_list_credit_card = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'default.payment.next.month']
    num_list_credit_card = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    credit_qai_columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE']
    credit_risk_column = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month']
    df_real_credit_card_train = DataLoader('../../data/credit_card_Train.csv').get_dataframe(cat_list_credit_card, str, drop_identation=True)
    df_real_credit_card_test = DataLoader('../../data/credit_card_Test.csv').get_dataframe(cat_list_credit_card, str, drop_identation=True)

    #take the sytnhetic data from all these generation
    results_none  = PipelineResults()
    results_0_3  = PipelineResults()
    results_0_15 = PipelineResults()
    results_0_25 = PipelineResults()
    load_all_results('./credit_ctgan_eps_none', results_none)
    load_all_results('./credit_ctgan_eps_0.3', results_0_3)
    load_all_results('./credit_ctgan_eps_0.15', results_0_15)
    load_all_results('./credit_ctgan_eps_0.25', results_0_25)
    print('epsilon - : ', eps)
    print('===========================================================: ')
    number = 0
    for result in [results_none, results_0_3, results_0_25, results_0_15] :

        print()

        print(f"Loading Model")

        ctgan = result['generation_results']['generator_model']['generator_model']
        ctgan.reset_sampling()
        synth_data = ctgan.sample(len(df_real_credit_card_train))
        # Your main code logic here
        print(f"Model loaded")

        print(f"Training pipeline")

        pipeline_builder = PipelineBuilder(df_real_credit_card_train, cat_list_credit_card, num_list_credit_card, ctgan)
        # pipeline_builder.add_sampling_and_reject_task(eps, synth_data)
        pipeline_builder.add_ressemblance_evaluation_task(df_real_credit_card_test, synth_data)
        classifier_types = [ClassifierType.CART, 
                            ClassifierType.KNN, 
                            ClassifierType.LDA, 
                            ClassifierType.NB, 
                            ClassifierType.LR, 
                            ClassifierType.RANDOM_FOREST,
                            ClassifierType.SVM,
                            ClassifierType.XGBOOST]
        pipeline_builder.add_utility_evaluation_task(df_real_credit_card_test, classifier_types)
        pipeline_builder.add_privacy_evaluation_task(df_real_credit_card_test,credit_qai_columns, credit_risk_column)
        pipeline_builder.add_privacy_anonymeter_evaluation_task(df_real_credit_card_test)
        pipeline_builder.build()
        results = pipeline_builder.run()

        print(f"pipeline trained")

        print(f"saving pipeline")
        folder_name = f"credit_ctgan_eps_none_detailed_{number}.pkl"
        save_all_results(results, folder_name)
        print('===========================================================: ')
        number += 1


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="A simple command-line Python script.")
    
    # # Adding arguments
    # parser.add_argument('-n', '--name', type=str, required=True, help="Name of the user")
    # parser.add_argument('-a', '--age', type=int, required=True, help="Age of the user")
    # parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose mode")

    # # Parse the arguments
    # args = parser.parse_args()
    
    # Call the main function
    for eps in range(0.1, 1.3, 0.2) :
        main(eps)
