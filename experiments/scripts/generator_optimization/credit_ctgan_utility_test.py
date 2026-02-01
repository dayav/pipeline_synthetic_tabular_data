#!/usr/bin/env python3

from data_loader import DataLoader
from data_synthesizer.sdv import SDVCTGAN_, SDVTVAE_
from data_evaluator import ClassifierType
from data_synthesizer.pipeline import PipelineBuilder
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import StratifiedKFold
import pickle
import pandas as pd


def main():

    cat_list_credit_card = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'default.payment.next.month']
    num_list_credit_card = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    credit_qai_columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE']
    credit_risk_column = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month']
    df_real_credit_card_train = DataLoader('../../data/credit_card_Train.csv').get_dataframe(cat_list_credit_card, str, drop_identation=True)
    df_real_credit_card_test = DataLoader('../../data/credit_card_Test.csv').get_dataframe(cat_list_credit_card, str, drop_identation=True)
    df_real_credit_card =  DataLoader(dataset=pd.concat([df_real_credit_card_train, df_real_credit_card_test], ignore_index=True)).get_dataframe( cat_list_credit_card, str)


    copy_data = df_real_credit_card.copy(deep=True)
    skf = StratifiedKFold(n_splits=5)  

    classifier_names = ['CART', 'KNN', 'LDA', 'NB', 'LR', 'Random Forest', 'SVM', 'XGBoost']
    classifier_accuracies = {name: [] for name in classifier_names}
    classifier_precisions = {name: [] for name in classifier_names}
    classifier_recalls = {name: [] for name in classifier_names}
    classifier_f1s = {name: [] for name in classifier_names}
    classifier_balanced_accuracies = {name: [] for name in classifier_names}
    classifier_precision_weighted = {name: [] for name in classifier_names}
    classifier_recall_weighted = {name: [] for name in classifier_names}
    classifier_f1_weighted = {name: [] for name in classifier_names}

    classifier_accuracies_real = {name: [] for name in classifier_names}
    classifier_precisions_real = {name: [] for name in classifier_names}
    classifier_recalls_real = {name: [] for name in classifier_names}
    classifier_f1s_real = {name: [] for name in classifier_names}
    classifier_balanced_accuracies_real = {name: [] for name in classifier_names}
    classifier_precision_weighted_real = {name: [] for name in classifier_names}
    classifier_recall_weighted_real = {name: [] for name in classifier_names}
    classifier_f1_weighted_real = {name: [] for name in classifier_names}
    
    

    for fold, (train_index, test_index) in enumerate(skf.split(copy_data, copy_data.iloc[:,-1:]), 1):

        df_real_credit_card_train, df_real_credit_card_test = copy_data.iloc[train_index], copy_data.iloc[test_index]


        print(f"Loading Model")
        metadata = SingleTableMetadata()

        metadata.detect_from_dataframe(data=df_real_credit_card_train)

        ctgan = SDVCTGAN_( metadata, df_real_credit_card_train, enforce_min_max_values=True, enforce_rounding=True, locales=None,
                    embedding_dim=128, generator_dim=[256, 256], discriminator_dim=[512, 512],
                    generator_lr=5.939177101221521e-05, generator_decay=1e-6, discriminator_lr=0.00019841154045204687,
                    discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                    log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, patience=400)
        # Your main code logic here
        print(f"Model loaded")

        print(f"Training pipeline")

        pipeline_builder = PipelineBuilder(df_real_credit_card_train, cat_list_credit_card, num_list_credit_card, ctgan)
        pipeline_builder.add_generation_task()
        # pipeline_builder.add_fine_tuning_generation_task()
        # pipeline_builder.add_ressemblance_evaluation_task(df_real_credit_card_test)
        classifier_types = [ClassifierType.CART, 
                            ClassifierType.KNN, 
                            ClassifierType.LDA, 
                            ClassifierType.NB, 
                            ClassifierType.LR, 
                            ClassifierType.RANDOM_FOREST,
                            ClassifierType.SVM,
                            ClassifierType.XGBOOST]
        pipeline_builder.add_utility_evaluation_task(df_real_credit_card_test, classifier_types)
        # pipeline_builder.add_privacy_evaluation_task(df_real_credit_card_test,credit_qai_columns, credit_risk_column)
        # pipeline_builder.add_privacy_anonymeter_evaluation_task(df_real_credit_card_test)
        pipeline_builder.build()
        results = pipeline_builder.run()

        print(f"pipeline trained")

        for name in classifier_names :
            classifier_accuracies[name].append(results['utility_evaluation_results']['train_synthetic_test_real_results']['accuracy'].loc[name])
            classifier_precisions[name].append(results['utility_evaluation_results']['train_synthetic_test_real_results']['precision'].loc[name])
            classifier_recalls[name].append(results['utility_evaluation_results']['train_synthetic_test_real_results']['recall'].loc[name])
            classifier_f1s[name].append(results['utility_evaluation_results']['train_synthetic_test_real_results']['f1'].loc[name])
            classifier_balanced_accuracies[name].append(results['utility_evaluation_results']['train_synthetic_test_real_results']['balanced_accuracy'].loc[name])
            classifier_precision_weighted[name].append(results['utility_evaluation_results']['train_synthetic_test_real_results']['precision_weighted'].loc[name])
            classifier_recall_weighted[name].append(results['utility_evaluation_results']['train_synthetic_test_real_results']['recall_weighted'].loc[name])
            classifier_f1_weighted[name].append(results['utility_evaluation_results']['train_synthetic_test_real_results']['f1_weighted'].loc[name])

            classifier_accuracies_real[name].append(results['utility_evaluation_results']['train_real_test_real_results']['accuracy'].loc[name])
            classifier_precisions_real[name].append(results['utility_evaluation_results']['train_real_test_real_results']['precision'].loc[name])
            classifier_recalls_real[name].append(results['utility_evaluation_results']['train_real_test_real_results']['recall'].loc[name])
            classifier_f1s_real[name].append(results['utility_evaluation_results']['train_real_test_real_results']['f1'].loc[name])
            classifier_balanced_accuracies_real[name].append(results['utility_evaluation_results']['train_real_test_real_results']['balanced_accuracy'].loc[name])
            classifier_precision_weighted_real[name].append(results['utility_evaluation_results']['train_real_test_real_results']['precision_weighted'].loc[name])
            classifier_recall_weighted_real[name].append(results['utility_evaluation_results']['train_real_test_real_results']['recall_weighted'].loc[name])
            classifier_f1_weighted_real[name].append(results['utility_evaluation_results']['train_real_test_real_results']['f1_weighted'].loc[name])
    
    classifier_metrics = {"accuracies": classifier_accuracies,
                        "precisions": classifier_precisions,
                        "recalls": classifier_recalls,
                        "f1s": classifier_f1s,
                        "balanced_accuracies": classifier_balanced_accuracies,
                        "precisions_weighted": classifier_precision_weighted,
                        "recalls_weighted": classifier_recall_weighted,
                        "f1s_weighted": classifier_f1_weighted,
                        "accuracies_real": classifier_accuracies_real,
                        "precisions_real": classifier_precisions_real,
                        "recalls_real": classifier_recalls_real,
                        "f1s_real": classifier_f1s_real,
                        "balanced_accuracies_real": classifier_balanced_accuracies_real,
                        "precisions_weighted_real": classifier_precision_weighted_real,
                        "recalls_weighted_real": classifier_recall_weighted_real,
                        "f1s_weighted_real": classifier_f1_weighted_real
                        }

    print(f"saving metrics")
    file_name = f"credit_ctgan_5kfold_optimized_300_early.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(classifier_metrics, file)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="A simple command-line Python script.")
    
    # # Adding arguments
    # parser.add_argument('-n', '--name', type=str, required=True, help="Name of the user")
    # parser.add_argument('-a', '--age', type=int, required=True, help="Age of the user")
    # parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose mode")

    # # Parse the arguments
    # args = parser.parse_args()
    
    # Call the main function
    main()
