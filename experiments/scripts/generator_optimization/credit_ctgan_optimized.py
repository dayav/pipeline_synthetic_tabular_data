#!/usr/bin/env python3

from data_loader import DataLoader
from data_synthesizer.sdv import SDVCTGAN_, SDVTVAE_
from data_evaluator import ClassifierType
from data_synthesizer.pipeline import PipelineBuilder
from sdv.metadata import SingleTableMetadata
import pickle


def main():

    cat_list_credit_card = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'default.payment.next.month']
    num_list_credit_card = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    credit_qai_columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE']
    credit_risk_column = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month']
    df_real_credit_card_train = DataLoader('../../data/credit_card_Train.csv').get_dataframe(cat_list_credit_card, str, drop_identation=True)
    df_real_credit_card_test = DataLoader('../../data/credit_card_Test.csv').get_dataframe(cat_list_credit_card, str, drop_identation=True)


    print(f"Loading Model")
    metadata = SingleTableMetadata()

    metadata.detect_from_dataframe(data=df_real_credit_card_train)
    ctgan = SDVCTGAN_( metadata, df_real_credit_card_train, enforce_min_max_values=True, enforce_rounding=True, locales=None,
            embedding_dim=128, generator_dim=[256,256], discriminator_dim=[512,512],
            generator_lr=5.939177101221521e-05, generator_decay=1e-6, discriminator_lr=0.00019841154045204687,
            discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                    log_frequency=True, verbose=False, epochs=1500, pac=10, cuda=True)
    print(f"Model loaded")

    print(f"Training pipeline")

    pipeline_builder = PipelineBuilder(df_real_credit_card_train, cat_list_credit_card, num_list_credit_card, ctgan)
    pipeline_builder.add_generation_task()
    # Your main code logic here
    pipeline_builder.add_fine_tuning_generation_task()
    # pipeline_builder.add_sampling_and_reject_task(eps)
    pipeline_builder.add_ressemblance_evaluation_task(df_real_credit_card_test)
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

    file_name = f"credit_ctgan_optimised.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(results, file)

    file_name = f"credit_ctgan_optimised_model.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(ctgan, file)


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
