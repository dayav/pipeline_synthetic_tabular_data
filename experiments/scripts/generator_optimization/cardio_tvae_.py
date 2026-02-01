#!/usr/bin/env python3

from data_loader import DataLoader
from data_synthesizer.sdv import SDVCTGAN_, SDVTVAE_
from data_evaluator import ClassifierType
from data_synthesizer.pipeline import PipelineBuilder
from sdv.metadata import SingleTableMetadata
import pickle




import argparse

def main():

    num_list_cardio = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    cat_list_cardio = ['gender','cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']

    cardio_qai_columns = ['age','gender','height','weight']
    cardio_risk_column = ['ap_lo','ap_hi','cholesterol','gluc','smoke','alco','active','cardio']

    df_real_cardio_train = DataLoader('../../data/cardio_train.csv').get_dataframe(cat_list_cardio, category_type=str, sep = ',')
    df_real_cardio_test = DataLoader('../../data/cardio_test.csv').get_dataframe(cat_list_cardio, category_type=str, sep = ',')


    print(f"Loading Model")
    metadata = SingleTableMetadata()

    metadata.detect_from_dataframe(data=df_real_cardio_train)
    ctgan = SDVTVAE_( metadata, df_real_cardio_train, enforce_min_max_values=True, enforce_rounding=True,
                    embedding_dim=256, compress_dims=[256, 512], decompress_dims=[256, 512],
                    l2scale=1e-5, batch_size=1000, epochs=1500,  loss_factor=2, cuda=True)
    # Your main code logic here
    print(f"Model loaded")

    print(f"Training pipeline")

    pipeline_builder = PipelineBuilder(df_real_cardio_train, cat_list_cardio, num_list_cardio, ctgan)
    pipeline_builder.add_generation_task()
    # pipeline_builder.add_fine_tuning_generation_task()
    # pipeline_builder.add_sampling_and_reject_task(eps)
    pipeline_builder.add_ressemblance_evaluation_task(df_real_cardio_test)
    classifier_types = [ClassifierType.CART, 
                        ClassifierType.KNN, 
                        ClassifierType.LDA, 
                        ClassifierType.NB, 
                        ClassifierType.LR, 
                        ClassifierType.RANDOM_FOREST,
                        ClassifierType.SVM,
                        ClassifierType.XGBOOST]
    pipeline_builder.add_utility_evaluation_task(df_real_cardio_test, classifier_types)
    pipeline_builder.add_privacy_evaluation_task(df_real_cardio_test,cardio_qai_columns, cardio_risk_column)
    pipeline_builder.add_privacy_anonymeter_evaluation_task(df_real_cardio_test)
    pipeline_builder.build()
    results = pipeline_builder.run()

    print(f"pipeline trained")

    file_name = f"cardio_tvae_optimised_test_no_finetuning.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(results, file)

    file_name = f"cardio_tvae_optimised_model_no_finetuning.pkl"
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
