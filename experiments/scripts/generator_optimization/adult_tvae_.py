#!/usr/bin/env python3

from data_loader import DataLoader
from data_synthesizer.sdv import SDVCTGAN_, SDVTVAE_
from data_evaluator import ClassifierType
from data_synthesizer.pipeline import PipelineBuilder
from sdv.metadata import SingleTableMetadata
import pickle




import argparse

def main():

    cat_list_adult = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','income']
    num_list_adult = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    adult_qai_columns = ['education','education-num','marital-status','occupation','relationship','race','sex', 'native-country']
    adult_risk_column = ['capital-gain','capital-loss','hours-per-week','native-country','income']
    df_real_adult_train = DataLoader('../../data/adult_train.csv').get_dataframe(cat_list_adult, str)
    df_real_adult_test = DataLoader('../../data/adult_test.csv').get_dataframe(cat_list_adult, str)


    print(f"Loading Model")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_real_adult_train)


    ctgan = SDVTVAE_( metadata, df_real_adult_train, enforce_min_max_values=True, enforce_rounding=True,
                    embedding_dim=128, compress_dims=[256, 512], decompress_dims=[256, 512],
                    l2scale=1e-5, batch_size=1000, epochs=1000,  loss_factor=2, cuda=True)
    # Your main code logic here
    print(f"Model loaded")

    print(f"Training pipeline")

    pipeline_builder = PipelineBuilder(df_real_adult_train, cat_list_adult, num_list_adult, ctgan)
    pipeline_builder.add_generation_task()
    # pipeline_builder.add_fine_tuning_generation_task()
    # pipeline_builder.add_sampling_and_reject_task()
    pipeline_builder.add_ressemblance_evaluation_task(df_real_adult_test)
    classifier_types = [ClassifierType.CART, 
                        ClassifierType.KNN, 
                        ClassifierType.LDA, 
                        ClassifierType.NB, 
                        ClassifierType.LR, 
                        ClassifierType.RANDOM_FOREST,
                        ClassifierType.SVM,
                        ClassifierType.XGBOOST]
    pipeline_builder.add_utility_evaluation_task(df_real_adult_test, classifier_types)
    pipeline_builder.add_privacy_evaluation_task(df_real_adult_test,adult_qai_columns, adult_risk_column)
    pipeline_builder.add_privacy_anonymeter_evaluation_task(df_real_adult_test)
    pipeline_builder.build()
    results = pipeline_builder.run()

    print(f"pipeline trained")

    print(f"saving pipeline")
    file_name = f"adult_tvae_optimize_test_no_finetuning.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(results, file)

    file_name = f"adult_tvae_optimize_model_no_finetuning.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(ctgan, file)


if __name__ == "__main__":

    main()
