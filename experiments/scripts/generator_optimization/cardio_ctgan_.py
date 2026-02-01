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

    ctgan = SDVCTGAN_( metadata, df_real_cardio_train, enforce_min_max_values=True, enforce_rounding=True, locales=None,
            embedding_dim=512, generator_dim=[256, 256], discriminator_dim=[256, 256],
            generator_lr=0.00017403142604971523, generator_decay=1e-6, discriminator_lr=0.00037251120555768314,
            discriminator_decay=1e-6, batch_size=100, discriminator_steps=1,
            log_frequency=True, verbose=False, epochs=1000, pac=10, cuda=True)
    
    # Your main code logic here
    print(f"Model loaded")

    print(f"Training pipeline")

    pipeline_builder = PipelineBuilder(df_real_cardio_train, cat_list_cardio, num_list_cardio, ctgan)
    pipeline_builder.add_generation_task()
    pipeline_builder.add_fine_tuning_generation_task()
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

    print(f"saving pipeline")
    file_name = f"cardio_ctgan_optimized_test.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(results, file)

    file_name = f"cardio_ctgan_optimized_model.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(ctgan, file)


if __name__ == "__main__":


    main()
