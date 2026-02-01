#!/usr/bin/env python3

from data_loader import DataLoader
from data_synthesizer.sdv import SDVCTGAN_, SDVTVAE_
from data_evaluator import ClassifierType
from data_synthesizer.pipeline import PipelineBuilder
from sdv.metadata import SingleTableMetadata
import pickle
import json
import optuna




import argparse

def export_optuna_study_to_json_dict(study):
    def dt_iso(x):
        return x.isoformat() if x is not None else None

    directions = (
        [str(d) for d in study.directions]
        if hasattr(study, "directions")
        else [str(study.direction)]
    )

    def trial_to_dict(t):
        return {
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            "values": t.values,
            "params": dict(t.params),
            "distributions": {k: str(v) for k, v in t.distributions.items()},
            "user_attrs": dict(t.user_attrs),
            "system_attrs": dict(t.system_attrs),
            "datetime_start": dt_iso(t.datetime_start),
            "datetime_complete": dt_iso(t.datetime_complete),
            "duration_seconds": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_start and t.datetime_complete
                else None
            ),
        }

    payload = {
        "study_name": study.study_name,
        "directions": directions,
        "n_trials": len(study.trials),
        "best_trial": trial_to_dict(study.best_trial)
        if study.trials
        else None,
        "trials": [trial_to_dict(t) for t in study.trials],
    }
    return payload

def main():


    def objective(trial):

        cat_list_adult = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','income']
        num_list_adult = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
        adult_qai_columns = ['education','education-num','marital-status','occupation','relationship','race','sex', 'native-country']
        adult_risk_column = ['capital-gain','capital-loss','hours-per-week','native-country','income']
        df_real_adult_train = DataLoader('../../../data/adult_train.csv').get_dataframe(cat_list_adult, str)
        df_real_adult_test = DataLoader('../../../data/adult_test.csv').get_dataframe(cat_list_adult, str)
            
        # Define the parameter space
        embedding_dim = trial.suggest_categorical('embedding_dim', [128, 256, 512])
        compress_dims = trial.suggest_categorical('generator_dim', [[256, 256], [512, 512], [256, 512]])
        decompress_dims = trial.suggest_categorical('discriminator_dim', [[256, 256], [512, 512], [256, 512]])
        batch_size = trial.suggest_categorical('batch_size', [100, 500, 1000])
        epochs = trial.suggest_categorical('epochs', [800, 1000, 1500])
        
        print(f"Loading Model")
        metadata = SingleTableMetadata()

        metadata.detect_from_dataframe(data=df_real_adult_train)
        ctgan = SDVTVAE_( metadata, df_real_adult_train, enforce_min_max_values=True, enforce_rounding=True,
                    embedding_dim=embedding_dim, compress_dims=compress_dims, decompress_dims=decompress_dims,
                    l2scale=1e-5, batch_size=batch_size, epochs=epochs,  loss_factor=2, cuda=True)
        # Your main code logic here
        print(f"Model loaded")

        print(f"Training pipeline")

        pipeline_builder = PipelineBuilder(df_real_adult_train, cat_list_adult, num_list_adult, ctgan)
        pipeline_builder.add_generation_task()
        pipeline_builder.add_ressemblance_evaluation_task(df_real_adult_test)
        classifier_types = [ ClassifierType.XGBOOST]
        pipeline_builder.add_utility_evaluation_task(df_real_adult_test, classifier_types)

        pipeline_builder.build()
        results = pipeline_builder.run()

        # Train the model and return evaluation metric (like resemblance or utility)
        xgboost_f1_synth = results['utility_evaluation_results']['train_synthetic_test_real_results']['f1']['XGBoost']
        xgboost_f1_real = results['utility_evaluation_results']['train_real_test_real_results']['f1']['XGBoost']
        score = abs(xgboost_f1_real- xgboost_f1_synth)
        return score

    # Perform Bayesian optimization
    study = optuna.create_study(direction='minimize')  # Specify directions for both objectives
    study.optimize(objective, n_trials=50)

    file_name = f"study_tvae_adult.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(study, file)
    json_file_name = f"study_tvae_adult.json"
    with open(json_file_name, 'w', encoding='utf-8') as file:
        json.dump(export_optuna_study_to_json_dict(study), file, indent=2)
    # print("Best trial:", study.best_trial)



if __name__ == "__main__":
    
    # Call the main function
    main()
