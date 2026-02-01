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

        cat_list_credit_card = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'default.payment.next.month']
        num_list_credit_card = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3','PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        credit_qai_columns = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE']
        credit_risk_column = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default.payment.next.month']
        df_real_credit_card_train = DataLoader('../../data/credit_card_Train.csv').get_dataframe(cat_list_credit_card, str, drop_identation=True)
        df_real_credit_card_test = DataLoader('../../data/credit_card_Test.csv').get_dataframe(cat_list_credit_card, str, drop_identation=True)
        
        # Define the parameter space
        embedding_dim = trial.suggest_categorical('embedding_dim', [128, 256, 512])
        generator_lr = trial.suggest_loguniform('generator_lr', 1e-5, 1e-3)
        discriminator_lr = trial.suggest_loguniform('discriminator_lr', 1e-5, 1e-3)
        generator_dim = trial.suggest_categorical('generator_dim', [[256, 256], [512, 512], [256, 512]])
        discriminator_dim = trial.suggest_categorical('discriminator_dim', [[256, 256], [512, 512]])
        batch_size = trial.suggest_categorical('batch_size', [100, 500, 1000])
        epochs = trial.suggest_categorical('epochs', [800, 1000, 1500])
        
        print(f"Loading Model")
        metadata = SingleTableMetadata()

        metadata.detect_from_dataframe(data=df_real_credit_card_train)
        ctgan = SDVCTGAN_( metadata, df_real_credit_card_train, enforce_min_max_values=True, enforce_rounding=True, locales=None,
                    embedding_dim=embedding_dim, generator_dim=generator_dim, discriminator_dim=discriminator_dim,
                    generator_lr=generator_lr, generator_decay=1e-6, discriminator_lr=discriminator_lr,
                    discriminator_decay=1e-6, batch_size=batch_size, discriminator_steps=1,
                    log_frequency=True, verbose=False, epochs=epochs, pac=10, cuda=True)
        # Your main code logic here
        print(f"Model loaded")

        print(f"Training pipeline")

        pipeline_builder = PipelineBuilder(df_real_credit_card_train, cat_list_credit_card, num_list_credit_card, ctgan)
        pipeline_builder.add_generation_task()
        pipeline_builder.add_ressemblance_evaluation_task(df_real_credit_card_test)
        classifier_types = [ ClassifierType.XGBOOST]
        pipeline_builder.add_utility_evaluation_task(df_real_credit_card_test, classifier_types)

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

    file_name = f"study_ctgan_credit.pkl"
    with open(file_name, 'wb') as file:
        # Serialize the object and write it to the file
        pickle.dump(study, file)
    json_file_name = f"study_ctgan_credit.json"
    with open(json_file_name, 'w', encoding='utf-8') as file:
        json.dump(export_optuna_study_to_json_dict(study), file, indent=2)
    # print("Best trial:", study.best_trial)



if __name__ == "__main__":

    main()
