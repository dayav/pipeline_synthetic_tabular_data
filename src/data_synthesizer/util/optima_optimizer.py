import os
import optuna
from data_synthesizer.sdv import SDVCTGAN, SDVTVAE
from data_evaluator.global_evaluator import GlobalEvaluator
from data_loader import DataLoader
from sdv.metadata import SingleTableMetadata
import pandas as pd


def objective_sdvgan(trial, data_real_path, cat_list, drop_identation=False) :


    df = DataLoader(data_real_path).get_dataframe(cat_list, str, drop_identation=drop_identation)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    
    # cat_list = ['checking_status', 'credit_history', 'purpose', 'savings_status',
    #    'employment', 'personal_status', 'other_parties', 'property_magnitude',
    #    'other_payment_plans', 'housing', 'job', 'own_telephone',
    #    'foreign_worker', 'class']
    generator_lr = trial.suggest_float("gen_lr", 1e-5, 1e-3)
    discr_lr = trial.suggest_float("discr_lr", 1e-5, 1e-3)

    model = SDVCTGAN(metadata=metadata, generator_lr=generator_lr, discriminator_lr=discr_lr,verbose=True, epochs=500)
    model.fit(df)
    loss = model._training_report['Generator Loss'][-20:]
    mean_loss = sum(loss) / 20
    data = model.sample(num_rows=1000)
    data.to_csv('data/_ctgan_trial.csv', sep=',', index=False)
    df_synth = DataLoader('data/_ctgan_trial.csv').get_dataframe(cat_list)
    df_real = DataLoader(data_real_path).get_dataframe(cat_list, drop_identation=drop_identation)
    global_evaluator = GlobalEvaluator(df_real, df_synth)
    pMSE, _ = global_evaluator.propensity_score()
    return mean_loss, pMSE[1]

class OptimizationHyperParam :
    def __init__(self, objective) :
        self._objective = objective

    def get_optimize_param(self) :      

        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(self._objective, n_trials=30, show_progress_bar=True)

        print("Number of finished trials: ", len(study.trials))

        print("Pareto front:")

        trials = sorted(study.best_trials, key=lambda t: t.values)

        for trial in trials:
            print("  Trial#{}".format(trial.number))
            print("    Values: mean_loss gen={}, pMSE={}".format(trial.values[0], trial.values[1]))
            print("    Params: {}".format(trial.params))