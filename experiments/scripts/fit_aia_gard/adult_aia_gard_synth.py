#!/usr/bin/env python3


from __future__ import annotations
import pickle


from data_loader import DataLoader
from data_synthesizer import sampling_reject_epsilon_tabnet_strong
from data_synthesizer.fit_aia_gard import fit_aia_guard_strong
from data_synthesizer.pipeline.storages.evaluation_results_storage import load_all_results
from privacy_sampling.embeddings import make_tabnet_embedder


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific configuration
# ──────────────────────────────────────────────────────────────────────────────
NUM_COLS_ADULT= ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
CAT_COLS_ADULT = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','income']

QAI_COLUMNS = ['education','education-num','marital-status','occupation','relationship','sex', 'native-country']
RISK_COLUMNS = ['capital-gain','capital-loss','hours-per-week','native-country','income', 'race']



# ──────────────────────────────────────────────────────────────────────────────
# Core routine
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # 1. Load data
    df_train = DataLoader("../../../data/adult_train.csv").get_dataframe(
        CAT_COLS_ADULT)
    adult_ctgan  = load_all_results('../baseline/adult_ctgan_baseline')

    
    model = adult_ctgan["generation_results"]["generator_model"]
    guards_path = "adult_aia_guards.pkl"
    with open(guards_path, 'rb') as file:
        guards = pickle.load(file)

    embed_fn = make_tabnet_embedder(df_train,
                                    NUM_COLS_ADULT,
                                    CAT_COLS_ADULT)   

    adult_synth, diag = sampling_reject_epsilon_tabnet_strong(
        model, df_train, min_eps=0.05, embed_fn=embed_fn, n_samples=len(df_train),guards=guards,
        apply_epsilon=False,
        apply_guard=True, guard_stage='prefilter',  trace_guard = True,
        explain_guard = True,debug_guard_table=True,
        return_diag=True
    )

    adult_synth.to_csv("adult_aiagards.csv", index=False)


def main() -> None:
    run()


if __name__ == "__main__":
    main()

