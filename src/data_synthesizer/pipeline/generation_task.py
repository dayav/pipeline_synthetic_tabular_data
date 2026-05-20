import copy
from typing import Optional

import pandas as pd

from data_evaluator.univariate_evaluator import UnivariateEvaluator
from data_loader.data_loader import DataLoader
from data_synthesizer.pipeline.base_pipeline import Task
from data_synthesizer.pipeline.pipeline_results import GenerationResults, PipelineResults
from data_synthesizer.privacy_sampling import (
    get_epsilon,
    get_epsilon_heom_any,
    get_epsilon_heom_any_weighted_k,
    get_epsilon_heom_hnsw,
    get_epsilon_numerical_only,
    sampling_reject_epsilon,
    sampling_reject_epsilon_heom_any_weighted_k,
    sampling_reject_epsilon_heom_faiss_any,
    sampling_reject_epsilon_heom_hnsw_any,
    sampling_reject_epsilon_heom_knn_any,
    sampling_reject_epsilon_hnsw,
    sampling_reject_epsilon_numerical_only,
)
from data_synthesizer.sampling_aia_guard import sampling_reject_attribute_inference

class GenerationTask(Task):
    """Task for generating synthetic data."""
    def __init__(self, model, train_data : pd.DataFrame , cat_features : list, num_features : list):
        super().__init__(model, train_data, cat_features, num_features)
        self.mode_collapse_corrected = False


    def process(self, results : PipelineResults) :
        """Process generation by fitting the model and sampling synthetic data."""
        print("Generation processing:")
        self.model.fit(self.train_data)
        self.synth_data = self.model.sample(len(self.train_data))
        generation_results = GenerationResults(synthetic_data =  self.synth_data, generator_model = self.model, mode_collapse_corrected = self.mode_collapse_corrected)

        results['generation_results'] = generation_results
    
class FineTuningGenerationTask(GenerationTask):
    """Task for fine-tuning generation based on mode collapse."""

    def __init__(self, model, train_data : pd.DataFrame , cat_features : list, num_features : list, synth_data : Optional[pd.DataFrame] = None):
        super().__init__(model, train_data, cat_features, num_features)
        self.mode_collapse_corrected = False
        self.synth_data = synth_data
        
    def process(self, results: PipelineResults) -> None:
        """
        Fine-tunes the generation process by addressing mode collapse and resampling rare values in synthetic data.

        This function refines the generated synthetic data by detecting mode collapse, a situation where some values are 
        underrepresented or missing. The function corrects this by resampling rare values from the real dataset and 
        fine-tuning the generator model. It iterates through the process until mode collapse is resolved, ensuring the 
        synthetic data better reflects the distribution of the real data.

        Args:
            results (PipelineResults): A dictionary-like object containing the generation results, including the synthetic data 
                and the generator model.

        Returns:
            None: The function modifies the synthetic data in place within the `results` object after correcting mode collapse.

        Example:
            process_results = process(results)

        Notes:
            - The function evaluates the synthetic data against the real data using a `UnivariateEvaluator` to detect mode collapse.
            - When mode collapse is detected, it filters the real dataset for rare values and retrains a copy of the generator model 
            using this subset.
            - Rare values from the real dataset are resampled and incorporated into the synthetic dataset.
            - The process continues iteratively until mode collapse is resolved.
            - The corrected synthetic data and generator model are stored back in the `results` object.
        """
        print("Fine Tuning Generation Task")


        #overwrite with the synthetic data from result
        if 'generation_results' in results :
            if 'synthetic_data' in results['generation_results'] :
                self.synth_data = results['generation_results']['synthetic_data']
        else :
            results['generation_results'] = GenerationResults(synthetic_data =  self.synth_data, generator_model = None, mode_collapse_corrected = False)
        # self.synth_data = results['generation_results']['synthetic_data']
        # self.model = results['generation_results']['generator_model']

        print('mode_collapse_corrected in FineTuningGenerationTask : ',results['generation_results']['mode_collapse_corrected'])

        train_data_to_evaluate = DataLoader(dataset=self.train_data).get_dataframe(self.cat_features)
        synth_data_to_evaluate = DataLoader(dataset=self.synth_data).get_dataframe(self.cat_features)

        self._uni_evaluators = UnivariateEvaluator(train_data_to_evaluate, synth_data_to_evaluate)
        mode_collapse = self._uni_evaluators.get_mode_collapse()
        if mode_collapse:
            while mode_collapse:
                print('mode collapse correction')
                mask = pd.Series([False] * len(self.train_data))
            
                for feature, values in mode_collapse.items():
                    print(values)
                    mask |= self.train_data[feature].isin(values)

                filtered_D = self.train_data[mask].reset_index(drop=True)
                print('rare from train : ', len(filtered_D))
                if len(filtered_D) < 10 :# change 10 by oversampling
                    to_sample = 10-len(filtered_D)
                    train_sample = train_data_to_evaluate.sample(n=to_sample, random_state=42)# random state to change
                    filtered_D = pd.concat([filtered_D, train_sample], ignore_index=True)
                self.deep_copied_model = copy.deepcopy(self.model)#
                self.deep_copied_model.fit_with_only_rare(filtered_D)# choose wisely filtered
                synth_data_new = self.deep_copied_model.sample(len(self.train_data))# change the len

                most_frequent = {col: self.synth_data[col].value_counts().idxmax() for col in self.synth_data.columns}
                df_combined = self.synth_data.copy()

                for feature, values in mode_collapse.items():
                    for value in values:
                        filter_mask = (synth_data_new[feature] == value)
                        print('filter_mask : ',  filter_mask.any())
                        filtered_df = synth_data_new[filter_mask]
                        print('len filtered_df :', len(filtered_df))
                        sampled_df = filtered_df.sample(n=10, random_state=42) if len(filtered_df) >= 10 else filtered_df
                        df_combined = pd.concat([df_combined, sampled_df], ignore_index=True)

                        most_common_value = most_frequent[feature]
                        filter_most_common = (df_combined[feature] == most_common_value)
                        random_indexes = df_combined[filter_most_common].sample(len(sampled_df)).index
                        df_combined.drop(random_indexes, inplace=True)
                        print('df_combined :', len(df_combined[df_combined[feature] == value]))
                        #evaluate uni synth vs real raw (measur-diff )


                self.synth_data = df_combined.reset_index(drop=True)
                synth_data_to_evaluate = DataLoader(dataset=self.synth_data).get_dataframe(self.cat_features)
                self._uni_evaluators = UnivariateEvaluator(train_data_to_evaluate, synth_data_to_evaluate)
                mode_collapse = self._uni_evaluators.get_mode_collapse()

            self.mode_collapse_corrected = True
            self.generation_results = GenerationResults(synthetic_data =  self.synth_data, generator_model = self.model, mode_collapse_corrected = self.mode_collapse_corrected)


            results['generation_results'] = self.generation_results
            print('mode_collapse_corrected in FineTuningGenerationTask (end) : ',results['generation_results']['mode_collapse_corrected'])
        else :
            print('No Mode Collpase')
     
class SamplingAndRejectTask(GenerationTask):
    """Task for generating data with rejection sampling based on privacy constraints."""
    def __init__(self, 
                 model, 
                 train_data : pd.DataFrame , 
                 cat_features : list, 
                 num_features : list,
                 epsilon: float, 
                 synth_data: Optional[pd.DataFrame] = None, 
                 dissimilarity_type:Optional[str] = 'euclidian', 
                 sensitives_cols : Optional[list] = None, 
                 aia_parameters: Optional[dict] = None):
        super().__init__(model, train_data, cat_features, num_features)
        self.epsilon = epsilon
        self.synth_data = synth_data
        self._dissimilarity_type = dissimilarity_type
        self._sensitives_cols = sensitives_cols
        self._aia_parameters = aia_parameters
    
    def _run_aia_rejection_sampling(self) -> pd.DataFrame:
        if not self._aia_parameters:
            raise ValueError(
                "AIA sampling requires `aia_parameters` with at least `sensitive_col` "
                "and (optionally) `real_val_df`."
            )

        params = dict(self._aia_parameters)
        sensitive_col = params.get("sensitive_col")
        if sensitive_col is None:
            sens_cols = params.get("sens_cols", [])
            if len(sens_cols) == 1:
                sensitive_col = sens_cols[0]
            elif len(sens_cols) > 1:
                raise ValueError(
                    "AIA sampling supports one sensitive attribute at a time. "
                    "Use `sensitive_col` for a single column."
                )

        if sensitive_col is None:
            raise ValueError("Missing `sensitive_col` in `aia_parameters`.")

        z_cols = params.get("Z_cols", params.get("z_cols"))
        if z_cols is not None:
            z_cols = list(z_cols)
            if len(z_cols) == 0:
                z_cols = None

        x_cols = params.get("X_cols", params.get("x_cols"))
        if x_cols is None:
            excluded = {sensitive_col}
            if z_cols:
                excluded.update(z_cols)
            x_cols = [c for c in self.train_data.columns if c not in excluded]
        x_cols = list(x_cols)
        if len(x_cols) == 0:
            raise ValueError("AIA sampling requires at least one non-sensitive feature in `X_cols`.")

        real_val_df = params.get("real_val_df", params.get("real_validation_df", self.train_data))
        attacker_families = params.get("attacker_families", ("logreg", "mlp_svd"))
        if isinstance(attacker_families, str):
            attacker_families = [a.strip() for a in attacker_families.split(",") if a.strip()]
        if len(attacker_families) == 0:
            raise ValueError("AIA sampling requires at least one attacker family.")

        tau_ai = float(params.get("tau_ai", self.epsilon))
        n_samples = int(params.get("n_samples", len(self.train_data)))
        sensitive_n_bins = params.get("sensitive_n_bins")
        if sensitive_n_bins is not None:
            sensitive_n_bins = int(sensitive_n_bins)
            if sensitive_n_bins <= 1:
                sensitive_n_bins = None

        return sampling_reject_attribute_inference(
            model=self.model,
            real_val_df=real_val_df,
            tau_ai=tau_ai,
            sensitive_col=sensitive_col,
            X_cols=x_cols,
            Z_cols=z_cols,
            num_cols=list(params.get("num_cols", self.num_features)),
            cat_cols=list(params.get("cat_cols", self.cat_features)),
            n_samples=n_samples,
            sensitive_n_bins=sensitive_n_bins,
            sensitive_bin_edges=params.get("sensitive_bin_edges"),
            sensitive_mode=str(params.get("sensitive_mode", "auto")),
            regression_unique_threshold=int(params.get("regression_unique_threshold", 50)),
            regression_target_transform=str(params.get("regression_target_transform", "auto")),
            linear_regressor_kind=str(params.get("linear_regressor_kind", "sgd")),
            attacker_families=tuple(attacker_families),
            retrain_every=int(params.get("retrain_every", 500)),
            max_swaps=int(params.get("max_swaps", 20000)),
            regression_normalize=bool(params.get("regression_normalize", True)),
            random_state=int(params.get("random_state", 0)),
            verbose=bool(params.get("verbose", True)),
        )

    def process(self, results : PipelineResults) -> pd.DataFrame:
        """Process data using epsilon-based rejection sampling."""
        print("Sampling and reject task")

        #overwrite with the synthetic data from result
        if 'generation_results' in results :
            if 'synthetic_data' in results['generation_results'] :
                self.synth_data = results['generation_results']['synthetic_data']
        else :
            results['generation_results'] = GenerationResults(synthetic_data =  self.synth_data, generator_model = None, mode_collapse_corrected = False)

        self.mode_collapse_corrected = results['generation_results']['mode_collapse_corrected']

        print('mode_collapse_corrected in SamplingAndRejectTask : ',results['generation_results']['mode_collapse_corrected'])
        if self._dissimilarity_type == "aia":
            self.synth_data = self._run_aia_rejection_sampling()
            self.generation_results = GenerationResults(
                synthetic_data=self.synth_data,
                generator_model=self.model,
                mode_collapse_corrected=self.mode_collapse_corrected,
            )
            results['generation_results'] = self.generation_results
            return self.synth_data

        self.synth_data = DataLoader(dataset=self.synth_data).get_dataframe(self.cat_features)

        if self._dissimilarity_type == "only_numerical" :
            self._epsilon_identifiability = get_epsilon_numerical_only(self.train_data, self.synth_data,
                                                        self.num_features)

        elif self._dissimilarity_type == "heom" :
                self._epsilon_identifiability = get_epsilon_heom_hnsw(
                    self.train_data, self.synth_data, self.num_features, self.cat_features)
        elif self._dissimilarity_type == "heom-knn-any" :
                self._epsilon_identifiability = get_epsilon_heom_any(
                    self.train_data, self.synth_data, self.num_features, self.cat_features)

        elif self._dissimilarity_type == "heom-knn-any-weighted-2" :
                self._epsilon_identifiability = get_epsilon_heom_any_weighted_k(
                        self.train_data, self.synth_data, self.num_features, self.cat_features, k=2)
        elif self._dissimilarity_type == "heom-knn-any-weighted-5" :
                self._epsilon_identifiability = get_epsilon_heom_any_weighted_k(
                        self.train_data, self.synth_data, self.num_features, self.cat_features, k=5)
        elif self._dissimilarity_type == "heom-knn-any-weighted-10" :
                self._epsilon_identifiability = get_epsilon_heom_any_weighted_k(
                        self.train_data, self.synth_data, self.num_features, self.cat_features, k=10)
        elif self._dissimilarity_type == "heom-knn-any-5" :
                self._epsilon_identifiability = get_epsilon_heom_any_weighted_k(
                        self.train_data, self.synth_data, self.num_features, self.cat_features, k=5, compute_inverse_entropy=False)
        elif self._dissimilarity_type == "heom-knn-any-10" :
                self._epsilon_identifiability = get_epsilon_heom_any_weighted_k(
                        self.train_data, self.synth_data, self.num_features, self.cat_features, k= 10, compute_inverse_entropy=False)
        elif self._dissimilarity_type == "heom-faiss-any" :
                self._epsilon_identifiability = get_epsilon_heom_any_weighted_k(
                    self.train_data, self.synth_data, self.num_features, self.cat_features)
        elif self._dissimilarity_type == "heom-hnsw-any" :
                self._epsilon_identifiability = get_epsilon_heom_any_weighted_k(
                    self.train_data, self.synth_data, self.num_features, self.cat_features)
        else :
            self._epsilon_identifiability = get_epsilon(self.train_data, self.synth_data,
                                                        self.cat_features, self.num_features)
        # only for test
        
        print('initial epsilon : ', self._epsilon_identifiability)
        if self._epsilon_identifiability > self.epsilon:
            if self._dissimilarity_type == "only_numerical" :
                self.synth_data = sampling_reject_epsilon_numerical_only(self.model, self.train_data, self.epsilon,
                                                    self.num_features, len(self.train_data))
            elif self._dissimilarity_type == "heom" :
                self.synth_data = sampling_reject_epsilon_hnsw(self.model, self.train_data, self.epsilon,
                    self.num_features, self.cat_features, len(self.train_data))
            elif self._dissimilarity_type == "heom-knn-any" :
                self.synth_data = sampling_reject_epsilon_heom_knn_any(self.model, self.train_data, self.epsilon,
                    self.num_features, self.cat_features, len(self.train_data))
            elif self._dissimilarity_type == "heom-knn-any-weighted-2" :
                self.synth_data = sampling_reject_epsilon_heom_any_weighted_k(self.model, self.train_data, self.epsilon,
                    self.num_features, self.cat_features, len(self.train_data), k=2)
            elif self._dissimilarity_type == "heom-knn-any-weighted-5" :
                self.synth_data = sampling_reject_epsilon_heom_any_weighted_k(self.model, self.train_data, self.epsilon,
                    self.num_features, self.cat_features, len(self.train_data), k=5)
            elif self._dissimilarity_type == "heom-knn-any-weighted-10" :
                self.synth_data = sampling_reject_epsilon_heom_any_weighted_k(self.model, self.train_data, self.epsilon,
                    self.num_features, self.cat_features, len(self.train_data), k=10)
            elif self._dissimilarity_type == "heom-knn-any-5" :
                self.synth_data = sampling_reject_epsilon_heom_any_weighted_k(self.model, self.train_data, self.epsilon,
                    self.num_features, self.cat_features, len(self.train_data), k=5, compute_inverse_entropy=False)
            elif self._dissimilarity_type == "heom-knn-any-10" :
                self.synth_data = sampling_reject_epsilon_heom_any_weighted_k(self.model, self.train_data, self.epsilon,
                    self.num_features, self.cat_features, len(self.train_data), k=10, compute_inverse_entropy=False)                
            
            elif self._dissimilarity_type == "heom-faiss-any" :
                self.synth_data = sampling_reject_epsilon_heom_hnsw_any(self.model, self.train_data, self.epsilon,
                    self.num_features, self.cat_features, len(self.train_data))
            elif self._dissimilarity_type == "heom-hnsw-any" :
                self.synth_data = sampling_reject_epsilon_heom_faiss_any(self.model, self.train_data, self.epsilon,
                    self.num_features, self.cat_features, len(self.train_data))
                
            else :
                self.synth_data = sampling_reject_epsilon(self.model, self.train_data, self.epsilon,
                                                    self.cat_features, self.num_features, len(self.train_data))
        #  only for test               
        
        self.generation_results = GenerationResults(synthetic_data =  self.synth_data, generator_model = self.model, mode_collapse_corrected = self.mode_collapse_corrected)
        # self.generation_results = GenerationResults(synthetic_data =  self.synth_data)

        results['generation_results'] = self.generation_results
        return self.synth_data
