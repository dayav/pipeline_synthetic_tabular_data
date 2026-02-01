from typing import List, Optional
import pandas as pd
import json

from data_evaluator import (
    MultivariateEvaluator,
    UnivariateEvaluator,
    UtilityEvaluation,
    ClassifierType
)

from data_evaluator.privacy_evaluation.privacy_evaluator import PrivacyEvaluator, SimilarityType
from data_evaluator.privacy_evaluation.privacy_evaluator_anonymeter import PrivacyEvaluatorAnonymeter
from data_loader.data_loader import DataLoader
from data_synthesizer.pipeline.base_pipeline import Task, TaskType
from data_synthesizer.pipeline.pipeline_results import PipelineResults, PrivacyAnonymeterEvaluationResults, PrivacyEvaluationResults, ResemblanceEvaluationResults, UtilityEvaluationResults, GenerationResults
from data_synthesizer.privacy_sampling import get_epsilon_any
  

class EvaluationTask(Task):
    def __init__(self, train_data : pd.DataFrame, test_data : pd.DataFrame, cat_features : list, num_features : list, synth_data: Optional[pd.DataFrame] = None):
        super().__init__(None, train_data, cat_features, num_features)
        self.train_data = DataLoader(dataset=train_data).get_dataframe(self.cat_features)
        self.test_data = DataLoader(dataset=test_data).get_dataframe(self.cat_features)
        self.synth_data = synth_data


class ResemblanceEvaluationTask(EvaluationTask):
    """Task for evaluating the resemblance between training and synthetic data."""

   
    def process(self, results : PipelineResults):
        """Processes the data to evaluate resemblance statistics."""
        
        #overwrite with the synthetic data from result
        if 'generation_results' in results :
            if 'synthetic_data' in results['generation_results'] :
                self.synth_data = results['generation_results']['synthetic_data']
        else :
            results['generation_results'] = GenerationResults(synthetic_data =  self.synth_data, generator_model = None, mode_collapse_corrected = False)
            print('new GenerationResults in  ResemblanceEvaluationTask',results['generation_results']['mode_collapse_corrected'])

        print('mode_collapse_corrected in ResemblanceEvaluationTask : ',results['generation_results']['mode_collapse_corrected'])
        self.synth_data = DataLoader(dataset=self.synth_data).get_dataframe(self.cat_features)
        
        self._uni_evaluators = UnivariateEvaluator(self.train_data, self.synth_data)
        self._multi_evaluator = MultivariateEvaluator(self.train_data, self.synth_data)
        self._uni_evaluators.evaluate_categorical_stat_evaluation()
        self._uni_evaluators.evaluate_numerical_stat_evaluation()
        self._multi_evaluator.evaluate_cramerV_correlation_matrix_diff()
        self._multi_evaluator.evaluate_pearson_correlation_matrix_diff()
        self._multi_evaluator.evaluate_correlation_ratio_matrix_diff()        
        self._uni_evaluators.evaluate_mode_collapse_values()

        self.ressemblance_evaluation_results = ResemblanceEvaluationResults(
            categorical_univariate=self._uni_evaluators.cat_uni,
            numerical_univariate=self._uni_evaluators.num_uni,
            categorical_multivariate=self._multi_evaluator.cat_multi,
            numerical_multivariate=self._multi_evaluator.num_multi,
            categorical_numerical_multivariate=self._multi_evaluator.num_cat_multi
        )

        results['resemblance_evaluation_results'] = self.ressemblance_evaluation_results

    def save(self, path : str) :
        
        evaluation_results_json = {
            'categorical_univariate': self.ressemblance_evaluation_results['categorical_univariate'],
            'numerical_univariate': self.ressemblance_evaluation_results['numerical_univariate'],
            'categorical_multivariate': self.ressemblance_evaluation_results['categorical_multivariate'],
            'numerical_multivariate': self.ressemblance_evaluation_results['numerical_multivariate'],
            'categorical_numerical_multivariate': self.ressemblance_evaluation_results['categorical_numerical_multivariate']
        }
        # Save to a JSON file
        with open(path, 'w') as f:
            json.dump(evaluation_results_json, f)



class UtilityEvaluationTask(EvaluationTask):
    """Task for evaluating the utility of synthetic data."""

    def __init__(self, train_data : pd.DataFrame, test_data : pd.DataFrame, cat_features : list, num_features : list, classifiers : List[ClassifierType],
                  synth_data: Optional[pd.DataFrame] = None):
        super().__init__(train_data, test_data, cat_features, num_features, synth_data)

        self._classifiers = classifiers
    
    def process(self, results : PipelineResults):
        """Processes data for utility evaluation."""
        print("Utility evaluation in progress.")

        
        #overwrite with the synthetic data from result
        if 'generation_results' in results :
            if 'synthetic_data' in results['generation_results'] :
                self.synth_data = results['generation_results']['synthetic_data']
        
        self._utility_evaluator = UtilityEvaluation(self.train_data, 
                                                    self.synth_data, 
                                                    self.test_data, 
                                                    self._classifiers)
        
       
        train_synthetic_test_real_results, permutation_importance_tstr, shap_importance_tstr, confusion_matrix_tstr, f1_mean_tstr = self._utility_evaluator.train_synthetic_test_real(True)
        train_real_test_real_results, permutation_importance_trtr, shap_importance_trtr, confusion_matrix_trtr, f1_mean_trtr = self._utility_evaluator.train_real_test_real(True)
        rbo_permutation_importance, rbo_shap =  self._utility_evaluator.rbo_compare_feature_importance(permutation_importance_trtr['XGBoost'],
                                                                shap_importance_trtr['XGBoost'],
                                                                permutation_importance_tstr['XGBoost'],
                                                                shap_importance_tstr['XGBoost'])
        
        
        spearman_permutation_importance, spearman_shap = self._utility_evaluator.spearman_compare_feature_importance(permutation_importance_trtr['XGBoost'],
                                                        shap_importance_trtr['XGBoost'],
                                                        permutation_importance_tstr['XGBoost'],
                                                        shap_importance_tstr['XGBoost'])
        
        kendall_permutation_importance, kendall_shap = self._utility_evaluator.kendall_compare_feature_importance(permutation_importance_trtr['XGBoost'],
                                                shap_importance_trtr['XGBoost'],
                                                permutation_importance_tstr['XGBoost'],
                                                shap_importance_tstr['XGBoost'])
        
        
        self.utility_evaluation_results = UtilityEvaluationResults(
            train_synthetic_test_real_results=train_synthetic_test_real_results,
            train_real_test_real_results=train_real_test_real_results,
            permutation_importance_tstr=permutation_importance_tstr,
            permutation_importance_trtr=permutation_importance_trtr,
            shap_importance_tstr=shap_importance_tstr,
            shap_importance_trtr=shap_importance_trtr,
            confusion_matrix_tstr=confusion_matrix_tstr,
            confusion_matrix_trtr=confusion_matrix_trtr,
            accuracy_mean_tstr=f1_mean_tstr,
            accuracy_mean_trtr=f1_mean_trtr,
            rbo_permutation_importance=rbo_permutation_importance,
            spearman_permutation_importance=spearman_permutation_importance,
            kendall_permutation_importance=kendall_permutation_importance,
            rbo_shap=rbo_shap,
            spearman_shap=spearman_shap,
            kendall_shap=kendall_shap)

        if 'generation_results' not in results :
        
            self.generation_results = GenerationResults(synthetic_data =  self.synth_data, generator_model = self.model, mode_collapse_corrected = False)

            results['generation_results'] = self.generation_results

        results['utility_evaluation_results'] = self.utility_evaluation_results


class PrivacyEvaluationTask(EvaluationTask):
    """Task for evaluating privacy concerns of synthetic data."""

    def __init__(self, train_data : pd.DataFrame, test_data : pd.DataFrame, cat_features : list, num_features : list, 
                 quasi_identifier_features : list, non_quasi_identifier_features : list, synth_data: Optional[pd.DataFrame] = None):
        super().__init__(train_data, test_data, cat_features, num_features, synth_data)

        self._quasi_identifier_features = quasi_identifier_features
        self._non_quasi_identifier_features = non_quasi_identifier_features
    
    def process(self, results : PipelineResults):
        """Processes data to evaluate privacy concerns."""
        print("Privacy evaluation in progress.")

         #overwrite with the synthetic data from result
        if 'generation_results' in results :
            if 'synthetic_data' in results['generation_results'] :
                self.synth_data = results['generation_results']['synthetic_data']

        self._univariate_evaluators = UnivariateEvaluator(self.train_data, 
                                                        self.synth_data)
        self._privacy_evaluator = PrivacyEvaluator(self.train_data, 
                                                    self.synth_data, 
                                                    self.test_data, 
                                                    self._quasi_identifier_features,
                                                    self._non_quasi_identifier_features)

        self._univariate_evaluators.evaluate_categorical_stat_evaluation()
        self._univariate_evaluators.evaluate_numerical_stat_evaluation()
        jensen_shanon_categorical = self._univariate_evaluators.cat_uni['jensen_shanon']
        jensen_shanon_numerical = self._univariate_evaluators.num_uni['univariate_num_js']
        results_synthetic, _, _, _, predictions_synthetic = self._privacy_evaluator.evaluate_attribute_synthetic_prediction()
        results_real, _, _, _, predictions_real = self._privacy_evaluator.evaluate_attribute_real_prediction()
        epsilon = get_epsilon_any(self.train_data, self.synth_data,
                                                        self.cat_features, self.num_features)
        diss_real, diss_test, min_diss_gen_idx_real, min_diss_gen_idx_test, share_real = self._privacy_evaluator.evaluate_similarity_stdg(SimilarityType.DISSIMILARITY)
        # eps_diss_real, eps_diss_test, min_diss_gen_idx_real, min_diss_gen_idx_test, epsilon = self._privacy_evaluator.evaluate_similarity_stdg(SimilarityType.EPSILON_DISSIMILARITY)

        self.privacy_evaluation_results = PrivacyEvaluationResults(
            jensen_shanon_categorical= jensen_shanon_categorical,
            jensen_shanon_numerical= jensen_shanon_numerical,
            dissimilarity_synthetic_real=diss_real,
            dissimilarity_synthetic_test=diss_test,
            epsilon_dissimilarity_synthetic_real=0.0,
            epsilon_dissimilarity_synthetic_test=0.0,
            share=share_real,
            epsilon = epsilon,
            attribute_synthetic_prediction=results_synthetic,
            attribute_real_prediction=results_real,
            attribute_synthetic_prediction_values=predictions_synthetic,
            attribute_real_prediction_values=predictions_real
        )

        results.privacy_evaluation_results = self.privacy_evaluation_results

class PrivacyAnonymeterEvaluationTask(EvaluationTask):
    """
    Evaluate privacy risks with Anonymeter and write the results into
    results["privacy_anonymeter_results"], which is now a dict that maps a
    *run label* → PrivacyAnonymeterEvaluationResults.
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        cat_features: list,
        num_features: list,
        synth_data: Optional[pd.DataFrame] = None,
        first_half: list = None,
        second_half: list = None,
        runs_count: int = 1,
    ) -> None:
        super().__init__(train_data, test_data, cat_features, num_features, synth_data)
        self._first_half = first_half
        self._second_half = second_half
        self._runs_count = runs_count

    # ──────────────────────────────────────────────────────────────
    # public
    # ──────────────────────────────────────────────────────────────
    def process(self, results: PipelineResults) -> None:
        """Run Anonymeter N times and store every run in results."""
        print("Privacy-Anonymeter evaluation in progress …")

        # Use freshly generated synthetic data if present
        synth_df = (
            results.get("generation_results", {}).get("synthetic_data")
            if "generation_results" in results
            else self.synth_data
        )
        self.synth_data = synth_df

        # Prepare the destination mapping
        results["privacy_anonymeter_results"] = {}

        for run in range(self._runs_count):
            evaluator = PrivacyEvaluatorAnonymeter(
                self.train_data, self.synth_data, self.test_data
            )

            evaluator.configure_singling_uni_attacks()
            evaluator.configure_singling_multi_attacks()
            evaluator.configure_linkability_attacks(
                attack_number=2000,
                neighbors_number=10,
                first_half=self._first_half,
                second_half=self._second_half,
            )

            # — run the three attack families —
            singling_uni = evaluator.evaluate_singling_uni_attacks()
            while singling_uni is None:          # defensive retry
                singling_uni = evaluator.evaluate_singling_uni_attacks()

            singling_multi = evaluator.evaluate_singling_multi_attacks()
            linkability    = evaluator.evaluate_linkability_attacks()

            run_key = f"run_{run}"
            results["privacy_anonymeter_results"][run_key] = (
                PrivacyAnonymeterEvaluationResults(
                    singling_univariate   = singling_uni,
                    singling_multivariate = singling_multi,
                    linkability_attacks   = linkability,
                )
            )

        print("Privacy-Anonymeter evaluation complete.")
