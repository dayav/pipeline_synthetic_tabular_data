from typing import NamedTuple, Optional

from anonymeter.evaluators import LinkabilityEvaluator, SinglingOutEvaluator
from pandas import DataFrame

from ..base_evaluator import BaseEvaluator

class AnonymeterResults(NamedTuple):
    attacks_numbers: int
    attacks_succeeded: int
    privacy_risk_original: int
    privacy_risk_control: int
    privacy_risk_naive: int
    specific_privacy: int
    specific_privacy_ci: int = 0


class PrivacyEvaluatorAnonymeter(BaseEvaluator) :

    def __init__(self,  real_train: DataFrame, synth: DataFrame, real_test: DataFrame) -> None:
        """
        Initialize the PrivacyEvaluatorAnonymeter.

        Parameters:
        - real_train: The real training data.
        - synth: The synthetic data.
        - real_test: The real testing data.
        """
        super().__init__(real_train, synth)
        self._real_test = real_test
        self._singling_uni : Optional[SinglingOutEvaluator] = None
        self._singling_multi : Optional[SinglingOutEvaluator] = None
        self.linkability : Optional[LinkabilityEvaluator] = None

    def configure_singling_uni_attacks(self, attack_number: int = 2000) -> None:
        """
        Configure the evaluator to perform univariate singling out attacks.

        Parameters:
        - attack_number: The number of attacks to attempt (default is 500).
        """

        self._singling_uni = SinglingOutEvaluator(ori=self._real, 
                                 syn=self._synth, 
                                 control=self._real_test,
                                 n_attacks=attack_number)

    def configure_singling_multi_attacks(self, attack_number : int =2000, column_number : int = 4) -> None:
        """
        Configure the evaluator to perform multivariate singling out attacks.

        Parameters:
        - attack_number: The number of attacks to attempt (default is 100).
        - column_number: The number of columns to consider in each attack (default is 4).
        """

        self._singling_multi = SinglingOutEvaluator(ori=self._real, 
                                 syn=self._synth, 
                                 control=self._real_test,
                                 n_attacks=attack_number, # this attack takes longer
                                 n_cols=column_number)     
      
    def configure_linkability_attacks(self, attack_number : int = 2000, neighbors_number : int = 10, first_half : list = None, second_half : list = None) -> None:
        """
        Configure the evaluator to perform linkability attacks.

        Parameters:
        - attack_number: The number of attacks to attempt (default is 2000).
        - neighbors_number: The number of nearest neighbors to consider in the evaluation (default is 10).
        """

        if (first_half == None and second_half == None) :
            columns = self._real.columns
            midpoint = len(columns) // 2

            # Split the columns into two halves
            first_half = columns[:midpoint]
            second_half = columns[midpoint:]


        self.linkability = LinkabilityEvaluator(ori=self._real, 
                                 syn=self._synth, 
                                 control=self._real_test,
                                 n_attacks=attack_number,
                                 aux_cols=[first_half, second_half],
                                 n_neighbors=neighbors_number)
        

    def evaluate_singling_uni_attacks(self) -> AnonymeterResults:

        if self._singling_uni is None:
            raise ValueError("The univariate singling out evaluator has not been configured."
                             " Please configure it before evaluation.")

        try:
            self._singling_uni.evaluate(mode='univariate')
            results_evaluation = self._singling_uni.results()
            results = AnonymeterResults(attacks_numbers = self._singling_uni._n_attacks,
                                        attacks_succeeded = self._singling_uni._n_success,
                                        privacy_risk_original = results_evaluation.attack_rate.value,
                                        privacy_risk_control = results_evaluation.control_rate.value,
                                        privacy_risk_naive = results_evaluation.baseline_rate.value,
                                        specific_privacy = results_evaluation.risk().value,
                                        specific_privacy_ci = results_evaluation.risk().ci)
            return results

        except RuntimeError as ex: 
            print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
                  "For more stable results increase `n_attacks`. Note that this will "
                  "make the evaluation slower.")
            
    def evaluate_singling_multi_attacks(self) -> AnonymeterResults:

        if self._singling_multi is None:
            raise ValueError("The multivariate singling out evaluator has not been configured."
                             " Please configure it before evaluation.")

        try:
            self._singling_uni.evaluate(mode='multivariate')
            results_evaluation = self._singling_uni.results()
            results = AnonymeterResults(attacks_numbers = self._singling_uni._n_attacks,
                                        attacks_succeeded = self._singling_uni._n_success,
                                        privacy_risk_original = results_evaluation.attack_rate.value,
                                        privacy_risk_control = results_evaluation.control_rate.value,
                                        privacy_risk_naive = results_evaluation.baseline_rate.value,
                                        specific_privacy = results_evaluation.risk().value,
                                        specific_privacy_ci = results_evaluation.risk().ci)
            return results

        except RuntimeError as ex: 
            print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
                  "For more stable results increase `n_attacks`. Note that this will "
                  "make the evaluation slower.")
            
    def evaluate_linkability_attacks(self) -> AnonymeterResults:

        if self.linkability is None:
            raise ValueError("The multivariate singling out evaluator has not been configured."
                             " Please configure it before evaluation.")

        try:
            self.linkability.evaluate(n_jobs=-2)
            results_evaluation = self.linkability.results()
            results = AnonymeterResults(attacks_numbers = self.linkability._n_attacks,
                                        attacks_succeeded = self.linkability._n_attacks,
                                        privacy_risk_original = results_evaluation.attack_rate.value,
                                        privacy_risk_control = results_evaluation.control_rate.value,
                                        privacy_risk_naive = results_evaluation.baseline_rate.value,
                                        specific_privacy = results_evaluation.risk().value,
                                        specific_privacy_ci = results_evaluation.risk().ci)
            return results

        except RuntimeError as ex: 
            print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
                  "For more stable results increase `n_attacks`. Note that this will "
                  "make the evaluation slower.")
