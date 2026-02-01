from abc import ABC
import pandas as pd

class BaseEvaluator(ABC) :
    def __init__(self, real : pd.DataFrame, synth : pd.DataFrame) -> None :
        """
        Initializes the BaseEvaluator class with real and synthetic datasets.

        This class serves as a base class for evaluating the similarity between real and synthetic datasets. 
        It identifies and stores the categorical and numerical columns for further analysis.

        Args:
            real (pd.DataFrame): The real dataset in the form of a pandas DataFrame.
            synth (pd.DataFrame): The synthetic dataset in the form of a pandas DataFrame.

        Attributes:
            _real (pd.DataFrame): Internal storage for the real dataset.
            _synth (pd.DataFrame): Internal storage for the synthetic dataset.
            _categorical_columns (pd.Index): The names of the categorical columns in the real dataset.
            _numerical_columns (pd.Index): The names of the numerical columns in the real dataset.

        """
        self._real = real
        self._synth = synth
        self._categorical_columns = real.select_dtypes(include='category').columns
        self._numerical_columns = real.select_dtypes(exclude='category').columns
