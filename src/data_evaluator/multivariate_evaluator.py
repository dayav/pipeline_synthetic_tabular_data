from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats.contingency import association
from dython.nominal import correlation_ratio, associations

from .base_evaluator import BaseEvaluator

class MultivariateEvaluator(BaseEvaluator) :

    def __init__(self, real : pd.DataFrame, synth : pd.DataFrame) -> None :
        """
        Initializes the MultivariateEvaluator class with real and synthetic datasets.

        This class extends the BaseEvaluator to handle multivariate evaluation of both numerical 
        and categorical data in the real and synthetic datasets.

        Args:
            real (pd.DataFrame): The real dataset in the form of a pandas DataFrame.
            synth (pd.DataFrame): The synthetic dataset in the form of a pandas DataFrame.

        Attributes:
            num_multi (None): Placeholder for storing multivariate evaluation results for numerical columns. 
                Initially set to None.
            cat_multi (None): Placeholder for storing multivariate evaluation results for categorical columns. 
                Initially set to None.
            num_cat_multi (None): Placeholder for storing multivariate evaluation results that involve both 
                numerical and categorical columns. Initially set to None.

        """
        super().__init__(real, synth)
        self.num_multi = None
        self.cat_multi = None
        self.num_cat_multi = None

    def _extract_upper_triangle(self,  df : pd.DataFrame):
        '''Returns the upper triangle of a correlation matrix.
        You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
        Args:
        df: pandas or numpy correlation matrix
        Returns:
        list of values from upper triangle
        '''
        try:
            assert(type(df)==np.ndarray)
        except:
            if type(df)==pd.DataFrame:
                df = df.values
            else:
                raise TypeError('Must be np.ndarray or pd.DataFrame')
        mask = np.triu_indices(df.shape[0], k=1)
        return df[mask]


    def _get_spearman_correlation_coefficient(
        self,
        real_correlation_matrix: pd.DataFrame,
        synthetic_correlation_matrix: pd.DataFrame,
        only_upper_matrix: bool
    ) -> Tuple[float, float]:
        """
        Compute the Spearman correlation coefficient and p-value between two correlation matrices.

        Parameters:
            real_correlation_matrix (pd.DataFrame): The real correlation matrix.
            synthetic_correlation_matrix (pd.DataFrame): The synthetic correlation matrix.
            only_upper_matrix (bool): If True, calculate Spearman correlation using only 
                the upper triangle of the matrices (excluding the diagonal).

        Returns:
            Tuple[float, float]: The Spearman correlation coefficient and p-value.

        """

        # Extract data based on the `only_upper_matrix` flag
        if only_upper_matrix:
            real_data = self._extract_upper_triangle(real_correlation_matrix)
            synthetic_data = self._extract_upper_triangle(synthetic_correlation_matrix)
        else:
            real_data = real_correlation_matrix
            synthetic_data = synthetic_correlation_matrix

        # Compute Spearman correlation
        correlation_result = spearmanr(real_data, synthetic_data)

        return correlation_result.statistic, correlation_result.pvalue

   
    def get_pearson_correlation_matrix(self, dataframe : pd.DataFrame ) -> tuple :
        """
        Computes the Pearson correlation matrix for the numerical columns of the given dataset and 
        calculates the Frobenius norm of the matrix.

        This function calculates the absolute values of the Pearson pairwise correlation matrix for 
        the numerical columns in the provided dataset. It also computes the Frobenius norm of the correlation 
        matrix as a summary measure of the overall correlation structure.

        Args:
            dataframe (pd.DataFrame): The dataset for which to compute the Pearson correlation matrix. 
                The dataset should contain numerical columns.

        Returns:
            tuple: A tuple containing:
                - cors (pd.DataFrame): The absolute Pearson correlation matrix of numerical columns.
                - cors_norm (float): The Frobenius norm of the Pearson correlation matrix, rounded to 4 decimal places.

        Example:
            cors_matrix, cors_norm = get_pearson_correlation_matrix(dataframe)
            This returns the Pearson correlation matrix and its normalized value for further analysis.

        Notes:
            - Pearson correlation is a measure of the linear relationship between two variables.
            - The Frobenius norm provides a summary measure of the magnitude of the correlations in the matrix.
            - The `numeric_only=True` argument ensures that only numerical columns are used in the correlation calculation.
        """
        #compute the pearson pairwise correlation matrix of numerical attributes of the dataset
        cors = np.absolute(dataframe.corr(method='pearson', numeric_only=True))

        #compute the norm of the pearson pairwise correlation matrix computed before
        cors_norm = np.round(np.linalg.norm(cors),4)

        #return the values
        return cors, cors_norm
    
    def evaluate_pearson_correlation_matrix_diff(self) -> None:
        """
        Evaluates the difference between the Pearson correlation matrices of the real and synthetic datasets.

        This function computes the Pearson correlation matrices for the numerical columns of both the real and 
        synthetic datasets. It calculates the absolute difference between the Frobenius norms of these matrices, 
        which quantifies how different the correlation structures are between the two datasets. The results are 
        stored in the object's attributes for further analysis.

        Returns:
            None: The function does not return a value but updates the following attributes:
                - pearson_real (pd.DataFrame): The Pearson correlation matrix for the real dataset.
                - pearson_synth (pd.DataFrame): The Pearson correlation matrix for the synthetic dataset.
                - pearson_norm_diff (float): The absolute difference between the Frobenius norms of the real 
                and synthetic Pearson correlation matrices.
                - num_multi (dict): A dictionary containing the real and synthetic Pearson correlation matrices 
                and the difference in their Frobenius norms.

        Attributes:
            pearson_real (pd.DataFrame): Stores the Pearson correlation matrix for the real dataset.
            pearson_synth (pd.DataFrame): Stores the Pearson correlation matrix for the synthetic dataset.
            pearson_norm_diff (float): Stores the absolute difference between the Frobenius norms of the real 
                and synthetic Pearson correlation matrices.
            num_multi (dict): Stores the Pearson correlation matrices and norm difference for multivariate 
                analysis of numerical data.

        Example:
            After calling this function, the Pearson correlation matrices and the difference in their 
            Frobenius norms will be available in `self.num_multi` for further analysis.

        Notes:
            - The Pearson correlation matrix measures the linear relationships between numerical columns.
            - The Frobenius norm difference quantifies how similar or different the correlation structures are 
            between the real and synthetic datasets.
        """
       
        corr_real , real_norm = self.get_pearson_correlation_matrix(self._real)
        corr_synth , synth_norm = self.get_pearson_correlation_matrix(self._synth)

        self.pearson_real = corr_real
        self.pearson_synth = corr_synth
        self.pearson_norm_diff = np.round(np.linalg.norm(pd.Series.abs(corr_real - corr_synth)),4)
        self.pearson_spearman_correlation_coefficient, _ =  self._get_spearman_correlation_coefficient(corr_real, corr_synth, True)

        self.num_multi = {
            'pearson_real' : corr_real, 
            'pearson_synth' : corr_synth,
            'pearson_norm_diff' : self.pearson_norm_diff,
            'pearson_spearman_correlation_coefficient' :  self.pearson_spearman_correlation_coefficient}
    
    def get_pearson_correlation_matrix_diff(self) -> tuple :
        """
        Retrieves the Pearson correlation matrices for the real and synthetic datasets and the difference in their norms.

        This function returns the previously computed Pearson correlation matrices for the real and synthetic datasets, 
        along with the absolute difference in their Frobenius norms. These values are stored in the `num_multi` attribute.

        Returns:
            tuple: A tuple containing:
                - pearson_real (pd.DataFrame): The Pearson correlation matrix for the real dataset.
                - pearson_synth (pd.DataFrame): The Pearson correlation matrix for the synthetic dataset.
                - pearson_norm_diff (float): The absolute difference between the Frobenius norms of the real 
                and synthetic Pearson correlation matrices.

        Example:
            pearson_real, pearson_synth, pearson_norm_diff = get_pearson_correlation_matrix_diff()
            This retrieves the Pearson correlation matrices for further analysis or comparison.

        Notes:
            - The Pearson correlation matrix measures the linear relationship between numerical variables.
            - The Frobenius norm difference quantifies how different the correlation structures are between 
            the real and synthetic datasets.
        """

        return self.num_multi['pearson_real'], self.num_multi['pearson_synth'] , self.num_multi['pearson_norm_diff'], self.num_multi['pearson_spearman_correlation_coefficient']
          
    
    def get_cramerV_correlation_matrix(self, data : pd.DataFrame) -> tuple :

        """
        Computes the Cramér's V correlation matrix for the categorical columns in the given dataset.

        This function calculates the pairwise Cramér's V correlation between all categorical columns in the dataset. 
        Cramér's V is a measure of association between two categorical variables. The function also computes 
        the Frobenius norm of the resulting correlation matrix.

        Args:
            data (pd.DataFrame): The dataset containing the categorical columns for which Cramér's V correlations 
                are computed.

        Returns:
            tuple: A tuple containing:
                - cramer (pd.DataFrame): A DataFrame containing the Cramér's V correlation matrix for the categorical columns.
                - cors_norm (float): The Frobenius norm of the Cramér's V correlation matrix, rounded to 4 decimal places.

        Example:
            cramer_matrix, cors_norm = get_cramerV_correlation_matrix(data)
            This returns the Cramér's V correlation matrix and its normalized value for further analysis.

        Notes:
            - The Cramér's V correlation is computed using the `association` function from `dython`, which takes 
            a contingency table created via `pd.crosstab`.
            - The function initializes an identity matrix for the Cramér's V correlations, and then updates 
            the matrix with pairwise Cramér's V values for each combination of categorical columns.
            - The Frobenius norm of the Cramér's V correlation matrix is calculated using `np.linalg.norm`.
        """

        cramer = pd.DataFrame(
        np.eye(len(self._categorical_columns)),
        columns=self._categorical_columns,
        index=self._categorical_columns)

        all_combinations = combinations(self._categorical_columns, r=2)

        for comb in all_combinations:
            i = comb[0]
            j = comb[1]

            input_tab = pd.crosstab(data[i], data[j])

            res_cramer = association(input_tab, method='cramer')
            cramer[i][j], cramer[j][i] = res_cramer, res_cramer

        cors_norm = np.round(np.linalg.norm(cramer),4)

        return cramer, cors_norm
    
    def get_correlation_ratio_matrix(self, data : pd.DataFrame) -> tuple :
        """
        Computes the correlation ratio matrix between categorical and numerical columns in the given dataset.

        This function calculates the correlation ratio matrix, which shows the relationships between 
        categorical and numerical columns in the provided dataset. It uses the `associations` function 
        to compute these correlations, and also returns the normalized Frobenius norm of the correlation matrix.

        Args:
            data (pd.DataFrame): The dataset for which to compute the correlation ratio matrix, containing 
                both categorical and numerical columns.

        Returns:
            tuple: A tuple containing:
                - correlation_matrix (dict): A dictionary containing the correlation matrix and additional 
                information calculated by the `associations` function.
                - cors_norm (float): The Frobenius norm of the correlation matrix, rounded to 4 decimal places.

        Example:
            correlation_matrix, cors_norm = get_correlation_ratio_matrix(data)
            This returns the correlation matrix and the normalized correlation value for further analysis.

        Notes:
            - The `associations` function is used to compute correlations between nominal (categorical) 
            and numerical columns, hiding rows and columns based on the input types.
            - The Frobenius norm of the correlation matrix is computed using `np.linalg.norm`.
        """
        
        if (self._categorical_columns.size == 0 or self._numerical_columns.size == 0) :
            return None, None

        correlation_matrix = associations(data, 
              nominal_columns=self._categorical_columns.to_list(),
              numerical_columns=self._numerical_columns.to_list(),
              hide_rows=self._categorical_columns.to_list(),
               hide_columns=self._numerical_columns.to_list(),
               plot=False
              )


        cors_norm = np.round(np.linalg.norm(correlation_matrix['corr']),4)

        return correlation_matrix, cors_norm
    
    def evaluate_cramerV_correlation_matrix_diff(self) -> dict :
        """
        Evaluates the difference between the Cramér's V correlation matrices of the real and synthetic datasets.

        This function computes the Cramér's V correlation matrices for the categorical columns in the real and 
        synthetic datasets. It then calculates the absolute difference between their Frobenius norms, which 
        quantifies the structural differences in their correlation patterns. The results are stored in the 
        object's attributes and returned as a dictionary for further analysis.

        Returns:
            dict: A dictionary containing the following:
                - 'cramer_real' (pd.DataFrame): The Cramér's V correlation matrix for the real dataset.
                - 'cramer_synth' (pd.DataFrame): The Cramér's V correlation matrix for the synthetic dataset.
                - 'diff_norm_cramer' (float): The absolute difference between the Frobenius norms of the real 
                and synthetic Cramér's V correlation matrices.

        Attributes:
            cramer_real (pd.DataFrame): The Cramér's V correlation matrix for the real dataset.
            cramer_synth (pd.DataFrame): The Cramér's V correlation matrix for the synthetic dataset.
            diff_norm_cramer (float): The absolute difference between the Frobenius norms of the real and 
                synthetic Cramér's V correlation matrices.
            cat_multi (dict): A dictionary containing:
                - 'cramer_real': The real dataset's Cramér's V correlation matrix.
                - 'cramer_synth': The synthetic dataset's Cramér's V correlation matrix.
                - 'diff_norm_cramer': The absolute Frobenius norm difference between the matrices.

        Example:
            After calling this function:
            >>> self.cat_multi
            {'cramer_real': cramer_real_matrix,
            'cramer_synth': cramer_synth_matrix,
            'diff_norm_cramer': 0.2567}

        Notes:
            - The Frobenius norm difference provides a quantitative measure of the similarity between 
            the correlation structures of the real and synthetic datasets.
            - This function is designed for datasets with categorical columns.
        """

        cramer_real, real_norm = self.get_cramerV_correlation_matrix(self._real)
        cramer_synth, synth_norm = self.get_cramerV_correlation_matrix(self._synth)

        self.cramer_real = cramer_real
        self.cramer_synth = cramer_synth
        self.diff_norm_cramer =  np.round(np.linalg.norm(pd.Series.abs(cramer_real - cramer_synth)),4)
        self.cramer_spearman_correlation_coefficient, _ =  self._get_spearman_correlation_coefficient(cramer_real, cramer_synth, True)

        self.cat_multi = {
            'cramer_real' : cramer_real,
            'cramer_synth' : cramer_synth,
            'diff_norm_cramer' : self.diff_norm_cramer,
            'cramer_spearman_correlation_coefficient' : self.cramer_spearman_correlation_coefficient}
    
    def get_cramerV_correlation_matrix_diff(self) -> tuple :
        """
        Retrieves the Cramér's V correlation matrices for the real and synthetic datasets and the difference in their norms.

        This function returns the previously computed Cramér's V correlation matrices for the real and synthetic datasets, 
        along with the absolute difference in their Frobenius norms. These values are stored in the `cat_multi` attribute.

        Returns:
            tuple: A tuple containing:
                - cramer_real (pd.DataFrame): The Cramér's V correlation matrix for the real dataset.
                - cramer_synth (pd.DataFrame): The Cramér's V correlation matrix for the synthetic dataset.
                - diff_norm_cramer (float): The absolute difference between the Frobenius norms of the real 
                and synthetic Cramér's V correlation matrices.

        Example:
            cramer_real, cramer_synth, diff_norm_cramer = get_cramerV_correlation_matrix_diff()
            This retrieves the Cramér's V correlation matrices and their norm difference for further analysis or comparison.

        Notes:
            - Cramér's V is a measure of association between categorical variables.
            - The Frobenius norm difference quantifies the difference in the correlation structure between 
            the real and synthetic datasets.
        """

        return self.cat_multi['cramer_real'], self.cat_multi['cramer_synth'] , self.cat_multi['diff_norm_cramer'], self.cat_multi['cramer_spearman_correlation_coefficient']
    
    def evaluate_correlation_ratio_matrix_diff(self) -> None:
        """
        Evaluates the difference between the correlation ratio matrices of the real and synthetic datasets.

        This function calculates the correlation ratio matrices for the real and synthetic datasets, 
        focusing on the relationships between numerical and categorical columns. It computes the absolute 
        difference in the Frobenius norms of the two matrices to quantify how different the correlation 
        structures are between the two datasets. The results are stored in the object's attributes.

        Returns:
            None: The function does not return a value but updates the following attributes:
                - corr_ratio_real (dict): The correlation ratio matrix for the real dataset.
                - corr_ratio_synth (dict): The correlation ratio matrix for the synthetic dataset.
                - diff_norm_corr_ratio (float): The absolute difference between the Frobenius norms 
                of the real and synthetic correlation ratio matrices.
                - num_cat_multi (dict): A dictionary containing the real and synthetic correlation ratio 
                matrices and the difference in their norms.

        Attributes:
            corr_ratio_real (dict): Stores the correlation ratio matrix for the real dataset.
            corr_ratio_synth (dict): Stores the correlation ratio matrix for the synthetic dataset.
            diff_norm_corr_ratio (float): Stores the absolute difference between the Frobenius norms 
                of the real and synthetic correlation ratio matrices.
            num_cat_multi (dict): Stores the correlation ratio matrices and norm difference for 
                multivariate analysis of numerical and categorical data.

        Example:
            After calling this function, the correlation ratio matrices and the difference in their 
            Frobenius norms will be available in `self.num_cat_multi` for further analysis.

        Notes:
            - The correlation ratio matrix helps assess the relationships between categorical and 
            numerical columns, and the Frobenius norm difference quantifies the similarity between 
            the real and synthetic datasets.
        """

        if (self._categorical_columns.size != 0 and self._numerical_columns.size != 0) :
            
            corr_ratio_real, real_norm = self.get_correlation_ratio_matrix(self._real)
            corr_ratio_synth, synth_norm = self.get_correlation_ratio_matrix(self._synth)
            
            self.corr_ratio_real = corr_ratio_real
            self.corr_ratio_synth = corr_ratio_synth
            self.diff_norm_corr_ratio = np.round(np.linalg.norm(pd.Series.abs(corr_ratio_real['corr'] - corr_ratio_synth['corr'])),4)
            self.corr_ratio_spearman_correlation_coefficient, _ =  self._get_spearman_correlation_coefficient(corr_ratio_real['corr'].to_numpy().ravel(), corr_ratio_synth['corr'].to_numpy().ravel(), False)

            self.num_cat_multi = {
                'corr_ratio_real' : corr_ratio_real,
                'corr_ratio_synth': corr_ratio_synth,
                'diff_norm_corr_ratio' : self.diff_norm_corr_ratio,
                'corr_ratio_spearman_correlation_coefficient' : self.corr_ratio_spearman_correlation_coefficient}
    
    def get_correlation_ratio_matrix_diff(self) -> tuple:
        """
        Retrieves the correlation ratio matrices for the real and synthetic datasets and the difference in their norms.

        This function returns the previously computed correlation ratio matrices for the real and synthetic datasets, 
        which describe the relationships between categorical and numerical variables, along with the absolute difference 
        in their Frobenius norms. These values are stored in the `num_cat_multi` attribute.

        Returns:
            tuple: A tuple containing:
                - corr_ratio_real (dict): The correlation ratio matrix for the real dataset.
                - corr_ratio_synth (dict): The correlation ratio matrix for the synthetic dataset.
                - diff_norm_corr_ratio (float): The absolute difference between the Frobenius norms of the real 
                and synthetic correlation ratio matrices.

        Example:
            corr_ratio_real, corr_ratio_synth, diff_norm_corr_ratio = get_correlation_ratio_matrix_diff()
            This retrieves the correlation ratio matrices and their norm difference for further analysis or comparison.

        Notes:
            - The correlation ratio matrix captures the relationships between numerical and categorical variables.
            - The Frobenius norm difference quantifies how different the correlation structures are between the 
            real and synthetic datasets.
        """
   
        return self.num_cat_multi['corr_ratio_real'], self.num_cat_multi['corr_ratio_synth'], self.num_cat_multi['diff_norm_corr_ratio'], self.num_cat_multi['corr_ratio_spearman_correlation_coefficient']