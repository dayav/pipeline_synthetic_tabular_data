from typing import List

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import (
    chi2_contingency,
    gaussian_kde,
    ks_2samp,
    kstest,
    mannwhitneyu,
    ttest_ind,
    wilcoxon,
)

from .base_evaluator import BaseEvaluator

class UnivariateEvaluator(BaseEvaluator) :

    def __init__(self, real : pd.DataFrame, synth : pd.DataFrame) -> None :
        """
        Initializes the class with real and synthetic DataFrames, and identifies 
        categorical and numerical columns from the real DataFrame.

        Args:
            real (pd.DataFrame): The real dataset in the form of a pandas DataFrame.
            synth (pd.DataFrame): The synthetic dataset in the form of a pandas DataFrame.

        Attributes:
            _real (pd.DataFrame): Internal storage for the real DataFrame.
            _synth (pd.DataFrame): Internal storage for the synthetic DataFrame.
            _categorical_columns (pd.Index): The column names from the real DataFrame 
                that are categorized as categorical.
            _numerical_columns (pd.Index): The column names from the real DataFrame 
                that are categorized as numerical.
            cat_uni (None or pd.Series): Placeholder attribute for storing categorical 
                univariate analysis. Initially set to None.
            num_uni (None or pd.Series): Placeholder attribute for storing numerical 
                univariate analysis. Initially set to None.
        """

        self._real = real
        self._synth = synth
        self._categorical_columns = real.select_dtypes(include='category').columns
        self._numerical_columns = real.select_dtypes(exclude='category').columns

        self.cat_uni = None
        self.num_uni = None
    
    def style_negative(v, props=''):
        return props if v < 0.05 else None
    
    def style_binary(v, props=''):
        return props if v > 0.5 else None
    
    
    def student_Ttest(self) -> List[float] :
        """
        Performs the Student's t-test to compare the means of numerical columns between 
        the real and synthetic datasets.

        This function applies the independent two-sample Student's t-test to determine 
        whether there is a significant difference between the means of each numerical 
        column in the real and synthetic datasets.

        Returns:
            List[float]: A list of p-values from the Student's t-test, one for each numerical column.

        Example:
            After calling this function, it returns a list of p-values corresponding to each numerical column.
            A small p-value suggests that the means of the real and synthetic data for that column are 
            significantly different.

        Notes:
            - The `ttest_ind` function from `scipy` is used to perform the t-test.
            - This test assumes that the data is normally distributed and that the samples are independent.
        """
        p_s = []
        for col in self._numerical_columns :
            _, p = ttest_ind(self._real[col], self._synth[col])
            p_s.append(p)
        
        return p_s
    
    def Welchs_Ttest(self)-> List[float] :
        """
        Performs Welch's t-test to compare the means of numerical columns between 
        the real and synthetic datasets, without assuming equal variances.

        This function applies Welch's t-test, a variation of the independent two-sample 
        t-test that does not assume equal variances between the two samples (real and synthetic datasets).

        Returns:
            List[float]: A list of p-values from Welch's t-test, one for each numerical column.

        Example:
            After calling this function, it returns a list of p-values corresponding to each numerical column.
            A small p-value indicates that the means of the real and synthetic data for that column are 
            significantly different, taking into account possible unequal variances.

        Notes:
            - The `ttest_ind` function from `scipy` is used with `equal_var=False` to perform Welch's t-test.
            - Welch's t-test is used when the assumption of equal variances is not valid, making it more robust 
            in the case of unequal sample variances.
        """
        p_s = []
        for col in self._numerical_columns :
            _, p = ttest_ind(self._real[col], self._synth[col], equal_var=False)
            p_s.append(p)
        
        return p_s

    def mann_whitney_Utest(self) -> List[float] :
        """
        Performs the Mann-Whitney U test to compare the distributions of numerical columns 
        between the real and synthetic datasets.

        This function applies the Mann-Whitney U test, a non-parametric test that compares 
        whether two independent samples (in this case, numerical columns from the real and 
        synthetic datasets) come from the same distribution.

        Returns:
            List[float]: A list of p-values from the Mann-Whitney U test, one for each numerical column.

        Example:
            After calling this function, it returns a list of p-values, each corresponding to a 
            numerical column in the dataset. A small p-value suggests that the distribution 
            of the real and synthetic data for that column are significantly different.

        Notes:
            - The `mannwhitneyu` function from `scipy` is used to perform the test.
            - This test is non-parametric and does not assume normal distribution of the data.
        """
        p_s = []
        for col in self._numerical_columns :
            stat, p = mannwhitneyu(self._real[col], self._synth[col])
            p_s.append(p)
        return p_s
    
    def wilcoxon_test(self) -> List[float]:
        """
        Performs the Wilcoxon signed-rank test to compare paired numerical columns 
        between the real and synthetic datasets.

        This function applies the Wilcoxon signed-rank test, a non-parametric test used to compare 
        two related samples (in this case, numerical columns from the real and synthetic datasets) 
        to determine whether their population mean ranks differ.

        Returns:
            List[float]: A list of p-values from the Wilcoxon signed-rank test, one for each numerical column.

        Example:
            After calling this function, it returns a list of p-values corresponding to each numerical column.
            A small p-value indicates a significant difference in the distributions of the real and synthetic data 
            for that column.

        Notes:
            - The `wilcoxon` function from `scipy` is used to perform the test.
            - The Wilcoxon test is designed for paired or related samples and does not assume normal distribution.
            - It should only be used when comparing matched or paired data.
        """
        p_s = []
        for col in self._numerical_columns :
            stat, p = wilcoxon(self._real[col], self._synth[col])
            p_s.append(p)
        return p_s
    
    def ks_test(self) -> List[float]:
        ks = []
        ks_2samp_ = []
        ks_2samp_p_value = []
        for col in self._numerical_columns:
            ks.append(kstest(self._real[col],'norm')[0]-kstest(self._synth[col],'norm')[0])
            ks_2 = ks_2samp(self._real[col], self._synth[col])
            ks_2samp_.append(ks_2[0])
            ks_2samp_p_value.append(ks_2[1])
        return ks, ks_2samp_, ks_2samp_p_value
    
    def chisquared_tests(self) -> List[float]:
        """
        Performs chi-squared tests to compare the categorical distributions of the real and synthetic datasets 
        for each categorical column.

        This function calculates the p-values from chi-squared tests on contingency tables of observed frequencies 
        in the real and synthetic datasets for each categorical column. The chi-squared test evaluates whether there 
        is a significant difference between the distributions of categorical values in the two datasets.

        Returns:
            List[float]: A list of p-values, one for each categorical column, indicating the result of the chi-squared test.

        Notes:
            - The function constructs contingency tables using `pd.crosstab` to compute observed frequencies 
            between the real and synthetic datasets for each categorical column.
            - The `chi2_contingency` function from `scipy` is used to perform the chi-squared test.

        Example:
            For each categorical column, the function creates a contingency table, performs the chi-squared test, 
            and appends the p-value to the result list.
        
        """
        p_values = []

        for c in self._categorical_columns :
            
            observed = pd.crosstab(self._real[c], self._synth[c], dropna=True)
            #perform chi-squared test
            stat, p, _, _ = chi2_contingency(observed)
            p_values.append(p)

        return p_values
    
    def get_mode_collapse_values(self) :
         return self.cat_uni['mode_collapse_dict']
    
    def get_mode_collapse(self) :
        mode_collapse_dict = {}
        for c in self._categorical_columns :
            observed = pd.crosstab(self._real[c], self._synth[c], dropna=False)
            
            not_mode_collapse = set(self._synth[c].unique()) == set(self._real[c].unique())
            if  not not_mode_collapse :
                cats = pd.Categorical(self._synth[c], categories=self._real[c].unique())
                cat_freq = cats.value_counts()
                cat_freq[cat_freq == 0].index.to_list()
                mode_collapse_dict[c] = cat_freq[cat_freq == 0].index.to_list()
        return mode_collapse_dict
                
    def kl_divergence_categorical(self) -> List[float] :
        """
        Computes the Kullback-Leibler (KL) divergence between the categorical distributions 
        of the real and synthetic datasets for each categorical column.

        This function calculates the KL divergence for each categorical column in the real 
        dataset by comparing the probability distributions of the categories in the real 
        and synthetic datasets.

        Returns:
            List[float]: A list of KL divergence values, one for each categorical column.
        
        Notes:
            - The function handles cases where categories in the real dataset may not 
            exist in the synthetic dataset by reindexing and filling missing values 
            with a small constant (1e-7).
            - KL divergence is computed using the `rel_entr` function, which calculates 
            the relative entropy.

        Example:
            For each categorical column, the function computes the real and synthetic 
            probabilities, adjusts for missing categories, and computes KL divergence.

        """
        kl_divergence = []

        for c in self._categorical_columns:
            real_prob = [i/self._real[c].shape[0] for i in self._real[c].value_counts().sort_index(ascending=True).reindex(self._real[c].unique(), fill_value=1e-7)]
            synthetic_prob = [i/self._synth[c].shape[0] for i in self._synth[c].value_counts().sort_index(ascending=True).reindex(self._real[c].unique(), fill_value=1e-7)]#ReAdjust epsilon
            
            kl_divergence.append(sum(rel_entr(real_prob, synthetic_prob)))

        return kl_divergence
    
    def jensen_shanon_divergence_categorical(self) -> List[float]:
        """
        Computes the Jensen-Shannon (JS) divergence between the categorical distributions 
        of the real and synthetic datasets for each categorical column.

        This function calculates the JS divergence, which is a symmetric measure of the 
        similarity between two probability distributions. It is calculated between the 
        categorical distributions in the real and synthetic datasets for each categorical column.

        Returns:
            List[float]: A list of Jensen-Shannon divergence values, one for each categorical column.
        
        Notes:
            - The function handles cases where categories in the real dataset may not exist 
            in the synthetic dataset by reindexing and filling missing values with a small 
            constant (1e-7).
            - JS divergence is computed using the `jensenshannon` function from `scipy`, with 
            base 2 logarithms.

        Example:
            For each categorical column, the function computes the real and synthetic 
            probabilities, adjusts for missing categories, and computes the JS divergence.
        
        """
        js_divergence = []

        for c in self._categorical_columns:
            real_prob = np.array([i/self._real[c].shape[0] for i in self._real[c].value_counts().sort_index(ascending=True).reindex(self._real[c].unique(), fill_value=1e-7)])
            synthetic_prob = np.array([i/self._synth[c].shape[0] for i in self._synth[c].value_counts().sort_index(ascending=True).reindex(self._real[c].unique(), fill_value=1e-7)])
            js_pq = jensenshannon(real_prob, synthetic_prob, base=2)

            js_divergence.append(js_pq)

        return js_divergence
    
    
    def kl_divergence_numerical(self) -> List[float] :
        """
        Computes the Kullback-Leibler (KL) divergence between the numerical distributions 
        of the real and synthetic datasets for each numerical column.

        This function estimates the probability density functions (PDFs) of the numerical data 
        in both the real and synthetic datasets using kernel density estimation (KDE). 
        It then calculates the KL divergence, which measures the difference between the two 
        distributions for each numerical column.

        Returns:
            List[float]: A list of KL divergence values, one for each numerical column.

        Process:
            1. For each numerical column, the function performs KDE to estimate the PDFs 
            of the real and synthetic datasets.
            2. The PDFs are evaluated over a common range of values (`x_range`) spanning 
            the minimum and maximum values from both datasets.
            3. KL divergence is computed using the `rel_entr` function, and small values in the PDFs 
            are clipped to avoid division by zero.

        Example:
            After calling this function, it returns a list of KL divergence values. A larger KL divergence 
            indicates a greater difference between the distributions of the real and synthetic data 
            for a particular column.

        Notes:
            - The function uses `gaussian_kde` from `scipy` for kernel density estimation.
            - The PDFs are evaluated over 1000 points in a range defined by the min and max values 
            from both datasets.
            - An epsilon value of `1e-10` is used to clip the PDFs to avoid computational issues 
            such as division by zero.

        """

        kl_divergence = []

        for c in self._numerical_columns:
            data1 = self._real[c].values
            data2 = self._synth[c].values

            # Perform kernel density estimation (KDE) for both datasets
            kde1 = gaussian_kde(data1)
            kde2 = gaussian_kde(data2)

            # Define the range over which to evaluate the PDFs 
            x_range = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)

            # Compute the PDFs using KDE for the x_range
            pdf1 = kde1(x_range)
            pdf2 = kde2(x_range)

            epsilon = 1e-10
            pdf1 = np.clip(pdf1, epsilon, None)
            pdf2 = np.clip(pdf2, epsilon, None)

            pdf1 /= np.sum(pdf1)
            pdf2 /= np.sum(pdf2)
    
            kl_divergence.append(np.sum(rel_entr(pdf1, pdf2)))

        return kl_divergence
    
    def jensen_shanon_divergence_numerical(self) -> List[float] :
        """
        Computes the Jensen-Shannon (JS) divergence between the numerical distributions 
        of the real and synthetic datasets for each numerical column.

        This function estimates the probability density functions (PDFs) of the numerical data 
        in both the real and synthetic datasets using kernel density estimation (KDE). 
        It then calculates the Jensen-Shannon divergence, which measures the similarity 
        between the two distributions for each numerical column.

        Returns:
            List[float]: A list of Jensen-Shannon divergence values, one for each numerical column.

        Process:
            1. For each numerical column, the function performs KDE to estimate the PDFs 
            of the real and synthetic datasets.
            2. The PDFs are evaluated over a common range of values (`x_range`) spanning 
            the minimum and maximum values from both datasets.
            3. The Jensen-Shannon divergence is computed using the `jensenshannon` function, 
            with base 2 logarithms. Small values in the PDFs are clipped to avoid 
            division by zero.

        Example:
            After calling this function, it returns a list of Jensen-Shannon divergence values. 
            A smaller value indicates more similarity between the distributions of the real and 
            synthetic data for a particular column.

        Notes:
            - The function uses `gaussian_kde` from `scipy` for kernel density estimation.
            - The PDFs are evaluated over 1000 points in a range defined by the min and max values 
            from both datasets.
            - An epsilon value of `1e-10` is used to clip the PDFs to avoid computational issues 
            such as division by zero.
            - Jensen-Shannon divergence is symmetric and bounded between 0 and 1, where 0 indicates 
            identical distributions.
        """

        js_divergence = []

        for c in self._numerical_columns:
            # Define or obtain your two numerical datasets (replace these with your data)
            data1 = self._real[c].values
            data2 = self._synth[c].values

            # Perform kernel density estimation (KDE) for both datasets
            kde1 = gaussian_kde(data1)
            kde2 = gaussian_kde(data2)

            # Define the range over which to evaluate the PDFs 
            x_range = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)

            # Compute the PDFs using KDE for the x_range
            pdf1 = kde1(x_range)
            pdf2 = kde2(x_range)

            epsilon = 1e-10
            pdf1 = np.clip(pdf1, epsilon, None)
            pdf2 = np.clip(pdf2, epsilon, None)
            
            js_pq = jensenshannon(pdf1, pdf2, base=2)

            js_divergence.append( js_pq)

        return js_divergence
    
    def cohen_s_d(self) -> List[float] :
        """
        Computes Cohen's d effect size between the numerical columns of the real and synthetic datasets.

        This function calculates Cohen's d, a measure of the effect size that quantifies the difference 
        between the means of two groups (the real and synthetic datasets) for each numerical column. 
        Cohen's d is calculated as the difference in means divided by the pooled standard deviation.

        Returns:
            List[float]: A list of absolute Cohen's d values, one for each numerical column.

        Process:
            1. For each numerical column, the function calculates the mean and sample variance for the 
            real and synthetic datasets.
            2. The pooled standard deviation is computed using the sample variances of the two groups.
            3. Cohen's d is calculated as the difference in means divided by the pooled standard deviation, 
            and the absolute value of Cohen's d is appended to the result list.

        Example:
            After calling this function, it returns a list of absolute Cohen's d values. A higher 
            Cohen's d value indicates a larger effect size, or greater difference, between the distributions 
            of the real and synthetic data for a particular column.

        Notes:
            - The pooled standard deviation accounts for the variability in both the real and synthetic datasets.
            - Cohen's d is a commonly used measure of effect size, where values of 0.2, 0.5, and 0.8 are often 
            interpreted as small, medium, and large effect sizes, respectively.
        """

        cohen_s_d_list = []

        for c in self._numerical_columns:
            # Get the data for the two groups
            group1 = self._real[c].values
            group2 = self._synth[c].values

            # Calculate means and variances
            mean1 = np.mean(group1)
            mean2 = np.mean(group2)
            variance1 = np.var(group1, ddof=1)  # Sample variance
            variance2 = np.var(group2, ddof=1)

            # Calculate pooled standard deviation
            n1 = len(group1)
            n2 = len(group2)
            pooled_std = np.sqrt(((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2))

            # Append the Cohen's d value (absolute to remove directionality)
            cohen_d = (mean1 - mean2) / pooled_std
            cohen_s_d_list.append(abs(cohen_d))
        return cohen_s_d_list
    
    def evaluate_categorical_stat_evaluation(self) -> dict :        
        """
        Evaluates and stores statistical metrics for comparing categorical columns between 
        the real and synthetic datasets.

        This function calculates multiple univariate statistical metrics for the categorical 
        columns in the real and synthetic datasets, including KL divergence, Jensen-Shannon (JS) 
        divergence, and chi-squared test p-values. The results are stored as attributes for 
        further analysis.

        Returns:
            dict: A dictionary containing the following univariate statistical evaluations:
                - 'chi_test' (pd.DataFrame): A DataFrame of chi-squared test p-values for 
                each categorical column.
                - 'jensen_shanon' (pd.DataFrame): A DataFrame containing KL and JS divergence 
                values for each categorical column.
                - 'jensen_shanon_data_mean' (pd.Series): The mean of the JS divergence values 
                across all categorical columns.

        Attributes:
            chi_test (pd.DataFrame): Stores the chi-squared test p-values.
            jensen_shanon (pd.DataFrame): Stores the calculated KL and JS divergence values.
            jensen_shanon_data_mean (pd.Series): Stores the mean of JS divergence values.
            cat_uni (dict): A dictionary with the results of chi-squared tests, JS divergence, 
                and the mean of the JS divergence values.

        Example:
            After calling this function, the categorical statistical metrics are available 
            for further analysis:
            
            - chi-squared test p-values are stored in `self.chi_test`.
            - KL and JS divergence values are stored in `self.jensen_shanon`.
            - The mean JS divergence is stored in `self.jensen_shanon_data_mean`.

        """
        univariate_categorical = dict()
        univariate_categorical['KL_divergence_tests'] = self.kl_divergence_categorical()
        univariate_categorical['JS_divergence'] = self.jensen_shanon_divergence_categorical()

        chi_test = pd.DataFrame(data={'chisquared_tests_p' : self.chisquared_tests()}, index=(self._categorical_columns))
        js = pd.DataFrame(data=univariate_categorical, index=(self._categorical_columns))
        
        self.chi_test = chi_test
        self.jensen_shanon = js
        self.jensen_shanon_data_mean = js.mean()

        self.cat_uni = {'chi_test' : chi_test, 'jensen_shanon' : js, 'jensen_shanon_data_mean': js.mean() }

    def evaluate_mode_collapse_values(self) -> dict :
        """
        Evaluates and identifies mode collapse in the synthetic dataset for each categorical column.

        This function compares the categorical distributions of the real and synthetic datasets 
        to detect mode collapse, where certain categories from the real dataset are missing entirely 
        in the synthetic dataset. For each categorical column, it identifies the categories from 
        the real dataset that do not appear in the synthetic dataset.

        Returns:
            dict: A dictionary where the keys are the names of the categorical columns, and the values 
            are strings of category labels from the real dataset that are missing in the synthetic dataset 
            (mode collapse). Each missing category is listed in the format `category1/category2/...`.

        Updates:
            - If the `cat_uni` attribute is `None`, it initializes it with the mode collapse dictionary.
            - Otherwise, it adds the mode collapse dictionary to the existing `cat_uni` attribute.

        Example:
            After calling this function, the attribute `cat_uni` will contain a 'mode_collapse_dict' 
            entry, which lists categories from the real dataset that are missing in the synthetic dataset 
            for each categorical column.

        Notes:
            - The function handles missing categories in the synthetic dataset by using a contingency table 
            (`pd.crosstab`) and identifying categories with zero frequency.
            - Mode collapse is detected if there is a mismatch in the number of unique categories between 
            the real and synthetic datasets for a given column.
        """
        mode_collapse_dict = self.get_mode_collapse()
        
        if self.cat_uni is None :
            self.cat_uni = {'mode_collapse_dict' : mode_collapse_dict }
        else :
            self.cat_uni['mode_collapse_dict'] = mode_collapse_dict
        
    def evaluate_numerical_stat_evaluation(self) -> dict :
        """
        Evaluates and stores statistical metrics for comparing numerical columns between 
        the real and synthetic datasets.

        This function calculates multiple univariate statistical tests and divergence metrics 
        for the numerical columns in the real and synthetic datasets, including Mann-Whitney U, 
        Wilcoxon, Student's t-test, Welch's t-test, and Kolmogorov-Smirnov (KS) test. Additionally, 
        it computes KL divergence, Jensen-Shannon (JS) divergence, and Cohen's d for each numerical column.

        Returns:
            dict: A dictionary containing the following univariate numerical evaluations:
                - 'univariate_num_s' (pd.DataFrame): A DataFrame of statistical test p-values 
                for Mann-Whitney U, Wilcoxon, Student's t-test, Welch's t-test, and KS test.
                - 'univariate_num_js' (pd.DataFrame): A DataFrame containing divergence metrics 
                (KL divergence, JS divergence, and Cohen's d) for each numerical column.
                - 'univariate_num_js_data_mean' (pd.Series): The mean of the JS divergence metrics 
                across all numerical columns.

        Attributes:
            univariate_num_s (pd.DataFrame): Stores the p-values for the various statistical tests.
            univariate_num_js (pd.DataFrame): Stores the calculated divergence metrics.
            univariate_num_js_data_mean (pd.Series): Stores the mean of JS divergence values.
            num_uni (dict): A dictionary with the results of the statistical tests and divergence metrics.

        Example:
            After calling this function, the numerical statistical metrics are available for further analysis:
            
            - Statistical test p-values are stored in `self.univariate_num_s`.
            - Divergence metrics (KL, JS, Cohen's d) are stored in `self.univariate_num_js`.
            - The mean JS divergence is stored in `self.univariate_num_js_data_mean`.

        Notes:
            - The function performs several statistical tests, including non-parametric (Mann-Whitney, Wilcoxon) 
            and parametric (Student's t-test, Welch's t-test) tests.
            - Divergence metrics (KL, JS) and effect size (Cohen's d) provide insights into the similarity 
            of numerical distributions between the real and synthetic datasets.
        """
        univariate_numeric = dict()
        univariate_numeric['mann_whitney'] = self.mann_whitney_Utest()
        univariate_numeric['wilcoxon'] = self.wilcoxon_test()
        univariate_numeric['student_t'] = self.student_Ttest()
        univariate_numeric['welchs_Ttest'] = self.Welchs_Ttest()
        _, _, univariate_numeric['ks_test_p'] = self.ks_test()

        univariate_num_div = dict()
        univariate_num_div['KL_divergence_tests'] = self.kl_divergence_numerical()
        univariate_num_div['JS_divergence'] = self.jensen_shanon_divergence_numerical()
        univariate_num_div['cohen_s_d'] = self.cohen_s_d()

        df = pd.DataFrame(data=univariate_numeric, index=(self._numerical_columns))        
        js = pd.DataFrame(data=univariate_num_div, index=(self._numerical_columns))

        self.univariate_num_s = df
        self.univariate_num_js = js
        self.univariate_num_js_data_mean = js.mean()
        self.num_uni = {'univariate_num_s' : df, 'univariate_num_js' : js, 'univariate_num_js_data_mean': js.mean() }

    def get_numerical_stat_evaluation(self) -> tuple :
        """
        Retrieves and formats the numerical statistical evaluation results for better visualization.

        This function applies custom styling to the previously calculated univariate statistical metrics 
        for numerical columns, including test results and divergence metrics. The styling highlights 
        specific values, such as negative values and binary conditions, to make interpretation easier.

        Returns:
            tuple: A tuple containing the following styled DataFrames and Series:
                - univariate_num_s (pd.DataFrame): A styled DataFrame of statistical test results, 
                with negative values highlighted in red.
                - univariate_num_js (pd.DataFrame): A styled DataFrame of divergence metrics, 
                with binary conditions in the 'JS_divergence' column highlighted in red.
                - univariate_num_js_data_mean (pd.Series): The mean of JS divergence metrics.

        Notes:
            - The `style_negative` function is applied to highlight negative values in the statistical tests.
            - The `style_binary` function is applied to highlight specific conditions in the JS divergence 
            column for better visualization.
            - The function returns styled DataFrames ready for display or further manipulation.

        Example:
            After calling this function, the returned DataFrames are styled to improve readability 
            and highlight important metrics such as negative values and binary divergence conditions.

        """

        univariate_num_s = self.num_uni['univariate_num_s'].style.applymap(UnivariateEvaluator.style_negative, props='color:red;')
        univariate_num_js = self.num_uni['univariate_num_js'].style.applymap(UnivariateEvaluator.style_binary, props='color:red;', subset= pd.IndexSlice[:, ['JS_divergence']])
        univariate_num_js_data_mean = self.num_uni['univariate_num_js_data_mean']
        
        return univariate_num_s, univariate_num_js, univariate_num_js_data_mean
    
    def get_categorical_stat_evaluation(self) -> tuple :
        """
        Retrieves and formats the categorical statistical evaluation results for better visualization.

        This function applies custom styling to the previously calculated univariate statistical metrics 
        for categorical columns, including chi-squared test results and Jensen-Shannon (JS) divergence metrics. 
        The styling highlights specific values, such as negative values and binary conditions, to make interpretation easier.

        Returns:
            tuple: A tuple containing the following styled DataFrames and Series:
                - chi_test (pd.DataFrame): A styled DataFrame of chi-squared test results, 
                with negative values highlighted in red.
                - jensen_shanon (pd.DataFrame): A styled DataFrame of JS divergence metrics, 
                with binary conditions in the 'JS_divergence' column highlighted in red.
                - jensen_shanon_mean (pd.Series): The mean of JS divergence metrics across all categorical columns.

        Notes:
            - The `style_negative` function is applied to highlight negative values in the chi-squared test results.
            - The `style_binary` function is applied to highlight specific conditions in the JS divergence 
            column for better visualization.
            - The function returns styled DataFrames ready for display or further manipulation.

        Example:
            After calling this function, the returned DataFrames are styled to improve readability 
            and highlight important metrics such as negative values and binary divergence conditions.
        """

        chi_test = self.cat_uni['chi_test'].style.applymap(UnivariateEvaluator.style_negative, props='color:red;')
        jensen_shanon = self.cat_uni['jensen_shanon'].style.applymap(UnivariateEvaluator.style_binary, props='color:red;', subset= pd.IndexSlice[:, ['JS_divergence']])
        jensen_shanon_mean = self.cat_uni['jensen_shanon_data_mean']
        
        return chi_test, jensen_shanon , jensen_shanon_mean
