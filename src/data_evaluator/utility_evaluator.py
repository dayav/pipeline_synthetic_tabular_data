from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.inspection import permutation_importance
from .base_evaluator import BaseEvaluator
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
import numpy as np
import rbo
from scipy import stats

from enum import Enum

class ClassifierType(Enum):
    LR = 'LR' 
    LDA = 'LDA'
    KNN = 'KNN' 
    CART = 'CART' 
    NB = 'GaussianNB' 
    SVM = 'SVM'
    XGBOOST = 'XGBoost'
    RANDOM_FOREST = 'Random Forest'


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if hasattr(X, 'toarray'):
            return X.toarray()
        else :
            return X
    
class MyLabelBinarizer(LabelEncoder):

    def fit(self, x, y=0):
        super().fit(x)
        return self
    
    def fit_transform(self, x, y=0):        
        return super().fit_transform(x)
    def transform(self, x, y=0):
        return super().transform(x)

class UtilityEvaluation(BaseEvaluator) :

    def __init__(self, real_train, synth , real_test, classifiers_ids) :
        super().__init__(real_train, synth)
        self._real_test = real_test
        self._classifiers_ids = classifiers_ids
        

    def get_model_name(self) :
        return [classifer[0] for classifer in  self._classifiers]

    def _get_model(self, classifier_id) :

        if classifier_id == ClassifierType.CART :
            return ('CART', DecisionTreeClassifier())
        elif classifier_id == ClassifierType.KNN :
            return ('KNN', KNeighborsClassifier())
        elif classifier_id == ClassifierType.LDA :
            return ('LDA', LinearDiscriminantAnalysis())
        elif classifier_id == ClassifierType.LR :
            return ('LR', LogisticRegression(solver='liblinear'))
        elif classifier_id == ClassifierType.NB :
            return ('NB', GaussianNB())
        elif classifier_id == ClassifierType.RANDOM_FOREST :
            return ('Random Forest', RandomForestClassifier(class_weight='balanced',n_estimators=100))
        elif classifier_id == ClassifierType.SVM :
            return ('SVM', SVC(class_weight='balanced'))
        elif classifier_id == ClassifierType.XGBOOST :

            return ('XGBoost', XGBClassifier())
        elif classifier_id == ClassifierType.XGBOOST_ :
            return ('XGBoost_', XGBClassifier(scale_pos_weight=3.55))
    
    def _get_data_target(self, data_df) :
        copy_data = data_df.copy(deep=True)
        y = copy_data.iloc[:,-1:].values
        X = copy_data.drop(copy_data.columns[-1], axis=1).values
        return X, y.ravel()
    
    def train_test(self, isSyntheticData: bool, with_preprocess: bool = True, with_feature_importance: bool = False, only_xgboost: bool = False) -> tuple:
        """
        Trains and evaluates multiple classifiers on real or synthetic data and computes performance metrics.

        This function trains a set of classifiers on either real or synthetic data and evaluates their performance 
        on a test set. Optionally, preprocessing steps such as one-hot encoding for categorical variables and 
        feature scaling for numerical variables are applied. It also computes feature importance using permutation 
        importance and SHAP values when applicable.

        Args:
            isSyntheticData (bool): Flag indicating whether to use synthetic data for training (`True`) or real data (`False`).
            with_preprocess (bool, optional): Whether to apply preprocessing to the data (default is `True`).
            with_feature_importance (bool, optional): Whether to compute feature importance for XGBoost (default is `False`).
            only_xgboost (bool, optional): Whether to restrict training to only the XGBoost model (default is `False`).

        Returns:
            tuple: A tuple containing:
                - results_df (pd.DataFrame): A DataFrame of evaluation metrics for each classifier, 
                including accuracy, precision, recall, F1 score, and weighted metrics.
                - p_is (dict): A dictionary of permutation importances for XGBoost if `with_feature_importance` is `True`.
                - shap_values (dict): A dictionary of SHAP values for XGBoost if `with_feature_importance` is `True`.
                - conf_matrices dict: A dict with confusion matrices for all classifier's predictions.
                - accuracy_mean (float): The mean F1 score across all classifiers.

        Example:
            results_df, p_is, shap_values, conf_matrix, accuracy_mean = train_test(isSyntheticData=True, with_preprocess=True, with_feature_importance=True)

        Notes:
            - Preprocessing steps include one-hot encoding for categorical variables and MinMax scaling for numerical variables.
            - Feature importance is computed using permutation importance for XGBoost, and SHAP values are used to explain model predictions.
            - Multiple classifiers are trained, and performance metrics such as accuracy, precision, recall, F1 score, and balanced accuracy are computed.
            - The function supports restricting training to XGBoost using the `only_xgboost` flag.
        """
        self._classifiers = [self._get_model(model_id) for model_id in self._classifiers_ids]
        if (isSyntheticData) :
            X_train, y_train = self._get_data_target(self._synth)
        else :
            X_train, y_train = self._get_data_target(self._real)

        X_real, y_real = self._get_data_target(self._real_test)

        #prepare target 
        le = LabelEncoder()
        le.fit(y_real)
        y_train_enc = le.transform(y_train)
        y_real_enc = le.transform(y_real)
            
            # define the data preparation for the columns
        categorical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._categorical_columns[0:-1]]
        numerical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._numerical_columns]
        t = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx), ('num', MinMaxScaler(), numerical_idx)]
        
        col_transform = ColumnTransformer(transformers=t)
        

        results = dict()
        p_is = dict()
        shap_values = dict()
        confusion_matrices = dict()

        for name , model in  self._classifiers:
            if ((only_xgboost) and (name != 'XGBoost')) : continue
            if with_preprocess :
                if (name == 'LDA' or name == 'NB') :
                    pipeline = Pipeline(steps=[('prep',col_transform), ('to_dense', DenseTransformer()), ('m', model)])
                else :
                    pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])

            else :
                pipeline = Pipeline(steps=[('m', model)])

            pipeline.fit(X_train, y_train_enc)

            if ((with_feature_importance) and (name == 'XGBoost')):
                perm = permutation_importance(pipeline, X_train, y_train_enc, scoring='accuracy')
                shap_value = self.compute_shapvalues(model, X_train,categorical_idx, numerical_idx)
                
                shap_values[name] = np.abs(shap_value.values).mean(axis=0)
                p_is[name] = perm.importances_mean

            pred = pipeline.predict(X_real)
            results[name] = [accuracy_score(y_real_enc, pred), 
                             precision_score(y_real_enc, pred),
                             recall_score(y_real_enc, pred),
                             f1_score(y_real_enc, pred),
                             balanced_accuracy_score(y_real_enc, pred),
                             precision_score(y_real_enc, pred, average='weighted'),
                             recall_score(y_real_enc, pred, average='weighted'),
                             f1_score(y_real_enc, pred, average='weighted')]
            confusion_matrices[name] = confusion_matrix(y_real_enc, pred)

        columns  = ['accuracy',
                    'precision',
                    'recall',
                    'f1', 
                    'balanced_accuracy', 
                    'precision_weighted',
                    'recall_weighted',
                    'f1_weighted']     
        accuracy_mean = sum([lst[3] for lst in results.values()]) / len(results)
        return pd.DataFrame.from_dict(results, orient='index', columns = columns), p_is,  shap_values, confusion_matrices, accuracy_mean
    
    def compute_shapvalues(self, model, X_train: pd.DataFrame, categorical_idx: list, numerical_idx: list) -> shap.Explanation:
        """
        Computes SHAP values for a trained model using a transformed dataset that includes both categorical and numerical features.

        This function computes SHAP values for a given model, using a pipeline that applies one-hot encoding to categorical 
        features and scaling to numerical features. It then adjusts the SHAP values for categorical variables by summing the 
        SHAP values of their one-hot encoded components.

        Args:
            model: The trained machine learning model for which SHAP values are to be computed.
            X_train (pd.DataFrame): The training dataset containing both categorical and numerical features.
            categorical_idx (list): The list of indices corresponding to the categorical columns in `X_train`.
            numerical_idx (list): The list of indices corresponding to the numerical columns in `X_train`.

        Returns:
            shap.Explanation: An object containing the computed SHAP values, adjusted for categorical features, 
            along with feature names and data.

        Example:
            shap_values = compute_shapvalues(model, X_train, categorical_idx, numerical_idx)

        Notes:
            - The function applies a preprocessing pipeline that one-hot encodes categorical features and scales numerical features.
            - SHAP values for one-hot encoded categorical features are summed to provide a single SHAP value for each original feature.
            - The `shap.Explainer` is used to compute SHAP values, which are then adjusted for categorical features.
            - The function returns SHAP values along with the original feature names and data (without encoding).
        """

        t_shap = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx), ('num', MinMaxScaler(), numerical_idx)]
        col_transform_shap = ColumnTransformer(transformers=t_shap)

        pipeline_shap = Pipeline(steps=[('prep',col_transform_shap)])
        X_train_shap = pipeline_shap.fit_transform(X_train)
        if not isinstance(X_train_shap, np.ndarray) :
            X_train_shap = X_train_shap.toarray()
        explainer = shap.Explainer(model)
        shap_value = explainer(X_train_shap)
        feature_names = self._real.columns
        n_categories = []
        for feat in feature_names[:-1]:
            if (feat in self._categorical_columns) :
                n = self._real[feat].nunique()
                n_categories.append(n)
            else : 
                n_categories.append(1)

        new_shap_values = []
        for values in shap_value.values:
            
            #split shap values into a list for each feature
            values_split = np.split(values , np.cumsum(n_categories))
            
            #sum values within each list
            values_sum = [sum(l) for l in values_split]
            
            new_shap_values.append(values_sum)

        
        #replace shap values
        shap_value.values = np.array(new_shap_values)

        #replace data with categorical feature values 
        new_data = X_train
        shap_value.data = np.array(new_data)

        #update feature names
        shap_value.feature_names = list(feature_names[:-1])

        return shap_value
   
    
    def rbo_compare_feature_importance(self, p_i_real: list, shap_value_real: list, p_i_synth: list, shap_value_synth: list) -> tuple:
        """
        Compares feature importance rankings between real and synthetic datasets using Ranked Biased Overlap (RBO).

        This function compares the feature importance rankings derived from real and synthetic datasets using 
        both permutation importance and SHAP values. It computes the Ranked Biased Overlap (RBO) similarity 
        between the feature importance rankings of the real and synthetic datasets for both methods.

        Args:
            p_i_real (list): The permutation importance values for the real dataset.
            shap_value_real (list): The SHAP values for the real dataset.
            p_i_synth (list): The permutation importance values for the synthetic dataset.
            shap_value_synth (list): The SHAP values for the synthetic dataset.

        Returns:
            tuple: A tuple containing:
                - rbo_perm_importance (float): The RBO similarity score between real and synthetic permutation importance rankings.
                - rbo_shap_importance (float): The RBO similarity score between real and synthetic SHAP importance rankings.

        Example:
            rbo_perm_importance, rbo_shap_importance = rbo_compare_feature_importance(p_i_real, shap_value_real, p_i_synth, shap_value_synth)

        Notes:
            - The function ranks the features based on their importance and compares the rankings between real and synthetic datasets.
            - Ranked Biased Overlap (RBO) is a measure of similarity between two ranked lists, accounting for the top-k features.
            - The `rbo.RankingSimilarity` method from the `rbo` library is used to compute the RBO similarity score.
        """


        importance_pi_real_df = pd.DataFrame([ self._real.columns[:-1], p_i_real]).T
        importance_pi_real_df.columns = ['column_name', 'shap_importance']
        importance_pi_real_df = importance_pi_real_df.sort_values('shap_importance', ascending=False)

        importance_pi_synth_df = pd.DataFrame([ self._real.columns[:-1], p_i_synth]).T
        importance_pi_synth_df.columns = ['column_name', 'shap_importance']
        importance_pi_synth_df = importance_pi_synth_df.sort_values('shap_importance', ascending=False)

        #--------------------------
        importance_shap_real_df = pd.DataFrame([ self._real.columns[:-1], shap_value_real]).T
        importance_shap_real_df.columns = ['column_name', 'shap_importance']
        importance_shap_real_df = importance_shap_real_df.sort_values('shap_importance', ascending=False)

        importance_shap_synth_df = pd.DataFrame([ self._real.columns[:-1], shap_value_synth]).T
        importance_shap_synth_df.columns = ['column_name', 'shap_importance']
        importance_shap_synth_df = importance_shap_synth_df.sort_values('shap_importance', ascending=False)

        return rbo.RankingSimilarity(importance_pi_real_df.index.to_list(), importance_pi_synth_df.index.to_list()).rbo(), rbo.RankingSimilarity(importance_shap_real_df.index.to_list(), importance_shap_synth_df.index.to_list()).rbo()

    def spearman_compare_feature_importance(self, p_i_real: list, shap_value_real: list, p_i_synth: list, shap_value_synth: list) -> tuple:
        """
        Compares the feature importance rankings between real and synthetic datasets using Spearman's rank correlation.

        This function computes Spearman's rank correlation between the feature importance values (permutation importance 
        and SHAP values) for the real and synthetic datasets. Spearman's correlation measures the strength and direction 
        of the monotonic relationship between two ranked variables.

        Args:
            p_i_real (list): Permutation importance values for the real dataset.
            shap_value_real (list): SHAP values for the real dataset.
            p_i_synth (list): Permutation importance values for the synthetic dataset.
            shap_value_synth (list): SHAP values for the synthetic dataset.

        Returns:
            tuple: A tuple containing:
                - spearman_perm_importance (float): The Spearman correlation coefficient for permutation importance between real and synthetic datasets.
                - spearman_shap_importance (float): The Spearman correlation coefficient for SHAP importance between real and synthetic datasets.

        Example:
            spearman_perm_importance, spearman_shap_importance = spearman_compare_feature_importance(p_i_real, shap_value_real, p_i_synth, shap_value_synth)

        Notes:
            - Spearman's rank correlation assesses the monotonic relationship between two ranked lists.
            - The `scipy.stats.spearmanr` function is used to calculate the Spearman correlation coefficient.
        """

        res_pi = stats.spearmanr(p_i_real, p_i_synth)
        res_shap = stats.spearmanr(shap_value_real, shap_value_synth)

        return res_pi.statistic, res_shap.statistic
    
    def kendall_compare_feature_importance(self, p_i_real: list, shap_value_real: list, p_i_synth: list, shap_value_synth: list) -> tuple:
        """
        Compares the feature importance rankings between real and synthetic datasets using Kendall's Tau correlation.

        This function computes Kendall's Tau correlation between the feature importance values (permutation importance 
        and SHAP values) for the real and synthetic datasets. Kendall's Tau measures the strength of association between 
        two ranked variables, specifically assessing the concordance between the rankings.

        Args:
            p_i_real (list): Permutation importance values for the real dataset.
            shap_value_real (list): SHAP values for the real dataset.
            p_i_synth (list): Permutation importance values for the synthetic dataset.
            shap_value_synth (list): SHAP values for the synthetic dataset.

        Returns:
            tuple: A tuple containing:
                - kendall_perm_importance (float): The Kendall Tau correlation coefficient for permutation importance between real and synthetic datasets.
                - kendall_shap_importance (float): The Kendall Tau correlation coefficient for SHAP importance between real and synthetic datasets.

        Example:
            kendall_perm_importance, kendall_shap_importance = kendall_compare_feature_importance(p_i_real, shap_value_real, p_i_synth, shap_value_synth)

        Notes:
            - Kendall's Tau assesses the ordinal association between two ranked lists.
            - The `scipy.stats.kendalltau` function is used to calculate Kendall's Tau correlation coefficient.
        """
        res_pi = stats.kendalltau(p_i_real, p_i_synth)
        res_shap = stats.kendalltau(shap_value_real, shap_value_synth)

        return res_pi.statistic, res_shap.statistic

    
    def train_real_test_real(self, with_feature_importance=False) :
        return self.train_test(False, True, with_feature_importance)
    
    def train_synthetic_test_real(self, with_feature_importance=False) :
        return self.train_test(True, True, with_feature_importance)
    
    def train_real_test_real_with_xgboost(self, with_feature_importance=False) :

        return self.train_test(False, True, False, True)
    
    def train_synthetic_test_real_xgboost(self, with_feature_importance=False) :
        return self.train_test(True, True, False, True)

    def feature_importance_analysis(self, isSyntheticData) :
        return
        
