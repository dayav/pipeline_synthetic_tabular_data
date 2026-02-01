from random import uniform
import sys
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from contextlib import suppress

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score
from ..base_evaluator import BaseEvaluator
from data_loader import DataLoader
from .mia_stdg import evaluate_membership_attack
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from scipy.stats import truncnorm
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyRegressor, DummyClassifier
import faiss

import gower
import gc

from statistics import mean
from enum import Enum

class SimilarityType(Enum):
    EUCLIDEAN = 0 
    HAUSDORFF = 1 
    COSINE = 2 
    MAHALANOBIS = 3
    GOWER = 4
    DISSIMILARITY = 5
    DISSIMILARITY_CATEGORICAL = 6
    DISSIMILARITY_NUMERICAL = 7
    DISSIMILARITY_MEAN_IMBALANCED = 8
    DISSIMILARITY_WITH_PREPROCESSING_CAT = 9
    EPSILON_DISSIMILARITY = 10


class DedupPolicy(Enum):
    NONE = "none"   # keep all rows, never drop duplicates
    QIDS = "qids"   # drop duplicates only on QID columns (legacy default)
    ALL  = "all"    # drop duplicates on all columns

class PrivacyEvaluator(BaseEvaluator) :

    def __init__(self, real_train, synth, real_test, qid_columns, non_quid_columns,
                 dedup_policy: DedupPolicy | str = DedupPolicy.NONE):
        """
        Initialize the PrivacyEvaluator.

        Parameters:
        - real_train: The real training data.
        - synth: The synthetic data.
        - real_test: The real testing data.
        """
        super().__init__(real_train, synth)
        self._real_test = real_test
        self._data_synthetic_qid = None
        self._data_synthetic_risk = None
        self._data_real_qid = None
        self._data_real_risk = None
        self._attacker_qid_data = None
        self._attacker_non_qid_data = None
        self.attributes_synthetic_models = {}
        self.attributes_real_models = {}
        self._all_qid_columns = qid_columns
        self._non_quid_columns = non_quid_columns

        # normalize policy
        if isinstance(dedup_policy, str):
            dedup_policy = DedupPolicy(dedup_policy.lower())
        self._dedup_policy = dedup_policy
        print(f"actual self._dedup_policy : {self._dedup_policy}")

        self._prepare_data(qid_columns)


    def _get_concatenate_data(self):

        backup_real = self._real.copy(deep=True)
        backup_synth = self._synth.copy(deep=True)

        backup_real['Label'] = np.zeros(self._real.shape[0]).astype('int8')
        backup_synth['Label'] = np.ones(self._synth.shape[0])

        #mix real and synthetic records

        frames = [backup_real, backup_synth]

        #return the concatenate dataframe with the mixed samples
        data = pd.concat(frames).sample(frac=1)

        last_ix = len(data.columns) - 1
        return data.drop(['Label'], axis=1), data[['Label']]

    def closest_distance_to_record_stdg(self, proportions, thresholds) :
        real_concat = pd.concat([self._real, self._real_test])
        train_data_indexes = self._real['ID'].tolist()

        precision_values = dict()
        accuracy_values = dict()
        
        for threshold in thresholds :
            precision_values[threshold] = []
            accuracy_values[threshold] = []
            for proportion in proportions :
                attacker_data = real_concat.sample(frac = proportion)

                precision, accuracy = evaluate_membership_attack(attacker_data, train_data_indexes, self._synth, threshold)
                precision_values[threshold].append(precision)
                accuracy_values[threshold].append(accuracy)
        return precision_values, accuracy_values
    
    # def _prepare_data(self, qid_columns):
    #     """Sample and deduplicate data for the attacker."""

    #     self._qid_columns = qid_columns

    #     real_data = self._real.drop_duplicates(subset=self._qid_columns, keep='first')
    #     synth_data = self._synth.drop_duplicates(subset=self._qid_columns, keep='first')
    #     test_data = self._real_test.drop_duplicates(subset=self._qid_columns, keep='first')

    #     self._data_synthetic_qid = synth_data[self._qid_columns] 
    #     self._data_synthetic_risk = synth_data[self._non_quid_columns]
    #     self._data_real_qid = real_data[self._qid_columns]
    #     self._data_real_risk = real_data[self._non_quid_columns]

    #     self._attacker_qid_data = test_data[self._qid_columns]
    #     self._attacker_non_qid_data = test_data[self._non_quid_columns]

    def _prepare_data(self, qid_columns):
        """Sample and deduplicate data for the attacker (according to policy)."""
        self._qid_columns = qid_columns  # store user's intent

        real_data, synth_data, test_data, eff_qids = self._resolve_splits()

        # X (QIDs) – can be empty (worst-case handled later)
        if eff_qids:
            self._data_synthetic_qid = synth_data[eff_qids]
            self._data_real_qid      = real_data[eff_qids]
            self._attacker_qid_data  = test_data[eff_qids]
        else:
            self._data_synthetic_qid = pd.DataFrame(index=synth_data.index)
            self._data_real_qid      = pd.DataFrame(index=real_data.index)
            self._attacker_qid_data  = pd.DataFrame(index=test_data.index)

        # Y (risk / non-QIDs)
        self._data_synthetic_risk   = synth_data[self._non_quid_columns]
        self._data_real_risk        = real_data[self._non_quid_columns]
        self._attacker_non_qid_data = test_data[self._non_quid_columns]


    def _train_attributes_prediction_models(self, isSyntheticData, is_dummy):
        """
        Train models to predict non-QID attributes based on QID attributes.

        Parameters:
        - categorical_columns_qid: Categorical columns among the QIDs.
        - numerical_columns_qid: Numerical columns among the QIDs.
        - non_quid_columns: Columns that are not QIDs (potentially sensitive attributes).
        """

        if (isSyntheticData) :
            data_quid = self._data_synthetic_qid
            data_risk = self._data_synthetic_risk
            attributes_models = self.attributes_synthetic_models
        else :
            data_quid = self._data_real_qid
            data_risk = self._data_real_risk
            attributes_models = self.attributes_real_models

        categorical_columns_qid = [x for x in self._qid_columns if x in self._categorical_columns]
        numerical_columns_qid = [x for x in self._qid_columns if x in self._numerical_columns]

        categorical_idx = [data_quid.columns.get_loc(cat_col) for cat_col in categorical_columns_qid]
        numerical_idx = [data_quid.columns.get_loc(num_col) for num_col in numerical_columns_qid]

        transformers = [
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx),
            ('num', StandardScaler(), numerical_idx)
        ]
        col_transform = ColumnTransformer(transformers=transformers)

        # for attribute in tqdm(self._non_quid_columns, desc="Training Models", unit="attribute"):
        for attribute in self._non_quid_columns :

            model = self._make_model(attribute in self._categorical_columns, is_dummy)
            
            pipeline = Pipeline(steps=[('prep', col_transform), ('m', model)])
            pipeline.fit(data_quid, data_risk[attribute])
            attributes_models[attribute] = pipeline

    def _make_model(self, is_categorical: bool, dummy: bool):
        """Return an unfitted estimator appropriate for the column type."""
        if dummy:
            return DummyClassifier(strategy="most_frequent") if is_categorical else DummyRegressor(strategy="mean")
        return DecisionTreeClassifier(random_state=64) if is_categorical else DecisionTreeRegressor(random_state=64)

    
    def train_attributes_synthetic_prediction_models(self, is_dummy = False):
        self._train_attributes_prediction_models(True, is_dummy)

    def train_attributes_real_prediction_models(self, is_dummy= False):
        self._train_attributes_prediction_models(False, is_dummy)

    def predict_attribute(self, isSynthetic) :
        # Dictionary to store predictions
        predictions = {}
        attributes_models = self.attributes_synthetic_models if isSynthetic else self.attributes_real_models
        # for attribute in tqdm(self._non_quid_columns, desc="Evaluating Predictions", unit="attribute"):
        for attribute in self._non_quid_columns :
            results = attributes_models[attribute].predict(self._attacker_qid_data)
            predictions[attribute] = results

        return predictions

    def _evaluate_attribute_prediction(self, isSynthetic):
        
        self._prepare_data(self._all_qid_columns)


        if isSynthetic :
            self.train_attributes_synthetic_prediction_models()
        else : 
            self.train_attributes_real_prediction_models()
        
        predictions_dict = self.predict_attribute(isSynthetic)
        results = {}
        results_std = {}
        results_std_att = {}
        results_acc_att = {}

        for attribute in predictions_dict :
            pred = []
            acc = []
            if attribute in self._categorical_columns :
                result = accuracy_score(self._attacker_non_qid_data[attribute] , predictions_dict[attribute])
                
                for true_label, prediction in  zip(self._attacker_non_qid_data[attribute], predictions_dict[attribute]) :
                    accuracy = accuracy_score([true_label], [prediction])
                    bool_r = True if accuracy == 1 else False
                    pred.append(bool_r)
                    acc.append(accuracy)
                results_std_attribute = pred
            else :
                result = sqrt(mean_squared_error(self._attacker_non_qid_data[attribute] , predictions_dict[attribute]))
                for true_label, prediction in  zip(self._attacker_non_qid_data[attribute], predictions_dict[attribute]) :
                    accuracy = sqrt(mean_squared_error([true_label], [prediction]))
                    bool_r = True if (accuracy < 0.06 )  else False
                    pred.append(bool_r)
                    acc.append(accuracy)
                results_std_attribute = pred
            results[attribute] = result
            results_std[attribute] =  np.round(np.sum(results_std_attribute)/len(results_std_attribute),2)
            results_std_att[attribute] = pred
            results_acc_att[attribute] = acc
        
        return results, results_std, results_std_att, results_acc_att, predictions_dict 
        # return results
    
    def _evaluate_attribute_prediction_dummy(self, isSynthetic):

        self._prepare_data(self._all_qid_columns)

        if isSynthetic :
            self.train_attributes_synthetic_prediction_models(True)
        else : 
            self.train_attributes_real_prediction_models(True)
        
        predictions_dict = self.predict_attribute(isSynthetic)
        results = {}
        results_std = {}
        results_std_att = {}
        results_acc_att = {}

        for attribute in predictions_dict :
            pred = []
            acc = []
            if attribute in self._categorical_columns :
                result = accuracy_score(self._attacker_non_qid_data[attribute] , predictions_dict[attribute])
                
                for true_label, prediction in  zip(self._attacker_non_qid_data[attribute], predictions_dict[attribute]) :
                    accuracy = accuracy_score([true_label], [prediction])
                    bool_r = True if accuracy == 1 else False
                    pred.append(bool_r)
                    acc.append(accuracy)
                results_std_attribute = pred
            else :
                result = sqrt(mean_squared_error(self._attacker_non_qid_data[attribute] , predictions_dict[attribute]))
                for true_label, prediction in  zip(self._attacker_non_qid_data[attribute], predictions_dict[attribute]) :
                    accuracy = sqrt(mean_squared_error([true_label], [prediction]))
                    bool_r = True if (accuracy < 0.06 )  else False
                    pred.append(bool_r)
                    acc.append(accuracy)
                results_std_attribute = pred
            results[attribute] = result
            results_std[attribute] =  np.round(np.sum(results_std_attribute)/len(results_std_attribute),2)
            results_std_att[attribute] = pred
            results_acc_att[attribute] = acc

        return results, results_std, results_std_att, results_acc_att, predictions_dict
        # return results


    def pairwise_euclidean_distance_stdg(self, scaled_real: pd.DataFrame, scaled_synth: pd.DataFrame) :
        # from stdg evaluation metric
        sum_distances = 0
        sum_sq_distances = 0
        n = 0

        for i in range(scaled_synth.shape[0]):
            # Compute distances for one row at a time
            distances_row = distance.cdist(scaled_synth[i:i+1], scaled_real, 'euclidean')[0]
            sum_distances += np.sum(distances_row)
            sum_sq_distances += np.sum(distances_row**2)
            n += len(distances_row)

        mean_distance = sum_distances / n
        std_distance = np.sqrt(sum_sq_distances/n - mean_distance**2)
        return np.round(mean_distance,4), np.round(std_distance,4)
    
    def hausdorff_distance_stdg(self, scaled_real: pd.DataFrame, scaled_synth: pd.DataFrame) :
        # from stdg evaluation metric
        hausdorff_dist = distance.directed_hausdorff(scaled_synth, scaled_real)[0]
        return  np.round(hausdorff_dist,4)
    
    def rts_similarity_stdg(self, scaled_real: pd.DataFrame, scaled_synth: pd.DataFrame) :
        # from stdg evaluation metric
        str_sim = cosine_similarity(scaled_synth, scaled_real)
        return  str_sim
     
    def mahalanobis_sim(self ) :
                    # define the data preparation for the columns
        categorical_idx = [self._real.columns.get_loc(cat_col) for cat_col in self._categorical_columns]
        t = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_idx)]
        
        col_transform = ColumnTransformer(transformers=t)
        _real = col_transform.fit_transform(self._real)
        _synth = col_transform.transform(self._synth)

        if isinstance(_real, np.ndarray) :
            data_real = _real
            data_synth = _synth
        
        elif isinstance(_synth, csr_matrix) :
            data_real = _real.toarray()
            data_synth = _synth.toarray()

        mean_vector = np.mean(data_real, axis=0)
        covariance_matrix = np.cov(data_real, rowvar=False)

        # Check condition number
        cond_number = np.linalg.cond(covariance_matrix)

        if cond_number < 1/sys.float_info.epsilon:
            # Safe to invert
            inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
        else:
            # Use pseudoinverse
            inverse_covariance_matrix = np.linalg.pinv(covariance_matrix)

        distances = []
        for row in data_synth :
            distance = self._mahalanobis_distance(row, mean_vector, inverse_covariance_matrix)
            distances.append(distance)

        return distances

    def _mahalanobis_distance(self, x, mean, inv_cov_matrix):
        x_moins_mean = x - mean
        return np.sqrt(np.dot(np.dot(x_moins_mean, inv_cov_matrix), x_moins_mean.T))
    

    def get_gower_matrix(self, save_path = None) :
        is_categorical = [col in self._categorical_columns for col in self._real.columns]

        real_synth = gower.gower_matrix(self._real, self._synth, cat_features=is_categorical)
        synth_real = gower.gower_matrix(self._synth, self._real, cat_features=is_categorical)

        dif_synth_real_u = np.triu(synth_real)
        dif_synth_real_l = np.tril(real_synth, k=-1)

        del real_synth
        del synth_real

        dif_synth_real = dif_synth_real_u + dif_synth_real_l

        del dif_synth_real_u
        del dif_synth_real_l

        if save_path == None :
            return dif_synth_real
        else :
            with open(save_path +'.pkl', 'wb') as file:
                pickle.dump(dif_synth_real, file)
            del dif_synth_real
            return None

    def get_gower_matrix_by_type(self, save_path = None) :
        is_categorical =[True] * self._categorical_columns
        is_not_categorical = [False] * self._numerical_columns

        real_synth = gower.gower_matrix(self._real[self._categorical_columns], self._synth[self._categorical_columns], cat_features=is_categorical)
        synth_real = gower.gower_matrix(self._synth[self._categorical_columns], self._real[self._categorical_columns], cat_features=is_categorical)

        dif_synth_real_u = np.triu(synth_real)
        dif_synth_real_l = np.tril(real_synth, k=-1)

        del real_synth
        del synth_real

        dif_synth_real_cat = dif_synth_real_u + dif_synth_real_l

        del dif_synth_real_u
        del dif_synth_real_l

        real_synth = gower.gower_matrix(self._real[self._numerical_columns], self._synth[self._numerical_columns], is_not_categorical)
        synth_real = gower.gower_matrix(self._synth[self._numerical_columns], self._real[self._numerical_columns], is_not_categorical)

        dif_synth_real_u = np.triu(synth_real)
        dif_synth_real_l = np.tril(real_synth, k=-1)

        del real_synth
        del synth_real

        dif_synth_real_num = dif_synth_real_u + dif_synth_real_l

        del dif_synth_real_u
        del dif_synth_real_l

        if save_path == None :
            return dif_synth_real_cat, dif_synth_real_num
        else :
            with open(save_path +'.pkl', 'wb') as file:
                pickle.dump(dif_synth_real, file)
            del dif_synth_real
            return None

    def gower_sim(self) :        
        dif_synth_real = self.get_gower_matrix()
        row_mins = np.min(dif_synth_real, axis=1)
        del dif_synth_real

        return row_mins

    def dissimilarity(self, weights_feature, read_data , synth_data, drop_id = False) :

        # df_synth = df_real_cardio_ctgan_first
        # df_real = df_real_cardio_train
        categorical_cols = read_data.select_dtypes(include='category').columns
        numeric_cols = read_data.select_dtypes(exclude='category').columns

        col_length = len(self._real.columns)
        df_length = len(self._real)

        min_diss_gen_idx = []
        min_diss_gen = []
        min_diss = []
        comp_results_df = []
        last_i = 0
        first_index = 0
        last_length = 5000

        while first_index != df_length :

            def compare_rows(i, df_real, df_synth, categorical_cols, numeric_cols, col_length, weights_feature):

                if drop_id : 
                    df_real_used = df_real.drop(i)
                else : 
                    df_real_used = df_real
                comp_results = pd.DataFrame(index=[i for i in range(len(df_real_used))], columns=df_real_used.columns)
                
                for col in categorical_cols:
                    comp_results[col] = df_real_used[col] == df_synth.iloc[i][col]
                    
                for col in numeric_cols:
                    comp_results[col] = abs(df_real_used[col] - df_synth.iloc[i][col])   
                
                
                comp_results = comp_results.dropna()
                comp_results[categorical_cols] = comp_results[categorical_cols].astype(int).replace({1: 0, 0: 1})
                
                for col in numeric_cols:
                    column_range = comp_results[col].max() - comp_results[col].min()
                    if column_range != 0:
                        comp_results[col] = (comp_results[col] - comp_results[col].min()) / column_range
                
                weighted_comp_results = comp_results * weights_feature
                weighted_comp_results['Sum'] = weighted_comp_results.sum(axis=1)
                # weighted_comp_results['Sum norm CTGAN'] = weighted_comp_results['Sum']
                
                min_comp_results = weighted_comp_results['Sum'].min()
                min_comp_results_idx = weighted_comp_results['Sum'].idxmin()
                del comp_results
                del weighted_comp_results
                return min_comp_results, min_comp_results_idx

            
            results = Parallel(n_jobs=-1)(delayed(compare_rows)(i, read_data, synth_data, categorical_cols, numeric_cols, col_length, weights_feature) for i in range(first_index, last_length))
            min_diss_values = []
            min_diss_idx_values = []
            for min_diss, min_diss_idx in results:
                min_diss_values.append(min_diss)
                min_diss_idx_values.append(min_diss_idx)
            min_diss_gen = min_diss_gen + min_diss_values
            min_diss_gen_idx = min_diss_gen_idx + min_diss_idx_values 

            
            del min_diss
            first_index = last_length
            left_length = df_length - last_length
            added_last_length = min(5000, left_length)
            last_length += added_last_length
            gc.collect()
            print(first_index,' ------finished------', last_length)
        return min_diss_gen, min_diss_gen_idx
        
    
    def dissimilarity_all(self) :
        list_ = [1] * len(self._real.columns)
        weights_feature = self.normalize_to_sum_1(list_)
        diss_real, min_diss_gen_idx_real = self.dissimilarity(weights_feature = weights_feature, read_data = self._real, synth_data = self._synth)
        diss_test, min_diss_gen_idx_test = self.dissimilarity(weights_feature = weights_feature, read_data = self._real_test , synth_data = self._synth)
        
        sum_real = sum(diss_real)
        sum_test = sum(diss_test)
        combined_total = sum_real + sum_test
        share_real = (sum_real / combined_total) * 100
        return diss_real, diss_test, min_diss_gen_idx_real, min_diss_gen_idx_test, share_real


    def _encode(self, df, num_cols, cat_cols):
        """Return numeric & categorical matrices, plus scaling info."""
        num_arr = df[num_cols].to_numpy(np.float32)

        # scale numeric to [0,1]
        num_min  = num_arr.min(0, keepdims=True)
        num_rng  = num_arr.ptp(0, keepdims=True)      # ptp = max-min
        num_rng[num_rng == 0] = 1.0                   # avoid /0
        num_arr = (num_arr - num_min) / num_rng

        # convert categoricals to integer codes
        cat_arr = df[cat_cols].apply(lambda s: s.cat.codes).to_numpy(np.int32)

        return num_arr, cat_arr, num_min, num_rng
    
    
    def dissimilarity_chunked(self, real_df: pd.DataFrame,
                            synth_df: pd.DataFrame,
                            weights_feature: np.ndarray,
                            batch: int = 5_000,
                            drop_id: bool = False):
        """
        Compute per-synthetic-row dissimilarity without blowing up RAM.

        Parameters
        ----------
        real_df, synth_df : DataFrames with identical columns
        weights_feature   : 1-D array of length n_features (must sum to 1)
        batch             : #synthetic rows processed at once
        drop_id           : if True, exclude the identical-index row from search
        """
        cat_cols = real_df.select_dtypes(include='category').columns.tolist()
        num_cols = real_df.select_dtypes(exclude='category').columns.tolist()

        Xr_num, Xr_cat, _, _ = self._encode(real_df,  num_cols, cat_cols)
        Xs_num, Xs_cat, _, _ = self._encode(synth_df, num_cols, cat_cols)

        if drop_id:
            mask = ~real_df.index.isin(synth_df.index)
            Xr_num, Xr_cat = Xr_num[mask], Xr_cat[mask]

        min_vals, min_idx = [], []
        n_real = Xr_num.shape[0]

        for start in range(0, len(synth_df), batch):
            end = min(start + batch, len(synth_df))

            # broadcast numeric and categorical separately
            num_diff = np.abs(Xr_num[:, None, :] - Xs_num[None, start:end, :])
            cat_diff = (Xr_cat[:, None, :] != Xs_cat[None, start:end, :]).astype(np.float32)

            diff = np.concatenate([num_diff, cat_diff], axis=2) * weights_feature

            sums = diff.sum(2)                 # shape (n_real, batch_size)
            min_vals.append(sums.min(0))       # nearest distance for each synth in batch
            min_idx.append(sums.argmin(0))     # index of closest real row

        return (np.concatenate(min_vals).tolist(), np.concatenate(min_idx).tolist())

    def dissimilarity_fast(self, real_df, synth_df, weights_feature, drop_id=False):
        cat_cols = real_df.select_dtypes(include='category').columns.tolist()
        num_cols = real_df.select_dtypes(exclude='category').columns.tolist()

        Xr_num, Xr_cat, num_min, num_rng = self._encode(real_df, num_cols, cat_cols)
        Xs_num, Xs_cat, _, _            = self._encode(synth_df, num_cols, cat_cols)

        if drop_id:
            # Remove identical rows once, not per-loop
            mask = ~real_df.index.isin(synth_df.index)
            Xr_num, Xr_cat = Xr_num[mask], Xr_cat[mask]

        # ---------- broadcast pairwise differences ---------
        # numeric: |xi-xj|
        num_diff = np.abs(Xr_num[:, None, :] - Xs_num[None, :, :])
        # categorical: 1 if different
        cat_diff = (Xr_cat[:, None, :] != Xs_cat[None, :, :]).astype(np.float32)

        # concatenate along feature axis → shape (n_real, n_synth, n_features)
        diff = np.concatenate([num_diff, cat_diff], axis=2)

        # apply per-feature weights (1D → broadcast)
        diff *= weights_feature

        # sum over features → distance matrix, take NN per synthetic row
        sums = diff.sum(2)                   # (n_real, n_synth)
        min_diss_gen  = sums.min(0)          # (n_synth,)
        min_diss_idx  = sums.argmin(0)       # (n_synth,)

        return min_diss_gen.tolist(), min_diss_idx.tolist()

    def dissimilarity_all_(self, fast=True, ann=False):
        weights_feature = np.ones(len(self._real.columns), dtype=np.float32)
        weights_feature /= weights_feature.sum()      # already all ones → 1/d
        if ann :
            fn = self.dissimilarity_ann
        else :
            fn = self.dissimilarity_fast if fast else self.dissimilarity

        diss_real,  idx_real  = fn(self._real,       self._synth, weights_feature)
        diss_test,  idx_test  = fn(self._real_test,  self._synth, weights_feature)

        share_real = 100 * sum(diss_real) / (sum(diss_real) + sum(diss_test))
        return diss_real, diss_test, idx_real, idx_test, share_real
    
    def dissimilarity_all_chuncked(self, batch_size=5_000):
        # 1. uniform weights that sum to 1  (change if you need per-feature importance)
        weights = np.ones(len(self._real.columns), dtype=np.float32)
        weights /= weights.sum()

        # 2. compute dissimilarities for the **train** portion
        diss_real, idx_real = self.dissimilarity_chunked(
            real_df   = self._real,
            synth_df  = self._synth,
            weights_feature = weights,
            batch = batch_size,
        )

        # 3. and for the **test** portion
        diss_test, idx_test = self.dissimilarity_chunked(
            real_df   = self._real_test,
            synth_df  = self._synth,
            weights_feature = weights,
            batch = batch_size,
        )

        # 4. post-processing exactly as before
        share_real = 100 * sum(diss_real) / (sum(diss_real) + sum(diss_test))
        return diss_real, diss_test, idx_real, idx_test, share_real


    def build_ann_matrix(self, X_num, X_cat, w_num, w_cat):
        # flatten mixed-type features into one vector per row
        v_num = X_num * w_num          # already [0,1]
        v_cat = X_cat.astype(np.float32) * w_cat
        V = np.hstack([v_num, v_cat]).astype(np.float32)
        index = faiss.IndexHNSWFlat(V.shape[1], 32)
        index.hnsw.efConstruction = 200
        index.add(V)
        return index

    def dissimilarity_ann(self, real_df, synth_df, weights_feature):
        cat_cols = real_df.select_dtypes(include='category').columns.tolist()
        num_cols = real_df.select_dtypes(exclude='category').columns.tolist()

        Xr_num, Xr_cat, _, _ = self._encode(real_df, num_cols, cat_cols)
        Xs_num, Xs_cat, _, _ = self._encode(synth_df, num_cols, cat_cols)

        w_num = weights_feature[:len(num_cols)]
        w_cat = weights_feature[len(num_cols):]

        index = self.build_ann_matrix(Xr_num, Xr_cat, w_num, w_cat)

        V_s = np.hstack([Xs_num * w_num,
                        Xs_cat.astype(np.float32) * w_cat]).astype(np.float32)

        # one exact NN search per synthetic row, very fast
        D, I = index.search(V_s, 1)     # D: distance², I: index
        return D.flatten().tolist(), I.flatten().tolist()

    
    def epsilon_dissimilarity(self) :
        list_ = [1] * len(self._real.columns)
        weights_feature = self.normalize_to_sum_1(list_)
        diss_real, min_diss_gen_idx_real = self.dissimilarity(weights_feature = weights_feature, read_data = self._real , synth_data = self._real, drop_id= True)
        diss_synth, min_diss_gen_idx_synth = self.dissimilarity(weights_feature = weights_feature, read_data = self._synth , synth_data = self._real)
    
        compare = np.array(diss_synth) < np.array(diss_real)
        epsilon = np.sum(compare)/ len(self._real)
        return diss_real, diss_synth, min_diss_gen_idx_real, min_diss_gen_idx_synth, epsilon
    
    def dissimilarity_numerical(self) :
        is_numerical = [int(col in self._numerical_columns) for col in self._real.columns]
        weights_feature = self.normalize_to_sum_1(is_numerical)
        return self.dissimilarity(weights_feature = weights_feature, read_data = self._real, synth_data = self._synth)

    def dissimilarity_categorical(self) :
        is_categorical = [int(col in self._categorical_columns) for col in self._real.columns]
        weights_feature = self.normalize_to_sum_1(is_categorical)
        return self.dissimilarity(weights_feature = weights_feature, read_data = self._real, synth_data = self._synth)

    def dissimilarity_mean_imbalanced(self) :
        mean_imbalance_ratios = []
        for column in self._real.columns :
            if column in self._categorical_columns :
                class_counts = self._real[column].value_counts()

                imbalance_ratios = []

                for label, count in class_counts.items():
                    imbalance_ratio = max(class_counts) / count
                    imbalance_ratios.append(imbalance_ratio)

                # Calculate the mean imbalance ratio
                mean_imbalance_ratio = sum(imbalance_ratios) / len(imbalance_ratios)
                mean_imbalance_ratios.append(mean_imbalance_ratio)
            else :
                mean_imbalance_ratios.append(1)

        mean_imbalance_ratios = [1.0 / x for x in mean_imbalance_ratios]
        
        weights_feature = self.normalize_to_sum_1(mean_imbalance_ratios)
        return self.dissimilarity(weights_feature = weights_feature, read_data = self._real, synth_data = self._synth)

    def dissimilarity_with_preprocessing_cat(self) :
        real_modified = self.create_preprocess_data_frame(self._real)
        real_modified = real_modified.select_dtypes(exclude='category')
        synth_modified = self.create_preprocess_data_frame(self._synth)
        synth_modified = synth_modified.select_dtypes(exclude='category')

        list_ = [1] * len(self._real.columns)
        weights_feature = self.normalize_to_sum_1(list_)

        return self.dissimilarity(weights_feature = weights_feature, read_data = real_modified, synth_data = synth_modified)


    
    def create_preprocess_data_frame(self, data_frame):
        data_modified = data_frame.copy(deep=True)

        for cat_feat in self._categorical_columns :
            sum_cum_cat = data_modified[cat_feat].value_counts(normalize=True).sort_values(ascending=False).cumsum()
            sum_cum_cat_dict = sum_cum_cat.to_dict()
            feature_interval = {}
            first = 0
            for cat in sum_cum_cat_dict :
                value = sum_cum_cat_dict[cat]
                feature_interval[cat] = (first, value)
                first = value

            truncated_dists = {}
            for cat, value in feature_interval.items():
                lower_bound = value[0]  # Lower bound in standard deviations
                upper_bound = value[1]
                mean = (upper_bound + lower_bound) / 2
                std_dev = (upper_bound - lower_bound) / 6
                # Convert the bounds to the scale of the distribution
                a, b = (lower_bound - mean) / std_dev, (upper_bound - mean) / std_dev
                truncated_dist = truncnorm(a, b, loc=mean, scale=std_dev)
                truncated_dists[cat] = truncated_dist

            def convert_to_distribution(value) :
                return truncated_dists[value].rvs(1)[0]

            num_feature_added = cat_feat+'_converted'
            data_modified[num_feature_added] = data_modified[cat_feat].apply(convert_to_distribution)
            data_modified[num_feature_added] = data_modified[num_feature_added].astype('float64')

        return data_modified


    def normalize_to_sum_1(self, numbers):
        total = sum(numbers)
        normalized_numbers = [x / total for x in numbers]
        return normalized_numbers
    
    # def evaluate_attribute_synthetic_prediction(self) :
    #     return self._evaluate_attribute_prediction(True)

    # def evaluate_attribute_real_prediction(self) :
    #     return self._evaluate_attribute_prediction(False)

    def evaluate_dummy(self, isSynthetic) :
        return self._evaluate_attribute_prediction_dummy(isSynthetic)
    
       # ---------- helpers for tuning / preprocessing ----------

    def _onehot_dense(self):
        """Return a dense OneHotEncoder, compatible across sklearn versions."""
        try:
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            # sklearn < 1.2
            return OneHotEncoder(handle_unknown='ignore', sparse=False)

    def _get_dedup_splits(self):
        real_data, synth_data, test_data, _ = self._resolve_splits()
        return real_data, synth_data, test_data

    def _build_preprocessor(self, feature_cols):
        """
        Build a ColumnTransformer with imputation + scaling (numeric)
        and imputation + one-hot (categorical). Outputs dense features.
        """
        cat_cols = [c for c in feature_cols if c in self._categorical_columns]
        num_cols = [c for c in feature_cols if c in self._numerical_columns]

        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', self._onehot_dense())
        ])

        pre = ColumnTransformer(
            transformers=[
                ('num', num_pipe, num_cols),
                ('cat', cat_pipe, cat_cols),
            ],
            remainder='drop'
        )
        return pre, num_cols, cat_cols

    def _get_feature_target_frames(self, isSynthetic, attribute, worst_case: bool):
        """
        Assemble (X_train, y_train, X_test, y_test, feature_cols) for one attribute.
        - worst_case=True → use all columns EXCEPT the target attribute as predictors.
        - else → use only the QIDs as predictors (your original threat model).
        """
        real_data, synth_data, test_data = self._get_dedup_splits()
        train_df = synth_data if isSynthetic else real_data

        if worst_case:
            feature_cols = [c for c in train_df.columns if c != attribute]
        else:
            feature_cols = list(self._qid_columns)  # may be empty → worst_case handled by caller

        X_train = train_df[feature_cols]
        y_train = train_df[attribute]
        # mirror the same feature set on the test side
        X_test  = test_data[feature_cols] if not worst_case else test_data[[c for c in test_data.columns if c != attribute]]
        y_test  = test_data[attribute]
        return X_train, y_train, X_test, y_test, feature_cols

    def _model_spaces(self, is_categorical: bool):
        if is_categorical:
            models = [
                ('dt', DecisionTreeClassifier(random_state=64), {
                    'model__max_depth': [None, 5, 10, 20, 40],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                }),
                ('rf', RandomForestClassifier(random_state=64, n_jobs=-1), {
                    'model__n_estimators': [200, 400, 800],
                    'model__max_depth': [None, 10, 20, 40],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__max_features': ['sqrt', 0.5, 1.0],
                }),
                ('logreg', LogisticRegression(max_iter=2000, multi_class='auto', solver='lbfgs'), {
                    'model__C': loguniform(1e-3, 1e2),
                    'model__penalty': ['l2'],
                }),
                # 👇 early_stopping=False avoids internal stratified split that fails with 1-sample classes
                ('hgb', HistGradientBoostingClassifier(random_state=64, early_stopping=False), {
                    'model__max_depth': [None, 5, 10, 20],
                    'model__learning_rate': loguniform(1e-2, 2e-1),
                    'model__max_leaf_nodes': [31, 63, 127],
                    'model__l2_regularization': loguniform(1e-4, 1.0),
                    'model__early_stopping': [False],  # keep fixed
                }),
            ]
        else:
            models = [
                ('dtr', DecisionTreeRegressor(random_state=64), {
                    'model__max_depth': [None, 10, 20, 40],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                }),
                ('rfr', RandomForestRegressor(random_state=64, n_jobs=-1), {
                    'model__n_estimators': [200, 400, 800],
                    'model__max_depth': [None, 10, 20, 40],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__max_features': ['sqrt', 0.5, 1.0],
                }),
                ('ridge', Ridge(random_state=64), {
                    'model__alpha': loguniform(1e-3, 1e2),
                }),
                # (regression early-stopping isn't stratified, but fixing it off keeps behavior consistent)
                ('hgr', HistGradientBoostingRegressor(random_state=64, early_stopping=False), {
                    'model__max_depth': [None, 5, 10, 20],
                    'model__learning_rate': loguniform(1e-2, 2e-1),
                    'model__max_leaf_nodes': [31, 63, 127],
                    'model__l2_regularization': loguniform(1e-4, 1.0),
                    'model__early_stopping': [False],
                }),
            ]
        return models

    
    def _cv_splitter(self, y, is_categorical: bool, cv: int):
        """Choose a robust CV splitter that respects class rarity."""
        y = pd.Series(y)
        n = len(y)
        # ensure splits never exceed #samples
        cv = max(2, min(int(cv), n))

        if is_categorical:
            counts = y.value_counts(dropna=False)
            min_count = int(counts.min()) if not counts.empty else 0

            # If every class has >=2 samples, we can stratify.
            if min_count >= 2:
                n_splits = min(cv, min_count)  # cannot ask more folds than the rarest class
                return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=64)
            else:
                # Cannot stratify (at least one class appears only once) → fall back to plain KFold
                return KFold(n_splits=cv, shuffle=True, random_state=64)
        else:
            return KFold(n_splits=cv, shuffle=True, random_state=64)


    def _fit_best_model_for_attribute(self,
                                      isSynthetic: bool,
                                      attribute: str,
                                      worst_case: bool,
                                      n_iter: int = 20,
                                      cv: int = 3):
        """
        Tune and select the best model pipeline for one attribute.
        Returns: fitted_best_pipeline, y_pred_on_test
        """
        # Build data
        X_train, y_train, X_test, y_test, feature_cols = self._get_feature_target_frames(
            isSynthetic=isSynthetic,
            attribute=attribute,
            worst_case=worst_case
        )

        is_categorical = attribute in self._categorical_columns
        preproc, _, _ = self._build_preprocessor(feature_cols)
        splitter = self._cv_splitter(y_train, is_categorical, cv)

        # If degenerate (e.g., single class), fallback to dummy
        if is_categorical and (pd.Series(y_train).nunique() < 2):
            model = DummyClassifier(strategy="most_frequent")
            pipe = Pipeline([('prep', preproc), ('model', model)])
            pipe.fit(X_train, y_train)
            return pipe, pipe.predict(X_test)

        if (not is_categorical) and (pd.Series(y_train).isna().all()):
            model = DummyRegressor(strategy="mean")
            pipe = Pipeline([('prep', preproc), ('model', model)])
            pipe.fit(X_train, y_train)
            return pipe, pipe.predict(X_test)

        candidates = self._model_spaces(is_categorical)
        scoring = 'balanced_accuracy' if is_categorical else 'neg_mean_squared_error'

        best_pipe = None
        best_score = -np.inf

        for name, estimator, param_dist in candidates:
            pipe = Pipeline([('prep', preproc), ('model', estimator)])

            try:
                if param_dist:
                    search = RandomizedSearchCV(
                        estimator=pipe,
                        param_distributions=param_dist,
                        n_iter=n_iter,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=splitter,
                        refit=True,                # will refit on full train; can raise
                        random_state=64,
                        verbose=0,
                        error_score='raise'        # make sure we catch issues here
                    )
                    search.fit(X_train, y_train)
                    score = float(search.best_score_)
                    cand  = search.best_estimator_
                else:
                    scores = cross_val_score(pipe, X_train, y_train, scoring=scoring, cv=splitter, n_jobs=-1)
                    score  = float(np.mean(scores))
                    cand   = pipe.fit(X_train, y_train)

            except Exception as e:
                # Skip this model family if it fails (e.g., degenerate class splits)
                # You can log the error if useful:
                # print(f"[skip {name}] {e}")
                continue

            if score > best_score:
                best_score = score
                best_pipe  = cand

        # As a final fallback, if every candidate failed, use a simple dummy
        if best_pipe is None:
            if is_categorical:
                best_pipe = Pipeline([('prep', preproc), ('model', DummyClassifier(strategy="most_frequent"))])
            else:
                best_pipe = Pipeline([('prep', preproc), ('model', DummyRegressor(strategy="mean"))])
            best_pipe.fit(X_train, y_train)

        # Predict on real test set
        y_pred = best_pipe.predict(X_test)
        return best_pipe, y_pred, (X_test, y_test)

    def _is_worst_case(self):
        q = self._qid_columns
        if q is None:
            return True
        if isinstance(q, (list, tuple, set, pd.Index)):
            return len(q) == 0
        return False  # single column provided
   
   
    def evaluate_attribute_synthetic_prediction(self, n_iter: int = 20, cv: int = 3, show_progress: bool = True):
        """
        Train + tune a set of models for each non-QID attribute using SYNTHETIC data.
        Predict on the real test set. Returns (results, results_std, results_std_att, results_acc_att, predictions_dict).
        `results` now matches the baseline_comparison notebook helpers: for each
        attribute it provides a dict of classification/regression metrics.
        Worst-case: if no QIDs were provided at init, uses all features except the target attribute.
        """
        # Recompute data views (safe) and decide if worst-case applies
        self._prepare_data(self._all_qid_columns)
        worst_case = self._is_worst_case()  # True if None or empty list

        results = {}
        results_std = {}
        results_std_att = {}
        results_acc_att = {}
        predictions_dict = {}
        # clear / reuse dict where we store fitted models
        self.attributes_synthetic_models = {}

        for attribute in self._iter_with_progress(self._non_quid_columns, "Synthetic attr", show_progress):
            # fit best model and predict on real test set
            best_pipe, y_pred, (X_test, y_test) = self._fit_best_model_for_attribute(
                isSynthetic=True,
                attribute=attribute,
                worst_case=worst_case,
                n_iter=n_iter,
                cv=cv
            )
            self.attributes_synthetic_models[attribute] = best_pipe
            predictions_dict[attribute] = y_pred

            # compute metrics like your original code
            pred_flags = []
            per_row_vals = []

            if attribute in self._categorical_columns:
                # classification metrics (align with baseline_comparison notebook)
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                    "weighted F1": f1_score(y_test, y_pred, average="weighted"),
                    "Macro F1": f1_score(y_test, y_pred, average="macro"),
                    "weighted recall": recall_score(y_test, y_pred, average="weighted"),
                    "weighted precision": precision_score(y_test, y_pred, average="weighted"),
                }
                results[attribute] = metrics

                # per-row 1/0 flags
                for t, p in zip(y_test, y_pred):
                    ok = 1 if t == p else 0
                    pred_flags.append(bool(ok))
                    per_row_vals.append(ok)     # in your code this is accuracy per row

                # thresholded success rate (same as accuracy here)
                results_std[attribute] = np.round(np.sum(pred_flags) / len(pred_flags), 2)

            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = sqrt(mse)
                metrics = {
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2_score(y_test, y_pred),
                }
                results[attribute] = metrics

                for t, p in zip(y_test, y_pred):
                    abs_err = sqrt(mean_squared_error([t], [p]))  # = |t - p|
                    ok = abs_err < 0.06
                    pred_flags.append(bool(ok))
                    per_row_vals.append(abs_err)

                results_std[attribute] = np.round(np.sum(pred_flags) / len(pred_flags), 2)

            results_std_att[attribute] = pred_flags
            results_acc_att[attribute] = per_row_vals

        return results, results_std, results_std_att, results_acc_att, predictions_dict

    def evaluate_attribute_real_prediction(self, n_iter: int = 20, cv: int = 3, show_progress: bool = True):
        """
        Train + tune a set of models for each non-QID attribute using REAL TRAINING data.
        Predict on the real test set. Returns (results, results_std, results_std_att, results_acc_att, predictions_dict).
        `results` mirrors the baseline_comparison notebook helpers: per-attribute
        dicts of classification/regression metrics.
        Worst-case: if no QIDs were provided at init, uses all features except the target attribute.
        """
        # Recompute data views (safe) and decide if worst-case applies
        self._prepare_data(self._all_qid_columns)
        worst_case = self._is_worst_case()  # True if None or empty list

        results = {}
        results_std = {}
        results_std_att = {}
        results_acc_att = {}
        predictions_dict = {}
        # metrics returned in the first position will mirror result_classification/result_regression

        # clear / reuse dict where we store fitted models
        self.attributes_real_models = {}

        for attribute in self._iter_with_progress(self._non_quid_columns, "Real attr", show_progress):
            best_pipe, y_pred, (X_test, y_test) = self._fit_best_model_for_attribute(
                isSynthetic=False,
                attribute=attribute,
                worst_case=worst_case,
                n_iter=n_iter,
                cv=cv
            )
            self.attributes_real_models[attribute] = best_pipe
            predictions_dict[attribute] = y_pred

            pred_flags = []
            per_row_vals = []

            if attribute in self._categorical_columns:
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                    "weighted F1": f1_score(y_test, y_pred, average="weighted"),
                    "Macro F1": f1_score(y_test, y_pred, average="macro"),
                    "weighted recall": recall_score(y_test, y_pred, average="weighted"),
                    "weighted precision": precision_score(y_test, y_pred, average="weighted"),
                }
                results[attribute] = metrics

                for t, p in zip(y_test, y_pred):
                    ok = 1 if t == p else 0
                    pred_flags.append(bool(ok))
                    per_row_vals.append(ok)

                results_std[attribute] = np.round(np.sum(pred_flags) / len(pred_flags), 2)

            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = sqrt(mse)
                metrics = {
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2_score(y_test, y_pred),
                }
                results[attribute] = metrics

                for t, p in zip(y_test, y_pred):
                    abs_err = sqrt(mean_squared_error([t], [p]))
                    ok = abs_err < 0.06
                    pred_flags.append(bool(ok))
                    per_row_vals.append(abs_err)

                results_std[attribute] = np.round(np.sum(pred_flags) / len(pred_flags), 2)

            results_std_att[attribute] = pred_flags
            results_acc_att[attribute] = per_row_vals

        return results, results_std, results_std_att, results_acc_att, predictions_dict
    
    def evaluate_similarity_stdg(self, distance=SimilarityType.EUCLIDEAN) :
        scaled_real = MinMaxScaler().fit_transform(self._real[self._numerical_columns])
        scaled_synth = MinMaxScaler().fit_transform(self._synth[self._numerical_columns])
        if distance==SimilarityType.EUCLIDEAN :
            distances = self.pairwise_euclidean_distance_stdg(scaled_real, scaled_synth)
        elif distance == SimilarityType.HAUSDORFF :
            distances = self.hausdorff_distance_stdg(scaled_real, scaled_synth)
        elif distance == SimilarityType.COSINE :
            distances = self.rts_similarity_stdg(scaled_real, scaled_synth)
        elif distance == SimilarityType.MAHALANOBIS :
            distances = self.mahalanobis_sim()
        elif distance == SimilarityType.GOWER :
            distances = self.gower_sim()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY :
            distances = self.dissimilarity_all()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY_NUMERICAL : 
            distances = self.dissimilarity_numerical()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY_CATEGORICAL : 
            distances = self.dissimilarity_categorical()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY_MEAN_IMBALANCED : 
            distances = self.dissimilarity_mean_imbalanced()
            gc.collect()
        elif distance == SimilarityType.DISSIMILARITY_WITH_PREPROCESSING_CAT : 
            distances = self.dissimilarity_with_preprocessing_cat()
            gc.collect()   
        elif distance == SimilarityType.EPSILON_DISSIMILARITY : 
            distances = self.epsilon_dissimilarity()
            gc.collect()  
        return distances

    
    def _effective_qids(self):
        """Return list of valid QID columns (may be empty)."""
        q = self._qid_columns
        if q is None:
            return []
        if isinstance(q, (list, tuple, set, pd.Index)):
            return [c for c in q if c in self._real.columns]
        return [q] if q in self._real.columns else []

    def _resolve_splits(self):
        """
        Apply dedup policy consistently and return:
        real_data, synth_data, test_data, eff_qids
        """
        eff_qids = self._effective_qids()

        if self._dedup_policy == DedupPolicy.NONE:
            # no dedup anywhere
            real_data  = self._real
            synth_data = self._synth
            test_data  = self._real_test
            return real_data, synth_data, test_data, eff_qids

        # Dedup on QIDs (if any) else on ALL columns
        if self._dedup_policy == DedupPolicy.QIDS and len(eff_qids) > 0:
            subset = eff_qids
        else:
            subset = None  # ALL columns

        real_data  = self._real.drop_duplicates(subset=subset, keep='first')
        synth_data = self._synth.drop_duplicates(subset=subset, keep='first')
        test_data  = self._real_test.drop_duplicates(subset=subset, keep='first')
        return real_data, synth_data, test_data, eff_qids

    def _iter_with_progress(self, iterable, desc: str, enabled: bool):
        """
        Wrap an iterable with tqdm when available and requested.
        """
        if not enabled:
            return iterable
        with suppress(ImportError):
            from tqdm import tqdm
            return tqdm(iterable, desc=desc)
        return iterable
