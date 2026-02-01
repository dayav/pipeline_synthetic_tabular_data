from enum import Enum
import warnings
import numpy as np
import pandas as pd
import shap
import rbo

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    make_scorer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from xgboost import XGBClassifier
from scipy import stats

from .base_evaluator import BaseEvaluator


class ClassifierType(Enum):
    LR = 'LR'
    LDA = 'LDA'
    KNN = 'KNN'
    CART = 'CART'
    NB = 'GaussianNB'
    SVM = 'SVM'
    XGBOOST = 'XGBoost'
    XGBOOST_WEIGHTS = 'XGBoost (pos-weighted)'
    RANDOM_FOREST = 'Random Forest'


def _safe_ohe_dense(**kwargs) -> OneHotEncoder:
    """Make OneHotEncoder dense across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False, **kwargs)


def _has_method(obj: object, name: str) -> bool:
    return callable(getattr(obj, name, None))


class UtilityEvaluation(BaseEvaluator):
    """
    Train/test utilities on (real|synthetic)->real settings with robust metrics,
    optional permutation/SHAP importance, and a fast hyperparameter tuner.
    """

    def __init__(self, real_train: pd.DataFrame, synth: pd.DataFrame,
                 real_test: pd.DataFrame, classifiers_ids: list[ClassifierType],
                 random_state: int = 42) -> None:
        super().__init__(real_train, synth)
        self._real_test = real_test
        self._classifiers_ids = classifiers_ids
        self._random_state = random_state
        self._classifiers: list[tuple[str, BaseEstimator]] = []
        self._best_params: dict[str, dict] = {}

    # -------------------------
    # Model factory & plumbing
    # -------------------------
    def get_model_name(self) -> list[str]:
        if hasattr(self, "_classifiers") and self._classifiers:
            return [clf_name for clf_name, _ in self._classifiers]
        return [self._get_model(cid)[0] for cid in self._classifiers_ids]

    def _get_model(self, classifier_id: ClassifierType) -> tuple[str, BaseEstimator]:
        rs = self._random_state

        if classifier_id == ClassifierType.CART:
            return ('CART', DecisionTreeClassifier(random_state=rs))

        elif classifier_id == ClassifierType.KNN:
            return ('KNN', KNeighborsClassifier(n_neighbors=11, weights='distance', n_jobs=-1))

        elif classifier_id == ClassifierType.LDA:
            return ('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))

        elif classifier_id == ClassifierType.LR:
            return ('LR', LogisticRegression(
                solver='lbfgs', max_iter=2000, n_jobs=-1, class_weight='balanced', random_state=rs
            ))

        elif classifier_id == ClassifierType.NB:
            return ('NB', GaussianNB())

        elif classifier_id == ClassifierType.RANDOM_FOREST:
            return ('Random Forest', RandomForestClassifier(
                n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=rs
            ))

        elif classifier_id == ClassifierType.SVM:
            return ('SVM', SVC(class_weight='balanced', probability=True, random_state=rs))

        elif classifier_id == ClassifierType.XGBOOST:
            return ('XGBoost', XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', n_jobs=-1, random_state=rs, tree_method='hist'
            ))

        elif classifier_id == ClassifierType.XGBOOST_WEIGHTS:
            return ('XGBoost (pos-weighted)', XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', n_jobs=-1, random_state=rs, tree_method='hist',
                scale_pos_weight=3.55
            ))

        else:
            raise ValueError(f"Unknown classifier id: {classifier_id}")

    # -------------------------
    # Data helpers
    # -------------------------
    def _split_features_target(self, data_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X_df = data_df.iloc[:, :-1].copy()
        y = data_df.iloc[:, -1].to_numpy().ravel()
        return X_df, y

    def _build_feature_indices(self) -> tuple[list[int], list[int], list[str]]:
        feature_cols = list(self._real.columns[:-1])
        cat_cols = [c for c in self._categorical_columns if c in feature_cols]
        num_cols = [c for c in self._numerical_columns if c in feature_cols]
        categorical_idx = [feature_cols.index(c) for c in cat_cols]
        numerical_idx  = [feature_cols.index(c) for c in num_cols]
        return categorical_idx, numerical_idx, feature_cols

    def _build_preprocessor(self, categorical_idx: list[int], numerical_idx: list[int]) -> ColumnTransformer:
        ohe = _safe_ohe_dense()
        col_transform = ColumnTransformer(
            transformers=[
                ('cat', ohe, categorical_idx),
                ('num', MinMaxScaler(), numerical_idx),
            ],
            sparse_threshold=0.0
        )
        return col_transform

    # -------------------------
    # FAST TUNING
    # -------------------------
    def _param_distributions_for(self, model_name: str) -> dict:
        """Small, high‑leverage spaces; all keys are prefixed with 'm__' for the final step."""
        if model_name == 'LR':
            return {
                'm__C': stats.loguniform(1e-3, 1e3),
            }
        if model_name == 'SVM':
            return {
                'm__C': stats.loguniform(1e-2, 1e2),
                'm__gamma': stats.loguniform(1e-4, 1e-1),
                'm__kernel': ['rbf', 'linear'],
            }
        if model_name == 'Random Forest':
            return {
                'm__n_estimators': [200, 300, 500],
                'm__max_depth': [None, 8, 16, 24],
                'm__min_samples_leaf': [1, 2, 4, 8],
                'm__max_features': ['sqrt', 'log2', 0.5, None],
            }
        if model_name.startswith('XGBoost'):
            return {
                'm__n_estimators': [200, 400, 800],
                'm__learning_rate': stats.loguniform(0.01, 0.3),
                'm__max_depth': [3, 4, 5, 6, 8],
                'm__subsample': stats.uniform(0.6, 0.4),
                'm__colsample_bytree': stats.uniform(0.6, 0.4),
                'm__min_child_weight': [1, 3, 5],
                'm__reg_lambda': stats.loguniform(1e-2, 1e2),
            }
        if model_name == 'KNN':
            return {
                'm__n_neighbors': [3, 5, 7, 11, 21, 31],
                'm__weights': ['uniform', 'distance'],
                'm__p': [1, 2],
            }
        if model_name == 'CART':
            return {
                'm__max_depth': [None, 5, 10, 20, 30],
                'm__min_samples_leaf': [1, 2, 4, 8],
                'm__ccp_alpha': stats.uniform(0.0, 0.02),
            }
        if model_name == 'LDA':
            # using solver='lsqr', shrinkage supports float in [0,1]
            return {'m__shrinkage': stats.uniform(0.0, 1.0)}
        if model_name == 'NB':
            return {'m__var_smoothing': stats.loguniform(1e-12, 1e-7)}
        return {}

    def _fast_tune_pipeline(
        self,
        name: str,
        pipeline: Pipeline,
        X_train_df: pd.DataFrame,
        y_train_enc: np.ndarray,
        tune_iter: int,
        tune_cv: int,
        tune_subsample: float,
    ) -> Pipeline:
        """Run a small RandomizedSearchCV on a stratified subsample, then refit on full train."""
        space = self._param_distributions_for(name)
        if not space:
            return pipeline  # nothing to tune for this model

        # Stratified subsample for speed (e.g., 50% by default)
        if 0 < tune_subsample < 1.0:
            X_tune, _, y_tune, _ = train_test_split(
                X_train_df, y_train_enc, test_size=(1.0 - tune_subsample),
                stratify=y_train_enc, random_state=self._random_state
            )
        else:
            X_tune, y_tune = X_train_df, y_train_enc

        cv = StratifiedKFold(n_splits=max(2, tune_cv), shuffle=True, random_state=self._random_state)
        scorer = make_scorer(f1_score, average='weighted', zero_division=0)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=space,
            n_iter=max(1, tune_iter),
            scoring=scorer,
            n_jobs=-1,
            cv=cv,
            refit=False,              # do NOT refit on subsample; we'll refit on full data
            random_state=self._random_state,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_tune, y_tune)

        best_params = getattr(search, 'best_params_', {})
        pipeline.set_params(**best_params)
        self._best_params[name] = best_params
        return pipeline

    def get_best_params(self) -> dict[str, dict]:
        """Access best params chosen by the tuner after train_test(..., fast_tune=True)."""
        return dict(self._best_params)

    # -------------------------
    # Public API
    # -------------------------
    def train_test(self,
                   isSyntheticData: bool,
                   with_preprocess: bool = True,
                   with_feature_importance: bool = False,
                   only_xgboost: bool = False,
                   *,
                   fast_tune: bool = False,
                   tune_iter: int = 20,
                   tune_cv: int = 3,
                   tune_subsample: float = 0.5) -> tuple[pd.DataFrame, dict, dict, dict, float]:
        """
        Train/evaluate classifiers. Optional fast tuning (small randomized search).
        Returns: results_df, p_is, shap_values, confusion_matrices, f1_mean
        """
        self._classifiers = [self._get_model(model_id) for model_id in self._classifiers_ids]

        # Choose train source
        X_train_df, y_train = self._split_features_target(self._synth if isSyntheticData else self._real)
        X_real_df,  y_real  = self._split_features_target(self._real_test)

        # Label encoding on train ∪ test
        le = LabelEncoder()
        le.fit(np.concatenate([y_train, y_real], axis=0))
        y_train_enc = le.transform(y_train)
        y_real_enc  = le.transform(y_real)
        n_classes = len(le.classes_)
        avg = 'binary' if n_classes == 2 else 'weighted'

        categorical_idx, numerical_idx, feature_cols = self._build_feature_indices()
        preprocessor = self._build_preprocessor(categorical_idx, numerical_idx)

        results: dict[str, list[float]] = {}
        p_is = {}
        shap_values = {}
        confusion_matrices: dict[str, np.ndarray] = {}
        self._best_params.clear()

        for name, model in self._classifiers:
            if only_xgboost and name not in ('XGBoost', 'XGBoost (pos-weighted)'):
                continue

            pipeline = Pipeline(steps=[('prep', preprocessor), ('m', model)]) if with_preprocess else Pipeline(steps=[('m', model)])

            # ---- Fast tune (optional)
            if fast_tune:
                pipeline = self._fast_tune_pipeline(
                    name, pipeline, X_train_df, y_train_enc,
                    tune_iter=tune_iter, tune_cv=tune_cv, tune_subsample=tune_subsample
                )

            # Final fit on full training set
            pipeline.fit(X_train_df, y_train_enc)

            # Predict
            pred = pipeline.predict(X_real_df)
            confusion_matrices[name] = confusion_matrix(y_real_enc, pred, labels=np.arange(n_classes))

            # Optional proba for AUC / PR
            proba = None
            if _has_method(pipeline, 'predict_proba'):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        proba = pipeline.predict_proba(X_real_df)
                    except Exception:
                        proba = None

            # Core metrics
            acc = accuracy_score(y_real_enc, pred)
            prec = precision_score(y_real_enc, pred, average=avg, zero_division=0)
            rec  = recall_score(y_real_enc, pred, average=avg, zero_division=0)
            f1   = f1_score(y_real_enc, pred, average=avg, zero_division=0)
            bal_acc = balanced_accuracy_score(y_real_enc, pred)

            # Additional metrics
            mcc = matthews_corrcoef(y_real_enc, pred) if n_classes == 2 else np.nan
            kappa = cohen_kappa_score(y_real_enc, pred)
            roc_auc = np.nan
            pr_ap = np.nan
            ll = np.nan
            if proba is not None:
                try:
                    if n_classes == 2:
                        roc_auc = roc_auc_score(y_real_enc, proba[:, 1])
                        pr_ap = average_precision_score(y_real_enc, proba[:, 1])
                        ll = log_loss(y_real_enc, proba)
                    else:
                        roc_auc = roc_auc_score(y_real_enc, proba, multi_class='ovr', average='macro')
                        y_true_bin = pd.get_dummies(y_real_enc).to_numpy()
                        pr_ap = average_precision_score(y_true_bin, proba, average='macro')
                        ll = log_loss(y_real_enc, proba)
                except Exception:
                    pass

            # Macro/weighted summaries
            prec_w = precision_score(y_real_enc, pred, average='weighted', zero_division=0)
            rec_w  = recall_score(y_real_enc, pred, average='weighted', zero_division=0)
            f1_w   = f1_score(y_real_enc, pred, average='weighted', zero_division=0)
            prec_m = precision_score(y_real_enc, pred, average='macro', zero_division=0)
            rec_m  = recall_score(y_real_enc, pred, average='macro', zero_division=0)
            f1_m   = f1_score(y_real_enc, pred, average='macro', zero_division=0)

            results[name] = [
                acc, prec, rec, f1, bal_acc,
                prec_w, rec_w, f1_w,
                prec_m, rec_m, f1_m,
                roc_auc, pr_ap, mcc, kappa, ll
            ]

            # Feature importance (optional; keep to XGBoost for speed)
            if with_feature_importance and name.startswith('XGBoost') and with_preprocess:
                scorer = make_scorer(f1_score, average='weighted', zero_division=0)
                perm = permutation_importance(
                    pipeline, X_train_df, y_train_enc,
                    scoring=scorer, n_repeats=10, random_state=self._random_state, n_jobs=-1
                )
                p_is[name] = pd.Series(perm.importances_mean, index=X_train_df.columns)

                shap_exp = self.compute_shapvalues(
                    fitted_pipeline=pipeline,
                    X_raw=X_train_df,
                    categorical_idx=categorical_idx,
                    numerical_idx=numerical_idx
                )
                shap_values[name] = pd.Series(
                    np.abs(shap_exp.values).mean(axis=0),
                    index=shap_exp.feature_names
                )

        # Results frame
        columns = [
            'accuracy',
            'precision_avg', 'recall_avg', 'f1_avg', 'balanced_accuracy',
            'precision_weighted', 'recall_weighted', 'f1_weighted',
            'precision_macro', 'recall_macro', 'f1_macro',
            'roc_auc', 'avg_precision', 'mcc', 'cohen_kappa', 'log_loss'
        ]
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=columns)
        f1_mean = float(np.nanmean(results_df['f1_weighted'].values))
        return results_df, p_is, shap_values, confusion_matrices, f1_mean

    # -------------------------
    # SHAP (pipeline-aware, OHE-collapsed)
    # -------------------------
    def compute_shapvalues(self,
                           fitted_pipeline: Pipeline,
                           X_raw: pd.DataFrame,
                           categorical_idx: list[int],
                           numerical_idx: list[int]) -> shap.Explanation:
        """Compute SHAP on the fitted pipeline; collapse OHE back to original features."""
        if 'prep' not in fitted_pipeline.named_steps:
            est = fitted_pipeline.named_steps['m']
            explainer = shap.Explainer(est, X_raw)
            return explainer(X_raw)

        prep = fitted_pipeline.named_steps['prep']
        est = fitted_pipeline.named_steps['m']

        X_t = prep.transform(X_raw)
        if hasattr(X_t, 'toarray'):
            X_t = X_t.toarray()

        enc = prep.named_transformers_['cat']
        cat_sizes = [len(c) for c in enc.categories_]
        num_sizes = [1] * len(numerical_idx)
        group_sizes = cat_sizes + num_sizes
        split_idx = np.cumsum(group_sizes)[:-1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.Explainer(est, X_t)
            exp = explainer(X_t)

        vals = exp.values
        if vals.ndim == 3:
            vals = vals[:, :, 1] if vals.shape[2] > 1 else vals.squeeze(2)

        collapsed = []
        for row in vals:
            chunks = np.split(row, split_idx)
            collapsed.append([float(np.sum(ch)) for ch in chunks])
        collapsed = np.asarray(collapsed)

        feature_cols = list(self._real.columns[:-1])
        cat_names = [feature_cols[i] for i in categorical_idx]
        num_names = [feature_cols[i] for i in numerical_idx]
        new_feature_names = cat_names + num_names

        exp.values = collapsed
        exp.data = X_raw[new_feature_names].to_numpy()
        exp.feature_names = new_feature_names
        return exp

    # -------------------------
    # Ranking comparisons
    # -------------------------
    def rbo_compare_feature_importance(self,
                                       p_i_real: list[float], shap_value_real: list[float],
                                       p_i_synth: list[float], shap_value_synth: list[float]) -> tuple[float, float]:
        features = list(self._real.columns[:-1])

        real_pi = sorted(zip(features, p_i_real), key=lambda x: x[1], reverse=True)
        synth_pi = sorted(zip(features, p_i_synth), key=lambda x: x[1], reverse=True)
        real_shap = sorted(zip(features, shap_value_real), key=lambda x: x[1], reverse=True)
        synth_shap = sorted(zip(features, shap_value_synth), key=lambda x: x[1], reverse=True)

        real_pi_rank = [f for f, _ in real_pi]
        synth_pi_rank = [f for f, _ in synth_pi]
        real_shap_rank = [f for f, _ in real_shap]
        synth_shap_rank = [f for f, _ in synth_shap]

        return (
            rbo.RankingSimilarity(real_pi_rank, synth_pi_rank).rbo(),
            rbo.RankingSimilarity(real_shap_rank, synth_shap_rank).rbo()
        )

    def spearman_compare_feature_importance(self,
                                            p_i_real: list[float], shap_value_real: list[float],
                                            p_i_synth: list[float], shap_value_synth: list[float]) -> tuple[float, float]:
        res_pi = stats.spearmanr(p_i_real, p_i_synth)
        res_shap = stats.spearmanr(shap_value_real, shap_value_synth)
        rho_pi = getattr(res_pi, 'statistic', getattr(res_pi, 'correlation', np.nan))
        rho_sh = getattr(res_shap, 'statistic', getattr(res_shap, 'correlation', np.nan))
        return rho_pi, rho_sh

    def kendall_compare_feature_importance(self,
                                           p_i_real: list[float], shap_value_real: list[float],
                                           p_i_synth: list[float], shap_value_synth: list[float]) -> tuple[float, float]:
        res_pi = stats.kendalltau(p_i_real, p_i_synth)
        res_shap = stats.kendalltau(shap_value_real, shap_value_synth)
        tau_pi = getattr(res_pi, 'statistic', np.nan)
        tau_sh = getattr(res_shap, 'statistic', np.nan)
        return tau_pi, tau_sh

    # -------------------------
    # Convenience wrappers
    # -------------------------
    def train_real_test_real(self, with_feature_importance: bool = False, **tune_kwargs):
        return self.train_test(False, True, with_feature_importance, **tune_kwargs)

    def train_synthetic_test_real(self, with_feature_importance: bool = False, **tune_kwargs):
        return self.train_test(True, True, with_feature_importance, **tune_kwargs)

    def train_real_test_real_with_xgboost(self, with_feature_importance: bool = False, **tune_kwargs):
        return self.train_test(False, True, with_feature_importance, True, **tune_kwargs)

    def train_synthetic_test_real_xgboost(self, with_feature_importance: bool = False, **tune_kwargs):
        return self.train_test(True, True, with_feature_importance, True, **tune_kwargs)

    def get_best_params(self) -> dict[str, dict]:
        return dict(self._best_params)

    def feature_importance_analysis(self, isSyntheticData: bool):
        return
