# --------- fit_aia_guard_strong (model-comparing, tuned; no LGBM/XGB) ---
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import warnings, math
import numpy as np, pandas as pd

from sklearn.model_selection import (
    StratifiedKFold, train_test_split, RandomizedSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor
)

# ---------------------------------------------------------------------
# small helpers
def _make_ohe(sparse: bool = False) -> OneHotEncoder:
    """Version-safe OHE: sklearn>=1.2 uses 'sparse_output'; older uses 'sparse'."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=sparse)

def _split_feats(X_cols: List[str], num_cols: List[str] | None,
                 cat_cols: List[str] | None, X: pd.DataFrame):
    if num_cols is None:
        num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_feats = [c for c in num_cols if c in X_cols]
    if cat_cols is None:
        cat_feats = [c for c in X_cols if c not in num_feats]
    else:
        cat_feats = [c for c in cat_cols if c in X_cols]
    return num_feats, cat_feats

def _pre_tree(num_feats, cat_feats):
    # Trees don’t need scaling; dense OHE keeps things simple
    return ColumnTransformer(
        [('num', 'passthrough', num_feats),
         ('cat', _make_ohe(sparse=False), cat_feats)],
        remainder='drop'
    )

def _pre_linear(num_feats, cat_feats):
    # Linear models like scaled numerics
    return ColumnTransformer(
        [('num', MinMaxScaler(), num_feats),
         ('cat', _make_ohe(sparse=False), cat_feats)],
        remainder='drop'
    )

def _adaptive_cv(y: pd.Series, max_cv: int = 5) -> int:
    """Choose CV folds that do not exceed the smallest class count."""
    vc = pd.Series(y).value_counts()
    return max(2, min(max_cv, int(vc.min()))) if len(vc) else 0

# ---------------------------------------------------------------------
def _clf_candidates(random_state: int) -> list[tuple[str, Pipeline, dict]]:
    """Return (name, pipeline(with placeholder pre), param_dist) triples."""
    cands: list[tuple[str, Pipeline, dict]] = []

    # Logistic Regression (good calibrated baseline)
    logreg = Pipeline([('pre', 'passthrough'),
                       ('clf', LogisticRegression(
                           solver='saga', max_iter=2000,
                           class_weight='balanced', multi_class='auto',
                           random_state=random_state
                       ))])
    cands.append(('logreg', logreg, {
        'clf__C': [0.1, 0.3, 1.0, 3.0, 10.0],
        'clf__penalty': ['l2']
    }))

    # HistGradientBoostingClassifier (fast, strong)
    hgbc = Pipeline([('pre', 'passthrough'),
                     ('clf', HistGradientBoostingClassifier(
                         random_state=random_state
                     ))])
    cands.append(('hgbc', hgbc, {
        'clf__learning_rate': [0.03, 0.05, 0.1],
        'clf__max_leaf_nodes': [31, 63, 127],
        'clf__min_samples_leaf': [20, 50],
        'clf__l2_regularization': [0.0, 1e-3, 1e-2],
    }))

    # RandomForest (robust, works well with OHE)
    rf = Pipeline([('pre', 'passthrough'),
                   ('clf', RandomForestClassifier(
                       n_estimators=400, class_weight='balanced_subsample',
                       random_state=random_state, n_jobs=-1
                   ))])
    cands.append(('rf', rf, {
        'clf__max_depth': [None, 12, 20],
        'clf__min_samples_leaf': [1, 2, 4]
    }))

    return cands

def _select_best_classifier(
    X: pd.DataFrame, y: pd.Series,
    num_feats: List[str], cat_feats: List[str],
    random_state: int = 42, n_iter: int = 12, n_jobs: int = -1
):
    """
    Tune several classifiers with neg_log_loss (probability quality),
    refit the best, then calibrate it.
    """
    # minimal support to proceed
    cls_counts = y.value_counts()
    minc = int(cls_counts.min())
    if minc < 2:
        return None, {'note': 'insufficient class support'}

    cv_splits = _adaptive_cv(y, max_cv=5)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    cands = _clf_candidates(random_state)

    best = None
    best_score = +np.inf
    best_report = {}

    for name, pipe, pdist in cands:
        # choose preprocessor: linear vs trees
        pipe.set_params(pre=_pre_linear(num_feats, cat_feats) if name == 'logreg'
                        else _pre_tree(num_feats, cat_feats))

        # cap draws so we don't exceed the full grid
        full_grid_size = 1
        for v in pdist.values():
            full_grid_size *= len(v)
        n_draws = min(n_iter, full_grid_size)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=pdist,
            n_iter=n_draws,
            scoring='neg_log_loss',
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X, y)

        score = -float(search.best_score_)  # log_loss (lower is better)
        if score < best_score:
            best_score = score
            best = search.best_estimator_
            best_report = {
                'model': name,
                'neg_log_loss_cv': float(search.best_score_),
                'best_params': search.best_params_
            }

    if best is None:
        return None, {'note': 'no classifier selected'}

    # Probability calibration on the chosen model (multiclass handled via OvR)
    method = 'isotonic' if (len(y) >= 3000 and y.nunique() <= 20) else 'sigmoid'
    calib_cv = _adaptive_cv(y, max_cv=5)
    calibrated = CalibratedClassifierCV(best, method=method, cv=calib_cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated.fit(X, y)

    best_report['calibration'] = dict(method=method, cv=calib_cv)
    return calibrated, best_report

def _select_best_quantile_interval(
    X: pd.DataFrame, y: pd.Series,
    num_feats: List[str], cat_feats: List[str],
    q_low: float, q_high: float,
    random_state: int = 42
):
    """
    Fit & compare quantile regressors (HGBR and GBR).
    Score on a validation split using coverage error + normalized width.
    Return (preprocess, low_model, high_model, report, iqr).
    """
    # preprocess once
    pre = ColumnTransformer(
        [('num', 'passthrough', num_feats),
         ('cat', _make_ohe(sparse=False), cat_feats)],
        remainder='drop'
    )

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    Xtr = pre.fit_transform(X_tr)
    Xva = pre.transform(X_va)
    ytr = y_tr.to_numpy()
    yva = y_va.to_numpy()

    # IQR for normalization
    q10, q90 = np.quantile(y.to_numpy(), [0.10, 0.90])
    iqr = (q90 - q10) if q90 > q10 else (np.std(y) + 1e-6)
    target_cov = q_high - q_low

    def eval_pair(low_pred, high_pred):
        width = np.maximum(high_pred - low_pred, 0.0)
        cover = ((yva >= low_pred) & (yva <= high_pred)).mean()
        cov_err = abs(cover - target_cov)
        norm_width = np.median(width) / (iqr + 1e-9)
        return cov_err + 0.25 * norm_width, cover, float(np.median(width))

    best = dict(score=+np.inf)

    # -------- Candidate family 1: HistGradientBoostingRegressor --------
    for params in [
        dict(learning_rate=0.05, max_leaf_nodes=63,  min_samples_leaf=20, l2_regularization=1e-3),
        dict(learning_rate=0.10, max_leaf_nodes=127, min_samples_leaf=20, l2_regularization=1e-3),
        dict(learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=50, l2_regularization=0.0)
    ]:
        low  = HistGradientBoostingRegressor(loss='quantile', quantile=q_low,
                                             random_state=random_state, **params).fit(Xtr, ytr)
        high = HistGradientBoostingRegressor(loss='quantile', quantile=q_high,
                                             random_state=random_state, **params).fit(Xtr, ytr)
        s, cov, w = eval_pair(low.predict(Xva), high.predict(Xva))
        if s < best['score']:
            best = dict(family='HGBR', score=s, cov=cov, width=w, params=params, low=low, high=high)

    # -------- Candidate family 2: GradientBoostingRegressor ------------
    for params in [
        dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=1.0),
        dict(n_estimators=500, learning_rate=0.06, max_depth=4, subsample=0.8),
    ]:
        low  = GradientBoostingRegressor(loss='quantile', alpha=q_low,
                                         random_state=random_state, **params).fit(Xtr, ytr)
        high = GradientBoostingRegressor(loss='quantile', alpha=q_high,
                                         random_state=random_state, **params).fit(Xtr, ytr)
        s, cov, w = eval_pair(low.predict(Xva), high.predict(Xva))
        if s < best['score']:
            best = dict(family='GBR', score=s, cov=cov, width=w, params=params, low=low, high=high)

    report = dict(
        family=best['family'], score=float(best['score']),
        val_coverage=float(best['cov']), val_median_width=float(best['width']),
        params=best['params'], target_coverage=float(target_cov)
    )
    return pre, best['low'], best['high'], report, float(iqr)

# ---------------------------------------------------------------------
def fit_aia_guard_strong(
    real_df   : pd.DataFrame,
    sens_cols : List[str],
    num_cols  : List[str],
    cat_cols  : List[str],
    *,
    min_count_per_class : int = 3,
    q_clf               : float = 0.90,   # probability quantile for τ (classification)
    q_low               : float = 0.10,   # lower quantile (regression)
    q_high              : float = 0.90,   # upper quantile (regression)
    tau_reg_width_frac  : float = 0.20,   # τ_width = frac * IQR(S) (regression)
    random_state        : int = 42,
    clf_n_iter          : int = 12,
    clf_n_jobs          : int = -1,
    train_row_cap       : int | None = 20000  # subsample per sensitive column for speed
) -> Dict[str, dict]:
    """
    Build one AIA guard per sensitive column by selecting & tuning the predictor of S from X\\{S}.

    Returns a dict[s] -> spec:
      - type: 'clf' or 'reg'
      - pipe: CalibratedClassifierCV (clf) OR {'pre','low','high'} (reg)
      - X_cols: feature columns used (all except s)
      - tau_clf: float (clf)  OR  tau_reg_width: float (reg)
      - model_report: dict (best model & params)
      - s_stats: dict (diagnostics)
    """
    guards: Dict[str, dict] = {}
    all_cols = real_df.columns.tolist()
    rng = np.random.default_rng(random_state)

    for s in sens_cols:
        X_cols = [c for c in all_cols if c != s]
        X_full = real_df[X_cols]
        y_full = real_df[s]

        # optional subsample for speed
        if train_row_cap is not None and len(X_full) > train_row_cap:
            idx = rng.choice(len(X_full), size=train_row_cap, replace=False)
            X = X_full.iloc[idx].copy()
            y = y_full.iloc[idx].copy()
        else:
            X, y = X_full, y_full

        # Decide numeric vs categorical by schema lists (fallback to dtype)
        is_numeric = (s in num_cols) if (num_cols is not None) else pd.api.types.is_numeric_dtype(y)
        num_feats, cat_feats = _split_feats(X_cols, num_cols, cat_cols, X_full)

        # ---------------- CATEGORICAL (classification) -----------------
        if not is_numeric:
            cls_counts = y.value_counts()
            valid = cls_counts[cls_counts >= min_count_per_class].index
            mask = y.isin(valid)
            X_fit = X.loc[mask]
            y_fit = y.loc[mask]

            if y_fit.nunique() < 2:
                # Too few classes to train → disabled guard for this S
                dummy = Pipeline([('pre', _pre_linear(num_feats, cat_feats)),
                                  ('clf', LogisticRegression(max_iter=1))]).fit(
                                      X.iloc[:2], pd.Series([0, 1])[:2]
                                  )
                guards[s] = dict(
                    type='clf', pipe=dummy, X_cols=X_cols,
                    tau_clf=1.0,
                    model_report={'note': 'disabled (insufficient classes)'},
                    s_stats={'n_train': int(mask.sum())}
                )
                continue

            # Select the best classifier + calibrate
            best_clf, report = _select_best_classifier(
                X_fit, y_fit, num_feats, cat_feats,
                random_state=random_state, n_iter=clf_n_iter, n_jobs=clf_n_jobs
            )
            if best_clf is None:
                # safety: fallback to HGBClassifier (uncalibrated)
                base = Pipeline([('pre', _pre_tree(num_feats, cat_feats)),
                                 ('clf', HistGradientBoostingClassifier(random_state=random_state))]).fit(X_fit, y_fit)
                best_clf, report = base, {'model': 'hgbc_fallback'}

            # Compute τ on *all* real rows (predict using X\{s})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                proba = best_clf.predict_proba(X_full)
            conf = proba.max(axis=1)
            tau  = float(np.quantile(conf, q_clf))

            guards[s] = dict(
                type='clf', pipe=best_clf, X_cols=X_cols,
                tau_clf=tau,
                model_report=report,
                s_stats={'q_conf': tau, 'n_train': int(mask.sum()),
                         'n_classes': int(y_fit.nunique())}
            )

        # ---------------- NUMERIC (regression intervals) ---------------
        else:
            pre, low, high, report, iqr = _select_best_quantile_interval(
                X, y, num_feats, cat_feats,
                q_low=q_low, q_high=q_high, random_state=random_state
            )
            tau_width = float(tau_reg_width_frac * iqr)

            guards[s] = dict(
                type='reg',
                pipe={'pre': pre, 'low': low, 'high': high},
                X_cols=X_cols,
                tau_reg_width=tau_width,
                model_report=report,
                s_stats={'iqr': float(iqr), 'tau_width': tau_width,
                         'q_low': q_low, 'q_high': q_high}
            )

    return guards
