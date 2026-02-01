# ---------- teacher for target Y ----------
from typing import List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def fit_teacher_gate(
    real_df: pd.DataFrame,
    target_col: str,
    num_cols: List[str],
    cat_cols: List[str],
    *,
    q_util_clf: float = 0.20,      # keep rows above this lower quantile of p(y|x) on real
    q_low_util: float = 0.05,      # numeric: lower conditional quantile
    q_high_util: float = 0.95,     # numeric: upper conditional quantile
    random_state: int = 42
) -> dict:
    """
    Build a 'teacher' on real data to preserve utility for Y.
    - If Y is categorical -> calibrated LR; tau_util = q_util_clf-quantile of p_y(x) on real.
    - If Y is numeric     -> HGB quantile regressors at (q_low_util, q_high_util).
    Returns dict with type, model(s), preprocessor, and thresholds.
    """
    X_cols = [c for c in real_df.columns if c != target_col]
    num_in = [c for c in num_cols if c in X_cols]
    cat_in = [c for c in cat_cols if c in X_cols]
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:  # sklearn<1.2
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    pre = ColumnTransformer(
        [('num', MinMaxScaler(), num_in),
         ('cat', ohe, cat_in)],
        remainder='drop'
    )

    y = real_df[target_col]
    if target_col in cat_cols:
        # categorical target -> calibrated LR
        y_cat = y.astype('category')
        base = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='auto', solver='lbfgs')
        # Calibrate with CV=5 if support allows; else fallback uncalibrated
        if y_cat.value_counts().min() >= 3:
            clf = CalibratedClassifierCV(base, method='sigmoid', cv=min(5, int(y_cat.value_counts().min())))
            pipe = Pipeline([('prep', pre), ('clf', clf)]).fit(real_df[X_cols], y_cat)
        else:
            pipe = Pipeline([('prep', pre), ('clf', base)]).fit(real_df[X_cols], y_cat)

        # threshold: lower quantile of real per-row realized-probability p_y(x)
        proba = pipe.predict_proba(real_df[X_cols])
        idx   = pipe.named_steps['clf'].classes_ if isinstance(pipe.named_steps['clf'], CalibratedClassifierCV) else pipe.named_steps['clf'].classes_
        # realized prob g_y(x):
        y_idx = pd.Series(y_cat).map({cls:i for i, cls in enumerate(idx)}).to_numpy()
        p_yx  = proba[np.arange(len(real_df)), y_idx]
        tau_util = float(np.quantile(p_yx, q_util_clf))
        return dict(type='clf', pipe=pipe, X_cols=X_cols, tau_util=tau_util,
                    stats=dict(q_util_clf=q_util_clf, p_yx_median=float(np.median(p_yx))))

    else:
        # numeric target -> quantile regressors
        Xtr = pre.fit_transform(real_df[X_cols])
        ytr = y.astype(float).to_numpy()
        low  = HistGradientBoostingRegressor(loss='quantile', quantile=float(q_low_util),  random_state=random_state).fit(Xtr, ytr)
        high = HistGradientBoostingRegressor(loss='quantile', quantile=float(q_high_util), random_state=random_state).fit(Xtr, ytr)
        return dict(type='reg', pipe=dict(pre=pre, low=low, high=high), X_cols=X_cols,
                    q_low=q_low_util, q_high=q_high_util)

def teacher_pass_mask(df_batch: pd.DataFrame, teacher: dict) -> np.ndarray:
    """Boolean mask: True if candidate passes the teacher gate."""
    if teacher['type'] == 'clf':
        pipe = teacher['pipe']; X_cols = teacher['X_cols']; tau = teacher['tau_util']
        proba = pipe.predict_proba(df_batch[X_cols])
        # realized p_y(x) for the batch:
        classes = pipe.named_steps['clf'].classes_ if isinstance(pipe.named_steps['clf'], CalibratedClassifierCV) else pipe.named_steps['clf'].classes_
        y_idx = df_batch[classes.name if hasattr(classes, 'name') else None]
        # safer: map from label -> index using pipe.classes_
        cls_map = {cls:i for i, cls in enumerate(classes)}
        y_idx = df_batch[teacher['pipe'].named_steps['clf'].classes_.name] if False else np.array([cls_map.get(y, -1) for y in df_batch.iloc[:, df_batch.columns.get_loc(classes.name)]])  # fallback not used; see score fn below
        # Simpler: score with top1 prob for the realized label if available; otherwise use max prob
        # Instead, we implement 'teacher_score' to avoid dtype issues.
        raise NotImplementedError


def teacher_score(df_batch: pd.DataFrame, teacher: dict, target_col: str) -> np.ndarray:
    """Utility score: larger is better. Categorical: log p_y(x) ; Numeric: 0 if inside band, negative outside distance."""
    if teacher['type'] == 'clf':
        pipe = teacher['pipe']; X_cols = teacher['X_cols']
        proba = pipe.predict_proba(df_batch[X_cols])
        classes = pipe.named_steps['clf'].classes_ if isinstance(pipe.named_steps['clf'], CalibratedClassifierCV) else pipe.named_steps['clf'].classes_
        # map realized labels in batch to indices
        cls_map = {cls:i for i, cls in enumerate(classes)}
        y_idx = df_batch[target_col].map(cls_map).to_numpy()
        # rows with unseen labels -> fall back to max prob
        score = np.full(len(df_batch), -np.inf, dtype=float)
        mask_ok = (y_idx >= 0)
        score[mask_ok] = np.log(np.maximum(1e-12, proba[np.arange(len(df_batch))[mask_ok], y_idx[mask_ok]]))
        score[~mask_ok] = np.log(np.maximum(1e-12, proba.max(axis=1)[~mask_ok]))
        return score
    else:
        pre  = teacher['pipe']['pre']
        low  = teacher['pipe']['low']
        high = teacher['pipe']['high']
        Xb   = pre.transform(df_batch[teacher['X_cols']])
        lo   = low.predict(Xb); hi = high.predict(Xb)
        lo2  = np.minimum(lo, hi); hi2 = np.maximum(lo, hi)
        y    = df_batch[target_col].astype(float).to_numpy()
        # 0 if inside interval, negative distance outside:
        below = lo2 - y
        above = y - hi2
        dist_outside = np.maximum(below, 0) + np.maximum(above, 0)  # one-sided distance
        return -dist_outside  # larger is better

def teacher_pass_mask(df_batch: pd.DataFrame, teacher: dict, target_col: str, tau_util: float = None) -> np.ndarray:
    """Pass if score >= tau (clf) or inside interval (reg). If tau_util is None and clf, use teacher['tau_util']."""
    if teacher['type'] == 'clf':
        tau = teacher['tau_util'] if tau_util is None else tau_util
        s   = teacher_score(df_batch, teacher, target_col)  # log p_y(x)
        return (s >= np.log(max(1e-12, tau)))   # compare in log-space
    else:
        # numeric: pass iff inside band -> score == 0
        s = teacher_score(df_batch, teacher, target_col)
        return (s >= 0.0)  # inside band


import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# ---------- utility: pinball loss for quantile regression ----------
def _pinball_loss(y: np.ndarray, u: np.ndarray, q: float) -> float:
    # L_q(y, u) = (q - 1_{y < u}) * (y - u)
    return float(np.mean((q - (y < u).astype(float)) * (y - u)))

# ---------- main: one‑row overview ----------
def audit_teacher_gate_overview(
    teacher: Dict[str, Any],
    real_df: pd.DataFrame,
    target_col: str,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    One-row audit of a teacher-gate fitted on REAL data.
    For categorical Y:
      - tau_util (threshold on p_y(x)); accept_rate (fraction with p_y(x) >= tau)
      - top1_acc, top1_prob_median
      - cross-entropy (mean -log p_y(x)), and Brier (multiclass) score
      - p_y(x) quantiles (q05, q50, q95)
    For numeric Y:
      - q_low, q_high, nominal_coverage = q_high - q_low
      - empirical_coverage, violations_low (y<q_low), violations_high (y>q_high)
      - width statistics (median, mean, p90, p95)
      - pinball losses at q_low and q_high
    """
    out = {}

    if teacher['type'] == 'clf':
        pipe   = teacher['pipe']
        X_cols = teacher['X_cols']
        tau    = float(teacher['tau_util'])
        proba  = pipe.predict_proba(real_df[X_cols])                       # (N, C)
        classes = pipe.named_steps['clf'].classes_
        # map realized labels to indices
        cls_map = {cls: i for i, cls in enumerate(classes)}
        y_idx   = real_df[target_col].map(cls_map).to_numpy()
        # realized probabilities p_y(x); fallback to max prob for any unseen label (shouldn't happen on real_df)
        p_yx = np.full(len(real_df), np.nan, dtype=float)
        mask_ok = (y_idx >= 0)
        p_yx[mask_ok]  = proba[np.arange(len(real_df))[mask_ok], y_idx[mask_ok]]
        p_yx[~mask_ok] = proba.max(axis=1)[~mask_ok]

        # acceptance & calibration summaries
        accept_rate = float(np.mean(p_yx >= tau))
        top1_idx = np.argmax(proba, axis=1)
        top1_acc = float(np.mean(top1_idx == y_idx))
        top1_prob_median = float(np.median(proba.max(axis=1)))

        # cross-entropy = mean(-log p_y(x)); Brier (multiclass) ~ sum_j (p_j - y_j)^2
        ce = float(np.mean(-np.log(np.clip(p_yx, 1e-12, 1.0))))
        # multiclass Brier
        Y_onehot = np.zeros_like(proba)
        valid = mask_ok & (y_idx < proba.shape[1])
        Y_onehot[np.arange(len(real_df))[valid], y_idx[valid]] = 1.0
        brier = float(np.mean(np.sum((proba - Y_onehot) ** 2, axis=1)))

        # pack overview
        q05, q50, q95 = np.quantile(p_yx, [0.05, 0.50, 0.95])
        out = dict(
            type='clf',
            n_classes=int(proba.shape[1]),
            tau_util=tau,
            accept_rate=accept_rate,
            top1_acc=top1_acc,
            top1_prob_median=top1_prob_median,
            ce_mean=ce,
            brier_mean=brier,
            p_yx_q05=float(q05),
            p_yx_q50=float(q50),
            p_yx_q95=float(q95)
        )
        return pd.DataFrame([out])

    else:  # numeric teacher
        pre   = teacher['pipe']['pre']
        low   = teacher['pipe']['low']
        high  = teacher['pipe']['high']
        X_cols = teacher['X_cols']
        q_low  = float(teacher['q_low'])
        q_high = float(teacher['q_high'])

        Xb  = pre.transform(real_df[X_cols])
        y   = real_df[target_col].astype(float).to_numpy()
        lo  = low.predict(Xb)
        hi  = high.predict(Xb)
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        width = hi2 - lo2

        inside = (y >= lo2) & (y <= hi2)
        emp_cov = float(np.mean(inside))
        nom_cov = float(q_high - q_low)
        v_low   = float(np.mean(y < lo2))
        v_high  = float(np.mean(y > hi2))

        # pinball losses at the two quantiles
        pl_low  = _pinball_loss(y, lo2, q_low)
        pl_high = _pinball_loss(y, hi2, q_high)

        out = dict(
            type='reg',
            q_low=q_low, q_high=q_high,
            nominal_coverage=nom_cov,
            empirical_coverage=emp_cov,
            violations_low=v_low,
            violations_high=v_high,
            width_median=float(np.median(width)),
            width_mean=float(np.mean(width)),
            width_p90=float(np.quantile(width, 0.90)),
            width_p95=float(np.quantile(width, 0.95)),
            pinball_low=pl_low,
            pinball_high=pl_high
        )
        return pd.DataFrame([out])

# ---------- details: reliability / calibration tables ----------
def audit_teacher_gate_details(
    teacher: Dict[str, Any],
    real_df: pd.DataFrame,
    target_col: str,
    n_bins: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Detailed diagnostics:
      - For categorical Y:
          * calib_top1: reliability curve of top1 probability vs empirical accuracy
          * realized_prob_bins: distribution of p_y(x) vs acceptance at tau_util
      - For numeric Y:
          * interval_stats: histogram of widths and coverage per width bin
          * quantile_calib: fraction below q_low and above q_high (should match q_low, 1-q_high)
    Returns a dict of DataFrames keyed by table name.
    """
    if teacher['type'] == 'clf':
        pipe   = teacher['pipe']
        X_cols = teacher['X_cols']
        tau    = float(teacher['tau_util'])
        proba  = pipe.predict_proba(real_df[X_cols])
        classes = pipe.named_steps['clf'].classes_
        cls_map = {cls: i for i, cls in enumerate(classes)}
        y_idx   = real_df[target_col].map(cls_map).to_numpy()

        # ---- 1) Top1 reliability bins ----
        top1_prob = proba.max(axis=1)
        top1_pred = np.argmax(proba, axis=1)
        is_correct = (top1_pred == y_idx)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.digitize(top1_prob, edges[1:-1], right=False)
        rows1 = []
        for b in range(n_bins):
            mask = (bins == b)
            n = int(np.sum(mask))
            if n == 0:
                rows1.append(dict(bin=b, left=edges[b], right=edges[b+1], n=0,
                                  avg_pred=np.nan, acc=np.nan))
            else:
                rows1.append(dict(bin=b, left=edges[b], right=edges[b+1], n=n,
                                  avg_pred=float(np.mean(top1_prob[mask])),
                                  acc=float(np.mean(is_correct[mask]))))
        calib_top1 = pd.DataFrame(rows1)

        # ---- 2) Realized p_y(x) bins and acceptance wrt tau ----
        p_yx = np.full(len(real_df), np.nan, dtype=float)
        mask_ok = (y_idx >= 0)
        p_yx[mask_ok]  = proba[np.arange(len(real_df))[mask_ok], y_idx[mask_ok]]
        p_yx[~mask_ok] = top1_prob[~mask_ok]  # fallback

        edges2 = np.linspace(0.0, 1.0, n_bins + 1)
        bins2 = np.digitize(p_yx, edges2[1:-1], right=False)
        rows2 = []
        for b in range(n_bins):
            mask = (bins2 == b)
            n = int(np.sum(mask))
            if n == 0:
                rows2.append(dict(bin=b, left=edges2[b], right=edges2[b+1], n=0,
                                  p_yx_mean=np.nan, pass_rate=np.nan))
            else:
                rows2.append(dict(bin=b, left=edges2[b], right=edges2[b+1], n=n,
                                  p_yx_mean=float(np.mean(p_yx[mask])),
                                  pass_rate=float(np.mean(p_yx[mask] >= tau))))
        realized_prob_bins = pd.DataFrame(rows2)

        return dict(calib_top1=calib_top1, realized_prob_bins=realized_prob_bins)

    else:  # numeric Y
        pre   = teacher['pipe']['pre']
        low   = teacher['pipe']['low']
        high  = teacher['pipe']['high']
        X_cols = teacher['X_cols']
        q_low  = float(teacher['q_low'])
        q_high = float(teacher['q_high'])

        Xb  = pre.transform(real_df[X_cols])
        y   = real_df[target_col].astype(float).to_numpy()
        lo  = low.predict(Xb)
        hi  = high.predict(Xb)
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        width = hi2 - lo2
        inside = (y >= lo2) & (y <= hi2)

        # ---- 1) Width histogram & inside-rate per bin ----
        w_edges = np.quantile(width, np.linspace(0.0, 1.0, n_bins + 1))
        w_edges = np.unique(w_edges)
        if len(w_edges) < 2:
            w_edges = np.array([float(width.min()), float(width.max())])
        bins = np.digitize(width, w_edges[1:-1], right=False)
        rows1 = []
        for b in range(len(w_edges) - 1):
            mask = (bins == b)
            n = int(np.sum(mask))
            if n == 0:
                rows1.append(dict(bin=b, left=w_edges[b], right=w_edges[b+1], n=0,
                                  width_mean=np.nan, inside_rate=np.nan))
            else:
                rows1.append(dict(bin=b, left=w_edges[b], right=w_edges[b+1], n=n,
                                  width_mean=float(np.mean(width[mask])),
                                  inside_rate=float(np.mean(inside[mask]))))
        interval_stats = pd.DataFrame(rows1)

        # ---- 2) Quantile calibration summary ----
        under = float(np.mean(y < lo2))
        over  = float(np.mean(y > hi2))
        coverage = float(np.mean(inside))
        quantile_calib = pd.DataFrame([dict(
            q_low=q_low, q_high=q_high,
            nominal_coverage=float(q_high - q_low),
            empirical_coverage=coverage,
            violations_low=under,
            violations_high=over
        )])

        return dict(interval_stats=interval_stats, quantile_calib=quantile_calib)
