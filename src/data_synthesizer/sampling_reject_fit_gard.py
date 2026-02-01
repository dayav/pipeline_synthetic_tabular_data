import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

from privacy_sampling.quota_matching_helpers import fit_quota_1d_spec, quota_apply_replace, quota_delta_replace, quota_init_state
from privacy_sampling.teacher_gate import audit_teacher_gate_details, audit_teacher_gate_overview, fit_teacher_gate, teacher_pass_mask, teacher_score

# ==============================================================
# 1) FIT: AIG with quantile regression for numeric S
# ==============================================================

def fit_aig_guard_qr(
    real_df: pd.DataFrame,
    sens_cols: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    *,
    # classification (categorical S)
    q_clf: float = 0.90,               # tau_clf = q_clf-quantile of max prob
    min_count_per_class: int = 3,
    random_state: int = 42,
    # regression (numeric S)
    q_low: float = 0.10,
    q_high: float = 0.90,
    use_width_quantile: bool = False,  # if True -> threshold by width quantile; else by IQR fraction
    q_width: float = 0.10,             # flag narrowest q_width fraction if use_width_quantile=True
    tau_reg_width_frac: float = 0.20   # otherwise, tau_w = tau_reg_width_frac * IQR(S)
) -> Dict[str, dict]:
    """
    Build Attribute-Inference Guards for each sensitive column S:
      • Categorical S -> 'clf' guard with calibrated LR (robust to small classes).
      • Numeric S     -> 'reg' guard with two quantile regressors (q_low, q_high).
                         Flag when predicted width <= tau_w.
    Returns a dict: s -> spec with keys:
      - 'type': 'clf' or 'reg'
      - 'pipe' (clf): sklearn Pipeline with predict_proba
      - 'tau_clf' (clf): float threshold on max-prob
      - 'X_cols' : features used (all columns except S)
      - 'pipe' (reg): dict {'pre': ColumnTransformer, 'low': HGBR, 'high': HGBR}
      - 'tau_w'  (reg): float threshold on interval width
      - 's_stats': diagnostics (IQR, quantiles, etc.)
    """
    guards: Dict[str, dict] = {}

    for s in sens_cols:
        X_cols = [c for c in real_df.columns if c != s]
        if len(X_cols) == 0:
            # no predictors -> disable guard for S
            guards[s] = dict(type='disabled', reason='no_predictors', X_cols=X_cols)
            continue

        # Build preprocessing for X\{s}
        num_in = [c for c in num_cols if c in X_cols]
        cat_in = [c for c in cat_cols if c in X_cols]
        pre = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), num_in),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_in)
            ],
            remainder='drop'
        )

        y_raw = real_df[s]

        # -------------------- CATEGORICAL S -> classification guard --------------------
        if not pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.astype('category')
            # If degenerate (single class), disable
            if y.nunique() < 2:
                pipe = Pipeline([('prep', pre), ('clf', LogisticRegression(max_iter=1000))]).fit(
                    real_df[X_cols].head(2), pd.Series([0, 1])[:2]
                )
                guards[s] = dict(type='clf', pipe=pipe, tau_clf=1.0, X_cols=X_cols,
                                 s_stats=dict(note='disabled_single_class'))
                continue

            counts = y.value_counts()
            minc = int(counts.min())

            base = LogisticRegression(
                max_iter=1000, class_weight='balanced', multi_class='auto', solver='lbfgs'
            )

            if minc >= 3:
                cv = min(5, minc)
                clf = CalibratedClassifierCV(base, method='sigmoid', cv=cv)
                pipe = Pipeline([('prep', pre), ('clf', clf)]).fit(real_df[X_cols], y)
            elif minc == 2 and len(y) >= 20:
                Xtr, Xva, ytr, yva = train_test_split(
                    real_df[X_cols], y, test_size=0.2, stratify=y, random_state=random_state
                )
                base_pipe = Pipeline([('prep', pre), ('clf', base)]).fit(Xtr, ytr)
                pipe = CalibratedClassifierCV(base_pipe, method='sigmoid', cv='prefit').fit(Xva, yva)
            else:
                # too small to calibrate -> uncalibrated LR
                pipe = Pipeline([('prep', pre), ('clf', base)]).fit(real_df[X_cols], y)

            conf = pipe.predict_proba(real_df[X_cols]).max(axis=1)
            tau = float(np.quantile(conf, q_clf))
            guards[s] = dict(type='clf', pipe=pipe, tau_clf=tau, X_cols=X_cols,
                             s_stats=dict(q_clf=q_clf, min_class=minc, tau=tau))
            continue

        # -------------------- NUMERIC S -> quantile-regression guard -------------------
        # Train features for X\{s}
        X = real_df[X_cols]
        Xtr = pre.fit_transform(X)
        ytr = y_raw.astype(float).to_numpy()

        # Two quantile regressors (lower/upper)
        low = HistGradientBoostingRegressor(
            loss='quantile', quantile=float(q_low), random_state=random_state
        )
        high = HistGradientBoostingRegressor(
            loss='quantile', quantile=float(q_high), random_state=random_state
        )
        low.fit(Xtr, ytr)
        high.fit(Xtr, ytr)

        # Predict on REAL to set threshold
        lo = low.predict(Xtr)
        hi = high.predict(Xtr)
        # ensure order (robustness)
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        width = hi2 - lo2

        
        width_q95 = float(np.quantile(width, 0.95))
        # if not np.isfinite(width_q95) or width_q95 <= 1e-9:
        #     # Degenerate: predicted widths are ~0 on real data.
        #     # Option 1 (recommended): disable this guard entirely.
        #     guards[s] = dict(type='disabled', reason='degenerate_numeric_width', X_cols=X_cols)
        #     continue

        # Global scale (IQR) of S on real data
        s_q10, s_q90 = np.quantile(ytr, [0.10, 0.90])
        iqr = (s_q90 - s_q10) if s_q90 > s_q10 else (np.std(ytr) + 1e-6)

        if use_width_quantile:
            tau_w = float(np.quantile(width, q_width))  # flag narrowest q_width fraction
            policy = dict(kind='width_quantile', q_width=q_width)
        else:
            tau_w = float(tau_reg_width_frac * iqr)     # absolute threshold
            policy = dict(kind='iqr_fraction', frac=tau_reg_width_frac, iqr=float(iqr))

        guards[s] = dict(
            type='reg',
            pipe=dict(pre=pre, low=low, high=high),
            X_cols=X_cols,
            tau_w=tau_w,
            s_stats=dict(q_low=q_low, q_high=q_high, tau_w=tau_w, policy=policy)
        )

    return guards


# ==============================================================
# 2) EVALUATE: block mask and severity using the mixed guard set
# ==============================================================

def aig_block_mask_v2(
    df_batch: pd.DataFrame,
    guards: Dict[str, dict],
    margin_prob: float = 0.0
) -> np.ndarray:
    """
    True if a row is blocked by ANY sensitive guard.
    - 'clf': block if max class prob > tau_clf  (optionally, could add a top1-top2 margin)
    - 'reg': block if predicted width <= tau_w
    """
    n = len(df_batch)
    risky = np.zeros(n, dtype=bool)

    for s, spec in guards.items():
        typ = spec.get('type', 'disabled')
        X_cols = spec.get('X_cols', [])

        if typ == 'clf':
            pipe = spec['pipe']
            proba = pipe.predict_proba(df_batch[X_cols])
            # top1 prob
            conf = proba.max(axis=1)
            # optional margin between top1 and top2
            if proba.shape[1] >= 2 and margin_prob > 0.0:
                idx_sorted = np.argsort(proba, axis=1)
                top1 = proba[np.arange(n), idx_sorted[:, -1]]
                top2 = proba[np.arange(n), idx_sorted[:, -2]]
                gap  = top1 - top2
            else:
                gap = np.zeros(n)
                top1 = conf
            tau = spec['tau_clf']
            risky |= (top1 > tau) & (gap >= margin_prob)

        elif typ == 'reg':
            pre  = spec['pipe']['pre']
            low  = spec['pipe']['low']
            high = spec['pipe']['high']
            tau_w = spec['tau_w']

            Xb = pre.transform(df_batch[X_cols])
            lo = low.predict(Xb)
            hi = high.predict(Xb)
            lo2 = np.minimum(lo, hi)
            hi2 = np.maximum(lo, hi)
            width = hi2 - lo2
            risky |= (width < tau_w)

        else:
            # disabled: never blocks
            continue

    return risky


def aig_surplus_v2(
    df_batch: pd.DataFrame,
    guards: Dict[str, dict],
    margin_prob: float = 0.0
) -> np.ndarray:
    """
    Severity per row (positive => flagged; negative => safe).
    - 'clf': severity = conf - tau_clf
    - 'reg': severity = tau_w - width
    Returns the maximum severity across sensitive columns for each row.
    """
    n = len(df_batch)
    sev = np.full(n, -np.inf, dtype=float)

    for s, spec in guards.items():
        typ = spec.get('type', 'disabled')
        X_cols = spec.get('X_cols', [])

        if typ == 'clf':
            pipe = spec['pipe']
            proba = pipe.predict_proba(df_batch[X_cols])
            conf = proba.max(axis=1)

            if proba.shape[1] >= 2 and margin_prob > 0.0:
                idx_sorted = np.argsort(proba, axis=1)
                top1 = proba[np.arange(n), idx_sorted[:, -1]]
                top2 = proba[np.arange(n), idx_sorted[:, -2]]
                gap  = top1 - top2
                conf = np.where(gap >= margin_prob, top1, -np.inf)

            tau = spec['tau_clf']
            sev = np.maximum(sev, conf - tau)

        elif typ == 'reg':
            pre  = spec['pipe']['pre']
            low  = spec['pipe']['low']
            high = spec['pipe']['high']
            tau_w = spec['tau_w']

            Xb = pre.transform(df_batch[X_cols])
            lo = low.predict(Xb)
            hi = high.predict(Xb)
            width = np.maximum(hi, lo) - np.minimum(hi, lo)
            sev = np.maximum(sev, tau_w - width)

        else:
            continue

    # For rows that no guard contributed (all -inf), set to a safe negative value
    mask = ~np.isfinite(sev)
    if np.any(mask):
        sev[mask] = -1.0
    return sev


def audit_aig_qr(guards, real_df):
    rows = []
    for s, g in guards.items():
        typ = g['type']
        X_cols = g.get('X_cols', [])
        row = dict(sensitive=s, type=typ)
        if typ == 'clf':
            pipe = g['pipe']
            tau  = g['tau_clf']
            conf = pipe.predict_proba(real_df[X_cols]).max(axis=1)
            row.update(dict(tau=float(tau),
                            conf_p50=float(np.median(conf)),
                            flag_rate=float((conf > tau).mean())))
        elif typ == 'reg':
            pre  = g['pipe']['pre']
            low  = g['pipe']['low']
            high = g['pipe']['high']
            Xb   = pre.transform(real_df[X_cols])
            lo   = low.predict(Xb); hi = high.predict(Xb)
            width = np.maximum(lo, hi) - np.minimum(lo, hi)
            tau_w = g['tau_w']
            row.update(dict(tau_w=float(tau_w),
                            width_p50=float(np.median(width)),
                            width_p95=float(np.quantile(width, 0.95)),
                            flag_rate=float((width < tau_w).mean())))
        else:
            row.update(dict(note='disabled'))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['type','flag_rate'], ascending=[True, False])


# ==============================================================
# 3) SAMPLER: rejection-with-replacement using only AIG (QR-enabled)
# ==============================================================

def sampling_reject_aig_only_qr(
    model,
    real_df: pd.DataFrame,
    *,
    sens_cols: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    n_samples: int,
    # guard hyper-params
    q_clf: float = 0.90,
    min_count_per_class: int = 3,
    q_low: float = 0.10,
    q_high: float = 0.90,
    use_width_quantile: bool = False,
    q_width: float = 0.10,
    tau_reg_width_frac: float = 0.20,
    # sampling loop hyper-params
    target_rate: float = 0.0,      # desired fraction of flagged rows
    pool_size: int = 16,
    max_swaps: int = 200_000,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Rejection-with-replacement driven SOLELY by the Attribute-Inference Guard (AIG),
    with quantile-regression intervals for NUMERIC sensitive columns.

    Accept candidate only if it passes AIG; replace the worst current row
    (highest severity) with the safest candidate (most negative severity).
    Stop when fraction of blocked rows ≤ target_rate.
    """

    rng = np.random.default_rng(random_state)

    # 1) fit guards (clf for categorical S; quantile regression for numeric S)
    guards = fit_aig_guard_qr(
        real_df, sens_cols, num_cols, cat_cols,
        q_clf=q_clf, min_count_per_class=min_count_per_class, random_state=random_state,
        q_low=q_low, q_high=q_high,
        use_width_quantile=use_width_quantile, q_width=q_width, tau_reg_width_frac=tau_reg_width_frac
    )

    print(audit_aig_qr(guards, real_df))

    # 2) initial synthetic
    synth = model.sample(n_samples).reset_index(drop=True)
    sev   = aig_surplus_v2(synth, guards)      # severity per row
    rate  = float((sev > 0).mean())

    if verbose:
        print(f"[init] AIG-flag rate = {rate:.4f} (target ≤ {target_rate})")

    swaps = 0
    while rate > target_rate and swaps < max_swaps:
        # 3) draw a pool and keep only AIG-safe candidates
        pool = model.sample(pool_size).reset_index(drop=True)
        blocked = aig_block_mask_v2(pool, guards)
        safe_pool = pool[~blocked]

        if safe_pool.empty:
            # No fully safe candidate -> take the least severe in the pool
            pool_sev = aig_surplus_v2(pool, guards)
            best_idx = int(np.argmin(pool_sev))   # most negative severity is safest
            cand = pool.iloc[[best_idx]]
            cand_sev = pool_sev[best_idx]
        else:
            safe_sev = aig_surplus_v2(safe_pool, guards)
            best_idx = int(np.argmin(safe_sev))
            cand = safe_pool.iloc[[best_idx]]
            cand_sev = safe_sev[best_idx]

        # 4) replace worst current row (largest severity)
        w = int(np.argmax(sev))
        synth.iloc[w] = cand.iloc[0]
        sev[w]        = cand_sev

        # 5) update rate
        rate = float((sev > 0).mean())
        swaps += 1
        if verbose and swaps % 200 == 0:
            print(f"[{swaps} swaps] AIG-flag rate = {rate:.4f}")

    if verbose:
        print(f"[done] AIG-flag rate = {rate:.4f} after {swaps} swaps")
    return synth.reset_index(drop=True)


def sampling_reject_aig_with_teacher_and_quota(
    model, real_df, *,
    sens_cols, num_cols, cat_cols,
    n_samples,
    # AIG params (as before) ...
    q_clf=0.90, min_count_per_class=3,
    q_low=0.10, q_high=0.90,
    use_width_quantile=False, q_width=0.10, tau_reg_width_frac=0.20,
    # TEACHER params
    target_col: Optional[str] = None,
    q_util_clf: float = 0.20,
    q_low_util: float = 0.05, q_high_util: float = 0.95,
    # QUOTA params (see next section)
    quota_features: Optional[List[str]] = None,
    quota_num_bins: int = 10,
    lambda_quota: float = 1.0,
    random_state=42, pool_size=16, target_rate=0.0, max_swaps=200_000, verbose=True
):
    rng = np.random.default_rng(random_state)

    # ---- AIG guards
    guards = fit_aig_guard_qr(
        real_df, sens_cols, num_cols, cat_cols,
        q_clf=q_clf, min_count_per_class=min_count_per_class, random_state=random_state,
        q_low=q_low, q_high=q_high,
        use_width_quantile=use_width_quantile, q_width=q_width, tau_reg_width_frac=tau_reg_width_frac
    )
    if verbose:
        print(audit_aig_qr(guards, real_df))

    # ---- Teacher
    teacher = None
    if target_col is not None:
        teacher = fit_teacher_gate(real_df, target_col, num_cols, cat_cols,
                                   q_util_clf=q_util_clf, q_low_util=q_low_util, q_high_util=q_high_util,
                                   random_state=random_state)
        
        ov = audit_teacher_gate_overview(teacher, real_df, target_col, n_bins=10)
        print(ov.to_string(index=False))

        details = audit_teacher_gate_details(teacher, real_df, target_col, n_bins=10)
        for name, df in details.items():
            print(f"\n{name}:\n{df.to_string(index=False)}")

    # ---- Quota spec (optional; defined below)
    quota = None
    if quota_features:
        quota = fit_quota_1d_spec(real_df, quota_features, num_cols, cat_cols, n_samples, num_bins=quota_num_bins)

    # ---- init synthetic + severities + quota state
    synth = model.sample(n_samples).reset_index(drop=True)
    sev   = aig_surplus_v2(synth, guards)
    rate  = float((sev > 0).mean())
    if verbose: print(f"[init] AIG-flag rate = {rate:.4f} (target ≤ {target_rate})")

    if quota is not None:
        quota_state = quota_init_state(quota, synth)

    swaps = 0
    while rate > target_rate and swaps < max_swaps:
        pool = model.sample(pool_size).reset_index(drop=True)

        # 1) AIG filter
        blocked = aig_block_mask_v2(pool, guards)
        cand1 = pool[~blocked]
        # fallback: if empty, use least-severe in pool
        if cand1.empty:
            pool_sev = aig_surplus_v2(pool, guards)
            best_idx = int(np.argmin(pool_sev))
            cand = pool.iloc[[best_idx]]
        else:
            # 2) Teacher gate (if any)
            if teacher is not None:
                pass_mask = teacher_pass_mask(cand1, teacher, target_col)
                cand2 = cand1[pass_mask]
                if cand2.empty:
                    # fallback: pick best teacher score from cand1
                    sc = teacher_score(cand1, teacher, target_col)
                    best_idx = int(np.argmax(sc))
                    cand = cand1.iloc[[best_idx]]
                else:
                    cand = cand2.copy()
            else:
                cand = cand1.copy()

            # 3) Among survivors, use quota improvement (if enabled)
            if quota is not None and len(cand) > 1:
                # choose worst current index to replace
                w = int(np.argmax(sev))
                deltas = quota_delta_replace(quota, quota_state, synth.iloc[[w]], cand)
                # pick candidate that most decreases quota cost (most negative delta)
                best_idx = int(np.argmin(deltas))
                cand = cand.iloc[[best_idx]]

        # Replace worst current row
        w = int(np.argmax(sev))
        # update quota state if used
        if quota is not None:
            quota_apply_replace(quota, quota_state, synth.iloc[[w]], cand)
        # apply replacement
        synth.iloc[w] = cand.iloc[0]
        # update severity & rate
        sev[w] = aig_surplus_v2(cand, guards)[0]
        rate = float((sev > 0).mean())
        swaps += 1
        if verbose and swaps % 200 == 0:
            print(f"[{swaps} swaps] AIG-flag rate = {rate:.4f}")

    if verbose:
        print(f"[done] AIG-flag rate = {rate:.4f} after {swaps} swaps")
    return synth.reset_index(drop=True)

