from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.metrics import log_loss
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier, MLPRegressor

import heapq


# -----------------------------
# Utilities
# -----------------------------

def _safe_softmax_probs(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip to avoid log(0) in KL/logloss."""
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, eps, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba

def _kl_rows(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise KL(p || q). p, q: (n, K) probability arrays."""
    p = _safe_softmax_probs(p, eps=eps)
    q = _safe_softmax_probs(q, eps=eps)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)

def _prior_probs_from_labels(y: np.ndarray, classes: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Laplace-smoothed prior p0(a) estimated from labels."""
    counts = np.zeros(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        counts[i] = np.sum(y == c)
    probs = (counts + alpha) / (counts.sum() + alpha * len(classes))
    return probs

def _compute_quantile_bin_edges(values: pd.Series, n_bins: int) -> Optional[np.ndarray]:
    """Compute monotonic quantile edges for numeric-sensitive binning."""
    if n_bins < 2:
        raise ValueError(f"`sensitive_n_bins` must be >= 2, got {n_bins}.")
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    edges = np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return None
    # Expand boundary bins slightly to robustly include clipped edge values.
    edges = edges.astype(np.float64)
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges

def _encode_sensitive_labels(values: pd.Series, *, bin_edges: Optional[np.ndarray]) -> np.ndarray:
    """Optionally bin numeric-sensitive values; otherwise return raw labels."""
    if bin_edges is None:
        return values.to_numpy()

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    fill = float(np.median(finite)) if finite.size > 0 else float(bin_edges[0])
    arr = np.where(np.isfinite(arr), arr, fill)
    arr = np.clip(arr, bin_edges[0], bin_edges[-1])
    binned = pd.cut(arr, bins=bin_edges, include_lowest=True, labels=False)
    return np.asarray(binned, dtype=np.int64)

def _log_loss_with_class_support(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    classes: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Log-loss that remains defined even if y_true has classes unseen during training.
    Unseen classes are assigned a tiny mass (`eps`) and probabilities are renormalized.
    """
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    p_pred = _safe_softmax_probs(p_pred, eps=eps)

    unseen = np.setdiff1d(np.unique(y_true), classes)
    if unseen.size == 0:
        return float(log_loss(y_true, p_pred, labels=classes))

    all_classes = np.concatenate([classes, unseen])
    p_ext = np.full((len(y_true), len(all_classes)), eps, dtype=np.float64)
    p_ext[:, : len(classes)] = p_pred
    p_ext = _safe_softmax_probs(p_ext, eps=eps)

    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    y_idx = np.fromiter((class_to_idx[v] for v in y_true), dtype=np.int64, count=len(y_true))
    chosen = np.clip(p_ext[np.arange(len(y_true)), y_idx], eps, 1.0)
    return float(-np.mean(np.log(chosen)))

def _make_preprocessor(num_cols: Sequence[str], cat_cols: Sequence[str]) -> ColumnTransformer:
    """
    Sparse-friendly preprocessing:
    - numeric: StandardScaler(with_mean=False) to keep sparse output feasible
    - categorical: OneHotEncoder(handle_unknown='ignore')
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), list(num_cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

@dataclass
class AttackModels:
    task_type: str                     # "classification" or "regression"

    # Full attacker(s)
    f_rel: object                      # predicts A from X
    f_aux: Optional[object]            # predicts A from X+Z (if Z provided)

    # Baselines
    prior: Optional[np.ndarray]        # shape (K,) for classification
    prior_mean: Optional[float]        # scalar for regression
    g_z: Optional[object]              # predicts A from Z only (if Z provided)

    # Preprocessors
    pre_X: ColumnTransformer
    pre_Z: Optional[ColumnTransformer]
    pre_XZ: Optional[ColumnTransformer]

    # Class support
    classes_: Optional[np.ndarray]     # shape (K,) for classification

    # Regression metadata
    regression_target_transform: str = "none"


def _fit_logreg_classifier(X, y, *, max_iter: int = 2000, C: float = 1.0, n_jobs: int = -1):
    # saga handles sparse; multinomial works for multi-class.
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        n_jobs=n_jobs,
        multi_class="auto",
    )
    clf.fit(X, y)
    return clf

def _fit_stronger_attacker(X_sparse, y, *, random_state: int = 0):
    """
    A stronger-but-still-practical option without external libs:
    - Reduce sparse one-hot to dense embedding with TruncatedSVD
    - Train an MLP on the embedding
    This is not the strongest possible, but is meaningfully stronger than linear in many tabular settings.
    """
    n_rows, n_features = X_sparse.shape
    max_components = min(n_rows - 1, n_features - 1, 256)
    if max_components < 2:
        # Not enough rank to build a meaningful SVD embedding; use linear fallback.
        return _fit_logreg_classifier(X_sparse, y)

    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    X_emb = svd.fit_transform(X_sparse)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        max_iter=50,
        # sklearn can fail in early-stopping validation scoring with string labels.
        early_stopping=False,
        random_state=random_state,
    )
    mlp.fit(X_emb, y)

    # Wrap into a tiny object with predict_proba
    class _SVDMLP:
        def __init__(self, svd, mlp):
            self.svd = svd
            self.mlp = mlp
        def predict_proba(self, X):
            return self.mlp.predict_proba(self.svd.transform(X))

    return _SVDMLP(svd, mlp)

def _fit_linear_regressor(X, y, *, kind: str = "sgd", random_state: int = 0):
    if kind == "ridge":
        reg = Ridge(alpha=1.0)
    elif kind == "sgd":
        reg = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown linear regressor kind: {kind}")
    reg.fit(X, y)
    return reg

def _fit_stronger_regressor(
    X_sparse,
    y,
    *,
    random_state: int = 0,
    linear_regressor_kind: str = "sgd",
):
    """
    Stronger regressor without external libs:
    - TruncatedSVD embedding
    - MLP regressor on the embedding
    """
    n_rows, n_features = X_sparse.shape
    max_components = min(n_rows - 1, n_features - 1, 256)
    if max_components < 2:
        return _fit_linear_regressor(
            X_sparse,
            y,
            kind=linear_regressor_kind,
            random_state=random_state,
        )

    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    X_emb = svd.fit_transform(X_sparse)

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        max_iter=200,
        early_stopping=True,
        random_state=random_state,
    )
    mlp.fit(X_emb, y)

    class _SVDMLPReg:
        def __init__(self, svd, mlp):
            self.svd = svd
            self.mlp = mlp
        def predict(self, X):
            return self.mlp.predict(self.svd.transform(X))

    return _SVDMLPReg(svd, mlp)

def _encode_sensitive_numeric(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    fill = float(np.median(finite)) if finite.size > 0 else 0.0
    arr = np.where(np.isfinite(arr), arr, fill)
    return arr.astype(np.float64)

def _transform_sensitive_regression(y: np.ndarray, transform: str) -> np.ndarray:
    """
    Optional transform for heavy-tailed nonnegative sensitive attributes.
    """
    y = np.asarray(y, dtype=np.float64)
    if transform == "none":
        return y
    if transform == "log1p_clip0":
        return np.log1p(np.clip(y, 0.0, None))
    raise ValueError(f"Unknown regression target transform: {transform}")

def _resolve_regression_target_transform(transform: str, sensitive_col: str) -> str:
    """
    Resolve configured transform mode into a concrete transform.
    """
    if transform == "auto":
        # Adult's capital-gain is heavy tailed with many zeros.
        return "log1p_clip0" if sensitive_col == "capital-gain" else "none"
    if transform not in {"none", "log1p_clip0"}:
        raise ValueError(
            "Invalid `regression_target_transform`. Expected one of: "
            "{'auto', 'none', 'log1p_clip0'}."
        )
    return transform


def _train_attack_models_on_S(
    S: pd.DataFrame,
    sensitive_col: str,
    X_cols: Sequence[str],
    *,
    Z_cols: Optional[Sequence[str]],
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    attacker_kind: str,
    use_regression: bool = False,
    regression_target_transform: str = "none",
    linear_regressor_kind: str = "sgd",
    bin_edges: Optional[np.ndarray] = None,
    random_state: int = 0,
) -> AttackModels:
    """
    Train attacker models using ONLY S (threat-model consistent).
    attacker_kind: 'logreg' or 'mlp_svd' (mapped to linear/strong regressor in regression mode).
    """
    if use_regression:
        y_raw = _encode_sensitive_numeric(S[sensitive_col])
        y = _transform_sensitive_regression(y_raw, regression_target_transform)
        classes = None
        prior = None
        prior_mean = float(np.mean(y))
    else:
        y = _encode_sensitive_labels(S[sensitive_col], bin_edges=bin_edges)
        classes = np.unique(y)
        prior = _prior_probs_from_labels(y, classes, alpha=1.0)
        prior_mean = None

    # --- build preprocessors for feature sets ---
    def split_num_cat(cols: Sequence[str]) -> Tuple[List[str], List[str]]:
        cols = list(cols)
        n = [c for c in cols if c in num_cols]
        k = [c for c in cols if c in cat_cols]
        return n, k

    # X
    X_num, X_cat = split_num_cat(X_cols)
    pre_X = _make_preprocessor(X_num, X_cat).fit(S[list(X_cols)])
    X_X = pre_X.transform(S[list(X_cols)])

    # Release-only attacker f_rel
    if use_regression:
        if attacker_kind == "logreg":
            f_rel = _fit_linear_regressor(
                X_X,
                y,
                kind=linear_regressor_kind,
                random_state=random_state,
            )
        elif attacker_kind == "mlp_svd":
            f_rel = _fit_stronger_regressor(
                X_X,
                y,
                random_state=random_state,
                linear_regressor_kind=linear_regressor_kind,
            )
        else:
            raise ValueError(f"Unknown attacker_kind={attacker_kind}")
    else:
        if attacker_kind == "logreg":
            f_rel = _fit_logreg_classifier(X_X, y)
        elif attacker_kind == "mlp_svd":
            f_rel = _fit_stronger_attacker(X_X, y, random_state=random_state)
        else:
            raise ValueError(f"Unknown attacker_kind={attacker_kind}")

    # Auxiliary models if Z provided
    pre_Z = None
    pre_XZ = None
    g_z = None
    f_aux = None
    if Z_cols is not None and len(Z_cols) > 0:
        Z_num, Z_cat = split_num_cat(Z_cols)
        pre_Z = _make_preprocessor(Z_num, Z_cat).fit(S[list(Z_cols)])
        X_Z = pre_Z.transform(S[list(Z_cols)])

        XZ_cols = list(X_cols) + list(Z_cols)
        XZ_num, XZ_cat = split_num_cat(XZ_cols)
        pre_XZ = _make_preprocessor(XZ_num, XZ_cat).fit(S[XZ_cols])
        X_XZ = pre_XZ.transform(S[XZ_cols])

        if use_regression:
            if attacker_kind == "logreg":
                g_z = _fit_linear_regressor(
                    X_Z,
                    y,
                    kind=linear_regressor_kind,
                    random_state=random_state,
                )
                f_aux = _fit_linear_regressor(
                    X_XZ,
                    y,
                    kind=linear_regressor_kind,
                    random_state=random_state,
                )
            else:
                g_z = _fit_stronger_regressor(
                    X_Z,
                    y,
                    random_state=random_state,
                    linear_regressor_kind=linear_regressor_kind,
                )
                f_aux = _fit_stronger_regressor(
                    X_XZ,
                    y,
                    random_state=random_state,
                    linear_regressor_kind=linear_regressor_kind,
                )
        else:
            if attacker_kind == "logreg":
                g_z = _fit_logreg_classifier(X_Z, y)
                f_aux = _fit_logreg_classifier(X_XZ, y)
            else:
                g_z = _fit_stronger_attacker(X_Z, y, random_state=random_state)
                f_aux = _fit_stronger_attacker(X_XZ, y, random_state=random_state)

    return AttackModels(
        task_type="regression" if use_regression else "classification",
        f_rel=f_rel,
        f_aux=f_aux,
        prior=prior,
        prior_mean=prior_mean,
        g_z=g_z,
        pre_X=pre_X,
        pre_Z=pre_Z,
        pre_XZ=pre_XZ,
        classes_=classes,
        regression_target_transform=regression_target_transform if use_regression else "none",
    )


def _risk_logloss_gain(
    models: AttackModels,
    real_val: pd.DataFrame,
    sensitive_col: str,
    X_cols: Sequence[str],
    *,
    bin_edges: Optional[np.ndarray] = None,
    Z_cols: Optional[Sequence[str]],
    regression_normalize: bool = True,
    regression_var_eps: float = 1e-12,
) -> float:
    """
    Compute R_AI(S) on a REAL validation set:
    - classification: max incremental log-loss gain
    - regression: max conditional-mean shift
      max( E[(mu_X - mu0)^2], E[(mu_XZ - mu_Z)^2] )
      optionally normalized by Var(A_val) for stable tau semantics.
    """
    if models.task_type == "regression":
        y_val_raw = _encode_sensitive_numeric(real_val[sensitive_col])
        y_val = _transform_sensitive_regression(
            y_val_raw,
            models.regression_target_transform,
        )

        X_val = models.pre_X.transform(real_val[list(X_cols)])
        y_full = np.asarray(models.f_rel.predict(X_val), dtype=np.float64)
        y_prior = float(models.prior_mean)
        r_rel = float(np.mean((y_full - y_prior) ** 2))

        r_aux = -np.inf
        if Z_cols is not None and len(Z_cols) > 0 and models.f_aux is not None and models.g_z is not None:
            Z_val = models.pre_Z.transform(real_val[list(Z_cols)])
            XZ_val = models.pre_XZ.transform(real_val[list(X_cols) + list(Z_cols)])

            y_base = np.asarray(models.g_z.predict(Z_val), dtype=np.float64)
            y_aux = np.asarray(models.f_aux.predict(XZ_val), dtype=np.float64)
            r_aux = float(np.mean((y_aux - y_base) ** 2))

        risk = float(max(r_rel, r_aux))
        if regression_normalize:
            var_y = float(np.var(y_val))
            risk = 0.0 if var_y <= regression_var_eps else risk / var_y
        return risk

    y_val = _encode_sensitive_labels(real_val[sensitive_col], bin_edges=bin_edges)
    classes = models.classes_

    # --- release-only: prior vs f_rel(X) ---
    X_val = models.pre_X.transform(real_val[list(X_cols)])
    p_full = models.f_rel.predict_proba(X_val)
    p_full = _safe_softmax_probs(p_full)
    p_prior = np.tile(models.prior.reshape(1, -1), (len(real_val), 1))

    L_prior = _log_loss_with_class_support(y_val, p_prior, classes=classes)
    L_full = _log_loss_with_class_support(y_val, p_full, classes=classes)
    r_rel = L_prior - L_full

    r_aux = -np.inf
    if Z_cols is not None and len(Z_cols) > 0 and models.f_aux is not None and models.g_z is not None:
        Z_val = models.pre_Z.transform(real_val[list(Z_cols)])
        XZ_val = models.pre_XZ.transform(real_val[list(X_cols) + list(Z_cols)])

        p_base = _safe_softmax_probs(models.g_z.predict_proba(Z_val))
        p_aux = _safe_softmax_probs(models.f_aux.predict_proba(XZ_val))

        L_base = _log_loss_with_class_support(y_val, p_base, classes=classes)
        L_aux = _log_loss_with_class_support(y_val, p_aux, classes=classes)
        r_aux = L_base - L_aux

    return float(max(r_rel, r_aux))


def _record_scores(
    models: AttackModels,
    S: pd.DataFrame,
    sensitive_col: str,
    X_cols: Sequence[str],
    *,
    Z_cols: Optional[Sequence[str]],
) -> np.ndarray:
    """
    Per-record leakage score.
    - classification: max(KL(full || baseline))
    - regression: max conditional-mean shift squared
      max( (mu_X - mu0)^2, (mu_XZ - mu_Z)^2 )
    """
    n = len(S)

    if models.task_type == "regression":
        X = models.pre_X.transform(S[list(X_cols)])
        y_rel = np.asarray(models.f_rel.predict(X), dtype=np.float64)
        y_prior = float(models.prior_mean)
        s_rel = (y_rel - y_prior) ** 2

        if Z_cols is None or len(Z_cols) == 0 or models.f_aux is None or models.g_z is None:
            return s_rel.astype(np.float64)

        Z = models.pre_Z.transform(S[list(Z_cols)])
        XZ = models.pre_XZ.transform(S[list(X_cols) + list(Z_cols)])
        y_base = np.asarray(models.g_z.predict(Z), dtype=np.float64)
        y_aux = np.asarray(models.f_aux.predict(XZ), dtype=np.float64)
        s_aux = (y_aux - y_base) ** 2
        return np.maximum(s_rel, s_aux).astype(np.float64)

    # release-only KL(f_rel(.|x) || prior)
    X = models.pre_X.transform(S[list(X_cols)])
    p_rel = _safe_softmax_probs(models.f_rel.predict_proba(X))
    q_prior = np.tile(models.prior.reshape(1, -1), (n, 1))
    s_rel = _kl_rows(p_rel, q_prior)

    if Z_cols is None or len(Z_cols) == 0 or models.f_aux is None or models.g_z is None:
        return s_rel.astype(np.float64)

    Z = models.pre_Z.transform(S[list(Z_cols)])
    XZ = models.pre_XZ.transform(S[list(X_cols) + list(Z_cols)])

    p_aux = _safe_softmax_probs(models.f_aux.predict_proba(XZ))
    q_z = _safe_softmax_probs(models.g_z.predict_proba(Z))
    s_aux = _kl_rows(p_aux, q_z)

    return np.maximum(s_rel, s_aux).astype(np.float64)


def sampling_reject_attribute_inference(
    model,
    real_val_df: pd.DataFrame,
    *,
    tau_ai: float,
    sensitive_col: str,
    X_cols: Sequence[str],
    Z_cols: Optional[Sequence[str]],
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    n_samples: int,
    sensitive_n_bins: Optional[int] = None,
    sensitive_bin_edges: Optional[Sequence[float]] = None,
    sensitive_mode: str = "auto",
    regression_unique_threshold: int = 50,
    regression_target_transform: str = "auto",
    linear_regressor_kind: str = "sgd",
    attacker_families: Sequence[str] = ("logreg", "mlp_svd"),
    retrain_every: int = 500,
    max_swaps: int = 20000,
    regression_normalize: bool = True,
    random_state: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Attack-aware rejection-with-replacement targeting attribute inference risk.

    Risk metric (measured on REAL validation set):
      R_AI(S) = max over attacker families of
          max( logloss(prior) - logloss(f_rel(X)),
               logloss(g(Z)) - logloss(f_aux(X,Z)) )

    In regression mode (numeric-sensitive):
      R_AI(S) = max( E[(mu_X - mu0)^2], E[(mu_XZ - mu_Z)^2] ),
      optionally normalized by Var(A_val).

    Per-record score:
      score(s) = max( KL(f_rel(.|x) || prior), KL(f_aux(.|x,z) || g(.|z)) )
      or, in regression mode:
      score(s) = max( (mu_X(x)-mu0)^2, (mu_XZ(x,z)-mu_Z(z))^2 )

    Returns a synthetic DataFrame with measured AIA risk < tau_ai if feasible within max_swaps.
    """
    if sensitive_mode not in {"auto", "classification", "regression"}:
        raise ValueError(
            "Invalid `sensitive_mode`. Expected one of: "
            "{'auto', 'classification', 'regression'}."
        )

    is_numeric = pd.api.types.is_numeric_dtype(real_val_df[sensitive_col])
    n_unique = int(real_val_df[sensitive_col].nunique(dropna=True))
    if sensitive_mode == "regression":
        if not is_numeric:
            raise ValueError(
                "`sensitive_mode='regression'` requires a numeric sensitive column."
            )
        use_regression = True
    elif sensitive_mode == "classification":
        use_regression = False
    else:
        use_regression = bool(is_numeric and n_unique > int(regression_unique_threshold))

    resolved_regression_transform = "none"
    bin_edges: Optional[np.ndarray] = None
    if use_regression:
        resolved_regression_transform = _resolve_regression_target_transform(
            regression_target_transform,
            sensitive_col,
        )
        if verbose:
            mode = "normalized" if regression_normalize else "unnormalized"
            print(
                f"[aia] using regression objective for numeric sensitive `{sensitive_col}` "
                f"({mode}, transform={resolved_regression_transform})"
            )
        if verbose and (sensitive_n_bins is not None or sensitive_bin_edges is not None):
            print("[aia] note: `sensitive_n_bins`/`sensitive_bin_edges` are ignored in regression mode")
    else:
        if verbose and sensitive_mode == "auto" and is_numeric:
            print(
                f"[aia] auto-selected classification mode for numeric sensitive `{sensitive_col}` "
                f"(unique={n_unique} <= threshold={regression_unique_threshold})"
            )
        if sensitive_bin_edges is not None:
            edges = np.asarray(sensitive_bin_edges, dtype=np.float64)
            edges = np.unique(edges)
            if edges.size < 3:
                raise ValueError("`sensitive_bin_edges` must contain at least 3 unique values.")
            edges[0] = np.nextafter(edges[0], -np.inf)
            edges[-1] = np.nextafter(edges[-1], np.inf)
            bin_edges = edges
        elif sensitive_n_bins is not None and int(sensitive_n_bins) > 1:
            bin_edges = _compute_quantile_bin_edges(real_val_df[sensitive_col], int(sensitive_n_bins))

        if verbose and bin_edges is not None:
            print(
                f"[aia] using binned sensitive labels for `{sensitive_col}` "
                f"with {len(bin_edges) - 1} bins"
            )

    # ---- initialize synthetic dataset ----
    S = model.sample(n_samples).reset_index(drop=True)

    # ---- train attackers, pick worst (max risk) ----
    def fit_all_attackers(S_cur: pd.DataFrame):
        fitted: Dict[str, AttackModels] = {}
        risks: Dict[str, float] = {}
        for kind in attacker_families:
            am = _train_attack_models_on_S(
                S_cur,
                sensitive_col=sensitive_col,
                X_cols=X_cols,
                Z_cols=Z_cols,
                num_cols=num_cols,
                cat_cols=cat_cols,
                attacker_kind=kind,
                use_regression=use_regression,
                regression_target_transform=resolved_regression_transform,
                linear_regressor_kind=linear_regressor_kind,
                bin_edges=bin_edges,
                random_state=random_state,
            )
            fitted[kind] = am
            risks[kind] = _risk_logloss_gain(
                am,
                real_val_df,
                sensitive_col=sensitive_col,
                X_cols=X_cols,
                bin_edges=bin_edges,
                Z_cols=Z_cols,
                regression_normalize=regression_normalize,
            )
        worst_kind = max(risks, key=risks.get)
        return worst_kind, fitted[worst_kind], risks

    worst_kind, am, risks = fit_all_attackers(S)
    R = risks[worst_kind]
    if verbose:
        print(f"[init] worst attacker={worst_kind}  R_AI={R:.6f}  target<{tau_ai}")

    # ---- score all records and build a max-heap ----
    scores = _record_scores(am, S, sensitive_col=sensitive_col, X_cols=X_cols, Z_cols=Z_cols)
    version = np.zeros(len(S), dtype=np.int64)
    heap: List[Tuple[float, int, int]] = []  # (-score, idx, version)

    for i, sc in enumerate(scores):
        heap.append((-float(sc), i, int(version[i])))
    heapq.heapify(heap)

    def pop_worst() -> int:
        while True:
            neg_sc, i, v = heap[0]
            if v == version[i]:
                return i
            heapq.heappop(heap)

    def push_idx(i: int):
        heapq.heappush(heap, (-float(scores[i]), i, int(version[i])))

    # ---- helper: score a single candidate row ----
    def score_candidate(row: pd.DataFrame) -> float:
        if am.task_type == "regression":
            X1 = am.pre_X.transform(row[list(X_cols)])
            y_rel = float(np.asarray(am.f_rel.predict(X1), dtype=np.float64)[0])
            y_prior = float(am.prior_mean)
            s1 = (y_rel - y_prior) ** 2

            if Z_cols is None or len(Z_cols) == 0 or am.f_aux is None or am.g_z is None:
                return s1

            Z1 = am.pre_Z.transform(row[list(Z_cols)])
            XZ1 = am.pre_XZ.transform(row[list(X_cols) + list(Z_cols)])
            y_base = float(np.asarray(am.g_z.predict(Z1), dtype=np.float64)[0])
            y_aux = float(np.asarray(am.f_aux.predict(XZ1), dtype=np.float64)[0])
            s2 = (y_aux - y_base) ** 2
            return max(s1, s2)

        # classification mode
        X1 = am.pre_X.transform(row[list(X_cols)])
        p1 = _safe_softmax_probs(am.f_rel.predict_proba(X1))
        q1 = am.prior.reshape(1, -1)
        s1 = float(_kl_rows(p1, q1)[0])

        if Z_cols is None or len(Z_cols) == 0 or am.f_aux is None or am.g_z is None:
            return s1

        Z1 = am.pre_Z.transform(row[list(Z_cols)])
        XZ1 = am.pre_XZ.transform(row[list(X_cols) + list(Z_cols)])
        p2 = _safe_softmax_probs(am.f_aux.predict_proba(XZ1))
        q2 = _safe_softmax_probs(am.g_z.predict_proba(Z1))
        s2 = float(_kl_rows(p2, q2)[0])

        return max(s1, s2)

    # ---- main replacement loop ----
    swaps = 0
    accepted = 0

    while R >= tau_ai and swaps < max_swaps:
        j = pop_worst()
        worst_sc = scores[j]

        cand = model.sample(1).reset_index(drop=True)
        cand_sc = score_candidate(cand)

        # Accept if it reduces leakage score for the worst record
        if cand_sc < worst_sc:
            S.iloc[j] = cand.iloc[0]
            scores[j] = cand_sc

            version[j] += 1
            push_idx(j)

            accepted += 1

            # Periodically retrain the attacker (important; scores drift otherwise)
            if accepted % retrain_every == 0:
                worst_kind, am, risks = fit_all_attackers(S)
                R = risks[worst_kind]
                scores = _record_scores(am, S, sensitive_col=sensitive_col, X_cols=X_cols, Z_cols=Z_cols)

                # rebuild heap
                version[:] = 0
                heap = [(-float(scores[i]), i, 0) for i in range(len(S))]
                heapq.heapify(heap)

                if verbose:
                    print(f"[swap {swaps}] retrain: worst={worst_kind}  R_AI={R:.6f}  target<{tau_ai}")

        swaps += 1

    if verbose:
        print(f"[done] swaps={swaps}, accepted={accepted}, final R_AI={R:.6f} (target<{tau_ai})")

    return S.reset_index(drop=True)
