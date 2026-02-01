# --------------- Quota matching helpers (1D marginals) -----------------

from typing import List
import numpy as np
import pandas as pd


def _bin_edges_quantiles(x: np.ndarray, num_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) <= 2:  # degenerate -> single bin
        edges = np.array([np.min(x), np.max(x)], dtype=float)
    return edges

def fit_quota_1d_spec(
    real_df: pd.DataFrame,
    features: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    n_samples: int,
    *,
    num_bins: int = 10
) -> dict:
    """
    Build a quota spec for 1D marginals: per-feature bin definitions and target counts.
    For numeric: quantile bins; for categorical: category bins.
    """
    spec = dict()
    for f in features:
        if f in cat_cols or not pd.api.types.is_numeric_dtype(real_df[f]):
            cats = real_df[f].astype('category').cat.categories.tolist()
            target_probs = real_df[f].astype('category').value_counts(normalize=True).reindex(cats).to_numpy()
            target_counts = (target_probs * n_samples).astype(float)
            spec[f] = dict(kind='cat', bins=cats, target_counts=target_counts)
        else:
            x = real_df[f].astype(float).to_numpy()
            edges = _bin_edges_quantiles(x, num_bins=num_bins)
            # bin indices for real to compute target probs
            idx = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges)-2)
            nb = len(edges) - 1
            counts = np.bincount(idx, minlength=nb).astype(float)
            target_counts = (counts / counts.sum()) * n_samples
            spec[f] = dict(kind='num', edges=edges, target_counts=target_counts)
    return spec

def _bin_index_for_value(spec_f: dict, val) -> int:
    if spec_f['kind'] == 'cat':
        cats = spec_f['bins']
        try:
            return cats.index(val)
        except ValueError:
            return -1  # unseen -> ignore in quota
    else:
        edges = spec_f['edges']
        v = float(val)
        j = np.searchsorted(edges, v, side='right') - 1
        return int(np.clip(j, 0, len(edges)-2))

def quota_init_state(spec: dict, df: pd.DataFrame) -> dict:
    """Current synthetic counts per feature/bin."""
    state = {}
    for f, sf in spec.items():
        m = len(sf['target_counts'])
        cnt = np.zeros(m, dtype=float)
        for v in df[f].to_numpy():
            j = _bin_index_for_value(sf, v)
            if j >= 0: cnt[j] += 1.0
        state[f] = cnt
    return state

def quota_cost(spec: dict, state: dict) -> float:
    """L1 discrepancy to targets summed over features."""
    cost = 0.0
    for f, sf in spec.items():
        tgt = sf['target_counts']; cur = state[f]
        cost += np.abs(cur - tgt).sum()
    return float(cost)

def quota_delta_replace(spec: dict, state: dict, row_out: pd.DataFrame, cand_pool: pd.DataFrame) -> np.ndarray:
    """
    For replacing row_out with each candidate in cand_pool, compute cost_after - cost_before.
    Negative delta is good (reduces discrepancy).
    """
    # precompute contrib of row_out
    out_bins = {}
    for f, sf in spec.items():
        j_out = _bin_index_for_value(sf, row_out.iloc[0][f])
        out_bins[f] = j_out

    base_cost = quota_cost(spec, state)
    deltas = np.zeros(len(cand_pool), dtype=float)

    for i, (_, row) in enumerate(cand_pool.iterrows()):
        # tentative update: state' = state - e_out + e_cand
        cost = 0.0
        for f, sf in spec.items():
            tgt = sf['target_counts']; cur = state[f].copy()
            j_out = out_bins[f]
            if j_out >= 0: cur[j_out] -= 1.0
            j_in  = _bin_index_for_value(sf, row[f])
            if j_in >= 0:  cur[j_in]  += 1.0
            cost += np.abs(cur - tgt).sum()
        deltas[i] = cost - base_cost
    return deltas

def quota_apply_replace(spec: dict, state: dict, row_out: pd.DataFrame, row_in: pd.DataFrame) -> None:
    """Mutate state in-place to reflect replacing row_out by row_in."""
    for f, sf in spec.items():
        j_out = _bin_index_for_value(sf, row_out.iloc[0][f])
        if j_out >= 0: state[f][j_out] -= 1.0
        j_in  = _bin_index_for_value(sf, row_in.iloc[0][f])
        if j_in >= 0:  state[f][j_in]  += 1.0
