# ---------------------------------------------  preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def make_matrix(
        df              : pd.DataFrame,
        num_cols        : list[str],
        cat_cols        : list[str],
        embed_map       : dict[str, dict[str, np.ndarray]] | None = None,
        scaler          : MinMaxScaler | None = None
):
    """
    Turns a mixed‑type DataFrame row into one flat *numeric* vector:

    [scaled numerics | concatenated entity embeddings | low‑card codes]

    Returned
    --------
    X            : np.ndarray, shape (n_samples, d)
    num_idx      : np.ndarray[int]   – positions of numeric dims
    cat_idx      : np.ndarray[int]   – positions of low‑card categorical dims
    ranges       : np.ndarray[float] – per‑dimension range for numeric+embed
    """
    df = df.copy()

    # ---------------------------------------------------- 1. numeric block
    X_num = df[num_cols].astype(float).values
    if scaler is None:
        scaler = MinMaxScaler().fit(X_num)
    X_num = scaler.transform(X_num)

    # ---------------------------------------------------- 2. entity‑embedding block
    embed_cols   = sorted(embed_map.keys()) if embed_map else []
    embed_parts  = []
    for c in embed_cols:
        # map() preserves order; fall back to zeros for unseen categories
        embed_vectors = df[c].map(embed_map[c]).apply(
            lambda vec: vec if vec is not None else np.zeros_like(next(iter(embed_map[c].values())))
        )
        embed_parts.append(np.vstack(embed_vectors.values))

    X_embed = np.hstack(embed_parts) if embed_parts else np.empty((len(df), 0))

    # ---------------------------------------------------- 3. low‑card categoricals (as int codes)
    low_card_cols = [c for c in cat_cols if c not in embed_cols]
    X_cat = (df[low_card_cols]
             .apply(lambda s: s.astype('category').cat.codes)
             .values.astype(np.float32))

    # ---------------------------------------------------- 4. assemble & meta
    X = np.hstack([X_num, X_embed, X_cat])

    num_idx = np.arange(X_num.shape[1] + X_embed.shape[1])
    cat_idx = np.arange(num_idx[-1] + 1, X.shape[1]) if X_cat.size else np.array([], dtype=int)

    # Per‑dimension range (needed by HEOM); zero range → constant column
    ranges = X[:, :len(num_idx)].ptp(axis=0)             # numeric + embed only
    return X, num_idx, cat_idx, ranges, scaler
