import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------ 1. fit
def fit_heom_encoder(real_df, num_cols, cat_cols):
    """Fit numeric scaler + category→column mapping on REAL data."""
    scaler = MinMaxScaler().fit(real_df[num_cols].to_numpy(np.float32))
    cat_maps, offset = {}, len(num_cols)
    scale = 1.0 / np.sqrt(2.0)          # makes ‖one‑hot diff‖ = 1
    for c in cat_cols:
        uniq = pd.factorize(real_df[c])[1]
        cat_maps[c] = {v: (offset+i, scale) for i, v in enumerate(uniq)}
        offset += len(uniq)
    return scaler, cat_maps, offset     # offset = final vector dim

# ------------------------------------------------------------------ 2. transform
def heom_to_vec(df, scaler, cat_maps, num_cols, cat_cols, dim):
    """Vectorise *any* DF with fixed encoder -> float32 matrix (n, dim)."""
    n = len(df)
    X = np.zeros((n, dim), dtype=np.float32)

    # numeric
    X[:, :len(num_cols)] = scaler.transform(df[num_cols].to_numpy(np.float32))

    # categorical 1‑hot scaled
    for c in cat_cols:
        mapping = cat_maps[c]
        codes   = (df[c].astype(object).map(mapping))    # dict or NaN
        mask    = codes.notna()
        if mask.any():
            rows  = np.nonzero(mask.to_numpy())[0]
            cols  = np.fromiter((mapping[v][0] for v in df[c][mask]), int)
            scale = np.fromiter((mapping[v][1] for v in df[c][mask]), float)
            X[rows, cols] = scale
    return X
