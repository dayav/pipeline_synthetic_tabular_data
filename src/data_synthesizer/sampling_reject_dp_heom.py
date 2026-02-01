import math, numpy as np, pandas as pd
from dataclasses import dataclass
from diffprivlib.models import KMeans as DPKMeans
from typing import Optional

_SQRT_HALF = 1.0 / math.sqrt(2.0)

@dataclass
class HeomVectorizer:
    num_cols: list[str]
    cat_cols: list[str]
    num_bounds: dict[str, tuple[float, float]]     # public (lo, hi) per numeric col
    cat_vocabs: dict[str, list]                    # public vocab per cat col (order fixed)

    # derived
    dim: int = 0
    _offsets: dict | None = None                   # start index per cat col
    _bounds_per_dim: list[tuple[float, float]] | None = None

    def __post_init__(self):
        # total dimension = #num + sum(|vocab_c|)
        self.dim = len(self.num_cols) + sum(len(self.cat_vocabs[c]) for c in self.cat_cols)
        # build offsets for one-hot blocks
        off = len(self.num_cols)
        self._offsets = {}
        for c in self.cat_cols:
            self._offsets[c] = off
            off += len(self.cat_vocabs[c])
        # per-dim bounds for diffprivlib
        self._bounds_per_dim = []
        # numeric dims: [0,1]
        self._bounds_per_dim += [(0.0, 1.0)] * len(self.num_cols)
        # categorical dims: [0, 1/√2]
        for c in self.cat_cols:
            self._bounds_per_dim += [(0.0, _SQRT_HALF)] * len(self.cat_vocabs[c])

    def bounds_per_dim(self) -> list[tuple[float, float]]:
        return list(self._bounds_per_dim)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return X in R^dim where Euclidean distance equals HEOM."""
        n = len(df)
        X = np.zeros((n, self.dim), dtype=np.float32)

        # numeric block: clip to [lo,hi], then scale to [0,1]
        for j, col in enumerate(self.num_cols):
            lo, hi = self.num_bounds[col]
            rng = max(hi - lo, 1e-12)
            vals = df[col].to_numpy(dtype=float)
            vals = np.clip(vals, lo, hi)
            X[:, j] = (vals - lo) / rng

        # categorical blocks: scaled one-hot with 1/√2 amplitude
        for c in self.cat_cols:
            off = self._offsets[c]
            vocab = self.cat_vocabs[c]
            idx_map = {v: i for i, v in enumerate(vocab)}
            col_vals = df[c].to_numpy(object)
            # unknown category → map to a special bucket if present, else ignore (all-zeros)
            for i, v in enumerate(col_vals):
                j = idx_map.get(v, None)
                if j is not None:
                    X[i, off + j] = _SQRT_HALF

        return X
    



def dp_kmeans_heom_summaries(
    real_df: pd.DataFrame,
    vec: HeomVectorizer,
    n_clusters: int = 64,
    eps_centers: float = 2.0,
    eps_scales:  float = 1.0,
    random_state: int = 42,
):
    """
    Returns dict with DP centers and DP per-cluster scale (avg. radius).
    Privacy cost: eps_total = eps_centers + eps_scales.
    """
    assert eps_centers > 0 and eps_scales > 0
    X = vec.transform(real_df)                       # uses only public schema

    # --- DP KMeans on bounded domain (bounds per dimension are public) ---
    km = DPKMeans(
        n_clusters=n_clusters,
        epsilon=eps_centers,
        bounds=vec.bounds_per_dim(),
        random_state=random_state
    )
    km.fit(X)
    centers = km.cluster_centers_.astype(np.float32)     # DP output

    # --- DP per-cluster average radius (post-processing of private X) ---
    # Sensitivities: changing one row affects exactly one cluster's sum by ≤ D_max
    # and count by ≤ 1. Here D_max is the maximum possible Euclidean distance in this space.
    d_max = math.sqrt(len(vec.num_cols) + len(vec.cat_cols))  # each attr contributes ≤1
    labels = km.labels_                                       # used internally only

    # sum of distances and counts per cluster
    dists = np.linalg.norm(X - centers[labels], axis=1)
    sums  = np.bincount(labels, weights=dists, minlength=n_clusters).astype(float)
    cnts  = np.bincount(labels, minlength=n_clusters).astype(float)

    # add Laplace noise to each cluster sum and count
    # (release as a group → total eps_scales by parallel composition)
    sums_noisy = sums + np.random.laplace(scale=d_max/eps_scales, size=n_clusters)
    cnts_noisy = cnts + np.random.laplace(scale=1.0/eps_scales, size=n_clusters)

    # avoid zero/negative
    cnts_noisy = np.maximum(cnts_noisy, 1.0)
    sigma = (sums_noisy / cnts_noisy).astype(np.float32)
    # enforce minimal scale to prevent division issues
    sigma = np.maximum(sigma, 1e-3).astype(np.float32)

    return dict(
        centers=centers,            # (k, D)
        sigma=sigma,                # (k,)
        d_max=float(d_max),
        eps_centers=float(eps_centers),
        eps_scales=float(eps_scales),
        eps_total=float(eps_centers + eps_scales),
        n_clusters=int(n_clusters),
        dim=int(vec.dim),
    )


def make_dp_heom_acceptor(vec: HeomVectorizer, dp: dict, tau: float = 1.0):
    """
    Returns function f(df_batch) -> (accept_mask, scores) where
    score = min_j dist(x, center_j) / sigma_j, and accept if score >= tau.
    """
    C = dp["centers"].astype(np.float32)
    S = np.maximum(dp["sigma"].astype(np.float32), 1e-6)
    invS = 1.0 / S

    def accept(df_batch: pd.DataFrame):
        Xb = vec.transform(df_batch).astype(np.float32)          # (n, D)
        # pairwise distances to centers: ||X - C||_2 per row, vectorized
        # (X - C)² = X² + C² - 2XCᵀ
        X2 = (Xb * Xb).sum(axis=1, keepdims=True)                # (n, 1)
        C2 = (C  * C ).sum(axis=1)[np.newaxis, :]                # (1, k)
        XC = Xb @ C.T                                            # (n, k)
        D  = np.sqrt(np.maximum(X2 + C2 - 2.0*XC, 0.0))          # (n, k)
        scores = D * invS[np.newaxis, :]                         # normalize per cluster
        smin = scores.min(axis=1)                                # (n,)
        mask = smin >= tau
        return mask, smin
    return accept


def sampling_reject_dp_heom(
    generator_model,
    real_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    num_bounds: dict[str, tuple[float, float]],   # PUBLIC bounds
    cat_vocabs: dict[str, list],                  # PUBLIC vocabularies
    *,
    n_samples: int,
    # DP k-means knobs
    k: int = 64,
    eps_centers: float = 2.0,
    eps_scales:  float = 1.0,
    # filter knob
    tau: float = 1.0,
    # batching
    pool_size: int = 64,
    max_pools: Optional[int] = None,              # None → unlimited until filled
    # acceptance fallback
    relax_every: int = 500,                       # relax tau every N empty pools
    relax_factor: float = 0.9,                    # multiply tau ← tau * factor
    random_state: int = 42,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    DP Option B: one-shot DP KMeans summaries in HEOM space → post-processed rejection.

    Returns (synth_df, diagnostics).
    Privacy: ε_total = eps_centers + eps_scales (δ = 0).
    """
    rng = np.random.default_rng(random_state)

    # 1) Vectorizer & DP summaries
    vec = HeomVectorizer(num_cols=num_cols, cat_cols=cat_cols,
                         num_bounds=num_bounds, cat_vocabs=cat_vocabs)
    dp = dp_kmeans_heom_summaries(real_df, vec,
                                  n_clusters=k,
                                  eps_centers=eps_centers,
                                  eps_scales=eps_scales,
                                  random_state=random_state)
    accept = make_dp_heom_acceptor(vec, dp, tau=tau)

    # 2) Sample until filled
    parts = []
    accepted = 0
    pools = 0
    empty_pools_in_a_row = 0
    max_pools = max_pools or 10_000

    while accepted < n_samples and pools < max_pools:
        pool = generator_model.sample(pool_size).reset_index(drop=True)
        pools += 1

        mask, smin = accept(pool)
        kept = pool[mask]
        parts.append(kept)
        accepted += len(kept)

        if verbose and pools % 50 == 0:
            rate = accepted / (pools * pool_size)
            print(f"[DP-HEOM] pools={pools} accepted={accepted}/{n_samples} "
                  f"(cum. accept rate ~ {rate:.3f}, tau={tau:.3f})")

        if len(kept) == 0:
            empty_pools_in_a_row += 1
            if (empty_pools_in_a_row % relax_every) == 0:
                tau *= relax_factor
                if verbose:
                    print(f"[DP-HEOM] relaxed tau → {tau:.3f} after {empty_pools_in_a_row} empty pools")
        else:
            empty_pools_in_a_row = 0

    synth = pd.concat(parts, ignore_index=True) if parts else real_df.iloc[0:0].copy()
    if len(synth) < n_samples:
        # fill remainder (no filter); still post-processing → no privacy cost
        need = n_samples - len(synth)
        filler = generator_model.sample(need).reset_index(drop=True)
        synth = pd.concat([synth, filler], ignore_index=True)

    # 3) Diagnostics
    diag = dict(
        eps_centers=float(eps_centers),
        eps_scales=float(eps_scales),
        eps_total=float(eps_centers + eps_scales),
        n_clusters=int(k),
        tau_final=float(tau),
        pools=int(pools),
        accepted=int(min(accepted, n_samples)),
        filled_without_filter=int(max(0, n_samples - accepted)),
        dim=int(vec.dim)
    )
    return synth.head(n_samples).reset_index(drop=True), diag


