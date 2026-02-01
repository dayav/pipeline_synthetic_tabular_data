# ---------------- sampling_reject_epsilon_tabnet_strong (logged) -------
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import time, logging
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

import faiss


# ---------- guard_debug.py -------------------------------------------------


def evaluate_guards(
    df_batch: pd.DataFrame,
    guards : Dict[str, dict],
    margin_prob: float = 0.0,
    build_table: bool = False,
    max_rows: int = 30,
) -> tuple[np.ndarray, pd.DataFrame | None, dict]:
    """
    Evaluate ALL guards on a batch once.

    Returns
    -------
    risky_union : (N,) bool
        True if row is flagged by ANY sensitive guard.
    table_df : pd.DataFrame | None
        Optional, small table explaining *which* guard triggered and *why*.
        One row per (row_index × sensitive_col) that triggered.
    per_col_counts : dict[str, int]
        Number of blocked rows per sensitive column.

    Expected guard spec per sensitive column `s`:
      • CLF: { 'type': 'clf', 'pipe': Pipeline with predict_proba,
               'X_cols': [...], 'tau_clf': float }
      • REG: { 'type': 'reg',
               'pipe': {'pre': ColumnTransformer, 'low': Regressor, 'high': Regressor},
               'X_cols': [...], 'tau_reg_width': float }
    """
    n = len(df_batch)
    risky_union = np.zeros(n, dtype=bool)# final OR over all sensitive columns
    per_col_counts = {s: 0 for s in guards.keys()}# how many rows blocked per column

    rows: list[dict[str, Any]] = []
    # We collect rows first and truncate to max_rows at the end to favor
    # “most severe” cases (highest conf / lowest width).
    clf_rows, reg_rows = [], []  # optional table rows

    for s, spec in guards.items():  # loop over sensitive columns
        typ    = spec['type']       # 'clf' or 'reg'
        X_cols = spec['X_cols']     # features used to predict S

        if typ == 'clf':
            pipe = spec['pipe']    # sklearn Pipeline with predict_proba
            tau  = spec.get('tau_clf', spec.get('tau', 0.9))  
            proba = pipe.predict_proba(df_batch[X_cols])   # (N, C) class probabilities
            # take top-1 (conf) and top-2 (sec) probabilities to form a margin

            # handle 1-class edge case gracefully
            if proba.shape[1] >= 2:
                idx_sorted = np.argsort(proba, axis=1)
                top1 = idx_sorted[:, -1]
                top2 = idx_sorted[:, -2]
                conf = proba[np.arange(n), top1]   # max class prob per row
                sec  = proba[np.arange(n), top2]   # second-best prob per row
            else:
                # single-prob "class"; treat 2nd-best as 0
                top1 = np.zeros(n, dtype=int)
                conf = proba[:, 0]
                sec  = np.zeros(n)

            gap   = conf - sec # top1 - top2
            # ---- Decision for classification guard ----
            # Flag if:
            #  (i) the model is confident in some class (conf > tau), AND
            # (ii) that class is decisively preferred (gap >= margin_prob).
            risky = (conf > tau) & (gap >= margin_prob)

            risky_union |= risky
            per_col_counts[s] = int(risky.sum())

            if build_table:
                flagged = np.where(risky)[0]
                # sort by severity: higher conf, then larger gap
                order = np.lexsort((-gap[flagged], -conf[flagged]))
                for j in flagged[order]:
                    clf_rows.append(dict(
                        row=int(j), sensitive=s, type='clf',
                        conf=float(conf[j]), gap=float(gap[j]),
                        tau=float(tau), reason='conf>tau & gap>=margin'
                    ))

        elif typ == 'reg':
            # bundle = {'pre': ColumnTransformer, 'low': HGBR, 'high': HGBR}
            bundle = spec['pipe'] 
            pre, low, high = bundle['pre'], bundle['low'], bundle['high']
            tau_w = spec['tau_reg_width']

            Xb  = pre.transform(df_batch[X_cols])
            lo  = low.predict(Xb)                   # q_low(x)
            hi  = high.predict(Xb)                  # q_high(x)
            lo2 = np.minimum(lo, hi)                # (robust) ensure order
            hi2 = np.maximum(lo, hi)
            width = hi2 - lo2                       # interval width

            # ---- Decision for regression guard ----
            # Flag if interval is too narrow → S is highly determined/precise.
            risky = (width <= tau_w)
            risky_union |= risky
            per_col_counts[s] = int(risky.sum())

            if build_table:
                flagged = np.where(risky)[0]
                # sort by severity: narrower first
                order = np.argsort(width[flagged])
                for j in flagged[order]:
                    reg_rows.append(dict(
                        row=int(j), sensitive=s, type='reg',
                        lo=float(lo2[j]), hi=float(hi2[j]),
                        width=float(width[j]), tau_w=float(tau_w),
                        reason='width<=tau_w'
                    ))
        else:
            raise ValueError(f"Unknown guard type for {s}: {typ!r}")

    # Optional, human-readable diagnostic table
    table_df = None
    if build_table:
        # combine and keep only the first max_rows (most severe per type already sorted)
        all_rows = clf_rows + reg_rows
        if all_rows:
            # We prefer to keep the most-severe CLF then most-severe REG up to max_rows
            # (this keeps the table readable and actionable)
            keep = all_rows[:max_rows]
            table_df = pd.DataFrame(keep)

    return risky_union, table_df, per_col_counts


def is_high_confidence_v2(
    df_batch: pd.DataFrame,
    guards: Dict[str, dict],
    margin: float = 0.0
) -> np.ndarray:
    """Compatibility wrapper: just return the union mask."""
    risky_union, _, _ = evaluate_guards(
        df_batch, guards, margin_prob=margin, build_table=False
    )
    return risky_union



# ----------------------------------------------------------------------
# FAISS helpers (unchanged)
def _build_faiss_index(X: np.ndarray, use_gpu: bool = False):
    d = X.shape[1]
    cpu = faiss.IndexFlatL2(d)
    if use_gpu and faiss.get_num_gpus() > 0:
        return faiss.index_cpu_to_all_gpus(cpu)
    return cpu

def _faiss_search(index, Q: np.ndarray, k: int):
    D, I = index.search(Q.astype(np.float32), k)      # returns squared L2
    return D, I


# Optional: per‑candidate explanation (which columns triggered & why)
def _explain_guard_row(
        guards: Dict[str, Tuple[Pipeline, float, List[str]]],
        row_df: pd.DataFrame,
        margin: float = 0.0) -> list[tuple[str, float, float, float, Any]]:
    """
    Returns a list of (sensitive_col, conf, tau, gap, predicted_class)
    for all guards that would flag this single row.
    """
    reasons = []
    for s, (pipe, tau, X_cols) in guards.items():
        proba = pipe.predict_proba(row_df[X_cols])
        p     = proba[0]
        # compute conf & gap
        if p.size >= 2:
            idx_sorted = np.argsort(p)
            conf = float(p[idx_sorted[-1]])
            gap  = float(conf - p[idx_sorted[-2]])
            pred = pipe.classes_[idx_sorted[-1]]
        else:
            conf = float(p.max())
            gap  = conf
            pred = pipe.classes_[np.argmax(p)]
        if (conf > tau) and (gap >= margin):
            reasons.append((s, conf, float(tau), gap, pred))
    return reasons

# ----------------------------------------------------------------------
def sampling_reject_epsilon_tabnet_strong(
    generator_model,
    real_df   : pd.DataFrame,
    min_eps   : float,
    embed_fn,                      # DataFrame -> np.ndarray[n, d]
    n_samples : int,
    use_faiss : bool = True,
    use_faiss_gpu: bool = False,
    guards    = None,              # <- from fit_aia_guard_strong
    guard_margin: float = 0.0,
    pool_size : int = 16,
    random_state: int = 42,
    max_swaps : int = 200_000,
    verbose   : bool = True,
    *,
    apply_guard: bool = True,
    guard_stage: str  = 'prefilter',
    apply_epsilon: bool = True,
    # -------- logging controls ----------
    logger: logging.Logger | None = None,
    log_every: int = 200,
    stats_every: int = 1000,
    trace_guard: bool = False,
    explain_guard: bool = False,
    return_diag: bool = False,
    # -------- guard table controls ----------
    debug_guard_table: bool = False,
    debug_guard_every: int  = 50,
    debug_guard_rows:  int  = 20
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:

    # ----------------- logging plumbing --------------------------------
    start_t = time.perf_counter()
    if logger is None:
        logger = logging.getLogger("epsilon_sampler")
        if not logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
            h.setFormatter(fmt); logger.addHandler(h)
        logger.setLevel(logging.INFO)

    def _pct(x): return f"{100.0 * x:.1f}%"

    diag: dict = dict(
        mode                = None,
        eps_history         = [],
        swaps               = 0,
        elapsed_sec         = 0.0,
        pool_drawn          = 0,
        pool_after_guard    = 0,
        total_guard_blocked = 0,
        guard_block_by_col  = {} if guards is not None else None,
        accept_count        = 0,
        veto_count          = 0,
        final_surplus_stats = {},
        radii_stats         = {},
        # NEW: padding diagnostics
        pad_guard_attempts  = 0,
        pad_guard_kept      = 0,
        pad_noguard_rows    = 0,
        pad_trims           = 0
    )
    if trace_guard and guards is not None:
        diag["guard_block_by_col"] = {s: 0 for s in guards.keys()}

    rng = np.random.default_rng(random_state)

    # ----------------- 0) radii in latent space ------------------------
    Zr = embed_fn(real_df).astype(np.float32)
    if use_faiss:
        index = _build_faiss_index(Zr, use_gpu=use_faiss_gpu); index.add(Zr)
        D_rr, _ = _faiss_search(index, Zr, 2)
        radii = np.sqrt(D_rr[:, 1])
        def qnn(Q):
            D, I = _faiss_search(index, Q, 1)
            return np.sqrt(D[:, 0]), I[:, 0]
    else:
        nn2 = NearestNeighbors(2, metric='euclidean').fit(Zr)
        nn1 = NearestNeighbors(1, metric='euclidean').fit(Zr)
        D_rr, _ = nn2.kneighbors(Zr)
        radii = D_rr[:, 1]
        def qnn(Q):
            D, I = nn1.kneighbors(Q, 1)
            return D[:, 0], I[:, 0]

    diag["radii_stats"] = dict(
        min=float(np.min(radii)),
        p25=float(np.quantile(radii, 0.25)),
        med=float(np.median(radii)),
        p75=float(np.quantile(radii, 0.75)),
        max=float(np.max(radii))
    )
    logger.info(
        f"[init] real={len(real_df)}  latent_dim={Zr.shape[1]}  "
        f"apply_epsilon={'on' if apply_epsilon else 'off'}  "
        f"guard={'on' if (apply_guard and guards is not None) else 'off'} "
        f"({guard_stage}) pool={pool_size} faiss={use_faiss}{'/gpu' if use_faiss_gpu else ''}"
    )
    logger.info(f"[init] radii min/med/max = "
                f"{diag['radii_stats']['min']:.3f} / {diag['radii_stats']['med']:.3f} / {diag['radii_stats']['max']:.3f}")

    # ----------------- helper: compute ε of any df ---------------------
    def epsilon_of(df: pd.DataFrame) -> tuple[float, np.ndarray]:
        Zs = embed_fn(df).astype(np.float32)
        d, i = qnn(Zs)
        surplus = d - radii[i]
        eps = float((surplus < 0).mean())
        return eps, surplus

    # ----------------- NEW: ensure_size helper -------------------------
    def ensure_size(df: pd.DataFrame) -> pd.DataFrame:
        """
        Guarantee the returned DataFrame has exactly n_samples rows.
        - If too many: trim head.
        - If too few: sample extra rows; prefer guard‑safe rows if prefilter is active,
          otherwise just fill; as last resort, fill ignoring the guard.
        """
        cur = len(df)
        if cur == n_samples:
            return df.reset_index(drop=True)
        if cur > n_samples:
            diag["pad_trims"] += (cur - n_samples)
            return df.head(n_samples).reset_index(drop=True)

        # Need to pad
        need = n_samples - cur
        parts = [df]
        max_attempts = max(1000, int(np.ceil(need / max(1, pool_size))) * 10)
        respect_guard = (apply_guard and guard_stage == 'prefilter' and guards is not None)

        # Try with guard first
        attempts = 0
        while need > 0 and attempts < max_attempts:
            attempts += 1
            batch = generator_model.sample(min(pool_size, need)).reset_index(drop=True)
            if respect_guard:
                diag["pad_guard_attempts"] += 1
                risky = is_high_confidence_v2(batch, guards, margin=guard_margin)
                batch = batch[~risky]
            if len(batch) > 0:
                take = min(len(batch), need)
                parts.append(batch.head(take))
                need -= take
                if respect_guard:
                    diag["pad_guard_kept"] += take

        # If still short, fill without guard
        while need > 0:
            batch = generator_model.sample(min(pool_size, need)).reset_index(drop=True)
            take = min(len(batch), need)
            parts.append(batch.head(take))
            diag["pad_noguard_rows"] += take
            need -= take

        out = pd.concat(parts, ignore_index=True)
        # Safety trim (should already be exact)
        if len(out) > n_samples:
            diag["pad_trims"] += (len(out) - n_samples)
            out = out.head(n_samples)
        return out.reset_index(drop=True)

    # ----------------- 1) initial batch --------------------------------
    synth0 = generator_model.sample(n_samples).reset_index(drop=True)

    # ---------- BRANCH A: ε filtering is OFF ---------------------------
    if not apply_epsilon:
        if apply_guard and guards is not None and guard_stage == 'prefilter':
            diag["mode"] = "guard_only_prefilter"
            logger.info("[guard-only] building synthetic dataset using guard prefilter only (ε disabled)")
            parts, kept, draws = [], 0, 0
            max_iter = max(10_000, int(np.ceil(n_samples / max(1, pool_size))) * 10)
            while kept < n_samples and draws < max_iter:
                pool = generator_model.sample(pool_size).reset_index(drop=True)
                draws += 1
                diag["pool_drawn"] += len(pool)

                build_tbl = debug_guard_table and (draws % debug_guard_every == 0)
                risky_union, tbl, per_counts = evaluate_guards(
                    pool, guards, margin_prob=guard_margin,
                    build_table=build_tbl, max_rows=debug_guard_rows
                )
                if build_tbl and tbl is not None:
                    logger.info("[guard-only] sample of blocked rows:\n" +
                                tbl.to_string(index=False, justify='left', max_cols=20))
                if trace_guard:
                    for s, cnt in per_counts.items():
                        diag["guard_block_by_col"][s] += int(cnt)

                safe_mask = ~risky_union
                blocked = int((~safe_mask).sum())
                diag["total_guard_blocked"] += blocked
                safe = pool[safe_mask]
                diag["pool_after_guard"] += len(safe)

                if len(safe):
                    parts.append(safe); kept += len(safe)

                if verbose and draws % 50 == 0:
                    logger.info(f"[guard-only] draws={draws} kept={kept}/{n_samples} "
                                f"(last pool blocked {blocked}/{len(pool)} = {_pct(blocked/len(pool))})")

            synth = (pd.concat(parts, ignore_index=True) if parts else synth0.copy())
            synth = ensure_size(synth)   # <---- guarantee exact size
        else:
            diag["mode"] = "no_epsilon_no_prefilter"
            synth = ensure_size(synth0.copy())  # <---- guarantee exact size
            if apply_guard and guards is not None and guard_stage == 'postfilter':
                logger.info("[note] guard_stage='postfilter' has no effect when apply_epsilon=False; returning initial batch.")

        eps, surplus = epsilon_of(synth)
        diag["eps_history"].append((0, eps))
        diag["final_surplus_stats"] = dict(
            min=float(np.min(surplus)),
            p25=float(np.quantile(surplus, 0.25)),
            med=float(np.median(surplus)),
            p75=float(np.quantile(surplus, 0.75)),
            max=float(np.max(surplus)),
            eps=float(eps)
        )
        diag["elapsed_sec"] = time.perf_counter() - start_t
        logger.info(f"[done] apply_epsilon=False → return with ε={eps:.4f}  "
                    f"filled_guard={diag['pad_guard_kept']}  filled_noguard={diag['pad_noguard_rows']}  "
                    f"elapsed={diag['elapsed_sec']:.1f}s")
        return (synth.reset_index(drop=True), diag) if return_diag else synth.reset_index(drop=True)

    # ---------- BRANCH B: ε filtering is ON ----------------------------
    diag["mode"] = f"epsilon_{'and_guard' if (apply_guard and guards is not None) else 'only'}_{guard_stage}"
    synth = synth0.copy()
    eps, surplus = epsilon_of(synth)
    diag["eps_history"].append((0, eps))
    logger.info(f"[init] ε={eps:.4f}  (target < {min_eps})")

    swaps = 0
    pools = 0
    while eps >= min_eps and swaps < max_swaps:
        pool = generator_model.sample(pool_size).reset_index(drop=True)
        pools += 1
        diag["pool_drawn"] += len(pool)

        if apply_guard and guard_stage == 'prefilter' and guards is not None:
            build_tbl = debug_guard_table and (pools % debug_guard_every == 0)
            risky_union, tbl, per_counts = evaluate_guards(
                pool, guards, margin_prob=guard_margin,
                build_table=build_tbl, max_rows=debug_guard_rows
            )
            if build_tbl and tbl is not None:
                logger.info(f"[pool {pools}] example of blocked rows:\n" +
                            tbl.to_string(index=False, justify='left', max_cols=20))
            if trace_guard:
                for s, cnt in per_counts.items():
                    diag["guard_block_by_col"][s] += int(cnt)

            safe_mask = ~risky_union
            blocked = int((~safe_mask).sum())
            diag["total_guard_blocked"] += blocked

            if verbose and pools % 50 == 0:
                logger.info(f"[pool {pools}] guard blocked {blocked}/{len(pool)} "
                            f"({_pct(blocked/len(pool))}) at margin={guard_margin:.2f}")

            if not safe_mask.any():
                continue
            pool = pool[safe_mask]

        diag["pool_after_guard"] += len(pool)

        Zc = embed_fn(pool).astype(np.float32)
        d_c, i_c = qnn(Zc)
        R_c = d_c - radii[i_c]

        worst    = surplus.argmin()
        safe_idx = np.where(R_c >= 0)[0]
        best     = safe_idx[np.argmax(R_c[safe_idx])] if safe_idx.size else int(np.argmax(R_c))
        accept   = (R_c[best] >= surplus[worst])

        if apply_guard and guard_stage == 'postfilter' and accept and guards is not None:
            cand_row = pool.iloc[[best]]
            veto_union, tbl_one, _ = evaluate_guards(
                cand_row, guards, margin_prob=guard_margin,
                build_table=explain_guard, max_rows=10
            )
            veto = bool(veto_union[0])
            if veto:
                diag["veto_count"] += 1
                if explain_guard and tbl_one is not None and len(tbl_one):
                    logger.info("[veto] candidate blocked by guard:\n" +
                                tbl_one.to_string(index=False, justify='left'))
                accept = False

        if accept:
            synth.iloc[worst] = pool.iloc[best]
            z_w = Zc[best].reshape(1, -1)
            d_w, i_w = qnn(z_w)
            surplus[worst] = d_w[0] - radii[i_w[0]]

            eps = float((surplus < 0).mean())
            swaps += 1
            diag["accept_count"] += 1
            diag["eps_history"].append((swaps, eps))

            if verbose and (swaps % log_every == 0):
                logger.info(f"[{swaps:>7d} swaps] ε={eps:.4f}  "
                            f"min/med/max surplus = "
                            f"{float(surplus.min()):.3f} / {float(np.median(surplus)):.3f} / {float(surplus.max()):.3f}")

            if verbose and (swaps % stats_every == 0):
                p10, p50, p90 = np.quantile(surplus, [0.10, 0.50, 0.90])
                logger.info(f"[stats] surplus p10/p50/p90 = {p10:.3f}/{p50:.3f}/{p90:.3f}  "
                            f"guard_blocks={diag['total_guard_blocked']}  "
                            f"accepts={diag['accept_count']}  vetoes={diag['veto_count']}")

    # ----------------- wrap up -----------------------------------------
    # Guarantee exact size even if something upstream misbehaved
    synth = ensure_size(synth)

    diag["swaps"] = swaps
    diag["elapsed_sec"] = time.perf_counter() - start_t
    eps, surplus = epsilon_of(synth)  # recompute final epsilon after any padding
    diag["final_surplus_stats"] = dict(
        min=float(np.min(surplus)),
        p25=float(np.quantile(surplus, 0.25)),
        med=float(np.median(surplus)),
        p75=float(np.quantile(surplus, 0.75)),
        max=float(np.max(surplus)),
        eps=float(eps)
    )
    if swaps >= max_swaps and verbose:
        logger.info(f"[done] reached max_swaps={max_swaps} with ε={eps:.4f}  "
                    f"filled_guard={diag['pad_guard_kept']}  filled_noguard={diag['pad_noguard_rows']}  "
                    f"elapsed={diag['elapsed_sec']:.1f}s")
    else:
        logger.info(f"[done] ε={eps:.4f} < {min_eps} in {swaps} swaps  "
                    f"filled_guard={diag['pad_guard_kept']}  filled_noguard={diag['pad_noguard_rows']}  "
                    f"elapsed={diag['elapsed_sec']:.1f}s")

    return (synth.reset_index(drop=True), diag) if return_diag else synth.reset_index(drop=True)


