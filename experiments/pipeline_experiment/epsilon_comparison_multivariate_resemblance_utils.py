"""Utility functions extracted from epsilon_comparison_multivariate_resemblance_evaluation.ipynb."""

import re
from collections import OrderedDict
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
from scipy.cluster.hierarchy import linkage


def extract_eps(key: str) -> float:
    key = key.strip()
    m = re.search(r'([0-9.]+)$', key)
    if m is None:
        return np.nan                    # baseline
    token = m.group(1)
    if '.' in token:
        return float(token)              # '0.3' → 0.3
    digits = token.lstrip('0') or '0'    # '025' → '25'
    return int(digits) / (10 ** (len(token) - 1))

def filter_by_prefix(results: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    prefix = prefix.rstrip()
    return {k: v for k, v in results.items() if k.startswith(prefix)}

def _norm_diff(res: dict, corr_type: str) -> float:
    """
    Return the *pre‑computed* Frobenius distance stored by your pipeline.
    """
    if corr_type == 'pearson':
        return res['resemblance_evaluation_results']['numerical_multivariate']['pearson_norm_diff']
    if corr_type == 'cramer':
        return res['resemblance_evaluation_results']['categorical_multivariate']['diff_norm_cramer']
    if corr_type == 'corr_ratio':
        return res['resemblance_evaluation_results']['categorical_numerical_multivariate']['diff_norm_corr_ratio']
    raise ValueError(corr_type)

def _paper_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=0.9)
    plt.rcParams.update({
        "axes.edgecolor": "0.2",
        "axes.linewidth": 0.6,
        "grid.color": "0.88",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.9,
        "axes.titleweight": "semibold",
    })

def _extract_tau(key):
    """
    Try to parse τ from a dict_result key.
    Returns float(τ) or np.nan if this key is a 'baseline' (or unparsable).
    Robust to strings like 'tau=0.1', 'eps_0.05', '0.005', etc.
    """
    if key is None:
        return np.nan
    if isinstance(key, (int, float, np.floating)):
        return float(key)
    s = str(key).lower()
    if "base" in s or "no_reject" in s or "none" in s:
        return np.nan
    m = re.search(r'(?:(?:tau|eps|epsilon)\s*=?\s*)?(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return np.nan   # treat as baseline-like

def _get_nested(d, path):
    """Safe lookup: d[path[0]][path[1]]..."""
    out = d
    for p in path:
        out = out[p]
    return out

def _avg_by_tau(dict_result, path, real_like):
    """
    Collect the correlation matrix for each τ and average replicates (nanmean).
    Returns: Ordered list of (tau, DataFrame) sorted by tau desc, tau is float or np.nan.
    All matrices are reindexed to match 'real_like' (index/columns).
    """
    from collections import defaultdict
    buckets = defaultdict(list)

    for k, res in dict_result.items():
        tau = _extract_tau(k)
        mat = _get_nested(res, path)  # should be a square DataFrame
        # align to real's shape/order
        mat = mat.reindex(index=real_like.index, columns=real_like.columns)
        buckets[tau].append(mat.astype(float))

    out = []
    for tau, lst in buckets.items():
        if len(lst) == 1:
            avg = lst[0]
        else:
            stack = np.stack([a.values for a in lst], axis=0)
            avg = pd.DataFrame(np.nanmean(stack, axis=0),
                               index=real_like.index, columns=real_like.columns)
        out.append((tau, avg))

    # sort τ numerically, largest first; keep NaN (baseline-like) at the end
    out.sort(key=lambda x: (np.isnan(x[0]), -x[0] if not np.isnan(x[0]) else 0))
    return out

def _plot_correlation_over_tau(dict_result: dict,
                               real: pd.DataFrame,
                               path: list,
                               metric_name: str,
                               cols: int = 2,
                               sort_rows_by_abs_real: bool = True,
                               show_annotations: bool = "auto"):
    """
    Core plotter used by the three wrappers below.
    - dict_result: {tau_key -> nested results dict}
    - real: DataFrame of real correlations (index == columns == feature names)
    - path: path inside each results dict to the correlation matrix
    """

    _paper_style()

    # colormap and range per metric
    if metric_name.lower().startswith("pearson"):
        vmin, vmax, center, cmap, cbar_label = -1.0, 1.0, 0.0, "coolwarm", "correlation (ρ)"
    else:
        vmin, vmax, center, cmap, cbar_label = 0.0, 1.0, None, "viridis", f"association ({metric_name})"

    # collect average matrix for each τ
    tau_mats = _avg_by_tau(dict_result, path, real)

    features = real.index.tolist()
    n_feat = len(features)
    rows = int(np.ceil(n_feat / cols))
    # fig, axes = plt.subplots(rows, cols,
    #                          figsize=(cols * 6.2, rows * 4.6),
    #                          squeeze=False)
    
    fig, axes = plt.subplots(rows, cols,
                         figsize=(cols * 6.2, rows * 4.6),
                         squeeze=False,
                         constrained_layout=True)
    axes = axes.ravel()

    mappable = None

    for i, feat in enumerate(features):
        ax = axes[i]

        # assemble one feature's table: columns = ['real', τ1, τ2, ...]
        cols_series = [("real", real.loc[feat])]
        for tau, mat in tau_mats:
            label = f"{tau:g}" if not np.isnan(tau) else "baseline"
            cols_series.append((label, mat.loc[feat]))

        df_feat = pd.concat([s for _, s in cols_series], axis=1)
        df_feat.columns = [lbl for lbl, _ in cols_series]

        # drop self-correlation row
        df_feat = df_feat.drop(index=feat, errors="ignore")


        all_cols = list(df_feat.columns)
        # order columns: real, then τ descending (baseline last if present)
        has_baseline = "baseline" in all_cols
        tau_only = [c for c in all_cols if c not in ("real", "baseline")]
        # sort tau labels numerically (strings like "0.4", "0.05", etc.)
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return -np.inf  # push weird labels to the right
        tau_sorted = sorted(tau_only, key=lambda c: -_to_float(c))
        new_order = ["real"]
        if has_baseline:
            new_order.append("baseline")       # put baseline right after real
        new_order += tau_sorted

        df_feat = df_feat[new_order]

        # optionally order rows by |real| (makes patterns pop)
        if sort_rows_by_abs_real and "real" in df_feat.columns:
            order = df_feat["real"].abs().sort_values(ascending=False).index
            df_feat = df_feat.loc[order]

        # annotations only if small enough
        if show_annotations == "auto":
            annot = (df_feat.shape[0] * df_feat.shape[1] <= 160)
        else:
            annot = bool(show_annotations)

        im = sns.heatmap(df_feat,
                         ax=ax,
                         cmap=cmap,
                         vmin=vmin, vmax=vmax, center=center,
                         annot=annot, fmt=".2f" if annot else "",
                         linewidths=0.4, linecolor="0.9",
                         cbar=False)

        ax.set_title(feat, pad=4)
        ax.set_xlabel("τ (threshold)")
        ax.set_ylabel("correlation with")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", va="top")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        mappable = im  # keep last for shared colorbar

    # remove empty axes (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # one shared colorbar
    # one shared colorbar (keep this AFTER the heatmaps)
    if mappable is not None:
        cbar = fig.colorbar(
            mappable.collections[0],
            ax=axes[:(i + 1)],
            location="right",
            fraction=0.035,   # a bit wider than before
            pad=0.02,         # small gap between subplots and cbar
            shrink=0.98,      # slightly shorter than subplot stack
            anchor=(0.0, 0.5) # vertically centered
        )
        cbar.set_label(cbar_label, rotation=90)

    fig.suptitle(f"{metric_name} over τ (largest on the left)", y=1.02,
                 fontsize=10, fontweight="bold")
    # fig.tight_layout()
    plt.show()

def plot_pearson_over_tau(dict_result: dict, real: pd.DataFrame) -> None:
    """
    Heatmaps of Pearson correlations (rows: other features, columns: real + τs).
    """
    path = ["resemblance_evaluation_results",
            "numerical_multivariate", "pearson_synth"]
    _plot_correlation_over_tau(dict_result, real, path, metric_name="Pearson (ρ)")

def plot_cramer_over_tau(dict_result: dict, real: pd.DataFrame) -> None:
    """
    Heatmaps of Cramér's V (rows: other features, columns: real + τs).
    """
    path = ["resemblance_evaluation_results",
            "categorical_multivariate", "cramer_synth"]
    _plot_correlation_over_tau(dict_result, real, path, metric_name="Cramér's V")

def plot_corr_ratio_over_tau(dict_result: dict, real: pd.DataFrame) -> None:
    """
    Heatmaps of correlation ratio η (rows: other features, columns: real + τs).
    """
    path = ["resemblance_evaluation_results",
            "categorical_numerical_multivariate", "corr_ratio_synth", "corr"]
    _plot_correlation_over_tau(dict_result, real, path, metric_name="Correlation ratio (η)")

def _paper_style():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=0.9)
    plt.rcParams.update({
        "axes.edgecolor": "0.2",
        "axes.linewidth": 0.6,
        "grid.color": "0.88",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.9,
        "axes.titleweight": "semibold",
        "legend.frameon": False,
    })

def _extract_tau(key):
    """Parse τ from dict_result key. NaN => baseline / unparsable."""
    if key is None:
        return np.nan
    if isinstance(key, (int, float, np.floating)):
        return float(key)
    s = str(key).lower()
    if "base" in s or "no_reject" in s or "none" in s:
        return np.nan
    m = re.search(r'(?:(?:tau|eps|epsilon)\s*=?\s*)?(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return np.nan

def _get_nested(d, path):
    out = d
    for p in path:
        out = out[p]
    return out

def _collect_by_tau(dict_result, path, real_like):
    """
    Return {tau_float_or_nan: [DataFrame, ...]} aligned to real_like.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for k, res in dict_result.items():
        tau = _extract_tau(k)
        mat = (_get_nested(res, path)
       .reindex(index=real_like.index, columns=real_like.columns)
       .astype(float))
        buckets[tau].append(mat)
    return buckets

def plot_delta_heatmaps_over_tau(dict_result: dict,
                                 real: pd.DataFrame,
                                 path: list,
                                 metric_name: str,
                                 absolute: bool = False,
                                 cols: int = 2,
                                 sort_rows_by_abs_real: bool = True):
    """
    One heatmap per 'anchor' feature.
    Columns are: baseline | τ (descending).
    Cells show Δ = synth - real (or |Δ| if absolute=True).
    """

    _paper_style()

    # Collect matrices and average per τ
    buckets = _collect_by_tau(dict_result, path, real)
    taus_sorted = sorted(buckets.keys(), key=lambda t: (np.isnan(t), -t if not np.isnan(t) else 0))

    features = real.index.tolist()
    n_feat = len(features)
    rows = int(np.ceil(n_feat / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 4.6),
                             squeeze=False, constrained_layout=True)
    axes = axes.ravel()

    # --- Global Δ limits for consistent color range ---
    all_deltas = []
    for tau in taus_sorted:
        mats = buckets[tau]
        avg = np.nanmean(np.stack([m.values for m in mats], axis=0), axis=0)
        delta = avg - real.values
        all_deltas.append(delta if not absolute else np.abs(delta))
    all_deltas = np.concatenate([d.ravel() for d in all_deltas]) if all_deltas else np.array([0.0])
    vmax = float(np.nanquantile(np.abs(all_deltas), 0.99)) or 1.0
    if absolute:
        vmin, vcenter, cmap = 0.0, None, "viridis"
    else:
        vmin, vcenter, cmap = -vmax, 0.0, "vlag"

    mappable = None
    last_ax_index = -1

    for i, feat in enumerate(features):
        ax = axes[i]
        last_ax_index = i

        # --- Build Δ values for this anchor feature ---
        cols_series = []
        for tau in taus_sorted:
            mats = buckets[tau]
            avg = np.nanmean(np.stack([m.values for m in mats], axis=0), axis=0)
            avg_df = pd.DataFrame(avg, index=real.index, columns=real.columns)
            delta = avg_df - real
            col = delta.loc[feat]
            if absolute:
                col = col.abs()
            label = f"{tau:g}" if not np.isnan(tau) else "baseline"
            cols_series.append((label, col))

        df_feat = pd.concat([s for _, s in cols_series], axis=1)
        df_feat.columns = [lbl for lbl, _ in cols_series]
        df_feat = df_feat.drop(index=feat, errors="ignore")

        # --- NEW: order columns baseline | τ↓, no "real" column ---
        all_cols = list(df_feat.columns)
        has_baseline = "baseline" in all_cols
        tau_only = [c for c in all_cols if c != "baseline"]

        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return -np.inf  # push bad labels to end

        tau_sorted = sorted(tau_only, key=lambda c: -_to_float(c))  # largest τ first
        new_order = (["baseline"] if has_baseline else []) + tau_sorted
        df_feat = df_feat[new_order]
        # -------------------------------------------------------------

        if sort_rows_by_abs_real:
            order = real.loc[feat].drop(index=feat, errors="ignore").abs().sort_values(ascending=False).index
            df_feat = df_feat.loc[order]

        im = sns.heatmap(df_feat, ax=ax, cmap=cmap,
                         vmin=vmin, vmax=vmax, center=vcenter,
                         annot=False, linewidths=0.4, linecolor="0.9",
                         cbar=False)
        ax.set_title(feat, pad=4)
        ax.set_xlabel("τ (threshold)")
        ax.set_ylabel("absolute difference" if absolute else "difference vs real")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", va="top")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        mappable = im

    # Remove unused subplots
    for j in range(last_ax_index + 1, len(axes)):
        fig.delaxes(axes[j])

    # Shared colorbar
    if mappable is not None:
        cbar = fig.colorbar(
            mappable.collections[0],
            ax=axes[: last_ax_index + 1],
            location="right", fraction=0.035, pad=0.02,
            shrink=0.98, anchor=(0.0, 0.5)
        )
        cbar.set_label(f"|Δ {metric_name}|" if absolute else f"Δ {metric_name}", rotation=90)

    fig.suptitle(f"{metric_name}: Δ vs real across τ (columns: baseline | τ↓)",
                 y=1.02, fontsize=10, fontweight="bold")
    plt.show()

def plot_matrix_error_over_tau(dict_result: dict,
                               real: pd.DataFrame,
                               path: list,
                               metric_name: str,
                               summary: str = "mae"):
    """
    Summarize Δ for each τ as one score:
      - 'mae': mean absolute difference (off-diagonal for square matrices; all cells otherwise)
      - 'rmse': root mean square error
      - 'fro': Frobenius / sqrt(N_pairs) (scale-free)
    Plots mean ±95% CI across runs for each τ (log x, largest left).
    """
    _paper_style()
    assert summary in {"mae", "rmse", "fro"}

    buckets = _collect_by_tau(dict_result, path, real)

    def summarise_mat(mat: pd.DataFrame) -> float:
        # ensure DataFrame, aligned to 'real' (keeps shape identical to real)
        if not isinstance(mat, pd.DataFrame):
            mat = pd.DataFrame(mat)
        mat = mat.reindex(index=real.index, columns=real.columns)

        # subtract with label alignment -> same shape as 'real'
        delta_df = mat - real
        delta = delta_df.to_numpy()

        # If square, drop diagonal; else use all entries (rectangular e.g. η)
        if delta.shape[0] == delta.shape[1]:
            mask = ~np.eye(delta.shape[0], dtype=bool)
            d = delta[mask]
        else:
            d = delta.ravel()

        if summary == "mae":
            return float(np.nanmean(np.abs(d)))
        elif summary == "rmse":
            return float(np.sqrt(np.nanmean(d ** 2)))
        else:  # 'fro' (scale-free)
            n_eff = np.isfinite(d).sum()
            if n_eff == 0:
                return np.nan
            return float(np.linalg.norm(np.nan_to_num(d), ord="fro") / np.sqrt(n_eff))

    # compute per-run summary, then mean & 95% CI per τ
    rows = []
    for tau, mats in buckets.items():
        vals = [summarise_mat(m) for m in mats]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) == 0:
            rows.append((tau, np.nan, 0.0, 0))
            continue
        n = len(vals)
        mean = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
        rows.append((tau, mean, ci, n))

    df = pd.DataFrame(rows, columns=["tau", "mean", "ci", "n"])

    # separate baseline
    base = df[df["tau"].isna()]
    df   = df[~df["tau"].isna()].sort_values("tau", ascending=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    if not df.empty:
        x  = df["tau"].to_numpy()
        y  = df["mean"].to_numpy()
        ci = df["ci"].to_numpy()

        ax.plot(x, y, marker="o", linewidth=1.15)
        ax.fill_between(x, y - ci, y + ci, alpha=0.18, linewidth=0)

        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:g}" for t in x], rotation=90, ha="center", va="top")
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * .1))
        ax.xaxis.set_minor_formatter(NullFormatter())

    # baseline horizontal band, if present
    if not base.empty and np.isfinite(base["mean"].iloc[0]):
        b, bci = float(base["mean"].iloc[0]), float(base["ci"].iloc[0])
        ax.axhline(b, linestyle=(0, (2, 2)), color="0.3", linewidth=1.0, label="baseline")
        if bci > 0:
            ax.axhspan(b - bci, b + bci, color="0.3", alpha=0.12)

    ylabel = {
        "mae": f"Mean |Δ {metric_name}|",
        "rmse": f"RMSE of Δ {metric_name}",
        "fro":  f"Frobenius/√N of Δ {metric_name}",
    }[summary]
    ax.set_ylabel(ylabel)
    ax.set_xlabel("τ (log scale, larger → left)")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="minor", axis="y", linewidth=0.3, alpha=0.45)
    ax.set_title(f"{metric_name} — summary error vs τ")
    plt.show()

def paper_style():
    sns.set_theme(style='whitegrid')
    sns.set_context('paper', font_scale=0.9)
    plt.rcParams.update({
        'axes.edgecolor':   '0.2',
        'axes.linewidth':   0.6,
        'grid.color':       '0.85',
        'grid.linewidth':   0.4,
        'grid.alpha':       0.9,
        'legend.frameon':   False,
        'legend.borderaxespad': 0.2,
        'axes.titleweight': 'semibold',
    })

def _agg_by_tau(df):
    g = (df.dropna(subset=['tau'])
           .groupby('tau', as_index=False)['dist']
           .agg(mean='mean', std='std', n='count'))
    ci = np.where(g['n'] > 1, 1.96 * g['std'] / np.sqrt(g['n']), 0.0)
    g['ci_low'] = g['mean'] - ci
    g['ci_high'] = g['mean'] + ci
    return g.sort_values('tau', ascending=False)

def plot_ressemblance_curves(results_dict,
                             dataset: str,
                             generator: str,
                             sample_size: int,
                             observed_rate: dict = None,
                             include_baseline: bool = True,
                             share_y_axis: bool = True,
                             harmonize_y_range: bool = True):

    paper_style()

    corr_types = OrderedDict([
        ('pearson',    r'Pearson ($\rho$)'),
        ('cramer',     "Cramér's V"),
        ('corr_ratio', r'Correlation ratio ($\eta$)'),
    ])

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2),
                             sharey=share_y_axis, constrained_layout=True)

    global_vals = []

    for ax, (ctype, ctitle) in zip(axes, corr_types.items()):
        # ---------- collect results ----------
        rows = []
        for k, res in results_dict.items():
            tau = extract_eps(k)                     # ε is τ here
            dist = _norm_diff(res, ctype)
            rows.append((tau, k, dist))
        df = pd.DataFrame(rows, columns=['tau', 'key', 'dist'])

        df_base = df[df['tau'].isna()]
        g = _agg_by_tau(df)

        # ---------- plot ----------
        if not g.empty:
            x  = g['tau'].to_numpy()
            y  = g['mean'].to_numpy()
            lo = g['ci_low'].to_numpy()
            hi = g['ci_high'].to_numpy()

            ax.plot(x, y, marker='o', markersize=4, linewidth=1.15, zorder=3)
            ax.fill_between(x, lo, hi, alpha=0.18, linewidth=0, zorder=2)

            # (keep subtle highlight but NO annotation)
            best_idx = np.argmin(y)
            ax.scatter([x[best_idx]], [y[best_idx]], s=28, zorder=4,
                       facecolors='white', edgecolors='black', linewidths=1)
            # If you don't want any highlight at all, remove the scatter() above.

            ax.set_xscale('log')
            ax.invert_xaxis()
            ax.set_xticks(x)
            ax.set_xticklabels([f'{t:g}' for t in x], rotation=90,
                               ha='center', va='top')
            ax.tick_params(axis='x', pad=1)
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*.1))
            ax.xaxis.set_minor_formatter(NullFormatter())

            if harmonize_y_range:
                global_vals.extend([*lo, *hi])

        # ---------- baseline ----------
        if include_baseline and not df_base.empty:
            base_mean = df_base['dist'].mean()
            base_std  = df_base['dist'].std()
            base_n    = df_base.shape[0]
            base_ci   = 1.96 * base_std / np.sqrt(base_n) if base_n > 1 and not np.isnan(base_std) else 0.0
            ax.axhline(base_mean, linestyle=(0, (2, 2)), linewidth=1.0, color='0.3', zorder=1)
            if base_ci > 0:
                ax.axhspan(base_mean - base_ci, base_mean + base_ci, color='0.3', alpha=0.12, zorder=1)
            ax.text(0.02, 0.98, 'baseline', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8, color='0.35')
            if harmonize_y_range:
                global_vals.extend([base_mean - base_ci, base_mean + base_ci])

        # ---------- cosmetics ----------
        ax.set_title(ctitle, pad=6)
        ax.set_xlabel(r'$\tau$ (log scale, larger → left)')
        if ax is axes[0]:
            ax.set_ylabel('Frobenius distance (lower is better)')
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', axis='y', linewidth=0.3, alpha=0.45)

    # harmonized y-range
    if harmonize_y_range and global_vals:
        y_min, y_max = min(global_vals), max(global_vals)
        rng = max(1e-6, y_max - y_min)
        pad_low, pad_high = 0.06*rng, 0.08*rng
        for ax in axes:
            ax.set_ylim(max(0.0, y_min - pad_low), y_max + pad_high)

    fig.suptitle(f'Resemblance loss — {dataset}, {generator} (n={sample_size:,})',
                 y=1.04, fontsize=10, fontweight='bold')
    fig.text(0.01, -0.02,
             'Dashed line = baseline (no rejection)',
             fontsize=8)

    plt.show()

def plot_ressemblance_curves_notscale(results_dict,
                                      dataset: str,
                                      generator: str,
                                      sample_size: int,
                                      observed_rate: dict = None,
                                      include_baseline: bool = True):
    return plot_ressemblance_curves(
        results_dict=results_dict,
        dataset=dataset,
        generator=generator,
        sample_size=sample_size,
        observed_rate=observed_rate,
        include_baseline=include_baseline,
        share_y_axis=False,
        harmonize_y_range=False,
    )

def print_evaluation_multivariate(dict_result, epsilon) :
    diff = {}
    for result , key in zip(dict_result.values(), epsilon) :
        pearson_diff = result['resemblance_evaluation_results']['numerical_multivariate']['pearson_norm_diff']
        corr_ratio_diff = result['resemblance_evaluation_results']['categorical_numerical_multivariate']['diff_norm_corr_ratio']
        cramer_diff = result['resemblance_evaluation_results']['categorical_multivariate']['diff_norm_cramer']

        diff[key] = [pearson_diff, cramer_diff, corr_ratio_diff]
    print(pd.DataFrame(diff, index=['Pearson', "Cramer's", "Correlation ratio"]).to_markdown())

def print_correlation_matrices(dict_results, epsilon, first_key) :
    spearman = {}
    pearson_diff = dict_results[first_key]['resemblance_evaluation_results']['numerical_multivariate']['pearson_real']
    corr_ratio_diff = dict_results[first_key]['resemblance_evaluation_results']['categorical_numerical_multivariate']['corr_ratio_real']['corr']
    cramer_diff = dict_results[first_key]['resemblance_evaluation_results']['categorical_multivariate']['cramer_real']

    print(f'\n \n pearson correlation matrix - Real data {first_key} : \n \n')
    print(pd.DataFrame(pearson_diff).to_markdown())
    print(f'\n \n cramer correlation matrix - Real data {first_key} : \n \n')
    print(pd.DataFrame(cramer_diff).to_markdown())
    print(f'\n \n correlation ratio matrix - Real data {first_key} : \n \n')
    print(pd.DataFrame(corr_ratio_diff).to_markdown())

    for result , key in zip(dict_results.values(), epsilon) :
        pearson_diff = result['resemblance_evaluation_results']['numerical_multivariate']['pearson_synth']
        corr_ratio_diff = result['resemblance_evaluation_results']['categorical_numerical_multivariate']['corr_ratio_synth']['corr']
        cramer_diff = result['resemblance_evaluation_results']['categorical_multivariate']['cramer_synth']
        print(f'\n \n pearson correlation matrix - {key} : \n \n')
        print(pd.DataFrame(pearson_diff).to_markdown())
        print(f'\n \n cramer correlation matrix - {key} : \n \n')
        print(pd.DataFrame(cramer_diff).to_markdown())
        print(f'\n \n correlation ratio matrix - {key} : \n \n')
        print(pd.DataFrame(corr_ratio_diff).to_markdown())

        pearson_spearman = result['resemblance_evaluation_results']['numerical_multivariate']['pearson_spearman_correlation_coefficient']
        corr_ratio_spearman = result['resemblance_evaluation_results']['categorical_numerical_multivariate']['corr_ratio_spearman_correlation_coefficient']['corr']
        cramer_spearman = result['resemblance_evaluation_results']['categorical_multivariate']['cramer_spearman_correlation_coefficient']
        spearman[key] = [pearson_spearman, cramer_spearman, corr_ratio_spearman]

    print(f'\n \n  Froebenus norm correlation matrices real vs synth : \n \n')
    print_evaluation_multivariate(dict_results, epsilon)
    
    print(f'\n \n  Spearman coefficent of  correlation matrices real vs synth : \n \n')
    print(pd.DataFrame(spearman, index=['Pearson', "Cramer's", "Correlation ratio"]).to_markdown())
