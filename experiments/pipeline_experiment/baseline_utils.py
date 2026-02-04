"""Utility functions extracted from baseline_comparison.ipynb."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Mapping
from matplotlib.lines import Line2D



def plot_per_feature_heatmap(
    metric_series_dict,
    metric_name="Cohen's d",
    dataset_name=None,
    sort_features=True,
    figsize=None,
    ax=None,
    vmin=None,
    vmax=None,
):
    # Build a DataFrame: rows = features, columns = models
    df = pd.DataFrame(metric_series_dict)

    if sort_features:
        df = df.loc[df.max(axis=1).sort_values(ascending=False).index]

    n_features, n_models = df.shape

    # Create fig/ax only if not provided
    created_fig = False
    if ax is None:
        if figsize is None:
            width = max(4, 1.2 * n_models)
            height = max(4, 0.4 * n_features)
            figsize = (width, height)
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(df.values, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(df.index)

    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")

    if dataset_name is not None:
        title = f"{dataset_name} — {metric_name} per feature"
    else:
        title = f"{metric_name} per feature"
    ax.set_title(title)

    # Only tighten if we own the figure
    if created_fig:
        fig.tight_layout()

    return fig, ax, im

def plot_per_feature_heatmap_multi(
    all_metrics_dict,
    metric_name="JS divergence",
    sort_features=True,
    figsize=None,
    datasets_order=None,
):
    # --- Group keys "<dataset> <model>" ---
    grouped = {}
    for key, series in all_metrics_dict.items():
        dataset, model = key.split(maxsplit=1)
        grouped.setdefault(dataset, {})[model] = series

    if datasets_order is None:
        datasets = sorted(grouped.keys())
    else:
        datasets = datasets_order

    n_datasets = len(datasets)

    # --- Shared color scale ---
    all_values = np.concatenate([s.values for s in all_metrics_dict.values()])
    vmin, vmax = all_values.min(), all_values.max()

    # --- Figure & subplots ---
    if figsize is None:
        figsize = (10, 4 * n_datasets)

    fig, axes = plt.subplots(n_datasets, 1, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    ims = []

    for ax, dataset in zip(axes, datasets):
        metric_series_dict = grouped[dataset]
        dataset_title = dataset.capitalize()

        fig, ax, im = plot_per_feature_heatmap(
            metric_series_dict=metric_series_dict,
            metric_name=metric_name,
            dataset_name=dataset_title,
            sort_features=sort_features,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

    # --- Leave room on the right for colorbar ---
    # everything left of x=0.88 is for subplots
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])

    # --- Explicit colorbar axes OUTSIDE the plots ---
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(ims[-1], cax=cbar_ax)
    cbar.set_label(metric_name)

    return fig, axes

def plot_per_feature_bar(
    metric_series_dict,
    metric_name="Cohen's d",
    dataset_name=None,
    sort_features=True,
    figsize=None,
    ax=None,
    ylim=None,
):
    """
    metric_series_dict: dict like {"CTGAN": Series, "TVAE": Series}
                        Series index = features, values = metric
    """

    # Build DataFrame: rows = features, cols = models
    df = pd.DataFrame(metric_series_dict).dropna(how="all")

    if sort_features:
        # Sort by max metric across models (descending)
        df = df.loc[df.max(axis=1).sort_values(ascending=False).index]

    n_features, n_models = df.shape
    models = list(df.columns)

    # Create fig/ax only if not provided
    created_fig = False
    if ax is None:
        if figsize is None:
            # Wider when many features
            width = max(6, 0.6 * n_features)
            height = 4
            figsize = (width, height)
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    x = np.arange(n_features)
    total_width = 0.8
    bar_width = total_width / n_models

    # Center grouped bars on x-ticks
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width

    for i, model in enumerate(models):
        ax.bar(x + offsets[i], df[model].values, width=bar_width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right")

    ax.set_xlabel("Feature")
    ax.set_ylabel(metric_name)

    if dataset_name is not None:
        title = f"{dataset_name} — {metric_name} per feature"
    else:
        title = f"{metric_name} per feature"
    ax.set_title(title)

    ax.legend(title="Model")

    if ylim is not None:
        ax.set_ylim(ylim)

    if created_fig:
        fig.tight_layout()

    return fig, ax

def plot_per_feature_bar_multi(
    all_metrics_dict,
    metric_name="JS distance",
    sort_features=True,
    figsize=None,
    datasets_order=None,
    sharey=True,
):
    """
    all_metrics_dict: dict with keys "<dataset> <model>", e.g. "credit CTGAN"
                      values are Series(feature -> metric)
    """

    # Group keys "<dataset> <model>"
    grouped = {}
    for key, series in all_metrics_dict.items():
        dataset, model = key.split(maxsplit=1)
        grouped.setdefault(dataset, {})[model] = series

    if datasets_order is None:
        datasets = sorted(grouped.keys())
    else:
        datasets = datasets_order

    n_datasets = len(datasets)

    # Shared y-range across all subplots
    all_values = np.concatenate([s.dropna().values for s in all_metrics_dict.values()])
    y_min = min(0, all_values.min())
    y_max = all_values.max() * 1.05
    ylim = (y_min, y_max) if sharey else None

    if figsize is None:
        figsize = (10, 3 * n_datasets)

    fig, axes = plt.subplots(
        n_datasets,
        1,
        figsize=figsize,
        squeeze=False,
        sharey=sharey,
    )
    axes = axes.ravel()

    for ax, dataset in zip(axes, datasets):
        metric_series_dict = grouped[dataset]
        dataset_title = dataset.capitalize()
        plot_per_feature_bar(
            metric_series_dict=metric_series_dict,
            metric_name=metric_name,
            dataset_name=dataset_title,
            sort_features=sort_features,
            ax=ax,
            ylim=ylim,
        )

    fig.tight_layout()
    return fig, axes

def get_cohen_result(evaluation_results):
    cohen = {}
    for key, result in evaluation_results.items() :
        cohen[key] = result['resemblance_evaluation_results']['numerical_univariate']['univariate_num_js']['cohen_s_d']
    return cohen

def get_jsdivergence_result(evaluation_results) :
    js = {}
    for key, result in evaluation_results.items() :
        js[key] = result['resemblance_evaluation_results']['categorical_univariate']['jensen_shanon']['JS_divergence']
    return js

def get_anonymeter_result(evaluation_results) :
    anonym = {}
    for key, result in evaluation_results.items() :
        anonym[key] = result['privacy_anonymeter_results']
    return anonym

def _extract_scalar(d, key):
    """Handle both {'corr': value} and plain float cases."""
    val = d[key]
    if isinstance(val, Mapping):
        # e.g. {'corr': 0.9369}
        return val.get("corr", list(val.values())[0])
    return val

def build_summary_df(corr_ratio_results, pearson_results, cramer_results):
    rows = []

    all_keys = sorted(corr_ratio_results.keys())  # e.g. "adult CTGAN"

    for name in all_keys:
        dataset, model = name.split()  # assumes "credit CTGAN" format

        corr_res    = corr_ratio_results[name]
        pearson_res = pearson_results[name]
        cramer_res  = cramer_results[name]

        rows.append({
            "dataset": dataset,
            "model": model,   # CTGAN / TVAE

            # corr-ratio
            "diff_norm_corr_ratio": corr_res["diff_norm_corr_ratio"],
            "corr_ratio_spearman": _extract_scalar(
                corr_res, "corr_ratio_spearman_correlation_coefficient"
            ),

            # pearson
            "pearson_norm_diff": pearson_res["pearson_norm_diff"],
            "pearson_spearman": pearson_res["pearson_spearman_correlation_coefficient"],

            # cramer
            "diff_norm_cramer": cramer_res["diff_norm_cramer"],
            "cramer_spearman": cramer_res["cramer_spearman_correlation_coefficient"],
        })

    return pd.DataFrame(rows)

def plot_matrix_errors_vs_rankcorr(summary_df):
    
    # (metric_col, ylabel, title)
    left_metrics = [
        ("diff_norm_corr_ratio",  "‖corr_ratio_real − corr_ratio_synth‖",      "Corr-ratio matrix error (lower is better)"),
        ("pearson_norm_diff",     "‖Pearson_real − Pearson_synth‖",            "Pearson matrix error (lower is better)"),
        ("diff_norm_cramer",      "‖Cramér_real − Cramér_synth‖",              "Cramér's V matrix error (lower is better)")
    ]

    right_metrics = [
        ("corr_ratio_spearman",   "Spearman(corr_ratio_real, corr_ratio_synth)", "Corr-ratio rank correlation (higher is better)"),
        ("pearson_spearman",      "Spearman(Pearson_real, Pearson_synth)",       "Pearson rank correlation (higher is better)"),
        ("cramer_spearman",       "Spearman(Cramér_real, Cramér_synth)",         "Cramér's V rank correlation (higher is better)")
    ]

    datasets = summary_df["dataset"].unique()
    models   = summary_df["model"].unique()
    x = np.arange(len(datasets))
    width = 0.35

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Left column = matrix errors
    for ax, (metric_col, ylabel, title) in zip(axes[:, 0], left_metrics):
        for i, model in enumerate(models):
            vals = [
                summary_df[(summary_df.dataset == ds) & (summary_df.model == model)][metric_col].values[0]
                for ds in datasets
            ]
            ax.bar(x + (i - 0.5) * width, vals, width, label=model)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Right column = rank correlations
    for ax, (metric_col, ylabel, title) in zip(axes[:, 1], right_metrics):
        for i, model in enumerate(models):
            vals = [
                summary_df[(summary_df.dataset == ds) & (summary_df.model == model)][metric_col].values[0]
                for ds in datasets
            ]
            ax.bar(x + (i - 0.5) * width, vals, width, label=model)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # One shared legend (outside)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.12, 0.5))

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    plt.show()
    return fig

def build_utility_long_meanstd(utility_dict):
    """
    utility_dict:
      key:  '<dataset>_<model>_<regime>' e.g. 'credit_ctgan_trtr'
      val:  metrics_df with columns like '<metric>_mean' and '<metric>_std'
            and index = classifier (CART, KNN, ...)

    Returns long df with columns:
      dataset, model, regime, classifier, metric, mean, std
    """
    rows = []

    for key, metrics_df in utility_dict.items():
        dataset, model, regime = key.split("_")
        model = model.upper()

        # identify metric base names by looking for *_mean columns
        mean_cols = [c for c in metrics_df.columns if c.endswith("_mean")]
        metrics = [c[:-5] for c in mean_cols]  # strip "_mean"

        for clf_name, r in metrics_df.iterrows():
            for m in metrics:
                mean_col = f"{m}_mean"
                std_col  = f"{m}_std"

                mean_val = r.get(mean_col, pd.NA)
                std_val  = r.get(std_col, pd.NA)

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "regime": regime,
                    "classifier": clf_name,
                    "metric": m,
                    "mean": float(mean_val) if pd.notna(mean_val) else pd.NA,
                    "std": float(std_val) if pd.notna(std_val) else pd.NA,
                })

    return pd.DataFrame(rows)

def scatter_trtr_vs_tstr(utility_long, dataset, model, metric="roc_auc"):
    """
    Plot TRTR vs TSTR for a given dataset, generator, and metric.

    dataset: 'credit', 'adult', 'cardio'
    model:   'CTGAN' or 'TVAE' (case insensitive)
    metric:  'roc_auc' or 'f1_macro'
    """
    model = model.upper()

    df = utility_long[
        (utility_long["dataset"] == dataset) &
        (utility_long["model"] == model) &
        (utility_long["metric"] == metric)
    ]

    # pivot to get columns 'trtr' and 'tstr' per classifier
    pivot = df.pivot(index="classifier", columns="regime", values="value")

    # make sure both regimes exist
    if not {"trtr", "tstr"}.issubset(pivot.columns):
        raise ValueError(f"Missing trtr/tstr for {dataset}, {model}, {metric}")

    trtr = pivot["trtr"].values
    tstr = pivot["tstr"].values
    clfs = pivot.index.values

    fig, ax = plt.subplots(figsize=(4, 4))

    # scatter points
    ax.scatter(trtr, tstr, s=30)

    # annotate with classifier names
    for x, y, clf in zip(trtr, tstr, clfs):
        ax.text(x, y, clf, fontsize=8, ha="left", va="bottom")

    # 45-degree line
    lo = min(trtr.min(), tstr.min())
    hi = max(trtr.max(), tstr.max())
    padding = (hi - lo) * 0.05
    lo -= padding
    hi += padding

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"{metric} – train on real, test on real (TRTR)")
    ax.set_ylabel(f"{metric} – train on synthetic, test on real (TSTR)")
    ax.set_title(f"{dataset.capitalize()} – {model} – {metric}")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig

def bar_delta_tstr_trtr(utility_long, dataset, model, metric="roc_auc"):
    df = utility_long[
        (utility_long["dataset"] == dataset) &
        (utility_long["model"].str.lower() == model.lower()) &
        (utility_long["metric"] == metric)
    ]
    pivot = df.pivot_table(index="classifier", columns="regime", values="value")

    delta = pivot["tstr"] - pivot["trtr"]  # could also use ratio = pivot["tstr"]/pivot["trtr"]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(delta.index, delta.values)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel(f"Δ {metric} (tstr − trtr)")
    ax.set_title(f"{dataset.capitalize()} – {model.upper()} – {metric} Δ per classifier")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig

def multi_scatter_trtr_vs_tstr_meanstd(utility_long, metric="roc_auc", use_errorbars=False):
    datasets = ["credit", "cardio", "adult"]
    models   = ["CTGAN", "TVAE"]

    # Filter metric
    sub = utility_long[utility_long["metric"] == metric]
    if sub.empty:
        raise ValueError(f"No data for metric '{metric}'")

    # Global axis range based on means
    lo = sub["mean"].min()
    hi = sub["mean"].max()
    pad = (hi - lo) * 0.05 if hi > lo else 0.01
    lo -= pad
    hi += pad

    # Get all classifiers
    all_clfs = sorted(sub["classifier"].unique())

    # marker + color mapping
    marker_cycle = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>']
    color_cycle  = plt.cm.tab10(np.linspace(0, 1, len(all_clfs)))

    clf_marker = {clf: marker_cycle[i % len(marker_cycle)] for i, clf in enumerate(all_clfs)}
    clf_color  = {clf: color_cycle[i % len(color_cycle)]  for i, clf in enumerate(all_clfs)}

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(11, 12),
        sharex=True,
        sharey=True
    )

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]

            df = sub[(sub.dataset == dataset) & (sub.model == model)]
            if df.empty:
                ax.set_visible(False)
                continue

            # Pivot means and stds
            pivot_mean = df.pivot(index="classifier", columns="regime", values="mean")
            if not {"trtr", "tstr"}.issubset(pivot_mean.columns):
                ax.set_visible(False)
                continue

            if use_errorbars:
                pivot_std = df.pivot(index="classifier", columns="regime", values="std")

            # Scatter (or errorbar) per classifier
            for clf, row in pivot_mean.iterrows():
                x = row["trtr"]
                y = row["tstr"]

                if use_errorbars:
                    sx = pivot_std.loc[clf, "trtr"] if "trtr" in pivot_std.columns else np.nan
                    sy = pivot_std.loc[clf, "tstr"] if "tstr" in pivot_std.columns else np.nan
                    ax.errorbar(
                        x, y,
                        xerr=sx, yerr=sy,
                        fmt=clf_marker[clf],
                        color=clf_color[clf],
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        markersize=7,
                        elinewidth=1,
                        capsize=2,
                        linestyle="none",
                    )
                else:
                    ax.scatter(
                        x, y,
                        marker=clf_marker[clf],
                        color=clf_color[clf],
                        edgecolors="black",
                        linewidths=0.5,
                        s=60
                    )

            # 45-degree diagonal
            ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="tab:blue", alpha=0.6)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            if i == 0:
                ax.set_title(model, fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{dataset.capitalize()}\nTSTR {metric}", fontsize=10)

    # Global axis labels
    fig.text(0.5, 0.04, f"{metric} – TRTR (train on real, test on real)", ha="center", fontsize=12)
    fig.text(0.04, 0.5, f"{metric} – TSTR (train on synthetic, test on real)",
             va="center", rotation="vertical", fontsize=12)

    # Legend
    legend_handles = [
        Line2D(
            [], [], linestyle='', marker=clf_marker[clf],
            markerfacecolor=clf_color[clf],
            markeredgecolor="black",
            markersize=8,
            label=clf
        )
        for clf in all_clfs
    ]

    fig.legend(
        handles=legend_handles,
        title="Classifier",
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        frameon=True,
        borderaxespad=0.4
    )

    fig.suptitle(f"TSTR vs TRTR – {metric} (mean{' ± std' if use_errorbars else ''})", fontsize=14, y=0.97)
    fig.tight_layout(rect=[0.06, 0.06, 0.86, 0.94])

    return fig

def multi_scatter_trtr_vs_tstr(utility_long, metric="roc_auc"):
    datasets = ["credit", "cardio", "adult"]
    models   = ["CTGAN", "TVAE"]

    sub = utility_long[utility_long["metric"] == metric]
    if sub.empty:
        raise ValueError(f"No data for metric '{metric}'")

    # Global axis range
    lo = sub["value"].min()
    hi = sub["value"].max()
    pad = (hi - lo) * 0.05 if hi > lo else 0.01
    lo -= pad
    hi += pad

    # Get all classifiers
    all_clfs = sorted(sub["classifier"].unique())

    # marker + color mapping
    marker_cycle = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>']
    color_cycle  = plt.cm.tab10(np.linspace(0, 1, len(all_clfs)))

    clf_marker = {
        clf: marker_cycle[i % len(marker_cycle)]
        for i, clf in enumerate(all_clfs)
    }
    clf_color = {
        clf: color_cycle[i % len(color_cycle)]
        for i, clf in enumerate(all_clfs)
    }

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(11, 12),
        sharex=True,
        sharey=True
    )

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]

            df = sub[(sub.dataset == dataset) & (sub.model == model)]
            if df.empty:
                ax.set_visible(False)
                continue

            pivot = df.pivot(index="classifier", columns="regime", values="value")
            if not {"trtr", "tstr"}.issubset(pivot.columns):
                ax.set_visible(False)
                continue

            # Scatter with custom marker + color
            for clf, row in pivot.iterrows():
                ax.scatter(
                    row["trtr"], row["tstr"],
                    marker=clf_marker[clf],
                    color=clf_color[clf],
                    edgecolors="black",
                    linewidths=0.5,
                    s=60
                )

            # 45-degree diagonal
            ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="tab:blue", alpha=0.6)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            if i == 0:
                ax.set_title(model, fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{dataset.capitalize()}\nTSTR {metric}", fontsize=10)

    # Global axis labels
    fig.text(0.5, 0.04,
             f"{metric} – TRTR (train on real, test on real)",
             ha="center", fontsize=12)
    fig.text(0.04, 0.5,
             f"{metric} – TSTR (train on synthetic, test on real)",
             va="center", rotation="vertical", fontsize=12)

    # Legend Handles: now with correct shapes AND correct colors
    legend_handles = [
        Line2D(
            [], [], linestyle='', marker=clf_marker[clf],
            markerfacecolor=clf_color[clf],
            markeredgecolor="black",
            markersize=8,
            label=clf
        )
        for clf in all_clfs
    ]

    fig.legend(
        handles=legend_handles,
        title="Classifier",
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        frameon=True,
        borderaxespad=0.4
    )

    # Leave space for legend
    fig.suptitle(f"TSTR vs TRTR – {metric}", fontsize=14, y=0.97)
    fig.tight_layout(rect=[0.06, 0.06, 0.86, 0.94])

    return fig

def build_attr_clf_long(results):
    rows = []
    for key, attrs in results.items():
        # key pattern: '<dataset>_<data_type>_<model>' e.g. 'credit_real_ctgan'
        dataset, data_type, model = key.split("_")
        model = model.upper()  # CTGAN / TVAE

        for attr_name, metrics in attrs.items():
            rows.append({
                "dataset": dataset,                 # credit / adult / cardio
                "model": model,                    # CTGAN / TVAE
                "data_type": data_type,            # 'real' or 'synth'
                "attribute": attr_name,            # PAY_0, income, etc.
                "balanced_accuracy": metrics["Balanced Accuracy"],
                "macro_f1": metrics["Macro F1"],
                "accuracy": metrics["Accuracy"],
            })
    return pd.DataFrame(rows)

def make_attr_clf_grid(attr_clf_long, metric="balanced_accuracy"):
    datasets = ["credit", "cardio", "adult"]
    models = ["CTGAN", "TVAE"]

    pretty_metric = {
        "balanced_accuracy": "Balanced accuracy",
        "macro_f1": "Macro F1",
        "accuracy": "Accuracy",
    }.get(metric, metric)

    # Per-dataset limits
    dataset_limits = {}
    for ds in datasets:
        vals = attr_clf_long[attr_clf_long["dataset"] == ds][metric].values
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * 0.15 if hi > lo else 0.01
        dataset_limits[ds] = (lo - pad, hi + pad)

    fig, axes = plt.subplots(3, 2, figsize=(8, 10), sharex=False, sharey=False)

    for i, dataset in enumerate(datasets):
        lo, hi = dataset_limits[dataset]

        for j, model in enumerate(models):
            ax = axes[i, j]

            sub = attr_clf_long[
                (attr_clf_long.dataset == dataset) &
                (attr_clf_long.model == model)
            ]

            real = sub[sub.data_type == "real"].set_index("attribute")[metric]
            synth = sub[sub.data_type == "synth"].set_index("attribute")[metric]

            # Scatter points
            ax.scatter(real, synth, s=55, alpha=0.85)

            # ---- smarter labeling: offset labels to avoid collapsing ----
            # sort by synth value so nearby points get different offsets
            pts = [(attr, real[attr], synth[attr]) for attr in real.index]
            pts_sorted = sorted(pts, key=lambda t: (t[2], t[1]))

            # vertical offsets in points (spread between -6 and +6)
            if len(pts_sorted) > 1:
                offsets = np.linspace(-6, 6, len(pts_sorted))
            else:
                offsets = [0.0]

            for (attr, x, y), dy in zip(pts_sorted, offsets):
                # put label slightly to the right, vertically offset by dy
                ax.annotate(
                    attr,
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, dy),
                    ha="left",
                    va="center",
                    fontsize=7
                )
            # -------------------------------------------------------------

            # Diagonal reference line
            ax.plot([lo, hi], [lo, hi], ls="--", lw=1,
                    color="tab:blue", alpha=0.5)

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            if i == 0:
                ax.set_title(model, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{dataset.capitalize()}\nSynth {pretty_metric}",
                              fontsize=9)

            ax.grid(True, ls=":", lw=0.5, alpha=0.5)

    fig.text(0.5, 0.04,
             f"{pretty_metric} – attack trained on real data",
             ha="center", fontsize=11)
    fig.text(0.02, 0.5,
             f"{pretty_metric} – attack trained on synthetic data",
             va="center", rotation="vertical", fontsize=11)

    fig.suptitle(f"Attribute inference (classification) – {pretty_metric}",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.05, 0.06, 1, 0.97])

    return fig

def build_attr_reg_long(results):
    rows = []
    for key, attrs in results.items():
        dataset, data_type, model = key.split("_")
        model = model.upper()

        for attr_name, metrics in attrs.items():
            rows.append({
                "dataset": dataset,
                "model": model,
                "data_type": data_type,       # real / synth
                "attribute": attr_name,
                "mse": metrics["MSE"],
                "rmse": metrics["RMSE"],
                "r2": metrics["R2"],
            })
    return pd.DataFrame(rows)

def scatter_real_vs_synth_reg_ax(ax, df, dataset, model, metric="r2",
                                 lo=None, hi=None):
    """Single panel: real vs synth regression metric for one dataset+model."""
    sub = df[(df["dataset"] == dataset) & (df["model"] == model)]
    pivot = sub.pivot(index="attribute", columns="data_type", values=metric)

    if not {"real", "synth"}.issubset(pivot.columns):
        ax.text(0.5, 0.5, "missing real/synth",
                ha="center", va="center", transform=ax.transAxes)
        return

    real  = pivot["real"].values
    synth = pivot["synth"].values
    attrs = pivot.index.values

    # scatter points
    ax.scatter(real, synth, s=45, alpha=0.9)

    # --- smarter labels: offset vertically so they don't collapse ---
    pts = [(attr, x, y) for attr, x, y in zip(attrs, real, synth)]
    pts_sorted = sorted(pts, key=lambda t: (t[2], t[1]))  # by y, then x

    if len(pts_sorted) > 1:
        offsets = np.linspace(-6, 6, len(pts_sorted))  # in points
    else:
        offsets = [0.0]

    for (attr, x, y), dy in zip(pts_sorted, offsets):
        ax.annotate(
            attr,
            (x, y),
            textcoords="offset points",
            xytext=(4, dy),  # right + vertical offset
            ha="left",
            va="center",
            fontsize=7
        )
    # ----------------------------------------------------------------

    # diagonal reference line & limits
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1,
            color="tab:blue", alpha=0.6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, linestyle=":", alpha=0.4)

def make_attr_reg_grid(attr_reg_long, metric="r2"):
    datasets = ["credit", "cardio", "adult"]
    models   = ["CTGAN", "TVAE"]

    # pretty metric name
    pretty_metric = {"r2": r"$R^2$"}.get(metric, metric)

    # per-dataset axis limits (shared across CTGAN/TVAE in the same row)
    dataset_limits = {}
    for ds in datasets:
        vals = attr_reg_long[attr_reg_long["dataset"] == ds][metric].values
        lo = vals.min()
        hi = vals.max()
        pad = (hi - lo) * 0.15 if hi > lo else 0.01
        dataset_limits[ds] = (lo - pad, hi + pad)

    fig, axes = plt.subplots(3, 2, figsize=(8, 10),
                             sharex=False, sharey=False)

    for i, dataset in enumerate(datasets):
        lo, hi = dataset_limits[dataset]

        for j, model in enumerate(models):
            ax = axes[i, j]
            scatter_real_vs_synth_reg_ax(
                ax, attr_reg_long, dataset, model, metric, lo, hi
            )

            if i == 0:
                ax.set_title(model, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{dataset.capitalize()}\nSynth {pretty_metric}",
                              fontsize=9)

    fig.text(0.5, 0.04,
             f"{pretty_metric} – attack trained on real data",
             ha="center", fontsize=11)
    fig.text(0.02, 0.5,
             f"{pretty_metric} – attack trained on synthetic data",
             va="center", rotation="vertical", fontsize=11)

    fig.suptitle(f"Attribute inference (regression) – {pretty_metric}",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0.05, 0.06, 1, 0.96])
    return fig

def build_anonymeter_df(results):
    rows = []
    for key, runs in results.items():
        # key like 'credit CTGAN'
        dataset, generator = key.split()
        generator = generator.upper()  # CTGAN / TVAE

        for run_name, attacks in runs.items():
            run_id = int(run_name.split("_")[-1])  # e.g. run_0 -> 0

            for attack_type, res in attacks.items():
                # res is AnonymeterResults(...)
                rows.append({
                    "dataset": dataset,                     # credit / adult / cardio
                    "generator": generator,                 # CTGAN / TVAE
                    "run": run_id,
                    "attack": attack_type,                  # singling_univariate, ...
                    "attacks_numbers": res.attacks_numbers,
                    "attacks_succeeded": res.attacks_succeeded,
                    "privacy_risk_original": res.privacy_risk_original,
                    "privacy_risk_control": res.privacy_risk_control,
                    "privacy_risk_naive": res.privacy_risk_naive,
                    "specific_privacy": res.specific_privacy,
                    "specific_privacy_ci": res.specific_privacy_ci,
                })
    df = pd.DataFrame(rows)
    df["excess_risk"] = df["privacy_risk_original"] - df["privacy_risk_control"]
    return df

def plot_anonymeter_summary(df_summary, metric="mean_risk"):
    attacks = ["singling_univariate", "singling_multivariate", "linkability_attacks"]
    attack_labels = ["Singling (uni-variate)", "Singling (multi-variate)", "Linkability"]
    datasets = ["credit", "adult", "cardio"]
    generators = ["CTGAN", "TVAE"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)

    plt.subplots_adjust(top=0.78, wspace=0.25)

    for i, (ax, attack, label) in enumerate(zip(axes, attacks, attack_labels)):
        sub = df_summary[df_summary["attack"] == attack]
        sub = sub.set_index(["dataset", "generator"])

        x = np.arange(len(datasets))
        width = 0.35

        means_ctgan, means_tvae = [], []
        stds_ctgan, stds_tvae = [], []

        for ds in datasets:

            if (ds, "CTGAN") in sub.index:
                row_c = sub.loc[(ds, "CTGAN")]
                means_ctgan.append(row_c[metric])
                stds_ctgan.append(row_c["std_risk"])
            else:
                means_ctgan.append(np.nan); stds_ctgan.append(0.0)

            if (ds, "TVAE") in sub.index:
                row_t = sub.loc[(ds, "TVAE")]
                means_tvae.append(row_t[metric])
                stds_tvae.append(row_t["std_risk"])
            else:
                means_tvae.append(np.nan); stds_tvae.append(0.0)

        # Only give labels on the FIRST subplot → avoids duplicated legend entries
        if i == 0:
            label_ctgan = "CTGAN"
            label_tvae = "TVAE"
        else:
            label_ctgan = None
            label_tvae = None

        ax.bar(x - width/2, means_ctgan, width, yerr=stds_ctgan,
               label=label_ctgan, capsize=3)
        ax.bar(x + width/2, means_tvae, width, yerr=stds_tvae,
               label=label_tvae, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([ds.capitalize() for ds in datasets])
        ax.set_title(label, pad=12)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Privacy risk (Anonymeter)")

    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    return fig


def anonymeter_pretty_tables(results_dict, float_fmt="{:.4f}"):
    """
    Build pretty tables from:
      results_dict[dataset_model][run_id] = PrivacyAnonymeterEvaluationResults(
          singling_univariate=AnonymeterResults(...),
          singling_multivariate=AnonymeterResults(...),
          linkability_attacks=AnonymeterResults(...)
      )

    Returns:
      summary_df: mean±std table by dataset_model
      runs_long_df: long table (one row per run)
    """

    def _extract(ar):
        # ar is AnonymeterResults
        return {
            "attacks_numbers": getattr(ar, "attacks_numbers", np.nan),
            "attacks_succeeded": getattr(ar, "attacks_succeeded", np.nan),
            "privacy_risk_original": getattr(ar, "privacy_risk_original", np.nan),
            "privacy_risk_control": getattr(ar, "privacy_risk_control", np.nan),
            "privacy_risk_naive": getattr(ar, "privacy_risk_naive", np.nan),
            "specific_privacy": getattr(ar, "specific_privacy", np.nan),
            "specific_privacy_ci": getattr(ar, "specific_privacy_ci", np.nan),
        }

    rows = []
    for dataset_model, runs in results_dict.items():
        # split "credit CTGAN" -> dataset="credit", model="CTGAN" (fallback safe)
        parts = dataset_model.rsplit(" ", 1)
        dataset = parts[0] if len(parts) == 2 else dataset_model
        model = parts[1] if len(parts) == 2 else ""

        for run_id, evalres in runs.items():
            u = _extract(evalres.singling_univariate)
            m = _extract(evalres.singling_multivariate)
            l = _extract(evalres.linkability_attacks)

            row = {
                "dataset_model": dataset_model,
                "dataset": dataset,
                "model": model,
                "run": run_id,
            }

            # prefix columns to keep them clean
            row.update({f"uni_{k}": v for k, v in u.items()})
            row.update({f"multi_{k}": v for k, v in m.items()})
            row.update({f"link_{k}": v for k, v in l.items()})

            rows.append(row)

    runs_long_df = pd.DataFrame(rows)

    # ---- summary: mean ± std across runs (numeric columns only) ----
    num_cols = runs_long_df.select_dtypes(include=[np.number]).columns.tolist()

    grp = runs_long_df.groupby(["dataset_model", "dataset", "model"], dropna=False)

    mean_df = grp[num_cols].mean().add_suffix("_mean")
    std_df  = grp[num_cols].std(ddof=1).add_suffix("_std")

    summary_df = pd.concat([mean_df, std_df], axis=1).reset_index()

    # ---- format selected “headline” metrics for a pretty view ----
    headline = [
        ("uni_privacy_risk_original", "Uni PR (orig)"),
        ("uni_privacy_risk_control",  "Uni PR (ctrl)"),
        ("uni_privacy_risk_naive",    "Uni PR (naive)"),
        ("uni_specific_privacy",      "Uni SpecificPrivacy"),
        ("multi_privacy_risk_original","Multi PR (orig)"),
        ("multi_privacy_risk_control", "Multi PR (ctrl)"),
        ("multi_privacy_risk_naive",   "Multi PR (naive)"),
        ("multi_specific_privacy",     "Multi SpecificPrivacy"),
        ("link_privacy_risk_original", "Link PR (orig)"),
        ("link_privacy_risk_control",  "Link PR (ctrl)"),
        ("link_privacy_risk_naive",    "Link PR (naive)"),
        ("link_specific_privacy",      "Link SpecificPrivacy"),
    ]

    pretty = summary_df[["dataset_model", "dataset", "model"]].copy()
    for base, label in headline:
        mcol = f"{base}_mean"
        scol = f"{base}_std"
        if mcol in summary_df.columns and scol in summary_df.columns:
            pretty[label] = summary_df.apply(
                lambda r: f"{float_fmt.format(r[mcol])} ± {float_fmt.format(r[scol])}",
                axis=1
            )

    # Optional: order rows nicely (dataset then model)
    pretty = pretty.sort_values(["dataset", "model"]).reset_index(drop=True)

    return pretty, runs_long_df, summary_df
