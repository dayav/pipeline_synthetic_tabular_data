"""
HDF5 I/O helpers for utility-evaluation artefacts.
"""

from __future__ import annotations

from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd

from data_synthesizer.pipeline.pipeline_results import UtilityEvaluationResults

UTILITY_GROUP = "utility_results"          # fixed typo

# ────────────────────────────────────────────
#  Field specification
# ────────────────────────────────────────────
_DF_FIELDS: tuple[str, ...] = (
    "train_synthetic_test_real_results",
    "train_real_test_real_results",
)

_SCALAR_FIELDS: tuple[str, ...] = (
    "accuracy_mean_tstr",
    "accuracy_mean_trtr",
    "rbo_permutation_importance",
    "spearman_permutation_importance",
    "kendall_permutation_importance",
    "rbo_shap",
    "spearman_shap",
    "kendall_shap",
)

# Groups that hold a   Dict[str, ndarray]
_DICT_GROUPS: tuple[str, ...] = (
    "confusion_matrix_tstr",
    "confusion_matrix_trtr",
    "permutation_importance_tstr",
    "permutation_importance_trtr",
    "shap_importance_tstr",
    "shap_importance_trtr",
)

_CONFUSION_MATRIX_GROUPS: tuple[str, ...] = (
    "confusion_matrix_tstr",
    "confusion_matrix_trtr",
)

_SERIES_GROUPS: tuple[str, ...] = (
    "permutation_importance_tstr",
    "permutation_importance_trtr",
    "shap_importance_tstr",
    "shap_importance_trtr",
)

# ────────────────────────────────────────────
#  Low-level helpers
# ────────────────────────────────────────────
def _write_dataset(grp: h5py.Group, name: str, data: Any) -> None:
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=data)


def _save_dataframe(hdf: h5py.File, path: str, df: pd.DataFrame) -> None:
    _write_dataset(hdf, path, df.values)
    hdf[path].attrs.update(columns=df.columns.to_list(), index=df.index.to_list())


def _load_dataframe(hdf: h5py.File, path: str) -> pd.DataFrame:
    ds = hdf[path]
    return pd.DataFrame(ds[()], columns=ds.attrs["columns"], index=ds.attrs["index"])


def _save_series(grp: h5py.Group, name: str, series: pd.Series) -> None:
    _write_dataset(grp, name, series.to_numpy())
    grp[name].attrs.update(index=series.index.to_list())


def _load_series(ds: h5py.Dataset) -> Any:
    data = ds[()]
    if "index" in ds.attrs:
        return pd.Series(data, index=ds.attrs["index"])
    return data


# ────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────
def save_utility_results(res: UtilityEvaluationResults, path: str) -> None:
    """Persist the full `UtilityEvaluationResults` structure."""
    with h5py.File(f"{path}.h5", "a") as hdf:
        grp = hdf.require_group(UTILITY_GROUP)

        # DataFrames
        for name in _DF_FIELDS:
            _save_dataframe(hdf, f"{UTILITY_GROUP}/{name}", res[name])

        # Scalars
        for name in _SCALAR_FIELDS:
            _write_dataset(grp, name, res[name])

        # Dict-of-arrays groups
        for group in _DICT_GROUPS:
            g = grp.require_group(group)
            for key, arr in res[group].items():
                if group in _CONFUSION_MATRIX_GROUPS and isinstance(arr, dict):
                    if key in g:
                        del g[key]
                    sub = g.create_group(key)
                    for stat_name, stat_arr in arr.items():
                        _write_dataset(sub, stat_name, stat_arr)
                elif group in _SERIES_GROUPS and isinstance(arr, pd.Series):
                    _save_series(g, key, arr)
                else:
                    _write_dataset(g, key, arr)


def load_utility_results(path: str) -> UtilityEvaluationResults | None:
    """Load the previously stored `UtilityEvaluationResults`."""
    with h5py.File(f"{path}.h5", "r") as hdf:
        root_ds = f"{UTILITY_GROUP}/train_synthetic_test_real_results"
        if root_ds not in hdf:
            return None

        out: Dict[str, Any] = {}

        # DataFrames
        for name in _DF_FIELDS:
            out[name] = _load_dataframe(hdf, f"{UTILITY_GROUP}/{name}")

        # Scalars
        grp = hdf[UTILITY_GROUP]
        out.update({name: grp[name][()] for name in _SCALAR_FIELDS})

        # Dict-of-arrays groups
        for group in _DICT_GROUPS:
            sub = {}
            for key, ds in grp[group].items():
                if isinstance(ds, h5py.Group):
                    nested = {stat: child[()] for stat, child in ds.items()}
                    sub[key] = nested
                else:
                    loaded = _load_series(ds)
                    if group in _CONFUSION_MATRIX_GROUPS:
                        sub[key] = {"mean": loaded, "std": np.zeros_like(loaded)}
                    else:
                        sub[key] = loaded
            out[group] = sub

    # mypy-friendly cast
    return UtilityEvaluationResults(**out)   # type: ignore[arg-type]
