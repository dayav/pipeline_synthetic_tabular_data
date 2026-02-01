"""
HDF5 I/O helpers for resemblance-evaluation artefacts.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import h5py
import pandas as pd

from data_synthesizer.pipeline.pipeline_results import ResemblanceEvaluationResults

RESEMBLANCE_GROUP = "resemblance_results"

# ────────────────────────────────────────────
#  Field specification
# ────────────────────────────────────────────
# kind: "df" = DataFrame, "series" = Series, "scalar" = number / float
_RESEMBLANCE_SPEC: dict[str, tuple[tuple[str, str], ...]] = {
    "categorical_univariate": (
        ("chi_test", "df"),
        ("jensen_shanon", "df"),
        ("jensen_shanon_data_mean", "series"),
        ("mode_collapse_dict", "dict"),
    ),
    "numerical_univariate": (
        ("univariate_num_s", "df"),
        ("univariate_num_js", "df"),
        ("univariate_num_js_data_mean", "series"),
    ),
    "categorical_multivariate": (
        ("cramer_real", "df"),
        ("cramer_synth", "df"),
        ("diff_norm_cramer", "scalar"),
        ("cramer_spearman_correlation_coefficient", "scalar"),
    ),
    "numerical_multivariate": (
        ("pearson_real", "df"),
        ("pearson_synth", "df"),
        ("pearson_norm_diff", "scalar"),
        ("pearson_spearman_correlation_coefficient", "scalar"),
    ),
    "categorical_numerical_multivariate": (  # optional block
        ("corr_ratio_real", "df"),   # stored as .['corr']
        ("corr_ratio_synth", "df"),
        ("diff_norm_corr_ratio", "scalar"),
        ("corr_ratio_spearman_correlation_coefficient", "scalar"),
    ),
}

# ────────────────────────────────────────────
#  Low-level helpers
# ────────────────────────────────────────────
def _write_dataset(grp: h5py.Group, name: str, data: Any) -> None:
    """Create or overwrite *name* inside *grp* with *data*."""
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=data)


def _save_dataframe(hdf: h5py.File, path: str, df: pd.DataFrame) -> None:
    _write_dataset(hdf, path, df.values)
    hdf[path].attrs.update(columns=df.columns.to_list(), index=df.index.to_list())


def _load_dataframe(hdf: h5py.File, path: str) -> pd.DataFrame:
    ds = hdf[path]
    return pd.DataFrame(ds[()], columns=ds.attrs["columns"], index=ds.attrs["index"])


def _save_series(hdf: h5py.File, path: str, s: pd.Series) -> None:
    _write_dataset(hdf, path, s.values)
    hdf[path].attrs["index"] = s.index.to_list()


def _load_series(hdf: h5py.File, path: str) -> pd.Series:
    ds = hdf[path]
    return pd.Series(ds[()], index=ds.attrs["index"])


def _save_value(hdf: h5py.File, path: str, value: Any) -> None:
    """
    Persist *value* at *path* where kind is inferred from the value.
    Supports DataFrame, Series, scalar, **and now dict[str, scalar/ndarray]**.
    """
    if isinstance(value, pd.DataFrame):
        _save_dataframe(hdf, path, value)

    elif isinstance(value, pd.Series):
        _save_series(hdf, path, value)

    elif isinstance(value, dict):                        
        # ensure <path> is a *group*
        if path in hdf and isinstance(hdf[path], h5py.Dataset):
            del hdf[path]                               
        grp = hdf.require_group(path)

        for k, v in value.items():                      
            _write_dataset(grp, k, v)

    else:  # scalar / NumPy number
        _write_dataset(hdf, path, value)


def _load_value(hdf: h5py.File, path: str, kind: str):
    if kind == "df":
        return _load_dataframe(hdf, path)
    if kind == "series":
        return _load_series(hdf, path)
    if kind == "dict":
        return {k: hdf[f"{path}/{k}"][()].tolist() for k in hdf[path]}
    return hdf[path][()]  # scalar


# ────────────────────────────────────────────
#  Public API
# ────────────────────────────────────────────
def save_resemblance_results(
    res: ResemblanceEvaluationResults,
    path: str,
) -> None:
    """Persist the full resemblance-evaluation structure."""
    with h5py.File(f"{path}.h5", "a") as hdf:
        for block, fields in _RESEMBLANCE_SPEC.items():
            block_data = res[block]
            if block_data is None:              # optional mixed-type block
                continue

            for name, kind in fields:
                if block == "categorical_numerical_multivariate" and name.startswith("corr_ratio_") and name != 'corr_ratio_spearman_correlation_coefficient':
                    # nested structure: {'corr_ratio_real': {'corr': DataFrame}}
                    value = block_data[name]["corr"]
                else:
                    value = block_data[name]

                _save_value(
                    hdf, f"{RESEMBLANCE_GROUP}/{block}/{name}", value
                )
        

def _resolve_resemblance_path(path: str | Path) -> Path | None:
    """
    Best-effort resolution of the resemblance results path.

    Accepts:
    - direct path to the .h5 file
    - base path without suffix (we append .h5)
    - folder/stem path (<folder>/<stem>_evaluation.h5)
    """
    base = Path(path)
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(p: Path) -> None:
        if p not in seen:
            candidates.append(p)
            seen.add(p)

    if base.suffix == ".h5":
        add_candidate(base)
    else:
        add_candidate(Path(f"{path}.h5"))  # preserves dots in base names

        eval_base = base / f"{base.name}_evaluation"
        add_candidate(Path(f"{eval_base}.h5"))

        # legacy fallbacks where we replace the suffix (e.g. foo.txt -> foo.h5)
        if base.suffix:
            add_candidate(base.with_suffix(".h5"))
            add_candidate(eval_base.with_suffix(".h5"))

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_resemblance_results(path: str) -> ResemblanceEvaluationResults | None:
    """Read the resemblance-evaluation artefacts back from disk."""
    out: dict[str, Any] = {}

    resolved = _resolve_resemblance_path(path)
    if resolved is None:
        return None

    with h5py.File(resolved, "r") as hdf:
        root_path = f"{RESEMBLANCE_GROUP}/categorical_univariate/chi_test"
        if root_path not in hdf:            # nothing persisted yet
            return None

        for block, fields in _RESEMBLANCE_SPEC.items():
            block_path = f"{RESEMBLANCE_GROUP}/{block}"
            if block_path not in hdf:       # optional block missing
                out[block] = None
                continue

            sub_dict: dict[str, Any] = {}
            for name, kind in fields:
                full_path = f"{block_path}/{name}"
                if name.startswith("corr_ratio_"):  # nested structure round-trip
                    sub_dict[name] = {"corr": _load_value(hdf, full_path, kind)}
                else:
                    sub_dict[name] = _load_value(hdf, full_path, kind)

            out[block] = sub_dict

    # mypy-friendly cast
    return ResemblanceEvaluationResults(**out)  # type: ignore[arg-type]
