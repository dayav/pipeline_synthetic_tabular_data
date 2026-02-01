"""
HDF5 I/O helpers for privacy-related evaluation artefacts.
"""

from __future__ import annotations

from typing import Dict
from pathlib import Path

import h5py
import pandas as pd

from data_evaluator.privacy_evaluation.privacy_evaluator_anonymeter import AnonymeterResults
from data_synthesizer.pipeline.pipeline_results import (
    PrivacyEvaluationResults,
    PrivacyAnonymeterEvaluationResults,
)

# ────────────────────────────────────────────
#  Constants
# ────────────────────────────────────────────
PRIVACY_RESULTS_GROUP      = "privacy_results"
PRIVACY_ANONYMETER_GROUP   = "privacy_anonymeter_results"

_ANONYMETER_METRICS: tuple[str, ...] = (
    "attacks_numbers",
    "attacks_succeeded",
    "privacy_risk_original",
    "privacy_risk_control",
    "privacy_risk_naive",
    "specific_privacy",
)


# ────────────────────────────────────────────
#  Generic helpers
# ────────────────────────────────────────────
def _write_dataset(grp: h5py.Group, name: str, data) -> None:
    """(Re)create *name* inside *grp* with *data*."""
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=data)


def _write_metrics_group(grp: h5py.Group, name: str, values: dict) -> None:
    """
    Persist metrics per attribute. Supports both scalar values and nested dicts
    (metric_name -> value) to match result_classification/result_regression.
    """
    if name in grp:
        del grp[name]
    sect_grp = grp.create_group(name)
    for attr, metrics in values.items():
        if isinstance(metrics, dict):
            # Clean existing attr group if present
            if attr in sect_grp:
                del sect_grp[attr]
            attr_grp = sect_grp.create_group(attr)
            for metric_name, metric_value in metrics.items():
                _write_dataset(attr_grp, metric_name, metric_value)
        else:
            _write_dataset(sect_grp, attr, metrics)


def _write_prediction_group(grp: h5py.Group, name: str, values: dict) -> None:
    """Persist raw prediction arrays under *name* if present."""
    if not values:
        return
    if name in grp:
        del grp[name]
    sect_grp = grp.create_group(name)
    for attr, preds in values.items():
        arr = pd.Series(preds).to_numpy()
        if arr.dtype == object:
            arr = arr.astype(h5py.string_dtype(encoding="utf-8"))
        _write_dataset(sect_grp, attr, arr)


def _read_dataset(ds: h5py.Dataset):
    """Read a dataset, decoding strings when needed."""
    if h5py.check_string_dtype(ds.dtype) is not None:
        return ds.asstr()[()]
    return ds[()]


def _load_metrics_group(grp: h5py.Group, name: str) -> dict:
    """Load metrics group back to a python dict."""
    if name not in grp:
        return {}
    section = grp[name]
    out: dict = {}
    for attr, node in section.items():
        if isinstance(node, h5py.Group):
            out[attr] = {metric: _read_dataset(node[metric]) for metric in node}
        else:
            out[attr] = _read_dataset(node)
    return out


def _save_dataframe(hdf: h5py.File, path: str, df: pd.DataFrame) -> None:
    """Store a DataFrame’s values + column/index metadata."""
    _write_dataset(hdf, path, df.values)
    hdf[path].attrs.update(columns=df.columns.to_list(), index=df.index.to_list())


def _load_dataframe(hdf: h5py.File, path: str) -> pd.DataFrame:
    ds = hdf[path]
    return pd.DataFrame(ds[()], columns=ds.attrs["columns"], index=ds.attrs["index"])


# ────────────────────────────────────────────
#  Anonymeter blocks
# ────────────────────────────────────────────
def _save_anonymeter_block(hdf: h5py.File, base: str, res: AnonymeterResults) -> None:
    grp = hdf.require_group(base)
    for m in _ANONYMETER_METRICS:
        value = getattr(res, m) if hasattr(res, m) else res[m]  # dataclass or dict
        _write_dataset(grp, m, value)


def _load_anonymeter_block(hdf: h5py.File, base: str) -> AnonymeterResults:
    return AnonymeterResults(**{m: hdf[f"{base}/{m}"][()] for m in _ANONYMETER_METRICS})


# ────────────────────────────────────────────
#  Public I/O – anonymeter
# ────────────────────────────────────────────
def save_privacy_anonymeter_results(
    results: Dict[str, PrivacyAnonymeterEvaluationResults],
    path: str,
) -> None:
    """Persist a mapping of anonymeter results (new schema)."""
    with h5py.File(f"{path}.h5", "a") as hdf:
        for key, block in results.items():
            prefix = f"{PRIVACY_ANONYMETER_GROUP}/{key}"
            _save_anonymeter_block(hdf, f"{prefix}/singling_univariate",   block["singling_univariate"])
            _save_anonymeter_block(hdf, f"{prefix}/singling_multivariate", block["singling_multivariate"])
            _save_anonymeter_block(hdf, f"{prefix}/linkability_attacks",   block["linkability_attacks"])


def load_privacy_anonymeter_results(
    path: str,
) -> Dict[str, PrivacyAnonymeterEvaluationResults]:
    """Load every anonymeter result set stored under the root group."""
    out: Dict[str, PrivacyAnonymeterEvaluationResults] = {}
    with h5py.File(f"{path}.h5", "r") as hdf:
        if PRIVACY_ANONYMETER_GROUP not in hdf:
            return out
        for key in hdf[PRIVACY_ANONYMETER_GROUP]:
            prefix = f"{PRIVACY_ANONYMETER_GROUP}/{key}"
            out[key] = PrivacyAnonymeterEvaluationResults(
                singling_univariate   = _load_anonymeter_block(hdf, f"{prefix}/singling_univariate"),
                singling_multivariate = _load_anonymeter_block(hdf, f"{prefix}/singling_multivariate"),
                linkability_attacks   = _load_anonymeter_block(hdf, f"{prefix}/linkability_attacks"),
            )
    return out


# ────────────────────────────────────────────
#  Public I/O – general privacy results
# ────────────────────────────────────────────
_SCALAR_PRIVACY_FIELDS: tuple[str, ...] = (
    "dissimilarity_synthetic_real",
    "dissimilarity_synthetic_test",
    "epsilon_dissimilarity_synthetic_real",
    "epsilon_dissimilarity_synthetic_test",
    "share",
    "epsilon",
)


def _resolve_privacy_path(path: str | Path) -> Path | None:
    """
    Best-effort resolution of the privacy results path.

    Accepts:
    - direct path to the .h5 file
    - base path without suffix (we append .h5)
    - folder/stem path (we try <folder>/<stem>_evaluation.h5)
    """
    base = Path(path)
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(p: Path) -> None:
        if p not in seen:
            candidates.append(p)
            seen.add(p)

    # direct .h5
    if base.suffix == ".h5":
        add_candidate(base)
    else:
        add_candidate(Path(f"{path}.h5"))  # preserves dots in folder names

        eval_base = base / f"{base.name}_evaluation"
        add_candidate(Path(f"{eval_base}.h5"))

        # legacy fallback where we replace the suffix (e.g. foo.txt -> foo.h5)
        if base.suffix:
            add_candidate(base.with_suffix(".h5"))
            add_candidate(eval_base.with_suffix(".h5"))

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def save_privacy_results(results: PrivacyEvaluationResults, path: str) -> None:
    """Store the full `PrivacyEvaluationResults` structure."""
    with h5py.File(f"{path}.h5", "a") as hdf:
        # DataFrames
        _save_dataframe(hdf, f"{PRIVACY_RESULTS_GROUP}/jensen_shanon_categorical",
                        results["jensen_shanon_categorical"])
        _save_dataframe(hdf, f"{PRIVACY_RESULTS_GROUP}/jensen_shanon_numerical",
                        results["jensen_shanon_numerical"])

        # Scalars
        grp = hdf.require_group(PRIVACY_RESULTS_GROUP)
        for name in _SCALAR_PRIVACY_FIELDS:
            _write_dataset(grp, name, results[name])

        # Nested dicts (metrics per attribute)
        _write_metrics_group(grp, "attribute_synthetic_prediction",
                             results.get("attribute_synthetic_prediction", {}))
        _write_metrics_group(grp, "attribute_real_prediction",
                             results.get("attribute_real_prediction", {}))

        # Raw predictions (optional)
        _write_prediction_group(
            grp,
            "attribute_synthetic_prediction_values",
            results.get("attribute_synthetic_prediction_values", {}),
        )
        _write_prediction_group(
            grp,
            "attribute_real_prediction_values",
            results.get("attribute_real_prediction_values", {}),
        )


def load_privacy_results(path: str) -> PrivacyEvaluationResults | None:
    """Retrieve previously stored `PrivacyEvaluationResults`."""
    resolved = _resolve_privacy_path(path)
    if resolved is None:
        return None
    with h5py.File(resolved, "r") as hdf:
        root = f"{PRIVACY_RESULTS_GROUP}/jensen_shanon_categorical"
        if root not in hdf:
            return None

        jsc = _load_dataframe(hdf, f"{PRIVACY_RESULTS_GROUP}/jensen_shanon_categorical")
        jsn = _load_dataframe(hdf, f"{PRIVACY_RESULTS_GROUP}/jensen_shanon_numerical")

        grp = hdf[PRIVACY_RESULTS_GROUP]
        scalars = {n: grp[n][()] for n in _SCALAR_PRIVACY_FIELDS}

        synth_pred = _load_metrics_group(grp, "attribute_synthetic_prediction")
        real_pred  = _load_metrics_group(grp, "attribute_real_prediction")
        synth_pred_values = (
            {k: _read_dataset(grp["attribute_synthetic_prediction_values"][k])
             for k in grp["attribute_synthetic_prediction_values"]}
            if "attribute_synthetic_prediction_values" in grp else {}
        )
        real_pred_values = (
            {k: _read_dataset(grp["attribute_real_prediction_values"][k])
             for k in grp["attribute_real_prediction_values"]}
            if "attribute_real_prediction_values" in grp else {}
        )

    return PrivacyEvaluationResults(
        jensen_shanon_categorical=jsc,
        jensen_shanon_numerical=jsn,
        attribute_synthetic_prediction=synth_pred,
        attribute_real_prediction=real_pred,
        attribute_synthetic_prediction_values=synth_pred_values,
        attribute_real_prediction_values=real_pred_values,
        **scalars,
    )
