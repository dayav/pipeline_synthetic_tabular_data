"""
High-level helpers to persist or restore complete PipelineResults
(generation + evaluation) in a folder structure:

    <folder>/
        <folder>_generation.h5
        <folder>_generation_mode_collapse_corrected.h5
        <folder>_generation_model.pkl
        <folder>_evaluation.h5
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping, MutableMapping

import h5py
import pandas as pd

from data_synthesizer.pipeline.pipeline_results import (
    GenerationResults,
    PipelineResults,
)
from data_synthesizer.pipeline.storages.privacy_results_storage import (
    load_privacy_anonymeter_results,
    load_privacy_results,
    save_privacy_anonymeter_results,
    save_privacy_results,
)
from data_synthesizer.pipeline.storages.utility_results_storage import (
    load_utility_results,
    save_utility_results,
)
from data_synthesizer.pipeline.storages.resemblance_results_storage import (
    load_resemblance_results,
    save_resemblance_results,
)

# ────────────────────────────────────────────
#  Dispatch maps
# ────────────────────────────────────────────
_SAVE_DISPATCH: Mapping[str, callable] = {
    "resemblance_evaluation_results": save_resemblance_results,
    "utility_evaluation_results": save_utility_results,
    "privacy_evaluation_results": save_privacy_results,
    "privacy_anonymeter_results": save_privacy_anonymeter_results,
}

_LOAD_DISPATCH: Mapping[str, callable] = {
    "resemblance_evaluation_results": load_resemblance_results,
    "utility_evaluation_results": load_utility_results,
    "privacy_evaluation_results": load_privacy_results,
    "privacy_anonymeter_results": load_privacy_anonymeter_results,
}

# ────────────────────────────────────────────
#  Generation results
# ────────────────────────────────────────────
def _generation_paths(base: Path) -> tuple[Path, Path, Path]:
    """Return HDF5, correction-flag HDF5, and model pickle paths."""
    return (
        base.with_suffix(".h5"),                       # synthetic_data
        base.with_name(f"{base.name}_mode_collapse_corrected.h5"),
        base.with_name(f"{base.name}_model.pkl"),
    )


def save_generation_results(res: GenerationResults, base: Path) -> None:
    """Persist synthetic data, mode-collapse flag, and (optionally) the model."""
    h5_path, flag_path, model_path = _generation_paths(base)

    # synthetic data (compressed)
    res["synthetic_data"].to_hdf(
        h5_path, key="synthetic_data", format="table",
        mode="w", complevel=9, complib="zlib"
    )

    # mode-collapse flag
    with h5py.File(flag_path, "w") as f:
        f.create_dataset("mode_collapse_corrected", data=res["mode_collapse_corrected"])

    # generator model (optional)
    if model := res.get("generator_model"):
        with model_path.open("wb") as fh:
            pickle.dump(model, fh)


def load_generation_results(base: Path, dst: MutableMapping) -> None:
    """Populate *dst['generation_results']* from files at *base*."""
    h5_path, flag_path, model_path = _generation_paths(base)
    if not h5_path.exists():
        return  # nothing saved

    gen = GenerationResults(
        synthetic_data=pd.read_hdf(h5_path, key="synthetic_data"),
        mode_collapse_corrected=bool(
            h5py.File(flag_path, "r")["mode_collapse_corrected"][()]
        ),
    )

    if model_path.exists():
        try:
            with model_path.open("rb") as fh:
                gen.generator_model = pickle.load(fh)
        except (pickle.UnpicklingError, OSError) as err:
            print(f"[warn] Could not load model pickle: {err}")

    dst["generation_results"] = gen  # type: ignore[index]


# ────────────────────────────────────────────
#  Evaluation results
# ────────────────────────────────────────────
def save_evaluation_results(res: PipelineResults, base: Path) -> None:
    """Iterate over evaluation sub-dicts and persist if present."""
    for key, fn in _SAVE_DISPATCH.items():
        payload = res.get(key)
        if payload is not None:
            fn(payload, str(base))  # storage helpers expect str-paths


def load_evaluation_results(base: Path, dst: MutableMapping) -> None:
    """Load evaluation artefacts into *dst* if they exist."""
    for key, fn in _LOAD_DISPATCH.items():
        result = fn(str(base))
        if result is not None:
            dst[key] = result  # type: ignore[index]


# ────────────────────────────────────────────
#  Public convenience
# ────────────────────────────────────────────
def save_all_results(res: PipelineResults, folder: str | Path) -> None:
    """
    Persist an entire PipelineResults object to *folder*.

    ├── <folder>/
    │   ├── <folder>_generation(.h5/.pkl/.h5)
    │   └── <folder>_evaluation.h5
    """
    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    stem = folder.name
    gen_base = folder / f"{stem}_generation"
    eval_base = folder / f"{stem}_evaluation"

    if "generation_results" in res:
        save_generation_results(res["generation_results"], gen_base)  # type: ignore[arg-type]

    save_evaluation_results(res, eval_base)


def load_all_results(folder: str | Path) -> PipelineResults:
    """
    Reconstruct a PipelineResults object from *folder*.

    Returns an empty dict if nothing is found.
    """
    folder = Path(folder)
    stem = folder.name
    gen_base = folder / f"{stem}_generation"
    eval_base = folder / f"{stem}_evaluation"

    out = PipelineResults()

    load_generation_results(gen_base, out)
    load_evaluation_results(eval_base, out)

    return out
