"""Centralized filesystem paths for experiment artifacts and results."""

from pathlib import Path

EXPERIMENTS_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = EXPERIMENTS_DIR.parent

RESULTS_DIR: Path = EXPERIMENTS_DIR / "pipeline_tabular_data_results"
DATA_DIR: Path = RESULTS_DIR / "data"
BASELINE_DIR: Path = RESULTS_DIR / "baseline"
GENERATOR_OPTIMIZATION_DIR: Path = RESULTS_DIR / "generator_optimization"
OPTUNA_DIR: Path = RESULTS_DIR / "optuna"
EXTRA_DIR: Path = RESULTS_DIR / "extra"
PRIVACY_ONLY_DIR: Path = RESULTS_DIR / "privacy_evaluation_pipeline"
PRIVACY_ONLY_DIR_WEIGHTED: Path = RESULTS_DIR / "privacy_evaluation_pipeline_weighted"
PRIVACY_ONLY_DIR_WEIGHTED_5: Path = RESULTS_DIR / "privacy_evaluation_pipeline_weighted_5"
UTILITY_ONLY_DIR: Path = RESULTS_DIR / "utility_evaluation_pipeline"


def results_path(*parts: str) -> Path:
    """Build a path under the pipeline results root."""
    return RESULTS_DIR.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    """Create the directory (and parents) if missing, returning the Path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "PROJECT_ROOT",
    "EXPERIMENTS_DIR",
    "RESULTS_DIR",
    "DATA_DIR",
    "BASELINE_DIR",
    "GENERATOR_OPTIMIZATION_DIR",
    "OPTUNA_DIR",
    "EXTRA_DIR",
    "PRIVACY_ONLY_DIR",
    "UTILITY_ONLY_DIR",
    "results_path",
    "ensure_dir",
]
