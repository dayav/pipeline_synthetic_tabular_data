"""Public API for privacy_sampling with lazy imports."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["make_tabnet_embedder", "fit_encoder"]

if TYPE_CHECKING:
    from .embeddings import make_tabnet_embedder
    from .ordenc import fit_encoder

_MODULE_FOR_NAME = {
    "make_tabnet_embedder": "privacy_sampling.embeddings",
    "fit_encoder": "privacy_sampling.ordenc",
}


def __getattr__(name: str):
    module_path = _MODULE_FOR_NAME.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
