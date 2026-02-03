"""Public API for data_synthesizer with lazy imports."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "sampling_reject",
    "sampling_reject_epsilon",
    "get_epsilon",
    "sampling_reject_epsilon_tabnet_strong",
    "sampling_reject_fit_gard",
    "fit_aia_gard",
]

if TYPE_CHECKING:
    from .privacy_sampling import sampling_reject, sampling_reject_epsilon, get_epsilon
    import data_synthesizer.sampling_reject_epsilon_tabnet_strong
    import data_synthesizer.sampling_reject_fit_gard
    import data_synthesizer.fit_aia_gard

_LAZY_SUBMODULES = {
    "sampling_reject_epsilon_tabnet_strong": "data_synthesizer.sampling_reject_epsilon_tabnet_strong",
    "sampling_reject_fit_gard": "data_synthesizer.sampling_reject_fit_gard",
    "fit_aia_gard": "data_synthesizer.fit_aia_gard",
}

_LAZY_ATTR_MODULES = [
    "data_synthesizer.privacy_sampling",
    "data_synthesizer.sampling_reject_epsilon_tabnet_strong",
    "data_synthesizer.sampling_reject_fit_gard",
    "data_synthesizer.fit_aia_gard",
]


def __getattr__(name: str):
    module_path = _LAZY_SUBMODULES.get(name)
    if module_path:
        module = import_module(module_path)
        globals()[name] = module
        return module

    for module_path in _LAZY_ATTR_MODULES:
        module = import_module(module_path)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
