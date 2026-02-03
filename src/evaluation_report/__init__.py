"""Public API for evaluation_report with lazy imports."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["ResemblanceReport", "UtilityReport"]

if TYPE_CHECKING:
    from .ressemblance_report import ResemblanceReport
    from .utility_report import UtilityReport

_MODULE_FOR_NAME = {
    "ResemblanceReport": "evaluation_report.ressemblance_report",
    "UtilityReport": "evaluation_report.utility_report",
}


def __getattr__(name: str):
    module_path = _MODULE_FOR_NAME.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
