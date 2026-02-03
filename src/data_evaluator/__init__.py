"""Public API for data_evaluator.

Keep optional/heavy dependencies lazy so a minimal install can import the package.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BaseEvaluator",
    "UnivariateEvaluator",
    "MultivariateEvaluator",
    "GlobalEvaluator",
    "UtilityEvaluation",
    "ClassifierType",
    "plot_correlation_ratio_diff_",
    "plot_correlation_diff_",
    "box_plot_results",
    "histo_plot_utility_compare",
    "accuracy_compare",
]

if TYPE_CHECKING:
    from .base_evaluator import BaseEvaluator
    from .univariate_evaluator import UnivariateEvaluator
    from .multivariate_evaluator import MultivariateEvaluator
    from .global_evaluator import GlobalEvaluator
    from .utility_evaluator_new import UtilityEvaluation, ClassifierType
    from .plot_evaluation import (
        plot_correlation_ratio_diff_,
        plot_correlation_diff_,
        box_plot_results,
        histo_plot_utility_compare,
        accuracy_compare,
    )

_MODULE_FOR_NAME = {
    "BaseEvaluator": "data_evaluator.base_evaluator",
    "UnivariateEvaluator": "data_evaluator.univariate_evaluator",
    "MultivariateEvaluator": "data_evaluator.multivariate_evaluator",
    "GlobalEvaluator": "data_evaluator.global_evaluator",
    "UtilityEvaluation": "data_evaluator.utility_evaluator_new",
    "ClassifierType": "data_evaluator.utility_evaluator_new",
    "plot_correlation_ratio_diff_": "data_evaluator.plot_evaluation",
    "plot_correlation_diff_": "data_evaluator.plot_evaluation",
    "box_plot_results": "data_evaluator.plot_evaluation",
    "histo_plot_utility_compare": "data_evaluator.plot_evaluation",
    "accuracy_compare": "data_evaluator.plot_evaluation",
}


def __getattr__(name: str):
    module_path = _MODULE_FOR_NAME.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
