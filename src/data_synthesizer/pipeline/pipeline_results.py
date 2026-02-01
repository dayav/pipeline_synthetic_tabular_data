from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from data_evaluator.privacy_evaluation.privacy_evaluator_anonymeter import AnonymeterResults

class _DictLikeDataclass:
    """
    Lightweight mapping mixin so dataclass instances can be used with both
    attribute access and dict-style access (results["field"], "field" in results).
    """

    __slots__ = ()

    @classmethod
    def _field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    def __getitem__(self, key: str):
        if not isinstance(key, str) or not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str) or not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        return (
            isinstance(key, str)
            and hasattr(self, key)
            and getattr(self, key) is not None
        )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            value = self[key]
        except KeyError:
            return default
        return default if value is None else value

    def items(self):
        for name in self._field_names():
            value = getattr(self, name)
            if value is not None:
                yield name, value

    def to_dict(self, drop_none: bool = False) -> dict:
        def convert(val: Any):
            if isinstance(val, _DictLikeDataclass):
                return val.to_dict(drop_none=drop_none)
            if isinstance(val, dict):
                return {k: convert(v) for k, v in val.items()}
            return val

        data = {name: convert(getattr(self, name)) for name in self._field_names()}
        if drop_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


@dataclass
class GenerationResults(_DictLikeDataclass):
    """Results from the generation step."""
    synthetic_data: pd.DataFrame
    generator_model: Optional[object] = None
    mode_collapse_corrected: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResults":
        if isinstance(data, cls):
            return data
        return cls(**data)


@dataclass
class ResemblanceEvaluationResults(_DictLikeDataclass):
    """Evaluation results for resemblance metrics."""
    categorical_univariate: dict
    numerical_univariate: dict
    categorical_multivariate: dict
    numerical_multivariate: dict
    categorical_numerical_multivariate: Optional[dict] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResemblanceEvaluationResults":
        if isinstance(data, cls):
            return data
        return cls(**data)


@dataclass
class UtilityEvaluationResults(_DictLikeDataclass):
    """Evaluation results for utility metrics."""
    train_synthetic_test_real_results: pd.DataFrame
    train_real_test_real_results: pd.DataFrame
    permutation_importance_tstr: Dict[str, pd.Series]
    permutation_importance_trtr: Dict[str, pd.Series]
    shap_importance_tstr: Dict[str, pd.Series]
    shap_importance_trtr: Dict[str, pd.Series]
    confusion_matrix_tstr: Dict[str, Dict[str, np.ndarray]]
    confusion_matrix_trtr: Dict[str, Dict[str, np.ndarray]]
    accuracy_mean_tstr: float
    accuracy_mean_trtr: float
    rbo_permutation_importance: float
    spearman_permutation_importance: float
    kendall_permutation_importance: float
    rbo_shap: float
    spearman_shap: float
    kendall_shap: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UtilityEvaluationResults":
        if isinstance(data, cls):
            return data
        return cls(**data)


@dataclass
class PrivacyEvaluationResults(_DictLikeDataclass):
    """Evaluation results for privacy metrics."""
    jensen_shanon_categorical: pd.DataFrame
    jensen_shanon_numerical: pd.DataFrame
    dissimilarity_synthetic_real: float
    dissimilarity_synthetic_test: float
    epsilon_dissimilarity_synthetic_real: float
    epsilon_dissimilarity_synthetic_test: float
    share: float
    epsilon: float
    attribute_synthetic_prediction: dict
    attribute_real_prediction: dict
    attribute_synthetic_prediction_values: dict
    attribute_real_prediction_values: dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrivacyEvaluationResults":
        if isinstance(data, cls):
            return data
        return cls(**data)

    def _target_names(self, obj: Any) -> set[str]:
        """Extract attribute names from a metrics table or mapping."""
        if isinstance(obj, pd.DataFrame):
            return set(obj.index)
        if isinstance(obj, dict):
            return set(obj.keys())
        return set()

    def _select_predictions(self, targets: set[str]) -> Dict[str, Dict[str, Any]]:
        """Return per-source prediction metrics for the given attributes."""
        def pick(values: dict) -> Dict[str, Any]:
            return {k: v for k, v in values.items() if k in targets} if targets else {}

        return {
            "synthetic": pick(self.attribute_synthetic_prediction),
            "real": pick(self.attribute_real_prediction),
        }

    def classification_results(
        self, categorical_targets: Optional[set[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return attribute-inference metrics for categorical targets (classification).

        If *categorical_targets* is not provided, the method falls back to the
        Jensen–Shannon categorical table to infer the attribute names.
        """
        targets = set(categorical_targets) if categorical_targets else self._target_names(
            self.jensen_shanon_categorical
        )
        return self._select_predictions(targets)

    def regression_results(
        self, numerical_targets: Optional[set[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return attribute-inference metrics for numerical targets (regression).

        If *numerical_targets* is not provided, the method falls back to the
        Jensen–Shannon numerical table to infer the attribute names.
        """
        targets = set(numerical_targets) if numerical_targets else self._target_names(
            self.jensen_shanon_numerical
        )
        return self._select_predictions(targets)


@dataclass
class PrivacyAnonymeterEvaluationResults(_DictLikeDataclass):
    singling_univariate: AnonymeterResults
    singling_multivariate: AnonymeterResults
    linkability_attacks: AnonymeterResults

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrivacyAnonymeterEvaluationResults":
        if isinstance(data, cls):
            return data
        return cls(**data)


@dataclass
class PipelineResults(_DictLikeDataclass):
    generation_results: Optional[GenerationResults] = None
    resemblance_evaluation_results: Optional[ResemblanceEvaluationResults] = None
    utility_evaluation_results: Optional[UtilityEvaluationResults] = None
    privacy_evaluation_results: Optional[PrivacyEvaluationResults] = None
    privacy_anonymeter_results: Optional[Dict[str, PrivacyAnonymeterEvaluationResults]] = None

    def __iter__(self):
        for name in self._field_names():
            if getattr(self, name) is not None:
                yield name

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Return a plain nested dict that mirrors the old TypedDict shape.

        None-valued optional fields are dropped, matching how the legacy
        PipelineResults omitted absent sections.
        """
        return self.to_dict(drop_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineResults":
        if isinstance(data, cls):
            return data

        def maybe(key: str, ctor):
            if key not in data or data[key] is None:
                return None
            return ctor(data[key])

        privacy_anon = None
        if "privacy_anonymeter_results" in data and data["privacy_anonymeter_results"] is not None:
            privacy_anon = {
                k: PrivacyAnonymeterEvaluationResults.from_dict(v)
                for k, v in data["privacy_anonymeter_results"].items()
            }

        return cls(
            generation_results=maybe("generation_results", GenerationResults.from_dict),
            resemblance_evaluation_results=maybe(
                "resemblance_evaluation_results", ResemblanceEvaluationResults.from_dict
            ),
            utility_evaluation_results=maybe(
                "utility_evaluation_results", UtilityEvaluationResults.from_dict
            ),
            privacy_evaluation_results=maybe(
                "privacy_evaluation_results", PrivacyEvaluationResults.from_dict
            ),
            privacy_anonymeter_results=privacy_anon,
        )
