import numpy as np
import pandas as pd
import pytest

from data_synthesizer.pipeline.pipeline_results import (
    GenerationResults,
    PipelineResults,
    PrivacyAnonymeterEvaluationResults,
    PrivacyEvaluationResults,
)


def _sample_generation_results():
    df = pd.DataFrame({"x": [1, 2]})
    return GenerationResults(synthetic_data=df, mode_collapse_corrected=True)


def _sample_privacy_results():
    jsc = pd.DataFrame([[0.1, 0.2]], columns=["c1", "c2"])
    jsn = pd.DataFrame([[0.3, 0.4]], columns=["n1", "n2"])
    return PrivacyEvaluationResults(
        jensen_shanon_categorical=jsc,
        jensen_shanon_numerical=jsn,
        dissimilarity_synthetic_real=0.5,
        dissimilarity_synthetic_test=0.6,
        epsilon_dissimilarity_synthetic_real=0.0,
        epsilon_dissimilarity_synthetic_test=0.0,
        share=0.7,
        epsilon=0.8,
        attribute_synthetic_prediction={"attr": {"m": 1.0}},
        attribute_real_prediction={"attr": {"m": 2.0}},
        attribute_synthetic_prediction_values={"attr": np.array([1, 2, 3])},
        attribute_real_prediction_values={"attr": np.array([4, 5, 6])},
    )


def test_generation_results_mapping_semantics():
    gen = _sample_generation_results()

    assert gen["synthetic_data"].equals(gen.synthetic_data)

    gen["mode_collapse_corrected"] = False
    assert gen.mode_collapse_corrected is False
    assert "mode_collapse_corrected" in gen
    assert "generator_model" not in gen  # None should not be considered “in”

    with pytest.raises(KeyError):
        _ = gen["missing_key"]


def test_pipeline_results_get_items_and_iter():
    gen = _sample_generation_results()
    results = PipelineResults(generation_results=gen)

    assert results.get("generation_results") is gen
    assert results.get("utility_evaluation_results", "missing") == "missing"
    assert dict(results.items()) == {"generation_results": gen}
    assert list(results) == ["generation_results"]


def test_to_dict_drop_none():
    gen = _sample_generation_results()
    privacy = _sample_privacy_results()
    results = PipelineResults(
        generation_results=gen,
        privacy_evaluation_results=privacy,
    )

    full = results.to_dict(drop_none=False)
    assert "utility_evaluation_results" in full
    assert full["utility_evaluation_results"] is None
    assert full["privacy_evaluation_results"]["jensen_shanon_categorical"].equals(
        privacy.jensen_shanon_categorical
    )

    dropped = results.to_dict(drop_none=True)
    assert "utility_evaluation_results" not in dropped
    assert "generator_model" not in dropped["generation_results"]


def test_from_dict_rebuilds_dataclasses():
    gen = _sample_generation_results()
    privacy = _sample_privacy_results()
    anonym = {
        "singling_univariate": {"metric": 1.0},
        "singling_multivariate": {"metric": 2.0},
        "linkability_attacks": {"metric": 3.0},
    }

    payload = {
        "generation_results": gen.to_dict(drop_none=True),
        "privacy_evaluation_results": privacy.to_dict(drop_none=True),
        "privacy_anonymeter_results": {"run_0": anonym},
    }

    rebuilt = PipelineResults.from_dict(payload)

    assert isinstance(rebuilt.generation_results, GenerationResults)
    assert isinstance(rebuilt.privacy_evaluation_results, PrivacyEvaluationResults)
    assert isinstance(
        rebuilt.privacy_anonymeter_results["run_0"],
        PrivacyAnonymeterEvaluationResults,
    )
    assert rebuilt.generation_results.synthetic_data.equals(gen.synthetic_data)
    assert (
        rebuilt.privacy_anonymeter_results["run_0"].linkability_attacks["metric"] == 3.0
    )
