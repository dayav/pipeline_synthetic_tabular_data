import pandas as pd
import pytest

from data_synthesizer.pipeline import evaluation_task as et
from data_synthesizer.pipeline.evaluation_task import PrivacyEvaluationTask
from data_synthesizer.pipeline.pipeline_results import GenerationResults, PipelineResults


def _patch_privacy_dependencies(monkeypatch):
    class DummyUnivariateEvaluator:
        def __init__(self, real, synth):
            self.real = real
            self.synth = synth
            self.cat_uni = {"jensen_shanon": pd.DataFrame({"city": [0.1]})}
            self.num_uni = {"univariate_num_js": pd.DataFrame({"age": [0.2]})}

        def evaluate_categorical_stat_evaluation(self):
            return None

        def evaluate_numerical_stat_evaluation(self):
            return None

    class DummyPrivacyEvaluator:
        def __init__(self, real_train, synth, real_test, qid, non_qid):
            self.real_train = real_train
            self.synth = synth
            self.real_test = real_test
            self.qid = qid
            self.non_qid = non_qid
            self.last_similarity_type = None

        def evaluate_attribute_synthetic_prediction(self):
            return {"synthetic_attr": 0.11}, None, None, None, {"synthetic_attr": [1, 2]}

        def evaluate_attribute_real_prediction(self):
            return {"real_attr": 0.22}, None, None, None, {"real_attr": [2, 3]}

        def evaluate_similarity_stdg(self, similarity_type):
            self.last_similarity_type = similarity_type
            return 0.33, 0.44, None, None, 0.55

    monkeypatch.setattr(et, "UnivariateEvaluator", DummyUnivariateEvaluator)
    monkeypatch.setattr(et, "PrivacyEvaluator", DummyPrivacyEvaluator)
    monkeypatch.setattr(et, "get_epsilon_any", lambda *args, **kwargs: 0.66)

    return DummyPrivacyEvaluator


def _sample_frames():
    train = pd.DataFrame(
        {
            "city": pd.Series(["a", "b"], dtype="category"),
            "age": [10, 20],
            "target": [0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "city": pd.Series(["b", "c"], dtype="category"),
            "age": [12, 22],
            "target": [1, 0],
        }
    )
    seed_synth = pd.DataFrame(
        {
            "city": pd.Series(["s1", "s2"], dtype="category"),
            "age": [30, 40],
            "target": [1, 1],
        }
    )
    generated_synth = pd.DataFrame(
        {
            "city": pd.Series(["g1", "g2"], dtype="category"),
            "age": [50, 60],
            "target": [0, 0],
        }
    )
    return train, test, seed_synth, generated_synth


def test_privacy_task_populates_pipeline_results_with_generation(monkeypatch):
    DummyPrivacyEvaluator = _patch_privacy_dependencies(monkeypatch)
    train, test, seed_synth, generated_synth = _sample_frames()

    task = PrivacyEvaluationTask(
        train_data=train,
        test_data=test,
        cat_features=["city"],
        num_features=["age", "target"],
        quasi_identifier_features=["city"],
        non_quasi_identifier_features=["age", "target"],
        synth_data=seed_synth,
    )

    results = PipelineResults()
    results.generation_results = GenerationResults(synthetic_data=generated_synth)

    task.process(results)

    assert results.privacy_evaluation_results is task.privacy_evaluation_results
    assert results.privacy_evaluation_results.share == pytest.approx(0.55)
    assert list(results) == ["generation_results", "privacy_evaluation_results"]
    assert task.synth_data.equals(generated_synth)
    assert not task.synth_data.equals(seed_synth)
    assert isinstance(task._privacy_evaluator, DummyPrivacyEvaluator)
    assert task._privacy_evaluator.last_similarity_type == et.SimilarityType.DISSIMILARITY


def test_privacy_task_populates_pipeline_results_without_generation(monkeypatch):
    _patch_privacy_dependencies(monkeypatch)
    train, test, seed_synth, _ = _sample_frames()

    task = PrivacyEvaluationTask(
        train_data=train,
        test_data=test,
        cat_features=["city"],
        num_features=["age", "target"],
        quasi_identifier_features=["city"],
        non_quasi_identifier_features=["age", "target"],
        synth_data=seed_synth,
    )

    results = PipelineResults()
    task.process(results)

    assert "privacy_evaluation_results" in results
    assert results.privacy_evaluation_results.epsilon == pytest.approx(0.66)
    assert list(results) == ["privacy_evaluation_results"]
