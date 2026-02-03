"""Public API for data_synthesizer.pipeline with lazy imports."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "Task",
    "TaskType",
    "GenerationTask",
    "FineTuningGenerationTask",
    "SamplingAndRejectTask",
    "EvaluationTask",
    "ResemblanceEvaluationTask",
    "UtilityEvaluationTask",
    "PrivacyEvaluationTask",
    "PrivacyAnonymeterEvaluationTask",
    "PipelineBuilder",
    "PipelineResults",
    "GenerationResults",
    "ResemblanceEvaluationResults",
    "UtilityEvaluationResults",
    "PrivacyEvaluationResults",
    "PrivacyAnonymeterEvaluationResults",
    "save_generation_results",
    "load_generation_results",
    "save_evaluation_results",
    "load_evaluation_results",
    "save_all_results",
    "load_all_results",
    "evaluation_task",
    "generation_task",
    "pipepline_builder",
]

if TYPE_CHECKING:
    from .base_pipeline import Task, TaskType
    from .generation_task import GenerationTask, FineTuningGenerationTask, SamplingAndRejectTask
    from .evaluation_task import (
        EvaluationTask,
        ResemblanceEvaluationTask,
        UtilityEvaluationTask,
        PrivacyEvaluationTask,
        PrivacyAnonymeterEvaluationTask,
    )
    from .pipepline_builder import PipelineBuilder
    from .pipeline_results import (
        PipelineResults,
        GenerationResults,
        ResemblanceEvaluationResults,
        UtilityEvaluationResults,
        PrivacyEvaluationResults,
        PrivacyAnonymeterEvaluationResults,
    )
    from .storages.evaluation_results_storage import (
        save_generation_results,
        load_generation_results,
        save_evaluation_results,
        load_evaluation_results,
        save_all_results,
        load_all_results,
    )

_MODULE_FOR_NAME = {
    "Task": ("data_synthesizer.pipeline.base_pipeline", "Task"),
    "TaskType": ("data_synthesizer.pipeline.base_pipeline", "TaskType"),
    "GenerationTask": ("data_synthesizer.pipeline.generation_task", "GenerationTask"),
    "FineTuningGenerationTask": ("data_synthesizer.pipeline.generation_task", "FineTuningGenerationTask"),
    "SamplingAndRejectTask": ("data_synthesizer.pipeline.generation_task", "SamplingAndRejectTask"),
    "EvaluationTask": ("data_synthesizer.pipeline.evaluation_task", "EvaluationTask"),
    "ResemblanceEvaluationTask": ("data_synthesizer.pipeline.evaluation_task", "ResemblanceEvaluationTask"),
    "UtilityEvaluationTask": ("data_synthesizer.pipeline.evaluation_task", "UtilityEvaluationTask"),
    "PrivacyEvaluationTask": ("data_synthesizer.pipeline.evaluation_task", "PrivacyEvaluationTask"),
    "PrivacyAnonymeterEvaluationTask": (
        "data_synthesizer.pipeline.evaluation_task",
        "PrivacyAnonymeterEvaluationTask",
    ),
    "PipelineBuilder": ("data_synthesizer.pipeline.pipepline_builder", "PipelineBuilder"),
    "PipelineResults": ("data_synthesizer.pipeline.pipeline_results", "PipelineResults"),
    "GenerationResults": ("data_synthesizer.pipeline.pipeline_results", "GenerationResults"),
    "ResemblanceEvaluationResults": ("data_synthesizer.pipeline.pipeline_results", "ResemblanceEvaluationResults"),
    "UtilityEvaluationResults": ("data_synthesizer.pipeline.pipeline_results", "UtilityEvaluationResults"),
    "PrivacyEvaluationResults": ("data_synthesizer.pipeline.pipeline_results", "PrivacyEvaluationResults"),
    "PrivacyAnonymeterEvaluationResults": (
        "data_synthesizer.pipeline.pipeline_results",
        "PrivacyAnonymeterEvaluationResults",
    ),
    "save_generation_results": (
        "data_synthesizer.pipeline.storages.evaluation_results_storage",
        "save_generation_results",
    ),
    "load_generation_results": (
        "data_synthesizer.pipeline.storages.evaluation_results_storage",
        "load_generation_results",
    ),
    "save_evaluation_results": (
        "data_synthesizer.pipeline.storages.evaluation_results_storage",
        "save_evaluation_results",
    ),
    "load_evaluation_results": (
        "data_synthesizer.pipeline.storages.evaluation_results_storage",
        "load_evaluation_results",
    ),
    "save_all_results": (
        "data_synthesizer.pipeline.storages.evaluation_results_storage",
        "save_all_results",
    ),
    "load_all_results": (
        "data_synthesizer.pipeline.storages.evaluation_results_storage",
        "load_all_results",
    ),
    "evaluation_task": ("data_synthesizer.pipeline.evaluation_task", None),
    "generation_task": ("data_synthesizer.pipeline.generation_task", None),
    "pipepline_builder": ("data_synthesizer.pipeline.pipepline_builder", None),
}


def __getattr__(name: str):
    if name not in _MODULE_FOR_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr = _MODULE_FOR_NAME[name]
    module = import_module(module_path)
    value = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value
