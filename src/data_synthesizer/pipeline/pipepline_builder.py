from typing import List, Optional

import pandas as pd

from data_evaluator.utility_evaluator_new import ClassifierType
from data_synthesizer.pipeline.base_pipeline import Task
from data_synthesizer.pipeline.generation_task import FineTuningGenerationTask, GenerationTask, SamplingAndRejectTask
from data_synthesizer.pipeline.evaluation_task import EvaluationTask, PrivacyAnonymeterEvaluationTask, PrivacyEvaluationTask, ResemblanceEvaluationTask, UtilityEvaluationTask
from data_synthesizer.pipeline.pipeline_results import PipelineResults


class PipelineBuilder :

    def __init__(self, train_data : pd.DataFrame, categorical_features : list, numerical_features : list, model = None) :
        self._train_data = train_data
        self._model = model
        self._categorical_features = categorical_features
        self._numerical_features = numerical_features
        self._synthetic_data = None
        self._tasks_buffer : List[Task] = []
        self._tasks : List[Task] = []

        self.results = PipelineResults()

    def add_generation_task(self) :
        self._tasks_buffer.append(GenerationTask(self._model, self._train_data, self._categorical_features, self._numerical_features))
        return self
    
    def add_fine_tuning_generation_task(self, synth_data : Optional[pd.DataFrame] = None) :
        self._tasks_buffer.append(FineTuningGenerationTask(self._model, self._train_data, self._categorical_features, self._numerical_features, synth_data))
        return self
    
    def add_sampling_and_reject_task(self, epsilon: float, synth_data : Optional[pd.DataFrame] = None, dissimilarity_type:Optional[str] = 'euclidian', non_quasi_identifier_features : Optional[list] =None, aia_parameters: Optional[dict] = None) :
        self._epsilon = epsilon
        self._tasks_buffer.append(SamplingAndRejectTask(self._model, self._train_data, self._categorical_features, self._numerical_features, epsilon, synth_data, dissimilarity_type, non_quasi_identifier_features, aia_parameters=aia_parameters))
        return self

    def add_privacy_evaluation_task(self, test_data : pd.DataFrame, quasi_identifier_features : list, non_quasi_identifier_features : list, 
                                    synth_data : Optional[pd.DataFrame] = None):
        self._test_data = test_data
        self._tasks_buffer.append(PrivacyEvaluationTask(self._train_data, test_data, self._categorical_features, self._numerical_features, 
                                                         quasi_identifier_features, non_quasi_identifier_features, synth_data))
        return self
    
    def add_privacy_anonymeter_evaluation_task(self, test_data : pd.DataFrame, synth_data : Optional[pd.DataFrame] = None, first_half : list = None, second_half : list = None, runs_count : int = 1):
        self._test_data = test_data
        self._tasks_buffer.append(PrivacyAnonymeterEvaluationTask(self._train_data, test_data, self._categorical_features, self._numerical_features,
                                                                   synth_data, first_half, second_half, runs_count))
        return self
    
    def add_ressemblance_evaluation_task(self, test_data : pd.DataFrame, synth_data : Optional[pd.DataFrame] = None):
        self._test_data = test_data
        self._tasks_buffer.append(ResemblanceEvaluationTask(self._train_data, test_data, self._categorical_features, self._numerical_features, synth_data))
        return self
    
    def add_utility_evaluation_task(self, test_data : pd.DataFrame, classifiers : List[ClassifierType], synth_data : Optional[pd.DataFrame] = None):
        self._test_data = test_data
        self._tasks_buffer.append(UtilityEvaluationTask(self._train_data, test_data, self._categorical_features, self._numerical_features, classifiers, synth_data))
        return self
    
    def build(self) :
        self._tasks.append(self._tasks_buffer[0])

        task_id = 0
        for current_task in self._tasks_buffer[1:]:
                        
            if (self._isTaskValid(self._tasks[task_id], current_task)) :
                self._tasks.append(current_task)
                task_id += 1
    
    def run(self) -> PipelineResults:
        for task in self._tasks :
            task.process(self.results)            
        return self.results
        
    def _isTaskValid(self, task : Task , next_task : Task) -> bool :
        if isinstance(task, EvaluationTask) and isinstance(next_task, EvaluationTask):
            return  True
        elif isinstance(task, GenerationTask) and isinstance(next_task, GenerationTask) :
            return True
        elif isinstance(task, GenerationTask) and isinstance(next_task, EvaluationTask) :
            return True
        else :
            return False

