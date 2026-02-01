
from typing import Optional
from enum import Enum, auto
from abc import ABC, abstractmethod

import pandas as pd

from data_synthesizer.pipeline.pipeline_results import PipelineResults

class TaskType(Enum):
    GENERATION_TASK = auto()
    FINE_TUNING_GENERATION_TASK = auto()
    RESEMBLANCE_EVALUATION_TASK = auto()
    UTILITY_EVALUATION_TASK = auto()
    PRIVACY_EVALUATION_TASK = auto()
    IDENTIFIABILITY_TASK = auto()


class Task(ABC) :
    def __init__(self, model, train_data : pd.DataFrame, cat_features : list, num_features : list):
        self.model = model
        self.cat_features = cat_features
        self.num_features = num_features

        self.load_data(train_data)
 
        self.results = PipelineResults()
    
    @abstractmethod
    def process(self, results : PipelineResults) :
        # Override this method in subclasses
        pass

    def load_data(self, train_data : pd.DataFrame) :
        self.train_data = train_data
