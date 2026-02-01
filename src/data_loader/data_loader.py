import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

class DataLoader:

    def __init__(self, path=None, dataset = None) :
        self.path = path
        self.dataset = dataset

    def get_dataframe(self, categorical_cols , category_type='category', drop_identation=False ,sep = ',', convert_float=False) :
        
        if (self.dataset is None):
            self.dataset = pd.read_csv(self.path, sep = sep)
        
        
        for column in categorical_cols :
            self.dataset[column] = self.dataset[column].astype(category_type)

        if(drop_identation) :
            self.dataset.pop(self.dataset.columns[0])
        return self.dataset
    
    @staticmethod
    def convert_type(dataset, categorical_cols) :
        columns = list(dataset)
        for column in columns :
            if column in categorical_cols:
                dataset[column] = dataset[column].astype(str)
            else :
                dataset[column] = dataset[column].astype(np.float64)
        
        return dataset

    @staticmethod
    def align_categoricals(dataset_to_align, dataset_reference, categorical_cols):
        for c in categorical_cols:
            # match base type first
            dataset_to_align[c] = dataset_to_align[c].astype(dataset_reference[c].cat.categories.dtype)
            # then match category set & order
            cat_dtype = CategoricalDtype(categories=dataset_reference[c].cat.categories,
                                        ordered=dataset_reference[c].cat.ordered)
            dataset_to_align[c] = dataset_to_align[c].astype(cat_dtype)
        return dataset_to_align
