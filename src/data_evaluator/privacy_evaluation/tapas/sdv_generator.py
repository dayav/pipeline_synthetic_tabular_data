from tapas.generators import Generator
from tapas.datasets import TabularDataset
from sdv.metadata import SingleTableMetadata
from data_loader import DataLoader

class CtganGenerator(Generator) :
    """A wrapper for ctgan objects."""

    def __init__(self, ctgan_class, metadata, categorical_cols):
        super().__init__()
        self.ctgan_class = ctgan_class
        self.metadata = metadata
        self.categorical_cols = categorical_cols

    def fit(self, dataset):
        assert isinstance(dataset, TabularDataset), 'dataset must be of class TabularDataset'
        self.dataset = dataset
        self.trained = True

    def generate(self, num_samples):
        """Instantiate a reprosyn model, run it, and return output."""
        assert self.trained, "No dataset provided to generator."
        model = self.ctgan_class(
            metadata=self.metadata,
            cuda=False
        )
        print('data')
        print(self.dataset.data.shape)
        _data_converted = self._convert_data(self.dataset.data)
        model.fit(_data_converted)
        return TabularDataset(model.sample(num_rows=num_samples), self.dataset.description)
    
    def _convert_data(self, data) :
        return DataLoader.convert_type(data, self.categorical_cols)



