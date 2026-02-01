import pandas as pd
import json
from sdv.single_table import TVAESynthesizer
from ctgan.synthesizers.tvae import *
from sdv.single_table.utils import detect_discrete_columns

from data_synthesizer.sdv.tvae.tvae_modified import ModifiedTVAE


class SDVTVAE_(TVAESynthesizer) :

    def __init__(self, metadata, data, enforce_min_max_values=True, enforce_rounding=True,
                 embedding_dim=128, compress_dims=(128, 128), decompress_dims=(128, 128), 
                 l2scale=1e-5, batch_size=500, epochs=300,  loss_factor=2, cuda=True) :
        super().__init__(metadata, enforce_min_max_values, enforce_rounding,
                 embedding_dim, compress_dims, decompress_dims,
                 l2scale, batch_size, epochs, loss_factor, cuda)        
        
        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(metadata, data, transformers)
        
        self._model_kwargs = {
            'data': data,
            'embedding_dim': embedding_dim,
            'compress_dims': compress_dims,
            'decompress_dims': decompress_dims,
            'l2scale': l2scale,
            'batch_size': batch_size,
            'epochs': epochs,
            'loss_factor': loss_factor,
            'cuda': cuda,
            'discrete_columns': discrete_columns
        }
        self._model = ModifiedTVAE(**self._model_kwargs)
        self._init_report(epochs)  
    
    
    def _init_report(self, epochs) -> None :

        self._training_report = dict()
        self._training_report['enforce_min_max_values'] = self.enforce_min_max_values
        self._training_report['enforce_rounding'] = self.enforce_rounding
        self._training_report['embedding_dim'] = self.embedding_dim
        self._training_report['compress_dims'] = self.compress_dims
        self._training_report['decompress_dims'] = self.decompress_dims
        self._training_report['l2scale'] = self.l2scale
        self._training_report['batch_size'] = self.batch_size
        self._training_report['nb_epochs'] = epochs
        self._training_report['loss_factor'] = self.loss_factor
        self._training_report['cuda'] = self.cuda
        


    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        
        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(self.get_metadata(), processed_data, transformers)
        self._model.fit(processed_data, discrete_columns=discrete_columns)
    
    
    def fit(self, data) -> None :
        # with capture_output() as output:
        super().fit(data)

        self._create_training_output()

    def _create_training_output(self) :

        self._training_report['Epoch'] = self._model.loss_values['Epoch'].values.tolist()
        self._training_report['Reconstruct Loss']= self._model.loss_values['Reconstruct Loss'].values.tolist()
        self._training_report['KLD Loss'] = self._model.loss_values['KLD Loss'].values.tolist()

    def save_training_report_to_json(self, path) :
        with open(path, "w") as fp:
            json.dump(self._training_report, fp, indent=2)

    def fit_with_only_rare(self, data_rare) :
        self._freeze_encoder()
        self._freeze_decoder()
        self.fit(data_rare)

    def _freeze_encoder(self) :
        for i, child in enumerate(self._model.encoder.seq.children()):
            if i < 2:
                for param in child.parameters():
                    param.requires_grad = False

    def _freeze_decoder(self) :
        for i, child in enumerate(self._model.decoder.seq.children()):
            if i < 4:
                for param in child.parameters():
                    param.requires_grad = False