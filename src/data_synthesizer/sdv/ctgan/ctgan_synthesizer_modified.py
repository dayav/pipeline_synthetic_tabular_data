import json

from sdv.single_table import CTGANSynthesizer
from .ctgan_modified import ModifiedCTGAN
from sdv.single_table.utils import detect_discrete_columns



class SDVCTGAN_(CTGANSynthesizer) :

    def __init__(self, metadata, data, enforce_min_max_values=True, enforce_rounding=True, locales=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, patience=None) :
        super().__init__(metadata, enforce_min_max_values, enforce_rounding, locales,
                 embedding_dim, generator_dim, discriminator_dim,
                 generator_lr, generator_decay, discriminator_lr,
                 discriminator_decay, batch_size, discriminator_steps,
                 log_frequency, verbose, epochs, pac, cuda)
        
        self._init_report(epochs)
        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(metadata, data, transformers)       
        self._model_kwargs = {
            'data': data,
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda,
            'discrete_columns': discrete_columns,
            'patience':patience
        }
        self._model = ModifiedCTGAN(**self._model_kwargs)

    def _init_report(self, epochs) -> None :

        self._training_report = dict()
        self._training_report['enforce_min_max_values'] = self.enforce_min_max_values
        self._training_report['enforce_rounding'] = self.enforce_rounding
        self._training_report['locales'] = self.locales
        self._training_report['embedding_dim'] = self.embedding_dim
        self._training_report['generator_dim'] = self.generator_dim
        self._training_report['discriminator_dim'] = self.discriminator_dim
        self._training_report['generator_lr'] = self.generator_lr
        self._training_report['discriminator_lr'] = self.discriminator_lr
        self._training_report['generator_decay'] = self.generator_decay
        self._training_report['discriminator_decay'] = self.discriminator_decay
        self._training_report['batch_size'] = self.batch_size
        self._training_report['discriminator_steps'] = self.discriminator_steps
        self._training_report['log_frequency'] = self.log_frequency
        self._training_report['verbose'] = self.verbose
        self._training_report['nb_epochs'] = epochs
        self._training_report['pac'] = self.pac
        self._training_report['cuda'] = self.cuda
          
          
    def fit(self, data) -> None :
        super().fit(data)
        self._create_training_output()

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        print('_fit from child')
        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(self.get_metadata(), processed_data, transformers)
        
        self._model.fit(processed_data, discrete_columns=discrete_columns)
        print('got it')

    def _create_training_output(self) :
        
        self._training_report['Epoch'] = self._model.loss_values['Epoch'].values.tolist()
        self._training_report['Generator Loss']= [tensor.item() for tensor in self._model.loss_values['Generator Loss'].values.tolist()]
        self._training_report['Discriminator Loss'] = [tensor.item() for tensor in self._model.loss_values['Discriminator Loss'].values.tolist()]

    def save_training_report_to_json(self, path) :
        with open(path, "w") as fp:
            json.dump(self._training_report, fp, indent=2)

    def fit_with_only_rare(self, data_rare) :
        self._freeze_generator()
        self._freeze_discriminator()
        self._model._batch_size = len(data_rare)
        self.fit(data_rare)

    def _freeze_generator(self) :
        for i, child in enumerate(self._model._generator.seq.children()):
            if i < 2:
                for param in child.parameters():
                    print(param.requires_grad)
                    param.requires_grad = False

    def _freeze_discriminator(self) :
        for i, child in enumerate(self._model._discriminator.seq.children()):
            if i < 6:
                for param in child.parameters():
                    print(param.requires_grad)
                    param.requires_grad = False





