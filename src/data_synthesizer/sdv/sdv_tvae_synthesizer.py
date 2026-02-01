import pandas as pd
import json
from sdv.single_table import TVAESynthesizer
from ctgan.synthesizers.tvae import *
from ctgan.synthesizers.tvae import _loss_function 
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table.utils import detect_discrete_columns


from IPython.utils.capture import capture_output

class TVAE_patch(TVAE) :

    def __init__(
        self,
        data,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        discrete_columns=None
    ):
        super().__init__(embedding_dim, 
                        compress_dims, 
                        decompress_dims, 
                        l2scale, 
                        batch_size,
                        epochs,
                        loss_factor,
                        cuda)

        self.transformer = DataTransformer()
        self.transformer.fit(data, discrete_columns)
        data_dim = self.transformer.output_dimensions
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

            # print(f'Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},'  # noqa: T001
            #           f'Loss D: {loss_d.detach().cpu(): .4f}',
            #           flush=True)
            print(f'Epoch {i+1}, Reconstruct Loss: {loss_1.detach().cpu(): .4f},'  # patch
                f'KLD Loss: {loss_2.detach().cpu(): .4f}',flush=True)



class SDVTVAE(TVAESynthesizer) :

    def __init__(self, metadata, data, enforce_min_max_values=True, enforce_rounding=True,
                 embedding_dim=128, compress_dims=(128, 128), decompress_dims=(128, 128), 
                 l2scale=1e-5, batch_size=500, epochs=300,  loss_factor=2, cuda=True) :
        super().__init__(metadata, enforce_min_max_values, enforce_rounding,
                 embedding_dim, compress_dims, decompress_dims,
                 l2scale, batch_size, epochs, loss_factor, cuda)        
        
        discrete_columns = detect_discrete_columns(metadata, data)
        
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
        self._model = TVAE_patch(**self._model_kwargs)
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
        discrete_columns = detect_discrete_columns(self.get_metadata(), processed_data)
        self._model.fit(processed_data, discrete_columns=discrete_columns)
    
    
    def fit(self, data) -> None :
        super().fit(data)

        self._create_training_output()

    def _create_training_output(self) :
        self._training_report['Epoch'] = loss_values['Epoch'].values.tolist()
        self._training_report['Reconstruct Loss']= loss_values['Reconstruct Loss'].values.tolist()
        self._training_report['KLD Loss'] = loss_values['KLD Loss'].values.tolist()

    def save_training_report_to_json(self, path) :
        with open(path, "w") as fp:
            json.dump(self._training_report, fp, indent=2)


    