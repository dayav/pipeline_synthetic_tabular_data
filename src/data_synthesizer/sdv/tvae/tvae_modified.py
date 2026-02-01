
from ctgan.synthesizers.tvae import *
from ctgan.synthesizers.tvae import _loss_function 

class ModifiedTVAE(TVAE) :

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

            reconstruction_loss = loss_1.detach().cpu()
            kld_loss = loss_2.detach().cpu()
            print(f'Epoch {i+1}, Reconstruct Loss: {reconstruction_loss: .4f},'  # patch
                f'KLD Loss: {kld_loss: .4f}',flush=True)
            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Reconstruct Loss': [reconstruction_loss],
                'KLD Loss': [kld_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df
