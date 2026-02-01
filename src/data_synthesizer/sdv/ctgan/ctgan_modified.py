import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from ctgan import CTGAN
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
import torch
from torch import optim
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        """
        :param patience: How long to wait after last time loss improved.
        :param delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0

    def __call__(self, val_loss, model_state_dict, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model_state_dict
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered")
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model_state_dict  # Save the best model
            self.counter = 0
            self.best_epoch = epoch

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data

class ModifiedCTGAN(CTGAN) :
    def __init__(self, data: pd.DataFrame, embedding_dim: int = 128, generator_dim: tuple = (256, 256), 
                 discriminator_dim: tuple = (256, 256), generator_lr: float = 2e-4, generator_decay: float = 1e-6, 
                 discriminator_lr: float = 2e-4, discriminator_decay: float = 1e-6, batch_size: int = 500, 
                 discriminator_steps: int = 1, log_frequency: bool = True, verbose: bool = False, 
                 epochs: int = 300, pac: int = 10, cuda: bool = True, discrete_columns: list = None, patience : int = None):
        """
        Initializes the ModifiedCTGAN class for training a CTGAN model with additional data transformation steps.

        This class extends the CTGAN model to include data transformations using the `DataTransformer` and 
        sets up the generator and discriminator components for training on the provided dataset. The model is 
        trained to generate synthetic data while preserving the structure of the real data, particularly for 
        discrete columns.

        Args:
            data (pd.DataFrame): The dataset used to fit the data transformer and train the CTGAN model.
            embedding_dim (int, optional): The size of the embedding for the generator input (default is 128).
            generator_dim (tuple, optional): Dimensions for the generator layers (default is (256, 256)).
            discriminator_dim (tuple, optional): Dimensions for the discriminator layers (default is (256, 256)).
            generator_lr (float, optional): Learning rate for the generator (default is 2e-4).
            generator_decay (float, optional): Weight decay for the generator (default is 1e-6).
            discriminator_lr (float, optional): Learning rate for the discriminator (default is 2e-4).
            discriminator_decay (float, optional): Weight decay for the discriminator (default is 1e-6).
            batch_size (int, optional): The batch size for training (default is 500).
            discriminator_steps (int, optional): The number of discriminator updates per generator update (default is 1).
            log_frequency (bool, optional): Whether to log the frequency of categorical values (default is True).
            verbose (bool, optional): Whether to output detailed logs during training (default is False).
            epochs (int, optional): The number of training epochs (default is 300).
            pac (int, optional): The number of pseudo-samples used in the discriminator (default is 10).
            cuda (bool, optional): Whether to use CUDA for training (default is True).
            discrete_columns (list, optional): List of discrete columns in the dataset (default is None).

        Attributes:
            _transformer (DataTransformer): The data transformer fitted to the real dataset for preprocessing.
            _transformer_test (DataTransformer): A secondary data transformer fitted to the real dataset.
            _data_sampler (DataSampler): A data sampler for creating conditional vectors for training.
            _generator (Generator): The generator model for the CTGAN.
            _discriminator (Discriminator): The discriminator model for the CTGAN.

        Notes:
            - The `DataTransformer` transforms the data for training and sets up output dimensions based on 
              discrete and continuous columns.
            - The `DataSampler` is used to sample conditional vectors for training the generator and discriminator.
            - This class uses the CTGAN architecture but extends it with additional preprocessing steps.
        """
        super().__init__(embedding_dim, generator_dim, discriminator_dim,
                    generator_lr, generator_decay, discriminator_lr,
                    discriminator_decay, batch_size, discriminator_steps,
                    log_frequency, verbose, epochs, pac, cuda)
        
        self._transformer = DataTransformer()
        print('discrete_columns : ', discrete_columns)
        self._transformer.fit(data, discrete_columns)

        self._transformer_test = DataTransformer()
        self._transformer_test.fit(data, discrete_columns)


        train_data = self._transformer.transform(data)
        self._data_sampler = DataSampler(
                train_data,
                self._transformer.output_info_list,
                self._log_frequency)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        if (patience != None) :
            self._early_stopping = EarlyStopping(patience=patience)
        else :
            self._early_stopping = None

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """
        Fit the Modified CTGAN model to the training data.

        This function trains the Modified CTGAN model using the provided training data. It applies data transformations,
        generates synthetic data, and updates the generator and discriminator models through multiple epochs. Optionally,
        the method accepts discrete columns and can handle both `numpy.ndarray` and `pandas.DataFrame` formats.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame): The training data used to fit the model. It should be a 
                2-dimensional `numpy` array or `pandas.DataFrame`.
            discrete_columns (list-like, optional): A list of discrete columns to generate the Conditional Vector. 
                If `train_data` is a `numpy` array, this list should contain the integer indices of the columns. 
                If it is a `pandas.DataFrame`, this list should contain the column names.
            epochs (int, optional): The number of training epochs. If not provided, the default number of epochs from 
                the constructor is used.

        Returns:
            None: This method trains the model and updates the loss values after each epoch.

        Attributes:
            loss_values (pd.DataFrame): A DataFrame containing the generator and discriminator loss values for each epoch.

        Example:
            model.fit(train_data, discrete_columns=['col1', 'col2'], epochs=100)

        Notes:
            - The `fit` method uses a data transformation pipeline to prepare the data for training, including handling 
            discrete columns.
            - Both the generator and discriminator are updated during training, with the discriminator being updated multiple times per generator update.
            - The training process computes and logs both generator and discriminator losses, which are stored in the `loss_values` attribute.
            - If the `epochs` argument is provided, a deprecation warning is issued, as the `epochs` should ideally be passed 
            to the constructor.
        """

        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer_for_sampling = DataTransformer()
        self._transformer_for_sampling.fit(train_data, discrete_columns)
        print('discrete_columns : ', discrete_columns)
        
        train_data_for_sampler = self._transformer_for_sampling.transform(train_data)
        print('train_data_for_sampler : ', len(train_data_for_sampler))

        train_data = self._transformer.transform(train_data)
        print('train_data : ', len(train_data))

        self._data_sampler = DataSampler(
                train_data_for_sampler,
                self._transformer_for_sampling.output_info_list,
                self._log_frequency)

    
        self._generator.to(self._device)

        self._discriminator.to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )
    

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    # print('epoch : ', i, ', step : ', id_,', discr step : ',n, '--------')
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        self._lastperm = perm
                        self._last_col_perm = col[perm]
                        self._last_opt_perm = opt[perm]
                        self._last_condvec = condvec
                        self._last_traindata = train_data
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)


                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)

                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self._discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu()
            discriminator_loss = loss_d.detach().cpu()    

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )
            
            if (self._early_stopping is not None) :
                self._early_stopping(generator_loss, self._generator.state_dict(), i)

                if self._early_stopping.early_stop:
                    print("Stopped early at epoch:", i+1)
                    break
        if (self._early_stopping is not None) :
            if self._early_stopping.best_model_state is not None:
                self._generator.load_state_dict(self._early_stopping.best_model_state)