import pandas as pd
import numpy as np
import json
from sdv.single_table import CTGANSynthesizer
from ctgan import CTGAN
from sdv.single_table.utils import detect_discrete_columns
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import random_state
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm
import warnings


from IPython.utils.capture import capture_output


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

class ModifCTGAN(CTGAN) :
    def __init__(self, data, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, discrete_columns=None):
        super().__init__(embedding_dim, generator_dim, discriminator_dim,
                    generator_lr, generator_decay, discriminator_lr,
                    discriminator_decay, batch_size, discriminator_steps,
                    log_frequency, verbose, epochs, pac, cuda)
        
        self._transformer = DataTransformer()
        print('discrete_columns : ', discrete_columns)
        self._transformer.fit(data, discrete_columns)


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

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
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

        train_data = self._transformer.transform(train_data)
        self._data_sampler = DataSampler(
                train_data,
                self._transformer.output_info_list,
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
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]
                    
                    # print('c1 : ', c1)
                    # print('m1 : ', m1)
                    # print('fakez : ', fakez)
                    # print('real sample : ', self._transformer.inverse_transform(real))
                    # print('c2 : ', c2)
                    

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)


                    real = torch.from_numpy(real.astype('float32')).to(self._device)
                    #fake = fake.astype('float32')).to('cpu')
                    #fakeact = torch.from_numpy(fakeact.astype('float32')).to('cpu')

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)
                    # print('y_fake : ', y_fake)
                    # print('y_real : ', y_real)

                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()
                    # print('fake : ', self._transformer.inverse_transform(fake.detach().to('cpu')))
                    # print('fake act: ', self._transformer.inverse_transform(fakeact.detach().to('cpu')))

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




class SDVCTGAN(CTGANSynthesizer) :

    def __init__(self, metadata, data, enforce_min_max_values=True, enforce_rounding=True, locales=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True) :
        super().__init__(metadata, enforce_min_max_values, enforce_rounding, locales,
                 embedding_dim, generator_dim, discriminator_dim,
                 generator_lr, generator_decay, discriminator_lr,
                 discriminator_decay, batch_size, discriminator_steps,
                 log_frequency, verbose, epochs, pac, cuda)
        
        self._init_report(epochs)
        discrete_columns = detect_discrete_columns(metadata, data)       
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
            'discrete_columns': discrete_columns
        }
        self._model = ModifCTGAN(**self._model_kwargs)

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
        if self.verbose :
            try :
                with capture_output() as output:
                    super().fit(data)
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                self._init_report(self.epochs)
                self._create_training_output()
        else :
            super().fit(data)

    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        print('_fit from child')
        discrete_columns = detect_discrete_columns(self.get_metadata(), processed_data)
        
        self._model.fit(processed_data, discrete_columns=discrete_columns)
        print('got it')

    def _create_training_output(self) :
        
        self._training_report['Epoch'] = self._model.loss_values['Epoch'].values.tolist()
        self._training_report['Generator Loss']= [tensor.item() for tensor in self._model.loss_values['Generator Loss'].values.tolist()]
        self._training_report['Discriminator Loss'] = [tensor.item() for tensor in self._model.loss_values['Discriminator Loss'].values.tolist()]

    def save_training_report_to_json(self, path) :
        with open(path, "w") as fp:
            json.dump(self._training_report, fp, indent=2)

    