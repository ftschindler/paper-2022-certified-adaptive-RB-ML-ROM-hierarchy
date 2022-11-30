import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# +
from pymor.core.base import BasicObject
from pymor.discretizers.builtin.cg import L2ProductP1
from pymor.discretizers.builtin.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.models.interface import Model
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu, Parameters
from pymor.vectorarrays.numpy import NumpyVectorSpace

from pymor.models.neural_network import FullyConnectedNN
from pymor.reductors.neural_network import multiple_restarts_training

from utils import kernels
from utils.utilities import ActivFunc

import pytorch_lightning as pl

class Network(pl.LightningModule):
    """
    This class corresponds to an optimization of the model in a straightforward way.
    """

    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        # Some logs
        self.log('train/loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        # Some logs
        self.log('val/loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(y_pred, y)

        # Some logs
        self.log('test/loss', loss, prog_bar=True)

        return loss
    
class LightningFullyConnectedSDKN(Network):

    def __init__(self, layer_sizes, activation_function=nn.ReLU(), settings={}):
        super().__init__()
        
        if layer_sizes is None or not len(layer_sizes) > 1 or not all(size >= 1 for size in layer_sizes):
            raise ValueError

        self.dim_input = layer_sizes[0]
        self.dim_output = layer_sizes[-1]
        
        self.layer_sizes = layer_sizes

        # General stuff
        self.centers = nn.Parameter(torch.rand(5, self.dim_input))
        self.centers.required_grad = False
        self.M = self.centers.shape[0]
 
        # Define linear maps
        self.width = layer_sizes[1]
        self.fc1 = nn.Linear(self.dim_input, self.width, bias=False)
        self.fc2 = nn.Linear(self.width, self.width, bias=False)
        self.fc3 = nn.Linear(self.width, self.width, bias=False)
        self.fc4 = nn.Linear(self.width, self.dim_output, bias=False)
 
        # Define activation maps
        self.activ1 = ActivFunc(self.width, self.M, kernel=kernels.Wendland_order_0(ep=1))
        self.activ2 = ActivFunc(self.width, self.M, kernel=kernels.Wendland_order_0(ep=1))
        self.activ3 = ActivFunc(self.width, self.M, kernel=kernels.Wendland_order_0(ep=1))        
        
        self.settings = settings
        
        self.loss_function = self.settings['loss_function']
        


    def forward(self, x):
        # First fully connect + activation function
        x = self.fc1(x)
        centers = self.fc1(self.centers)
        x, centers = self.activ1(x, centers)

        # Second fully connect + activation function
        x = self.fc2(x)
        centers = self.fc2(centers)
        x, centers = self.activ2(x, centers)

        # Third fully connect + activation function
        x = self.fc3(x)
        centers = self.fc3(centers)
        x, centers = self.activ3(x, centers)

        # Fourth fully connect
        x = self.fc4(x)

        return x


    def configure_optimizers(self):
        optimizer = self.settings['optimizer'](self.parameters(), lr=self.settings['learning_rate'])

        if self.settings['lr_scheduler'] is not None:
            scheduler = self.settings['lr_scheduler'](optimizer, **self.settings['lr_scheduler_params'])
            return [optimizer], [scheduler]

        return optimizer    


class LightningFullyConnectedNN(Network):

    def __init__(self, layer_sizes, activation_function=nn.ReLU(), settings={}):
        super().__init__()
        
        if layer_sizes is None or not len(layer_sizes) > 1 or not all(size >= 1 for size in layer_sizes):
            raise ValueError

        self.dim_input = layer_sizes[0]
        self.dim_output = layer_sizes[-1]
        
        self.layer_sizes = layer_sizes
        
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(int(layer_sizes[i]), int(layer_sizes[i+1]))
                            for i in range(len(layer_sizes) - 1)])

        self.activation_function = activation_function
            
        self.settings = settings
        
        self.loss_function = self.settings['loss_function']

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation_function(self.layers[i](x))
        return self.layers[len(self.layers)-1](x)

    def configure_optimizers(self):
        optimizer = self.settings['optimizer'](self.parameters(), lr=self.settings['learning_rate'])

        if self.settings['lr_scheduler'] is not None:
            scheduler = self.settings['lr_scheduler'](optimizer, **self.settings['lr_scheduler_params'])
            return [optimizer], [scheduler]

        return optimizer


def apply_additional_components(obj, mu):
    return np.array([action(x) for x, action in zip(mu.to_numpy(), obj.additional_components) if action])


class ANNStateModel(Model):

    _compute_allowed_kwargs = frozenset(('return_error_sequence',))

    def __init__(self, mlm, solution_space, output_functional=None,
                 time_stepper=None,
                 products=None, parameters=None, name=None, error_estimator=None,
                 visualizer=None, temporal_l2_matrix=None, input_scaling=None,
                 inverse_output_scaling=None, additional_components=[]):
        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name or 'ANNStateModel')

        if time_stepper is not None:
            self.T = time_stepper.end_time

        if output_functional is not None:
            assert isinstance(output_functional, Operator)
            self.output_space = output_functional.range

            # define temporal norms
            def output_sup_norm(output):
                assert output in self.output_space
                return np.linalg.norm(output._array, ord=np.inf)
            self.output_sup_norm = output_sup_norm

            if temporal_l2_matrix is not None:
                def output_l2_norm(output):
                    assert output in self.output_space
                    output = output._array.reshape(-1, 1)
                    return (output.T @ (temporal_l2_matrix @ output))[0][0]
                self.output_l2_norm = output_l2_norm

        if time_stepper.num_values:
            self._nt = time_stepper.num_values
        elif hasattr(time_stepper, 'nt'):
            self._nt = time_stepper.nt + 1

        if input_scaling is None:
            self._input_scaling = lambda x: x
        else:
            self._input_scaling = input_scaling

        if inverse_output_scaling is None:
            self._inverse_output_scaling = lambda x: x
        else:
            self._inverse_output_scaling = inverse_output_scaling

        self.__auto_init(locals())
        
    def _compute_solution(self, mu=None, **kwargs):
        if self.mlm is None:
            # empty model
            if self.time_stepper.num_values:
                return self.solution_space.zeros(self.time_stepper.num_values)
            elif hasattr(self.time_stepper, 'nt'):
                return self.solution_space.zeros(self.time_stepper.nt + 1)
            else:
                return self.solution_space.empty()
        else:
            # scale parameters without time component first
            scaled_mu = self.parameters.parse(self._input_scaling(mu.to_numpy()))
            if self.additional_components:
                applied_additional_components = apply_additional_components(self, mu)

                inputs = torch.DoubleTensor([np.concatenate([scaled_mu.with_(t=t).to_numpy(), applied_additional_components])
                                             for t in np.linspace(0., self.T, self._nt)])
            else:
                inputs = torch.DoubleTensor([scaled_mu.with_(t=t).to_numpy()
                                             for t in np.linspace(0., self.T, self._nt)])

            # pass batch of inputs to neural network and reshape properly
            state = self.mlm(inputs).data.numpy().reshape(-1, self.solution_space.dim)
            state = self._inverse_output_scaling(state)
            # transform to object from solution space and return result
            return self.solution_space.make_array(state)

    def current_statistics(self):
        return {}


class ANNStateReductor(BasicObject):

    def __init__(self, fom, training_inputs=None, additional_training_data=None,
                 validation_ratio=0.1, hidden_layers='[3*(N+P), 3*(N+P), 3*(N+P)]', activation_function=nn.ReLU(),
                 optimizer=optim.Adam, epochs=1000, batch_size=128, learning_rate=5e-3,
                 target_loss=None, lr_scheduler=optim.lr_scheduler.StepLR,
                 lr_scheduler_params={'step_size': 10, 'gamma': 0.9}, seed=0,
                 automatic_input_scaling=True, automatic_output_scaling=True,
                 input_scaling=None, output_scaling=None, inverse_output_scaling=None,
                 additional_components=[], loss_function='weighted MSE',
                 time_sample_frequency=10, num_workers=1, gpus=0, early_stopping_patience=3,
                 tb_path='ML_MORE_TB_LOGS/'):
        # If activation_function = 'SDKN', a SDKN instead of a NN is trained
        
        assert isinstance(fom, Model)
        self.__auto_init(locals())
        self.computed_inputs, self.computed_states = self._parse_data(additional_training_data)
        self.new_inputs = self._parse_inputs(training_inputs)
        self._new_training_data = True
        self._mlm = None
        self._ann_mlm = None
        
        self.settings = {'learning_rate': learning_rate,
                         'num_epochs_nn': epochs,
                         'num_workers': num_workers,
                         'gpus': gpus,
                         'batch_size': batch_size,
                         'log_every_n_steps': 5,
                         'progress_bar_refresh_rate': 1,
                         'optimizer': optimizer,
                         'lr_scheduler': lr_scheduler,
                         'lr_scheduler_params': lr_scheduler_params,
                         'tb_path': tb_path,
                         'early_stopping_patience': early_stopping_patience}

        if self.fom.time_stepper.num_values:
            self._nt = self.fom.time_stepper.num_values
        elif hasattr(self.fom.time_stepper, 'nt'):
            self._nt = self.fom.time_stepper.nt + 1

        if self.automatic_input_scaling or input_scaling is None:
            if self.automatic_input_scaling:
                self.logger.info('Using automatic input scaling ...')
            self._input_scaling = lambda x: x
        else:
            self.logger.info('Using input scaling ...')
            self._input_scaling = input_scaling

        if self.automatic_output_scaling or output_scaling is None:
        # Watch out!! In .reduce() below there is another change of the _output_scaling!
            if self.automatic_output_scaling:
                self.logger.info('Using automatic output scaling ...')
            self._output_scaling = lambda x: x
            self._inverse_output_scaling = lambda x: x
        else:
            self.logger.info('Using output scaling ...')
            self._output_scaling = output_scaling
            self._inverse_output_scaling = inverse_output_scaling

        self._statistics = {
                'collected_data_samples': 0,
                'num_training_samples': 0,
                'num_validation_samples': 0,
                }

        assert not self.additional_components or len(self.additional_components) == self.fom.parameters.dim

        self._mlm = self.build_rom()

    def replace_fom(self, fom):
        self._locked = False
        self.fom = fom
        self._locked = True
        self._mlm = self.build_rom()

    def reduce(self):
        if self._new_training_data or self._mlm is None:
            # compute training data
            if len(self.new_inputs) > 0:
                with self.logger.block(f'Computing {len(self.new_inputs)} samples ...'):
                    self.computed_inputs += [mu for mu in self.new_inputs]
                    self.computed_states += [self.fom.solve(mu=mu).to_numpy()
                                             for mu in self.new_inputs]
                self.new_inputs = []
            if len(self.computed_inputs) > 0:
                training_inputs = [mu for mu in self.computed_inputs]
                training_states = [state for state in self.computed_states]
                self._statistics['collected_data_samples'] = len(training_inputs)

                if self.automatic_input_scaling:
                    bounds_input = np.array([np.min(np.array([mu.to_numpy() for mu in training_inputs]), axis=0),
                                       np.max(np.array([mu.to_numpy() for mu in training_inputs]), axis=0)])
                    assert bounds_input.shape == (2, self.fom.parameters.dim)
                    self._input_scaling = lambda x: 2. * (x - bounds_input[0]) / (bounds_input[1] - bounds_input[0]) - 1.

                if self.automatic_output_scaling:
                    bounds_output = np.array([np.min(np.array([state for state in training_states]), axis=(0,1)),
                                              np.max(np.array([state for state in training_states]), axis=(0,1))])
                    assert bounds_output.shape == (2, self.fom.solution_space.dim)
                    self._output_scaling = lambda x: 2. * (x - bounds_output[0]) / (bounds_output[1] - bounds_output[0]) - 1.
                    self._inverse_output_scaling = lambda x: (1. + x) / 2. * (bounds_output[1] - bounds_output[0]) + bounds_output[0]

                if self.additional_components:
                    # collect training data
                    training_data = [(np.concatenate([mu_t.to_numpy(), apply_additional_components(self, mu)]), self._output_scaling(u_t))
                                     for mu, u in zip(training_inputs, training_states)
                                     for mu_t, u_t in zip([self.fom.parameters.parse(self._input_scaling(mu.to_numpy())).with_(t=t)
                                                           for t in np.linspace(0, self.fom.T, self._nt)][::self.time_sample_frequency], u[::self.time_sample_frequency])]
                else:
                    training_data = [(mu_t.to_numpy(), self._output_scaling(u_t))
                                     for mu, u in zip(training_inputs, training_states)
                                     for mu_t, u_t in zip([self.fom.parameters.parse(self._input_scaling(mu.to_numpy())).with_(t=t)
                                                           for t in np.linspace(0, self.fom.T, self._nt)][::self.time_sample_frequency], u[::self.time_sample_frequency])]

                self._statistics['size'] = len(self.computed_inputs)
                    
                # fit model
                with self.logger.block(f'Fitting ANN model to {len(self.computed_inputs)} samples ...'):
                    # compute validation data
                    with self.logger.block('Splitting training data in training and validation set ...'):
                        number_validation_snapshots = int(len(training_data)*self.validation_ratio)
                        # randomly shuffle training data before splitting into two sets
                        np.random.shuffle(training_data)
                        # split training data into validation and training set
                        validation_data = training_data[0:number_validation_snapshots]
                        training_data = training_data[number_validation_snapshots+1:]
                        self._statistics['num_training_samples'] = len(training_data)
                        self._statistics['num_validation_samples'] = len(validation_data)

                    # run the actual training of the neural network
                    with self.logger.block('Training of neural network ...'):
                        # no target loss required at the moment; could be a potential option in the future to reduce the number of restarts
                        target_loss = None

                        # set parameters for neural network and training
                        neural_network_parameters = {'layer_sizes': self._compute_layer_sizes(self.hidden_layers),
                                                     'activation_function': self.activation_function}
                        
                        loss_function = nn.MSELoss()
                        if self.loss_function == 'weighted MSE':
                            weights = torch.Tensor(self.fom._svals)

                            def weighted_mse_loss_function(inputs, targets):
                                return (weights * (inputs - targets)**2).mean()

                            loss_function = weighted_mse_loss_function
                            self.logger.info('Using weighted MSE loss function ...')

                        self.settings['loss_function'] = loss_function

                        # initialize the neural network
                        self.logger.info('Initializing neural network ...')
                        
                        self.logger.info(f"Layer sizes: {neural_network_parameters['layer_sizes']}")
                        
                        if not (hasattr(self, 'neural_network') and 
                                self.neural_network.layer_sizes == neural_network_parameters['layer_sizes']) and self.activation_function == 'SDKN':
                            self.logger.info(f'Training new SDKN ...')
                            self.neural_network = LightningFullyConnectedSDKN(layer_sizes=neural_network_parameters['layer_sizes'],
                                                                            settings=self.settings).double()    

                        if not (hasattr(self, 'neural_network') and self.neural_network.layer_sizes == neural_network_parameters['layer_sizes']):
                            self.logger.info(f'Training new neural network ...')
                            self.neural_network = LightningFullyConnectedNN(layer_sizes=neural_network_parameters['layer_sizes'],
                                                                            settings=self.settings).double()
                        else:
                            self.logger.info(f'Starting with pretrained network ...')
                        self.logger.info(f'Using {type(self.neural_network).__name__} ...')

                        # run training algorithm with multiple restarts
                        self.logger.info('Starting training procedure ...')
                        self._ann_mlm = multiple_restarts_training(training_data, validation_data,
                                                                   self.neural_network, target_loss=self.target_loss,
                                                                   settings=self.settings, seed=self.seed)

            self._mlm = self.build_rom()
        return self._mlm

    def build_rom(self):
        if hasattr(self.fom, 'temporal_l2_matrix'):
            temporal_l2_matrix = self.fom.temporal_l2_matrix
        else:
            # assemble temporal norms
            from pymor.discretizers.builtin.cg import L2ProductP1
            from pymor.discretizers.builtin.grids.boundaryinfos import EmptyBoundaryInfo
            from pymor.discretizers.builtin.grids.oned import OnedGrid

            time_stepper = self.fom.time_stepper
            assert time_stepper.num_values or time_stepper.nt
            temporal_grid = OnedGrid(
                    domain=(time_stepper.initial_time, time_stepper.end_time),
                    num_intervals=(time_stepper.num_values - 1) if time_stepper.num_values else time_stepper.nt)
            temporal_l2_matrix = L2ProductP1(temporal_grid, EmptyBoundaryInfo(temporal_grid))
            temporal_l2_matrix = temporal_l2_matrix.assemble().matrix

        # build model
        return ANNStateModel(
            mlm=self._ann_mlm,
            solution_space=self.fom.solution_space,
            time_stepper=self.fom.time_stepper,
            output_functional=self.fom.output_functional,
            products=self.fom.products,
            parameters=self.fom.parameters,
            name=f'{self.fom.name}_ann_reduced',
            error_estimator=self.fom.error_estimator,
            visualizer=self.fom.visualizer,
            temporal_l2_matrix=temporal_l2_matrix,
            input_scaling=self._input_scaling,
            inverse_output_scaling=self._inverse_output_scaling,
            additional_components=self.additional_components)

    def extend_training_data(self, training_inputs=None, additional_training_data=None):
        training_inputs = self._parse_inputs(training_inputs)
        if len(training_inputs) > 0:
            self.new_inputs += training_inputs
            self._new_training_data = True
        for mu, U in zip(*self._parse_data(additional_training_data)):
            if mu not in self.computed_inputs:
                self.computed_inputs += [mu,]
                self.computed_states += [U,]
                self._new_training_data = True

    def current_statistics(self):
        return self._statistics

    def _compute_layer_sizes(self, hidden_layers):
        """Compute the number of neurons in the layers of the neural network."""

        # determine the numbers of neurons in the hidden layers, '[128, 128, 128]' also possible
        if isinstance(hidden_layers, str):
            hidden_layers = eval(hidden_layers, {'N': self.fom.solution_space.dim, 'P': self.fom.parameters.dim})
        # input and output size of the neural network are prescribed by the
        # dimension of the parameter space (P) and the reduced basis size (N)

        assert isinstance(hidden_layers, list)

        # Compute the final sizes. For the input, also respect the additional components
        if self.additional_components:
            additional_length = len(self.additional_components) - self.additional_components.count(False)
        else:
            additional_length = 0
        return [self.fom.parameters.dim + 1 + additional_length, ] + hidden_layers + [self.fom.solution_space.dim, ]
    
    def _parse_inputs(self, inputs):
        if inputs is None:
            return []
        else:
            assert isinstance(inputs, (tuple, list))
            return [m.parameters.parse(mu) if not isinstance(mu, Mu) else mu for mu in inputs]

    def _parse_data(self, data):
        if data is None:
            return [], []
        else:
            assert len(data) == 2
            inputs, states = data
            assert len(inputs) == len(states)
            inputs = [m.parameters.parse(mu) if not isinstance(mu, Mu) else mu
                      for mu in inputs]
            assert np.all(len(state) == len(states[0]) for output in states)
            assert np.all(state in self.m.solution_space for state in states)
            states = [state.to_numpy() if not isinstance(state, np.ndarray) else state
                      for state in states]
            return inputs, states
