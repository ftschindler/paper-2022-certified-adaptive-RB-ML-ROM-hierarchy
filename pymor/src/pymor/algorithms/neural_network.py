# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


from pymor.core.config import config


if config.HAVE_TORCH:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils as utils

    from pymor.core.base import BasicObject
    from pymor.core.exceptions import NeuralNetworkTrainingFailed
    from pymor.core.logger import getLogger


    def train_ann(training_data, validation_data, layers,
                  activation_function=torch.tanh, optimizer=optim.LBFGS,
                  epochs=1000, batch_size=20, learning_rate=1., seed=None):

        logger = getLogger('pymor.algorithms.neural_network.train_ann')

        for data in training_data, validation_data:
            assert isinstance(data, list)
            assert all(isinstance(datum, tuple) and len(datum) == 2 for datum in data)
            assert all(isinstance(datum[0], torch.DoubleTensor) for datum in data)
            assert all(isinstance(datum[1], torch.DoubleTensor) for datum in data)

        # set a seed for the PyTorch initialization of weights and biases and further PyTorch methods
        if seed:
            torch.manual_seed(seed)

        # LBFGS-optimizer does not support mini-batching, so the batch size needs to be adjusted
        if optimizer == optim.LBFGS:
            batch_size = max(len(training_data), len(validation_data))

        logger.info('Initializing neural network ...')

        # initialize the neural network
        neural_network = FullyConnectedNN(layers,
                activation_function=activation_function).double()

        # initialize the optimizer
        optimizer = optimizer(neural_network.parameters(),
                lr=learning_rate)

        loss_function = nn.MSELoss()
        early_stopping_scheduler = EarlyStoppingScheduler(len(training_data) + len(validation_data))

        # create the training and validation sets as well as the respective data loaders
        training_dataset = CustomDataset(training_data)
        validation_dataset = CustomDataset(validation_data)
        phases = ['train', 'val']
        training_loader = utils.data.DataLoader(training_dataset,
                batch_size=batch_size)
        validation_loader = utils.data.DataLoader(validation_dataset,
                batch_size=batch_size)
        dataloaders = {'train':  training_loader, 'val': validation_loader}

        logger.info('Starting optimization procedure ...')

        # perform optimization procedure
        for epoch in range(epochs):
            losses = {'full': 0.}

            # alternate between training and validation phase
            for phase in phases:
                if phase == 'train':
                    neural_network.train()
                else:
                    neural_network.eval()

                running_loss = 0.0

                # iterate over batches
                for batch in dataloaders[phase]:
                    inputs = batch[0]
                    targets = batch[1]

                    with torch.set_grad_enabled(phase == 'train'):
                        def closure():
                            if torch.is_grad_enabled():
                                optimizer.zero_grad()
                            outputs = neural_network(inputs)
                            loss = loss_function(outputs, targets)
                            if loss.requires_grad:
                                loss.backward()
                            return loss

                        # perform optimization step
                        if phase == 'train':
                            optimizer.step(closure)

                        # compute loss of current batch
                        loss = closure()

                    # update overall absolute loss
                    running_loss += loss.item() * len(batch[0])

                # compute average loss
                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                losses[phase] = epoch_loss

                losses['full'] += running_loss

                # check for early stopping
                if phase == 'val' and early_stopping_scheduler(losses, neural_network):
                    logger.info(f'Stopping training process early after {epoch + 1} epochs with validation loss '
                                f'of {early_stopping_scheduler.best_losses["val"]}')
                    return early_stopping_scheduler.best_neural_network, early_stopping_scheduler.best_losses


    def restarted_ann_training(training_data, validation_data, layers, target_loss=None, max_restarts=10, seed=None,
                               train_ann_params=None):

        logger = getLogger('pymor.algorithms.neural_network.restarted_ann_training')

        # if applicable, set a common seed for the PyTorch initialization
        # of weights and biases and further PyTorch methods for all training runs ...
        train_ann_params = train_ann_params or {}
        assert isinstance(train_ann_params, dict)
        if not seed and 'seed' in train_ann_params:
            seed = train_ann_params['seed']
        train_ann_params.pop('seed', None) # ... and consequntly remove it here
        if seed:
            torch.manual_seed(seed)

        if target_loss:
            logger.info(f'Performing up to {max_restarts} restart{"s" if max_restarts > 1 else ""} to train an ANN '
                        f'with a loss below {target_loss}')
        else:
            logger.info(f'Performing up to {max_restarts} restart{"s" if max_restarts > 1 else ""} to find the ANN '
                        'with the lowest loss')

        with logger.block('Training ANN #0 ...'):
            neural_network, losses = train_ann(training_data, validation_data, layers, **train_ann_params)

        for run in range(1, max_restarts):

            if target_loss and losses['full'] <= target_loss:
                logger.info(f'Finished training after {run} restart{"s" if run > 1 else ""}, '
                            f'found ANN with {losses["full"]} loss')
                return neural_network, losses

            with logger.block(f'Training ANN #{run}'):
                current_nn, current_losses = train_ann(training_data, validation_data, layers, **train_ann_params)

            if current_losses['val'] < losses['val']:
                logger.info(f'Found better ANN (loss of {current_losses["val"]} instead of {losses["val"]})')
                neural_network = current_nn
                losses = current_losses
            else:
                logger.info(f'Rejecting ANN with loss of {current_losses["val"]} (instead of {losses["val"]})')

        if target_loss:
            raise NeuralNetworkTrainingFailed(f'Could not find ANN with prescribed loss of {target_loss} '
                                              f'(best one found was {losses["val"]})!')
        logger.info(f'Found ANN with validation error of {losses["val"]}')
        return neural_network, losses


    class FullyConnectedNN(nn.Module, BasicObject):
        """Class for neural networks with fully connected layers.

        This class implements neural networks consisting of linear and fully connected layers.
        Furthermore, the same activation function is used between each layer, except for the
        last one where no activation function is applied.

        Parameters
        ----------
        layers_sizes
            List of sizes (i.e. number of neurons) for the layers of the neural network.
        activation_function
            Function to use as activation function between the single layers.
        """

        def __init__(self, layers_sizes, activation_function=torch.tanh):
            super().__init__()

            if layers_sizes is None or not len(layers_sizes) > 1 or not all(size >= 1 for size in layers_sizes):
                raise ValueError

            self.input_dimension = layers_sizes[0]
            self.output_dimension = layers_sizes[-1]

            self.layers = nn.ModuleList()
            self.layers.extend([nn.Linear(int(layers_sizes[i]), int(layers_sizes[i+1]))
                for i in range(len(layers_sizes) - 1)])

            self.activation_function = activation_function

            if not self.logging_disabled:
                self.logger.info(f'Architecture of the neural network:\n{self}')

        def forward(self, x):
            """Performs the forward pass through the neural network.

            Applies the weights in the linear layers and passes the outcomes to the
            activation function.

            Parameters
            ----------
            x
                Input for the neural network.

            Returns
            -------
            The output of the neural network for the input x.
            """
            for i in range(len(self.layers) - 1):
                x = self.activation_function(self.layers[i](x))
            return self.layers[len(self.layers)-1](x)


    class EarlyStoppingScheduler(BasicObject):
        """Class for performing early stopping in training of neural networks.

        If the validation loss does not decrease over a certain amount of epochs, the
        training should be aborted to avoid overfitting the training data.
        This class implements an early stopping scheduler that recommends to stop the
        training process if the validation loss did not decrease by at least `delta`
        over `patience` epochs.

        Parameters
        ----------
        size_training_validation_set
            Size of both, training and validation set together.
        patience
            Number of epochs of non-decreasing validation loss allowed, before early
            stopping the training process.
        delta
            Minimal amount of decrease in the validation loss that is required to reset
            the counter of non-decreasing epochs.
        """

        def __init__(self, size_training_validation_set, patience=10, delta=0.):
            self.__auto_init(locals())

            self.best_losses = None
            self.best_neural_network = None
            self.counter = 0

        def __call__(self, losses, neural_network=None):
            """Returns `True` if early stopping of training is suggested.

            Parameters
            ----------
            losses
                Dictionary of losses on the validation and the training set in
                the current epoch.
            neural_network
                Neural network that produces the current validation loss.

            Returns
            -------
            `True` if early stopping is suggested, `False` otherwise.
            """
            if self.best_losses is None:
                self.best_losses = losses
                self.best_losses['full'] /= self.size_training_validation_set
                self.best_neural_network = neural_network
            elif self.best_losses['val'] - self.delta <= losses['val']:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            else:
                self.best_losses = losses
                self.best_losses['full'] /= self.size_training_validation_set
                self.best_neural_network = neural_network
                self.counter = 0

            return False


    class CustomDataset(utils.data.Dataset):
        """Class that represents the dataset to use in PyTorch.

        Parameters
        ----------
        training_data
            Set of training parameters and the respective coefficients of the
            solution in the reduced basis.
        """

        def __init__(self, training_data):
            self.training_data = training_data

        def __len__(self):
            return len(self.training_data)

        def __getitem__(self, idx):
            t = self.training_data[idx]
            return t
