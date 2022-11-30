# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Example script for the usage of neural networks in model order reduction (approach by Hesthaven and Ubbiali)

Usage:
    neural_networks.py [--fv] [--vis] GRID_INTERVALS TRAINING_SAMPLES VALIDATION_SAMPLES

Arguments:
    GRID_INTERVALS       Grid interval count.
    TRAINING_SAMPLES     Number of samples used for training the neural network.
    VALIDATION_SAMPLES   Number of samples used for validation during the training phase.

Options:
    -h, --help   Show this message.
    --fv         Use finite volume discretization instead of finite elements.
    --vis        Visualize full order solution and reduced solution for a test set.
"""

from docopt import docopt

import numpy as np

from pymor.basic import *

from pymor.core.config import config
from pymor.core.exceptions import TorchMissing


def create_fom(args):
    problem = StationaryProblem(
        domain=RectDomain(),
        rhs=LincombFunction(
            [ExpressionFunction('ones(x.shape[:-1]) * 10', 2, ()), ConstantFunction(1., 2)],
            [ProjectionParameterFunctional('mu'), 0.1]),
        diffusion=LincombFunction(
            [ExpressionFunction('1 - x[..., 0]', 2, ()), ExpressionFunction('x[..., 0]', 2, ())],
            [ProjectionParameterFunctional('mu'), 1]),
        dirichlet_data=LincombFunction(
            [ExpressionFunction('2 * x[..., 0]', 2, ()), ConstantFunction(1., 2)],
            [ProjectionParameterFunctional('mu'), 0.5]),
        name='2DProblem'
    )

    print('Discretize ...')
    discretizer = discretize_stationary_fv if args['--fv'] else discretize_stationary_cg
    fom, _ = discretizer(problem, diameter=1. / int(args['GRID_INTERVALS']))

    return fom, fom.l2_product if args['--fv'] else fom.h1_product


def neural_networks_demo(args):
    if not config.HAVE_TORCH:
        raise TorchMissing()

    fom, product = create_fom(args)

    parameter_space = fom.parameters.space((0.1, 1))

    from pymor.reductors.neural_network import StationaryNeuralNetworkRBReductor

    training_set = parameter_space.sample_uniformly(int(args['TRAINING_SAMPLES']))
    training_snapshots = fom.solution_space.empty()
    for mu in training_set:
        training_snapshots.append(fom.solve(mu))
    training_data = (training_set, training_snapshots)

    desired_MSE = 1e-5
    desired_l2_err = np.sqrt(len(training_set)*desired_MSE)
    RB, _ = pod(training_snapshots, product=product, l2_err=desired_l2_err)

    validation_set = parameter_space.sample_randomly(int(args['VALIDATION_SAMPLES']))
    validation_snapshots = fom.solution_space.empty()
    for mu in validation_set:
        validation_snapshots.append(fom.solve(mu))
    validation_data = (validation_set, validation_snapshots)

    reductor = StationaryNeuralNetworkRBReductor(
            fom, training_data, validation_data, RB=RB, product=product,
            ann_mse=desired_MSE, max_restarts=100, torch_seed=42)
    rom = reductor.reduce()

    test_set = parameter_space.sample_randomly(10)

    speedups = []

    import time

    print(f'Performing test on set of size {len(test_set)} ...')

    U = fom.solution_space.empty(reserve=len(test_set))
    U_red = fom.solution_space.empty(reserve=len(test_set))

    for mu in test_set:
        tic = time.time()
        U.append(fom.solve(mu))
        time_fom = time.time() - tic

        tic = time.time()
        U_red.append(reductor.reconstruct(rom.solve(mu)))
        time_red = time.time() - tic

        speedups.append(time_fom / time_red)

    absolute_errors = (U - U_red).norm(product)
    relative_errors = (U - U_red).norm(product) / U.norm(product)

    if args['--vis']:
        fom.visualize((U, U_red),
                      legend=('Full solution', 'Reduced solution'))

    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')


if __name__ == '__main__':
    args = docopt(__doc__)
    neural_networks_demo(args)
