# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Example script for the usage of neural networks in model order reduction (approach by Hesthaven and Ubbiali)

Usage:
    neural_networks.py [--vis] TRAINING_SAMPLES VALIDATION_SAMPLES

Arguments:
    TRAINING_SAMPLES     Number of samples used for training the neural network.
    VALIDATION_SAMPLES   Number of samples used for validation during the training phase.

Options:
    -h, --help   Show this message.
    --vis        Visualize full order solution and reduced solution for a test set.
"""

from docopt import docopt

import numpy as np

from pymor.basic import *

from pymor.core.config import config
from pymor.core.exceptions import TorchMissing


def create_fom(args):
    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.functions import (
        ConstantFunction,
        ExpressionFunction,
        GenericFunction,
        LincombFunction,
    )
    from pymor.analyticalproblems.domaindescriptions import RectDomain
    from pymor.parameters.functionals import ProjectionParameterFunctional

    problem = StationaryProblem(
        domain=RectDomain(([0, 0], [1, 1]),
                          left='dirichlet', top='neumann', right='neumann', bottom='neumann'),

        rhs=ConstantFunction(0, dim_domain=2),

        # diffusion in the lower part determined by parameter, upper part is 1
        diffusion=LincombFunction(
            [ExpressionFunction('1.*(x[..., 1] < 0.1)', dim_domain=2),
             ExpressionFunction('1.*(x[..., 1] > 0.1)', dim_domain=2)],
            [ProjectionParameterFunctional('diffusion', 1, 0),
             1]
        ),

        # pointing up
        advection=GenericFunction(
            mapping=lambda x: np.vstack([np.zeros(x.shape[0]), x[..., 0]]).T,
            dim_domain=2, shape_range=(2,)),

        # reaction in the upper part determined by parameter, lower part is 1
        reaction=LincombFunction(
            [ExpressionFunction('1.*(x[..., 1] > 0.9)', dim_domain=2),
             ExpressionFunction('1.*(x[..., 1] < 0.9)', dim_domain=2)],
            [ProjectionParameterFunctional('reaction', 1, 0),
             1]
        ),

        dirichlet_data=ConstantFunction(0, dim_domain=2),

        # -1 on the right boundary, 0 else
        neumann_data=ExpressionFunction('-1.*(x[..., 0] > (1 - 1e-7))', dim_domain=2),

        # the average on the right
        outputs=(('l2', ExpressionFunction('20.*(x[..., 0] > 0.95)', dim_domain=2)),),

        # parameter bounds, lower end to ensure solvability of the problem
        parameter_ranges={'diffusion': (0.1, 1), 'reaction': (0, 10)},    
    )

    print('Discretize ...')
    fom, _ = discretize_stationary_cg(problem, diameter=np.sqrt(2)/100, grid_type=RectGrid)

    return fom


def neural_networks_demo(args):
    if not config.HAVE_TORCH:
        raise TorchMissing()

    fom = create_fom(args)

    parameter_space = fom.parameters.space({'diffusion': (0.1, 1), 'reaction': (0, 10)})

    from pymor.reductors.neural_network import NeuralNetworkOutputReductor

    training_set = parameter_space.sample_uniformly(int(args['TRAINING_SAMPLES']))
    validation_set = parameter_space.sample_randomly(int(args['VALIDATION_SAMPLES']))

    reductor = NeuralNetworkOutputReductor(fom, training_set, validation_set)
    rom = reductor.reduce(hidden_layers=[30, 30, 30], restarts=10)

    test_set = parameter_space.sample_randomly(10)

    speedups = []

    import time

    print(f'Performing test on set of size {len(test_set)} ...')

    U = fom.output_space.empty(reserve=len(test_set))
    U_red = fom.output_space.empty(reserve=len(test_set))

    for mu in test_set:
        tic = time.time()
        U.append(fom.compute(output=True, mu=mu)['output'])
        time_fom = time.time() - tic

        tic = time.time()
        U_red.append(rom.compute(output=True, mu=mu)['output'])
        time_red = time.time() - tic

        speedups.append(time_fom / time_red)

    absolute_errors = (U - U_red).norm()
    relative_errors = (U - U_red).norm() / U.norm()

    if args['--vis']:
        fom.visualize((U, U_red),
                      legend=('Full solution', 'Reduced solution'))

    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')


if __name__ == '__main__':
    args = docopt(__doc__)
    neural_networks_demo(args)
