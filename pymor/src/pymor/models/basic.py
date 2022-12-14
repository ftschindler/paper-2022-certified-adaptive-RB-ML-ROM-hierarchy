# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.timestepping import TimeStepper
from pymor.models.interface import Model
from pymor.operators.constructions import IdentityOperator, VectorOperator, ZeroOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


class StationaryModel(Model):
    """Generic class for models of stationary problems.

    This class describes discrete problems given by the equation::

        L(u(μ), μ) = F(μ)

    with a vector-like right-hand side F and a (possibly non-linear) operator L.

    Note that even when solving a variational formulation where F is a
    functional and not a vector, F has to be specified as a vector-like
    |Operator| (mapping scalars to vectors). This ensures that in the complex
    case both L and F are anti-linear in the test variable.

    Parameters
    ----------
    operator
        The |Operator| L.
    rhs
        The vector F. Either a |VectorArray| of length 1 or a vector-like
        |Operator|.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    """

    def __init__(self, operator, rhs, output_functional=None, products=None,
                 error_estimator=None, visualizer=None, name=None):

        if isinstance(rhs, VectorArray):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')

        assert rhs.range == operator.range and rhs.source.is_scalar and rhs.linear
        assert output_functional is None or output_functional.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = operator.source
        self.linear = operator.linear and (output_functional is None or output_functional.linear)
        if output_functional is not None:
            self.output_space = output_functional.range

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    {"linear" if self.linear else "non-linear"}\n'
            f'    solution_space:  {self.solution_space}\n'
            f'    output_space:    {self.output_space}\n'
        )

    def _compute_solution(self, mu=None, **kwargs):
        return self.operator.apply_inverse(self.rhs.as_range_array(mu), mu=mu)


class InstationaryModel(Model):
    """Generic class for models of instationary problems.

    This class describes instationary problems given by the equations::

        M * ∂_t u(t, μ) + L(u(μ), t, μ) = F(t, μ)
                                u(0, μ) = u_0(μ)

    for t in [0,T], where L is a (possibly non-linear) time-dependent
    |Operator|, F is a time-dependent vector-like |Operator|, and u_0 the
    initial data. The mass |Operator| M is assumed to be linear.

    Parameters
    ----------
    T
        The final time T.
    initial_data
        The initial data `u_0`. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`) which
        applied to `NumpyVectorArray(np.array([1]))` will yield the
        initial data for given |parameter values|.
    operator
        The |Operator| L.
    rhs
        The right-hand side F.
    mass
        The mass |Operator| `M`. If `None`, the identity is assumed.
    time_stepper
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
        to be used by :meth:`~pymor.models.interface.Model.solve`.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    assemble_temporal_norms
        If True, provides L^2- and L^\infty-Bochner-norms for the solution
        and (if a scalar output is present) temporal L^2- and L^\infty-norms
        for the output.
    """


    _compute_allowed_kwargs = frozenset(('incremental', 'return_error_sequence'))


    def __init__(self, T, initial_data, operator, rhs, mass=None, time_stepper=None,
                 output_functional=None, products=None, error_estimator=None, visualizer=None, name=None,
                 assemble_temporal_norms=True):

        if isinstance(rhs, VectorArray):
            assert rhs in operator.range
            rhs = VectorOperator(rhs, name='rhs')
        if isinstance(initial_data, VectorArray):
            assert initial_data in operator.source
            initial_data = VectorOperator(initial_data, name='initial_data')
        mass = mass or IdentityOperator(operator.source)
        rhs = rhs or ZeroOperator(operator.source, NumpyVectorSpace(1))

        assert isinstance(time_stepper, TimeStepper)
        assert initial_data.source.is_scalar
        assert operator.source == initial_data.range
        assert rhs.linear and rhs.range == operator.range and rhs.source.is_scalar
        assert mass.linear and mass.source == mass.range == operator.source
        assert output_functional is None or output_functional.source == operator.source

        super().__init__(products=products, error_estimator=error_estimator, visualizer=visualizer, name=name)

        self.parameters_internal = {'t': 1}
        self.__auto_init(locals())
        self.solution_space = operator.source
        self.linear = operator.linear and (output_functional is None or output_functional.linear)
        if output_functional is not None:
            self.output_space = output_functional.range

        if assemble_temporal_norms:
            # assemble temporal L^2 matrix, TODO: determine temporal polynomial order from time_stepper
            from pymor.discretizers.builtin.cg import L2ProductP1
            from pymor.discretizers.builtin.grids.boundaryinfos import EmptyBoundaryInfo
            from pymor.discretizers.builtin.grids.oned import OnedGrid

            assert time_stepper.num_values or time_stepper.nt
            temporal_grid = OnedGrid(
                    domain=(time_stepper.initial_time, time_stepper.end_time),
                    num_intervals=(time_stepper.num_values -1 ) if time_stepper.num_values else time_stepper.nt)
            temporal_l2_product = L2ProductP1(temporal_grid, EmptyBoundaryInfo(temporal_grid))
            self.temporal_l2_product = temporal_l2_product.assemble()

            # define base Bochner norms
            def bochner_sup_norm(spatial_norm, U, mu=None):
                assert len(U) == self.temporal_l2_product.matrix.shape[0], \
                    f'Given VectorArray (len={len(U)}) does not match TimeStepper ({self.temporal_l2_product.matrix.shape[0]} values)!'
                spatial_norms = spatial_norm(U, mu=mu)
                return np.linalg.norm(spatial_norms, ord=np.inf)

            def bochner_l2_norm(spatial_norm, U, mu=None):
                assert len(U) == self.temporal_l2_product.matrix.shape[0], \
                    f'Given VectorArray (len={len(U)}) does not match TimeStepper ({self.temporal_l2_product.matrix.shape[0]} values)!'
                spatial_norms = spatial_norm(U, mu=mu).reshape(-1, 1)
                return np.sqrt(spatial_norms.T @ (self.temporal_l2_product.matrix @ spatial_norms))[0][0]

            # build full Bochner norms
            for kk in self.products.keys():
                setattr(self, f'bochner_sup_{kk}_norm',
                        lambda U, mu=None: bochner_sup_norm(getattr(self, f'{kk}_norm'), U, mu))
                setattr(self, f'bochner_l2_{kk}_norm',
                        lambda U, mu=None: bochner_l2_norm(getattr(self, f'{kk}_norm'), U, mu))

            # build output norms, if applicable
            if output_functional is not None and self.output_space.dim == 1:

                def output_sup_norm(output):
                    assert output in self.output_space
                    return np.linalg.norm(output._array, ord=np.inf)

                def output_l2_norm(output):
                    assert output in self.output_space
                    output = output._array.reshape(-1, 1)
                    return np.sqrt(output.T @ (self.temporal_l2_product.matrix @ output))[0][0]

                self.output_sup_norm = output_sup_norm
                self.output_l2_norm = output_l2_norm

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    {"linear" if self.linear else "non-linear"}\n'
            f'    T: {self.T}\n'
            f'    solution_space:  {self.solution_space}\n'
            f'    output_space:    {self.output_space}\n'
        )

    def with_time_stepper(self, **kwargs):
        return self.with_(time_stepper=self.time_stepper.with_(**kwargs))

    def _compute(self, solution=False, output=False,
                 solution_error_estimate=False, output_error_estimate=False,
                 mu=None, **kwargs):

        # delegate standard case to other methods
        if not ('incremental' in kwargs and kwargs['incremental']):
            return {}

        # the incremental output case
        assert not solution
        assert not solution_error_estimate
        assert not output_error_estimate
        if not hasattr(self, 'output_functional'):
            raise NotImplementedError
        if self.output_functional is None:
            raise ValueError('Model has no output')

        outputs = self.output_space.empty()
        t, _, data = self.time_stepper.bootstrap(initial_data=self.initial_data, operator=self.operator, rhs=self.rhs,
                                                 mass=self.mass, mu=mu, reserve=False)

        while not (t > self.time_stepper.end_time or np.allclose(t, self.time_stepper.end_time)):
            t, U_t = self.time_stepper.step(t, data, mu=mu)
            outputs.append(self.output_functional.apply(U_t, mu=mu.with_(t=t) if mu else Mu({'t': t})))

        return {'output': outputs, 'solution': None}

    def _compute_solution(self, mu=None, **kwargs):
        return self.time_stepper.solve(
                initial_data=self.initial_data, operator=self.operator, rhs=self.rhs, mass=self.mass, mu=mu)

    def _compute_solution_bootstrap(self, mu=None):
        return self.time_stepper.bootstrap(
                initial_data=self.initial_data, operator=self.operator, rhs=self.rhs, mass=self.mass, mu=mu,
                reserve=False)

    def _compute_solution_step(self, t, data, mu=None):
        return self.time_stepper.step(t, data, mu=mu)

    def to_lti(self):
        """Convert model to |LTIModel|.

        This method interprets the given model as an |LTIModel|
        in the following way::

            - self.operator        -> A
            self.rhs               -> B
            self.output_functional -> C
            None                   -> D
            self.mass              -> E
        """
        if self.output_functional is None:
            raise ValueError('No output defined.')
        A = - self.operator
        B = self.rhs
        C = self.output_functional
        E = self.mass

        if not all(op.linear for op in [A, B, C, E]):
            raise ValueError('Operators not linear.')

        from pymor.models.iosys import LTIModel
        return LTIModel(A, B, C, E=E, visualizer=self.visualizer)
