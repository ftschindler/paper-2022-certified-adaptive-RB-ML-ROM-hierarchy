import numpy as np

from vkoga import VKOGA, Gaussian

from pymor.core.base import BasicObject
from pymor.discretizers.builtin.cg import L2ProductP1
from pymor.discretizers.builtin.grids.boundaryinfos import EmptyBoundaryInfo
from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.models.interface import Model
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu, Parameters
from pymor.vectorarrays.numpy import NumpyVectorSpace


class VkogaStateModel(Model):

    _compute_allowed_kwargs = frozenset(('return_error_sequence',))

    def __init__(self, mlm, solution_space, output_functional=None,
                 time_stepper=None,
                 products=None, parameters=None, name=None, error_estimator=None,
                 visualizer=None, temporal_l2_matrix=None, input_scaling=None):
        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name or 'VkogaStateModel')

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

        if input_scaling is None:
            self.input_scaling = lambda x: x
        else:
            self.input_scaling = input_scaling

        self.__auto_init(locals())

    def _compute_solution(self, mu=None, **kwargs):
        if self.mlm.ctrs_ is None:
            # empty model
            if self.time_stepper.num_values:
                return self.solution_space.zeros(self.time_stepper.num_values)
            elif hasattr(self.time_stepper, 'nt'):
                return self.solution_space.zeros(self.time_stepper.nt + 1)
            else:
                return self.solution_space.empty()
        else:
            input = mu.to_numpy().reshape(1, -1)

            state = self.mlm.predict(self.input_scaling(input)).reshape(-1, self.solution_space.dim)
            return self.solution_space.make_array(state)


class VkogaStateReductor(BasicObject):

    def __init__(self, fom, training_inputs=None, max_iter=None, tol_p=1e-10, kernel_par=1, greedy_type='p_greedy',
                 vkoga_verbose=False, warm_restart=False, additional_training_data=None, input_scaling=None):
        assert isinstance(fom, Model)
        self.__auto_init(locals())
        self.computed_inputs, self.computed_states = self._parse_data(additional_training_data)
        self.new_inputs = self._parse_inputs(training_inputs)
        self._new_training_data = True
        self._mlm = None
        self._build_VKOGA = lambda : VKOGA( \
            tol_p=self.tol_p, kernel_par=self.kernel_par, greedy_type=self.greedy_type, \
            kernel=Gaussian(ep=1/np.sqrt(fom.parameters.dim)) \
        )
        self._vkoga_mlm = self._build_VKOGA()
        self._vkoga_mlm.verbose = self.vkoga_verbose
        self._mlm = self.build_rom()

        if input_scaling is None:
            self.input_scaling = lambda x: x
        else:
            self.input_scaling = input_scaling

        self._statistics = {
                'collected_data_samples': 0,
                'num_training_samples': 0,
                'num_validation_samples': 0,
                }

    def replace_fom(self, fom):
        self._locked = False
        self.fom = fom
        self._locked = True
        self._mlm = self.build_rom()

    def reduce(self):
        if self._new_training_data or self._mlm is None:
            if not self.warm_restart:
                self._vkoga_mlm = self._build_VKOGA()
                self._vkoga_mlm.verbose = self.vkoga_verbose
            # compute training data
            if len(self.new_inputs) > 0:
                with self.logger.block(f'Computing {len(self.new_inputs)} samples ...'):
                    self.computed_inputs += [mu for mu in self.new_inputs]
                    self.computed_states += [self.fom.solve(mu=mu).to_numpy().ravel()
                                             for mu in self.new_inputs]
                self.new_inputs = []
            if len(self.computed_inputs) > 0:
                training_inputs = np.stack([mu.to_numpy() for mu in self.computed_inputs])
                training_states = np.stack([state.ravel() for state in self.computed_states])
                self._statistics['collected_data_samples'] = len(self.computed_inputs)
                self._statistics['num_training_samples'] = len(training_inputs)
                self._statistics['num_validation_samples'] = 0
                # fit model
                with self.logger.block(f'Fitting VKOGA model to {len(self.computed_inputs)} samples ...'):
                    self._vkoga_mlm.fit(self.input_scaling(training_inputs),
                                        training_states,
                                        maxIter=self.max_iter or len(training_inputs))
                self._statistics['size'] = len(
                        self._vkoga_mlm.ctrs_ if self._vkoga_mlm.ctrs_ is not None else [])
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
                    num_intervals=(time_stepper.num_values -1 ) if time_stepper.num_values else time_stepper.nt)
            temporal_l2_matrix = L2ProductP1(temporal_grid, EmptyBoundaryInfo(temporal_grid))
            temporal_l2_matrix = temporal_l2_matrix.assemble().matrix

        # build model
        return VkogaStateModel(
            mlm=self._vkoga_mlm,
            solution_space=self.fom.solution_space,
            time_stepper=self.fom.time_stepper,
            output_functional=self.fom.output_functional,
            products=self.fom.products,
            parameters=self.fom.parameters,
            name=f'VOGA-{self.fom.name}',
            error_estimator=self.fom.error_estimator,
            visualizer=self.fom.visualizer,
            temporal_l2_matrix=temporal_l2_matrix,
            input_scaling=self.input_scaling)

    def extend_training_data(self, training_inputs=None, additional_training_data=None):
        training_inputs = self._parse_inputs(training_inputs)
        if len(training_inputs) > 0:
            self.new_inputs += training_inputs
            self._new_training_data = True
        additional_training_inputs, additional_training_states = self._parse_data(additional_training_data)
        if len(additional_training_inputs) > 0:
            self.computed_inputs += additional_training_inputs
            self.computed_states += additional_training_states
            self._new_training_data = True

    def current_statistics(self):
        return self._statistics

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
            states = [state.to_numpy().ravel() if not isinstance(state, np.ndarray) else state.ravel()
                      for state in states]
            return inputs, states
