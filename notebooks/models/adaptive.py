import numpy as np
from timeit import default_timer as timer

from pymor.algorithms.hapod import inc_model_hapod, inc_vectorarray_hapod
from pymor.models.interface import Model
from pymor.reductors.basic import ExtensionError
from pymor.tools.floatcmp import float_cmp_all


class AdaptiveModel(Model):
    '''
    Note: we currently train the ML-ROM after prolonging data due to RB-ROM update. This happens
          unconditionally, though we otherwise respect training_batch_size.
    '''

    def __init__(self, fom, rom_reductor_generator, mlm_reductor_generator,
                 solution_abs_tol=1e-2, output_abs_tol=1e-2, training_accuracy_factor=1,
                 training_batch_size=10, minimum_mlm_usage=0.6,
                 pod_l2_err=1e-4,
                 ensure_snapshot_reproduction=None,
                 name=None):
        super().__init__(name=name or 'AdaptiveModel')
        self.solution_space = fom.solution_space
        assert fom.output_space.dim == 1
        assert 0. <= minimum_mlm_usage <= 1.
        self.output_space = fom.output_space
        self.__auto_init(locals())

        self.T = fom.T
        if hasattr(fom, 'output_l2_norm'):
            self.output_l2_norm = fom.output_l2_norm
        if hasattr(fom, 'output_sup_norm'):
            self.output_sup_norm = fom.output_sup_norm

        self._rom_reductor = rom_reductor_generator(fom)
        self._rom = self._rom_reductor.reduce().with_(name=f'HaPOD-{fom.name}')
        self._svals = []
        self._rom._svals = self._svals
        self._mlm_reductor = mlm_reductor_generator(self._rom)
        self._mlm = self._mlm_reductor.reduce().with_(name=f'HaPOD-VKOGA-{fom.name}')
        self._collected_training_mus = []
        self._collected_training_states = []
        self._collection_counter = 0

        self._statistics = {
            'mus': [],
            'models': [],
            'ROM': {'dim': [],},
            'MLM': {'size': [],},
            'timings': {
                'eval_model': [],
                'build_model': []}
        }

    def _collect_return_statistics(self, mu, model_type, eval_model_time, build_model_time=None):
        self._statistics['mus'].append(mu)
        self._statistics['models'].append(model_type)
        self._statistics['ROM']['dim'].append(len(self._rom_reductor.bases['RB']))
        self._statistics['timings']['eval_model'].append(eval_model_time)
        self._statistics['timings']['build_model'].append(build_model_time)
        ML_statistics = self._mlm_reductor.current_statistics()
        for kk, vv in self._mlm_reductor.current_statistics().items():
            if kk not in self._statistics['MLM']:
                self._statistics['MLM'][kk] = []
            self._statistics['MLM'][kk].append(vv)

    # some things we don't want to support ATM
    def _compute(self, solution=False, output=False,
                 solution_error_estimate=False, output_error_estimate=False,
                 mu=None, **kwargs):
        raise NotImplementedError

    def _compute_output(self, solution, mu=None, **kwargs):
        raise NotImplementedError

    def compute(self, solution=False, output=False,
            solution_error_estimate=False, output_error_estimate=False, *,
            mu=None, **kwargs):
        raise NotImplementedError

    def estimate_error(self, mu=None, **kwargs):
        raise NotImplementedError

    def estimate_output_error(self, mu=None, **kwargs):
        raise NotImplementedError

    def solve(self, mu=None, return_error_estimate=False, **kwargs):
        with_wo_estimate = lambda U, err: (U, err) if return_error_estimate else U
        mu = self.parameters.parse(mu)
        with self.logger.block(f'solving for mu={mu}:'):
            self.logger.debug('solving MLM:')
            tic = timer()
            U_mlm, mlm_err = self._mlm.solve(mu=mu, return_error_estimate=True)
            eval_model_time = timer() - tic
            assert not np.isnan(mlm_err)
            if mlm_err < self.solution_abs_tol:
                self.logger.info(f'returning MLM solution (estimated error: {mlm_err})')
                tic = timer()
                U_mlm = self._rom_reductor.reconstruct(U_mlm)
                eval_model_time += timer() - tic
                self._collect_return_statistics(mu, 'MLM', eval_model_time)
                return with_wo_estimate(U_mlm, mlm_err)
            else:
                self.logger.debug(f'  estimated error ({mlm_err}) above tolerance, discarding MLM solution!')
                del U_mlm, mlm_err, eval_model_time
                self.logger.debug('solving ROM:')
                tic = timer()
                U_rb, rb_err = self._rom.solve(mu=mu, return_error_estimate=True)
                eval_model_time = timer() - tic
                assert not np.isnan(rb_err)
                if rb_err < self.solution_abs_tol*self.training_accuracy_factor:
                    tic = timer()
                    self._extend_mlm(mu, U_rb)
                    build_model_time = timer() - tic
                else:
                    build_model_time = None
                if rb_err < self.solution_abs_tol:
                    self.logger.info(f'returning ROM solution (estimated error: {rb_err})')
                    tic = timer()
                    U_rb = self._rom_reductor.reconstruct(U_rb)
                    eval_model_time += timer() - tic
                    self._collect_return_statistics(mu, 'ROM', eval_model_time, build_model_time)
                    return with_wo_estimate(U_rb, rb_err)
                else:
                    self.logger.debug(f'  estimated error ({rb_err}) above tolerance, discarding ROM solution!')
                    self.logger.debug('solving FOM:')
                    tic = timer()
                    U_h = self.fom.solve(mu=mu)
                    eval_FOM_time = timer() - tic
                    self.logger.debug('  performing HAPOD ...')

                    def _RB_projection(U):
                        RB = self._rom_reductor.bases['RB']
                        if len(RB) == 0:
                            return U
                        else:
                            return U.inner(RB, self._rom_reductor.products['RB'])

                    projection_error = lambda U: \
                        U - self._rom_reductor.bases['RB'].lincomb(_RB_projection(U_h)) \
                        if len(self._rom_reductor.bases['RB']) != 0 else U

                    tic = timer()
                    pod_modes, svals, _ = inc_vectorarray_hapod(
                        steps=len(U_h)/20, U=projection_error(U_h), eps=self.pod_l2_err, omega=0.01,
                        product=self._rom_reductor.products['RB'])
                    self._extend_rom(mu, pod_modes, svals)
                    build_ROM_time = timer() - tic
                    self.logger.debug('  computing orthogonal projection ...')
                    # compute anew, RB has been updated
                    tic = timer()
                    U_rb = _RB_projection(U_h)
                    eval_ROM_time = timer() - tic
                    tic = timer()
                    self._extend_mlm(mu, U_rb)
                    build_MLM_time = timer() - tic
                    self.logger.info('returning FOM solution!')
                    self._collect_return_statistics(
                            mu, 'FOM', eval_FOM_time, (build_ROM_time, eval_ROM_time, build_MLM_time))
                    return with_wo_estimate(U_h, 0.)

    def output(self, mu=None, return_error_estimate=False, **kwargs):
        with_wo_estimate = lambda U, err: (U, err) if return_error_estimate else U
        mu = self.parameters.parse(mu)
        with self.logger.block(f'computing output for mu={mu}:'):
            self.logger.debug('computing MLM output:')
            tic = timer()
            s_mlm, mlm_err = self._mlm.output(mu=mu, return_error_estimate=True)
            eval_model_time = timer() - tic
            assert not np.isnan(mlm_err)
            if mlm_err < self.output_abs_tol:
                self.logger.info(f'returning MLM output (estimated error: {mlm_err})')
                self._collect_return_statistics(mu, 'MLM', eval_model_time)
                return with_wo_estimate(s_mlm, mlm_err)
            else:
                self.logger.debug(f'  estimated error ({mlm_err}) above tolerance, discarding MLM output!')
                del s_mlm, mlm_err, eval_model_time
                self.logger.debug('computing ROM output:')
                tic = timer()
                rom_data = self._rom.compute(mu=mu,
                                             solution=True, solution_error_estimate=True,
                                             output=True, output_error_estimate=True)
                eval_model_time = timer() - tic
                U_rb, U_rb_err = rom_data['solution'], rom_data['solution_error_estimate']
                assert not np.isnan(U_rb_err)
                if U_rb_err < self.solution_abs_tol*self.training_accuracy_factor:
                    tic = timer()
                    self._extend_mlm(mu, U_rb)
                    build_model_time = timer() - tic
                else:
                    self.logger.debug(f'  ROM solution not accurate enough for training ({U_rb_err})!')
                    build_model_time = None
                s_rb, s_rb_err = rom_data['output'], rom_data['output_error_estimate']
                assert not np.isnan(s_rb_err)
                if s_rb_err < self.output_abs_tol:
                    self.logger.info(f'returning ROM output (estimated error: {s_rb_err})')
                    self._collect_return_statistics(mu, 'ROM', eval_model_time, build_model_time)
                    return with_wo_estimate(s_rb, s_rb_err)
                else:
                    self.logger.debug(f'  estimated error ({s_rb_err}) above tolerance, discarding ROM output!')
                    self.logger.debug('computing FOM output (+live solution HAPOD):')
                    s_h = self.fom.output_space.empty()

                    def transform(U):
                        s_h.append(self.fom.output_functional.apply(U))
                        RB = self._rom_reductor.bases['RB']
                        if len(RB) == 0:
                            return U
                        else:
                            return U - RB.lincomb(U.inner(RB, self._rom_reductor.products['RB']))

                    tic = timer()
                    pod_modes, svals, _ = inc_model_hapod(
                        m=self.fom,
                        mus=[mu,],
                        num_steps_per_chunk=20,
                        l2_err=self.pod_l2_err,
                        omega=0.01,
                        product=self._rom_reductor.products['RB'],
                        transform=transform)
                    eval_FOM_time = timer() - tic
                    tic = timer()
                    self._extend_rom(mu, pod_modes, svals)
                    build_ROM_time = timer() - tic
                    self.logger.debug('  solving ROM for MLM training ...')
                    tic = timer()
                    U_rb, rb_err = self._rom.solve(mu=mu, return_error_estimate=True)
                    eval_ROM_time = timer() - tic
                    if rb_err < self.solution_abs_tol*self.training_accuracy_factor:
                        tic = timer()
                        self._extend_mlm(mu, U_rb)
                        build_MLM_time = timer() - tic
                    else:
                        build_MLM_time = None
                    self.logger.info('returning FOM output!')
                    self._collect_return_statistics(
                            mu, 'FOM', eval_FOM_time, (build_ROM_time, eval_ROM_time, build_MLM_time))
                    return with_wo_estimate(s_h, 0.)

    def update_tolerances(self, solution_abs_tol=None, output_abs_tol=None):
        self._locked = False
        if solution_abs_tol is not None:
            # we can unconditionally set solution_abs_tol
            self.solution_abs_tol = solution_abs_tol
        if output_abs_tol is not None:
            if output_abs_tol < self.output_abs_tol and len(self._collected_training_mus) > 0:
                self.logger.debug(f'validating {len(self._collected_training_mus)} existing training samples against updated tolerance ...')
                # we need to check and possibly invalidate the training data for the MLM
                validated_training_mus = []
                validated_training_data = []
                for mu, U in zip(self._collected_training_mus, self._collected_training_states):
                    abs_output_err_est = self._mlm.error_estimator.estimate_output_error(U=U, mu=mu, m=self._mlm)
                    if abs_output_err_est < output_abs_tol:
                        # keep training data
                        validated_training_mus.append(mu)
                        validated_training_data.append(U)
                if len(validated_training_mus) < len(self._collected_training_mus):
                    self.logger.debug(f'  dropped {len(self._collected_training_mus) - len(validated_training_mus)} bad training samples')
                    # retrain the MLM
                    self._collected_training_mus = validated_training_mus
                    self._collected_training_states = validated_training_data
                    self.logger.debug('  training MLM ...')
                    self._mlm_reductor = self.mlm_reductor_generator(self._rom)
                    self._mlm_reductor.extend_training_data(additional_training_data=(
                        self._collected_training_mus, self._collected_training_states))
                    self._mlm = self._mlm_reductor.reduce().with_(name=f'HaPOD-VKOGA-{self.fom.name}')
                else:
                    self.logger.debug('  kept all training samples!')
        self.output_abs_tol = output_abs_tol
        self._locked = True

    def _extend_mlm(self, mu, U):
        assert len(self._collected_training_mus) == len(self._collected_training_states)
        if not U in self._rom.solution_space:
            U = self._rom.solution_space.from_numpy(U)
        do_train = False
        if self._collected_training_mus.count(mu) > 0:
            assert self._collected_training_mus.count(mu) == 1, 'This should not happen!'
            mu_ind = self._collected_training_mus.index(mu)
            U_old = self._collected_training_states[mu_ind]
            if not float_cmp_all(U._array, U_old._array):
                self.logger.debug(f'  updating already collected state for mu={mu}')
                self._collected_training_states[mu_ind] = U
                del U_old
                do_train = True
        else:
            self.logger.debug(f'  collecting state for mu={mu}')
            self._collected_training_mus.append(mu)
            self._collected_training_states.append(U)
            self._collection_counter += 1
            fraction_mlm_solves = self._statistics['models'][-self.training_batch_size:].count('MLM') / self.training_batch_size
            if self._collection_counter >= self.training_batch_size and fraction_mlm_solves < self.minimum_mlm_usage:
                self._collection_counter = 0
                do_train = True
        if do_train:
            self.logger.debug('  training MLM ...')
            if self._mlm_reductor.fom.solution_space.dim == self._rom.solution_space.dim:
                self.logger.info('Replacing RB-ROM in MLM ...')
                self._mlm_reductor.replace_fom(self._rom)
            else:
                self.logger.info('Constructing new ML-ROM-Reductor ...')
                self._mlm_reductor = self.mlm_reductor_generator(self._rom)
            self._mlm_reductor.extend_training_data(additional_training_data=(
                self._collected_training_mus, self._collected_training_states))
            self._mlm = self._mlm_reductor.reduce().with_(name=f'HaPOD-VKOGA-{self.fom.name}')
            if self.ensure_snapshot_reproduction is not None:
                assert isinstance(self.ensure_snapshot_reproduction, str)
                error_norm = getattr(self._rom, self.ensure_snapshot_reproduction)
                self.logger.debug('  ensuring snapshot reproduction ...')
                for mu, U_ref in zip(self._collected_training_mus, self._collected_training_states):
                    U_mlm = self._mlm.solve(mu=mu)
                    err = error_norm(U_ref - U_mlm)
                    assert not err > self.solution_abs_tol

    def _extend_rom(self, mu, pod_modes, svals):
        assert len(pod_modes) > 0, 'This should not happen, we only extend the ROM if the ROM is insufficient!'
        old_RB_size = len(self._rom_reductor.bases['RB'])
        self.logger.debug('  extending ROM ...')
        try:
            self._rom_reductor.extend_basis(pod_modes, method='gram_schmidt')
            self._svals.extend(svals)
            new_RB_size = len(self._rom_reductor.bases['RB'])
            assert new_RB_size > old_RB_size, 'This should have triggered an ExtensionError!'
            self._rom = self._rom_reductor.reduce().with_(name=f'HaPOD-{self.fom.name}')
            self._rom._svals = self._svals
            if len(self._collected_training_states) > 0:
                self.logger.debug('  prolonging MLM training data ...')
                # the basis extension is hierarchical, we can simply fill with 0
                for ii, U_old in enumerate(self._collected_training_states):
                    U_new = np.pad(U_old._array, ((0, 0), (0, new_RB_size - old_RB_size)))
                    self._collected_training_states[ii] = self._rom.solution_space.from_numpy(U_new)
                fraction_mlm_solves = self._statistics['models'][-self.training_batch_size:].count('MLM') / self.training_batch_size
                if self._collection_counter >= self.training_batch_size and fraction_mlm_solves < self.minimum_mlm_usage:
                    self.logger.debug('  training MLM ...')
                    self._mlm_reductor = self.mlm_reductor_generator(self._rom)
                    self._mlm_reductor.extend_training_data(additional_training_data=(
                        self._collected_training_mus, self._collected_training_states))
                    self._mlm = self._mlm_reductor.reduce().with_(name=f'HaPOD-VKOGA-{self.fom.name}')
        except ExtensionError:
            pass
