---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from itertools import cycle
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../VKOGA')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # tools.py

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings

# from matplotlib import cm # required for colors
from matplotlib import pyplot as plt
%matplotlib inline
from matplotlib.colors import TABLEAU_COLORS as COLORS

from functools import partial
from tools import draw_current_plot

# for C++ output
%load_ext wurlitzer
```

```python
prefix = 'KLAIM21__RBMLROM_minimization__run'
try:
    os.mkdir(prefix)
except FileExistsError:
    pass

# FOM
num_refines = 0
num_timesteps = 1000
scheme = 'dune-CG-P1'
# MLM
MLM_opts = {
    'type': 'VKOGA',
    'training_batch_size': 1,
}

desired_abs_output_err = 0.00001

filename = f"{prefix}/{prefix}_{num_refines}_{num_timesteps}_{MLM_opts['type']}_{MLM_opts['training_batch_size']}_{desired_abs_output_err}"

with open(f'{filename}.txt', 'w'):
    pass  # clears the log file

from pymor.core.defaults import set_defaults
set_defaults({'pymor.core.logger.getLogger.filename': f'{filename}.txt'})

from pymor.core.logger import set_log_levels, getLogger
set_log_levels({ # disable logging for some components
    'main': 'DEBUG',
    'pymor': 'WARN',
    'pymor.models': 'WARN',
    'pymor.discretizers.builtin': 'WARN',
    'pymor.discretizers.dunegdt': 'DEBUG',
    'pymor.analyticalproblems.functions.BitmapFunction': 'ERROR',
    'models.ann.ANNReductor': 'DEBUG',
    'models.vkoga.VkogaStateModel': 'INFO',
    'models.vkoga.VkogaStateReductor': 'DEBUG',
    'models.adaptive': 'DEBUG'})

logger = getLogger('main.main')
```

## FOM

```python
def setup_problem_and_discretize(num_refines, nt):
    from models.spe10channel import make_problem, discretize

    grid, boundary_info, problem, parameter_space, mu_bar = make_problem(
        regime='diffusion dominated', num_global_refines=num_refines)

    fom, fom_data, coercivity_estimator = discretize(
        grid, boundary_info, problem, mu_bar, nt=nt)

    return parameter_space, fom, fom_data, coercivity_estimator


spatial_product = lambda m: m.energy_0_product
spatial_norm = lambda m: m.energy_0_norm
output_norm = lambda m: m.output_l2_norm
bochner_norm = lambda m: m.bochner_l2_energy_norm
```

```python
logger.info('creating FOM:')
tic = time.time()

parameter_space, fom, fom_data, coercivity_estimator = setup_problem_and_discretize(
    num_refines, num_timesteps)

fom_offline_time = time.time() - tic
logger.info(f'  discretizing took {fom_offline_time}s')
logger.info(f'  grid has {fom_data["grid"].size(0)} elements, FOM has {fom.solution_space.dim} DoFs, uses {fom.time_stepper.nt} time steps')

logger.info(f'  input parameter space is {parameter_space.parameters.dim}-dimensional:')
logger.info(f'    {parameter_space}')
```

```python
from pymor.algorithms.to_matrix import to_matrix
from pymor.bindings.dunegdt import DuneXTMatrixOperator
from pymor.core.base import ImmutableObject
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.operators.numpy import NumpyMatrixOperator

class ConvertedVisualizer(ImmutableObject):
    
    def __init__(self, visualizer, vector_space):
        self.__auto_init(locals())
        
    def visualize(self, U, *args, **kwargs):
        V = self.vector_space.zeros(len(U))
        for ii in range(len(U)):
            v = np.array(V._list[ii].real_part.impl, copy=False)
            v[:] = U[ii].to_numpy()[:]
        self.visualizer.visualize(V, *args, **kwargs)


def to_numpy(obj):
    
    if isinstance(obj, (DuneXTMatrixOperator, VectorArrayOperator)):
        return NumpyMatrixOperator(to_matrix(obj))
    elif isinstance(obj, LincombOperator):
        return obj.with_(
            operators=[to_numpy(op) for op in obj.operators],
            solver_options=None,
        )
    elif isinstance(obj, InstationaryModel):
        return obj.with_(
            operator = to_numpy(obj.operator),
            rhs = to_numpy(obj.rhs),
            mass = to_numpy(obj.mass),
            products = {kk: to_numpy(vv) for kk, vv in obj.products.items()},
            initial_data = to_numpy(obj.initial_data),
            output_functional = to_numpy(obj.output_functional),
            visualizer=ConvertedVisualizer(fom.visualizer, fom.solution_space)
        )
    
    assert False, "We should not get here!"
```

#### learn some constants from the FOM

```python
logger.info('computing dual norm of output functional:')

assert not fom.output_functional.parametric
riesz_representative = spatial_product(fom).apply_inverse(fom.output_functional.as_vector())
dual_norm_output = np.sqrt(spatial_product(fom).apply2(riesz_representative, riesz_representative)[0][0])
del riesz_representative

logger.info(f'  {dual_norm_output}')
```

```python
mu_ref = parameter_space.sample_uniformly(1)[0]

logger.info(f'computing f(mu_ref={mu_ref}) ...')

tic = time.time()
f_mu_ref = fom.output(mu=mu_ref, incremental=True)  # no need to keep the state trajectory in memory
fom_online_output_time = time.time() - tic

logger.info(f'average FOM output (solve + apply functional) time: {fom_online_output_time}s')

logger.info('computing ||f_mu_ref||_{L^2(0, T)}:')
ref_output_norm = output_norm(fom)(f_mu_ref)
logger.info(f'  {ref_output_norm}')
```

```python
bounds = [[], []]
for kk in parameter_space.ranges.keys():
    for jj in range(parameter_space.parameters[kk]):
        bounds[0].append(parameter_space.ranges[kk][0])
        bounds[1].append(parameter_space.ranges[kk][1])
bounds = np.array(bounds)
```

## RB-ML-ROM

```python
from pymor.reductors.parabolic import ParabolicRBReductor

from models.adaptive import AdaptiveModel
from models.vkoga import VkogaStateReductor
from models.ann import ANNStateReductor


def make_adaptive_model(abs_output_tol):

    abs_state_tol = abs_output_tol/dual_norm_output
    pod_l2_tol = 1e-15

    logger.info(f'creating adaptive {MLM_opts["type"]} model (for abs_output_tol={abs_output_tol}) with')
    logger.info(f'- abs_output_tol={abs_output_tol}')
    logger.info(f'- abs_state_tol={abs_state_tol}')
    logger.info(f'- pod_l2_tol={pod_l2_tol}')

    input_scaling = lambda x: 2. * (x - bounds[0, :]) / (bounds[1, :] - bounds[0, :]) - 1.
    additional_components = [lambda x: 2. * (np.log10(x)-np.log10(bounds[0][0])) / (np.log10(bounds[0][1])-np.log10(bounds[0][0])) - 1., False]
    
    if MLM_opts['type'] == 'ANN':
        mlm_reductor = partial(ANNStateReductor, **MLM_opts['ml_params'])
    elif MLM_opts['type'] == 'VKOGA':
        mlm_reductor = partial(VkogaStateReductor, input_scaling=input_scaling)
    else:
        raise RuntimeError(f'Unknown MLM type "{MLM_opts["type"]}" requested!')

    return AdaptiveModel(
        fom=fom,
        rom_reductor_generator=partial(ParabolicRBReductor, product=spatial_product(fom), coercivity_estimator=coercivity_estimator),
        mlm_reductor_generator=mlm_reductor,
        solution_abs_tol=abs_state_tol,
        output_abs_tol=abs_output_tol,
        pod_l2_err=pod_l2_tol,
        training_batch_size=MLM_opts['training_batch_size'],
        training_accuracy_factor=1,
    )

tic = time.time()
adaptive_model = make_adaptive_model(desired_abs_output_err)
adaptive_model.logger.setLevel('DEBUG')
adaptive_model_offline_time = time.time() - tic
logger.info(f'took {adaptive_model_offline_time}s')
```

```python
def l_inf_subdomain(m, subdomain, f):
    time_points = np.linspace(0, m.T, len(f))
    subdomain_indices = np.where(1.*(time_points >= subdomain[0])*(time_points <= subdomain[1]))[0]
    return np.max(np.abs(f.to_numpy()[subdomain_indices]))

def objective(m, x, f_x=None, f_ref=None, print_status=False):
    if print_status:
        print('.', end='', flush=True)
    mu = m.parameters.parse(x)
    if f_x is None:
        f_x = m.output(mu, incremental=True)
    if f_ref is None:
        f_ref = m.output(mu=mu_ref, incremental=True)
    return l_inf_subdomain(m, [0, m.T], f_ref - f_x)
```

## optimization

```python
from pymor.tools.floatcmp import float_cmp


def plot_objective_surface(N):
    _, plot_fom, _, _ = setup_problem_and_discretize(0, 9)

    x = np.linspace(bounds[0][0], bounds[0][1], N)
    y = np.linspace(bounds[1][0], bounds[1][1], N)
    x, y = np.meshgrid(x, y)

    values = np.zeros(x.shape)
    for ii in range(values.shape[0]):
        for jj in range(values.shape[1]):
            values[ii][jj] = objective(plot_fom, x=[x[ii][jj], y[ii][jj]], print_status=True)

    fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    axs[0].set(xlabel='Da')
    axs[0].set(ylabel='Pe')
    im = axs[0].imshow(values, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], origin="lower")
    fig.colorbar(im, location='bottom', ax=axs[0])
    return fig, axs

# define this outside, to have access to the data even upon failure
data = {
    'num_evals': 0,
    'points': [],
    'values': [],
    'timings': []}


def minimize(m, initial_guess, method, opts, verbose=True, plot=False):

    from scipy.optimize import minimize as scipy_optimize
    from scipy import stats as scipy_stats

    logger = getLogger(f'main.{method}')

    if plot:
        logger.info('preparing plot:')
        colors = cycle(COLORS)
        color = next(colors)
        fig, axs = plot_objective_surface(10)
        fig.suptitle(f'{method}-minimization of adaptive model (bg: coarse model objective)')
        axs[0].set_title('trajectory in parameter space')
        axs[1].set_title(f'raw objective')
        objective_marker = '.'
        axs[1].plot([], [], objective_marker, color=color, label='objective')
        axs[1].legend()
        indicator_marker = 's'
        indicator_color = 'red'
        ratio_marker = '.'
        axs[0].plot(initial_guess[0], initial_guess[1], 'o', color='white')
        axs[1].grid()
        draw_current_plot()

    _global_index = [-1]

    def wrapper(mu):
        data['num_evals'] += 1
        data['points'].append(parameter_space.parameters.parse(mu))  # copy of mu required, changed inplace
        logger.debug(f'evaluating model for mu={mu} ...')
        try:
            logger.debug(f"- used FOM evals: {np.sum(np.array([m._statistics['models']]) == 'FOM')}")
            logger.debug(f"- used ROM evals: {np.sum(np.array([m._statistics['models']]) == 'ROM')}")
            logger.debug(f"- used MLM evals: {np.sum(np.array([m._statistics['models']]) == 'MLM')}")
        except AttributeError:
            pass
        tic = time.time()
        QoI = objective(m, x=mu, f_ref=f_mu_ref)
        data['timings'].append(time.time() - tic)
        logger.debug(f'... objective is {QoI}')
        data['values'].append(QoI)
        _global_index[0] += 1
        if plot:
            # TOP LEFT
            if (bounds[0][0] <= mu[0] <= bounds[0][1]) and (bounds[1][0] <= mu[1] <= bounds[1][1]):
                axs[0].plot(mu[0], mu[1], '.', color=color)
            # TOP RIGHT
            axs[1].semilogy(_global_index, data['values'][-1], objective_marker, color=color)
            axs[1].set_xlim([0, np.ceil(_global_index[0]/100)*100])
            draw_current_plot()
        return QoI

    tic = time.time()
    results = scipy_optimize(wrapper,
        x0=initial_guess,
        method=method,
        bounds=bounds,
        options=opts)

    data['elapsed'] = time.time() - tic,
    data['results'] = results

    if (results.status != 0):
        if plot:
            x = data['points'][-1].to_numpy()
            axs[0].plot(x[0], x[1], 's', color='white')
            draw_current_plot()
        logger.info(' failed!')
        logger.info(' These are the results:')
        logger.info(f'\n{results}')
    else:
        if plot:
            axs[0].plot(results.x[0], results.x[1], 'x', color='white')
            draw_current_plot()
        logger.info(' succeded!')
        logger.info('  minimizer:       {}'.format(parameter_space.parameters.parse(results.x)))
        logger.info('  objective value: {}'.format(results.fun))
        logger.info(' These are the results:')
        logger.info(f'\n{results}')

    return data


mu_initial_guess = parameter_space.parameters.parse({'Da': 2, 'Pe': 10.5})
```

```python
results = minimize(
    m=adaptive_model,
    initial_guess=mu_initial_guess.to_numpy(),
    method='Nelder-Mead',
    opts={},
    plot=True
)
```

```python
# fill in some data
results['num_dofs'] = fom.solution_space.dim
results['mu_ref'] = mu_ref
results['desired_abs_output_err'] = desired_abs_output_err
```

```python
from pymor.core.pickle import dump

logger.info(f'writing results to {filename}.pickle ...')

with open(f'{filename}.pickle', 'wb') as file:
    dump((results, adaptive_model._statistics), file)
```

```python
logger.info("Number of solves using the different models:")
logger.info(f"FOM: {np.sum(np.array([adaptive_model._statistics['models']]) == 'FOM')}")
logger.info(f"ROM: {np.sum(np.array([adaptive_model._statistics['models']]) == 'ROM')}")
logger.info(f"MLM: {np.sum(np.array([adaptive_model._statistics['models']]) == 'MLM')}")
```
