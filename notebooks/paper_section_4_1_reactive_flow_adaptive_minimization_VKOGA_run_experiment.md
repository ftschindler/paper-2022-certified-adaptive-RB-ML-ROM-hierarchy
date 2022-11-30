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

#### options, most of which determine the filename for unique saving of results

```python
headless = False
prefix = 'KLAIM21__RBMLROM_adaptive_tol_minimization'
# FOM
num_refines = 0
num_timesteps = 10
# stagnation detection for adaption
window_size = 6
max_num_violations = 10
bad_slope_threshold = -1e-15
bad_ratio_threshold = 5e-5
filename = f"{prefix}/{prefix}_{num_refines}_{num_timesteps}_{window_size}_{max_num_violations}_{bad_slope_threshold:.2e}_{bad_ratio_threshold:.2e}"

# MLM
# - VKOGA
MLM_opts = {
   'type': 'VKOGA',
   'training_batch_size': 1,
}
filename += f"_{MLM_opts['type']}_{MLM_opts['training_batch_size']}"
```

#### configure matplotlib before anyone else imports it

```python
import os, sys, glob
from timeit import default_timer as timer
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../VKOGA')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # tools.py

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings

from functools import partial

if headless:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    draw_current_plot = lambda : plt.savefig(f'{filename}_CURRENT_RUN.png')
else:
    %matplotlib inline
    from tools import draw_current_plot
```

#### configure pyMOR defaults

```python
# ensure directory for run data
os.makedirs(prefix, exist_ok=True)
# clear logfile, if exists
with open(f'{filename}.txt', 'w'):
    pass
# clear intermediate results
for file in glob.glob(f'{filename}_CURRENT_RUN_*.pickle'):
    os.remove(file)

from pymor.core.defaults import set_defaults
set_defaults({'pymor.core.logger.getLogger.filename': f'{filename}.txt'})

from pymor.core.pickle import dump
from pymor.core.logger import set_log_levels, getLogger
set_log_levels({ # disable logging for some components
    'main': 'DEBUG',
    'pymor': 'WARN',
    'pymor.models': 'WARN',
    'pymor.discretizers.builtin': 'WARN',
    'pymor.discretizers.dunegdt': 'DEBUG',
    'pymor.analyticalproblems.functions.BitmapFunction': 'ERROR',
    'models.ann.ANNStateReductor': 'INFO',
    'models.vkoga.VkogaStateModel': 'INFO',
    'models.vkoga.VkogaStateReductor': 'DEBUG',
    'models.adaptive': 'DEBUG',
    'algorithms.optimization': 'DEBUG'})

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
```

```python
logger.info('creating FOM:')
tic = timer()

parameter_space, fom, fom_data, coercivity_estimator = setup_problem_and_discretize(
    num_refines, num_timesteps)

fom_offline_time = timer() - tic
logger.info(f'  discretizing took {fom_offline_time}s')
logger.info(f'  grid has {fom_data["grid"].size(0)} elements, FOM has {fom.solution_space.dim} DoFs, uses {fom.time_stepper.nt} time steps')

logger.info(f'  input parameter space is {parameter_space.parameters.dim}-dimensional:')
logger.info(f'    {parameter_space}')
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

logger.info(f'computing f_h(mu={mu_ref}) ...')

tic = timer()
f_mu_ref = fom.output(mu=mu_ref, incremental=True)  # no need to keep the state trajectory in memory
fom_online_output_time = timer() - tic

logger.info(f'average FOM output (solve + apply functional) time: {fom_online_output_time}s')

logger.info('computing ||f_mu_ref||_{L^2(0, T)}:')
initial_abs_output_tol = fom.output_l2_norm(f_mu_ref)
logger.info(f'  {initial_abs_output_tol}')
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
    
    # rescale input to -1, 1
    bounds = [[], []]
    for kk in parameter_space.ranges.keys():
        for jj in range(parameter_space.parameters[kk]):
            bounds[0].append(parameter_space.ranges[kk][0])
            bounds[1].append(parameter_space.ranges[kk][1])
    bounds = np.array(bounds)
    input_scaling = lambda x: 2. * (x - bounds[0, :]) / (bounds[1, :] - bounds[0, :]) - 1.

    if MLM_opts['type'] == 'ANN':
        mlm_reductor = partial(ANNStateReductor, input_scaling=input_scaling, **MLM_opts["training_params"])
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

tic = timer()
adaptive_model = make_adaptive_model(initial_abs_output_tol)
logger.info(f'took {timer() - tic}s')
```

## optimization

```python
def Linf_misfit(m, x, f_ref=None, print_status=False):
    if print_status:
        print('.', end='', flush=True)
    mu = m.parameters.parse(x)
    f_x = m.output(mu, incremental=True)
    if f_ref is None:
        f_ref = m.output(mu=mu_ref, incremental=True)
    return np.max(np.abs((f_ref - f_x).to_numpy()))
```

```python
def plot_objective_surface(fig, ax):
    bounds = []
    for kk in parameter_space.ranges.keys():
        for jj in range(parameter_space.parameters[kk]):
            bounds.append((parameter_space.ranges[kk][0], parameter_space.ranges[kk][1]))
    bounds = np.array(tuple(np.array(b) for b in bounds))
    N = 10
    _, plot_fom, _, _ = setup_problem_and_discretize(0, 9)

    x = np.linspace(bounds[0][0], bounds[0][1], N)
    y = np.linspace(bounds[1][0], bounds[1][1], N)
    x, y = np.meshgrid(x, y)

    values = np.zeros(x.shape)
    for ii in range(values.shape[0]):
        for jj in range(values.shape[1]):
            values[ii][jj] = Linf_misfit(plot_fom, x=[x[ii][jj], y[ii][jj]], print_status=True)

    ax.set(xlabel='Da')
    ax.set(ylabel='Pe')
    im = ax.imshow(values, extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]], origin="lower")
    fig.colorbar(im, location='bottom', ax=ax)

```

```python
from algorithms.optimization import adaptive_minimization

results = adaptive_minimization(
    m=adaptive_model,
    parameter_space=parameter_space,
    initial_guess=parameter_space.parameters.parse({'Da': 2, 'Pe': 10.5}),
    objective=partial(Linf_misfit, f_ref=f_mu_ref),
    method='Nelder-Mead',
    opts={},
    stagnation_detection=(
        window_size,
        max_num_violations,
        bad_slope_threshold,
        bad_ratio_threshold),
    plotter=(plot_objective_surface, draw_current_plot),
)
```

```python
# fill in some data
results['num_dofs'] = fom.solution_space.dim
results['mu_ref'] = mu_ref
results['window_size'] = window_size
results['max_num_violations'] = max_num_violations
results['bad_slope_threshold'] = bad_slope_threshold
results['bad_ratio_threshold'] = bad_ratio_threshold
```

```python
from pymor.core.pickle import dump

logger.info(f'writing results to {filename}.pickle ...')

with open(f'{filename}.pickle', 'wb') as file:
    dump((results, adaptive_model._statistics), file)
```
