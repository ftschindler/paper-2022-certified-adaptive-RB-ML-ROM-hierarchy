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
headless = True
prefix = 'MC_adaptive_model'
# FOM
num_refines = 0
num_timesteps = 100
# adaptive model
abs_output_tol = 5e-2

filename = f"{prefix}/{prefix}_{num_refines}_{num_timesteps}_{abs_output_tol:.2e}"

# - ANN
import torch
import torch.nn as nn
import torch.optim as optim
MLM_opts = {
    'type': 'ANN',
    'training_batch_size': 10,
    'training_params': {'training_inputs': None,
                        'additional_training_data': None,
                        'validation_ratio': 0.05,
                        'hidden_layers': '[128, 128, 128, 128]',
                        'optimizer': optim.Adam,
                        'epochs': 1000,
                        'batch_size': 128,
                        'learning_rate': 5e-3,
                        'target_loss': None,
                        'lr_scheduler': optim.lr_scheduler.StepLR,
                        'lr_scheduler_params': {'step_size': 10, 'gamma': 0.7},
                        'seed': 0,
                        'automatic_input_scaling': False,
                        'automatic_output_scaling': False,
                        'output_scaling': None,
                        'inverse_output_scaling': None,
                        'additional_components': [],
                        'loss_function': None,
                        'time_sample_frequency': 1,
                        'num_workers': 1,
                        'gpus': 0,
                        'early_stopping_patience': 10,
                        'tb_path': prefix + '/ML_MORE_TB_LOGS/'}
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
    'algorithms.mc': 'DEBUG'})

logger = getLogger('main.main')
```

## FOM

```python
def setup_problem_and_discretize(num_refines, nt):
    from models.MM_EXC import make_problem, discretize
    
    problem, parameter_space, mu_bar = make_problem()
    
    fom, coercivity_estimator = discretize(
        problem, mu_bar, num_global_refines=num_refines, nt=nt)

    return parameter_space, fom, coercivity_estimator


spatial_product = lambda m: m.energy_product
```

```python
logger.info('creating FOM:')
tic = timer()

parameter_space, fom, coercivity_estimator = setup_problem_and_discretize(
    num_refines, num_timesteps)

fom_offline_time = timer() - tic
logger.info(f'  discretizing took {fom_offline_time}s')
logger.info(f'  FOM has {fom.solution_space.dim} DoFs, uses {fom.time_stepper.nt} time steps')

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

    if MLM_opts['type'] == 'SDKN':
        mlm_reductor = partial(ANNStateReductor, input_scaling=input_scaling, activation_function='SDKN',
                               **MLM_opts["training_params"])
    elif MLM_opts['type'] == 'ANN':
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
adaptive_model = make_adaptive_model(abs_output_tol=abs_output_tol)
logger.info(f'took {timer() - tic}s')
```

## Monte-Carlo

```python
# since we don't adapt in time, we can keep the indices
time_points = np.linspace(0, fom.time_stepper.end_time, fom.time_stepper.nt + 1)
subdomain_indices = np.where(1.*(time_points >= .9*fom.time_stepper.end_time)*(time_points <= fom.time_stepper.end_time))[0]

def average_end_temperature(m, mu):
    return np.mean(m.output(mu, incremental=True).to_numpy()[subdomain_indices])
```

```python
from algorithms.mc import adaptive_monte_carlo

results = adaptive_monte_carlo(
    m=adaptive_model,
    parameter_space=parameter_space,
    QoI=average_end_temperature,
    stagnation_detection=(
        5,  # does not matter
        10,  # does not matter
        0.),  # continue forever
    max_num_refinements=0,  # do not adapt
    min_variance_variation=np.inf,  # does not matter
    plotter=draw_current_plot,
    intermediate_dump_filename_prefix=filename + '_CURRENT_RUN',
)
```

```python
# fill in some data
results['num_dofs'] = fom.solution_space.dim
```

```python
from pymor.core.pickle import dump

logger.info(f'writing results to {filename}.pickle ...')

with open(f'{filename}.pickle', 'wb') as file:
    dump((results, adaptive_model._statistics), file)
```
