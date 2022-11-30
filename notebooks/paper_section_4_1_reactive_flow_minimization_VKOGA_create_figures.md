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

# Results for Section 4.1.1: Minimization with an a priori fixed tolerance $\varepsilon$

```python
from itertools import cycle

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings

from matplotlib import pyplot as plt
%matplotlib inline
from matplotlib.colors import TABLEAU_COLORS as COLORS

from pymor.core.logger import set_log_levels, getLogger

set_log_levels({
    'main': 'DEBUG',
    'pymor': 'WARN',
})
logger = getLogger('main.main')
```

## available finished runs

```python
!ls -lh KLAIM21__RBMLROM_minimization__run/*.pickle
```

```python
from pymor.core.pickle import load
from pymor.tools.floatcmp import float_cmp_all

def filename(num_refines, num_timesteps, MLM_opts, desired_abs_output_err):
    prefix = 'KLAIM21__RBMLROM_minimization__run'
    return f"{prefix}/{prefix}_{num_refines}_{num_timesteps}_{MLM_opts['type']}_{MLM_opts['training_batch_size']}_{desired_abs_output_err}.pickle"
```

### the FOM

```python
num_dofs = {0: 2121, 1: 8241, 2: 32481, 3: 128961}

global_refines = 0
nt = 1000
MLM_opts = {
    'type': 'VKOGA',
    'training_batch_size': 1,
}

with open(filename(global_refines, nt, MLM_opts, 0.), 'rb') as file:
    FOM_results = load(file)[0]
```

### the ROMs

```python
ROM_results = {}
mu_ref = None
colors = cycle(COLORS)
best_color = 'tab:red'

for tol in (0.1, 0.015, 0.0125, 0.01, 0.001, 0.0001, 0.00001):
    with open(filename(global_refines, nt, MLM_opts, tol), 'rb') as file:
        res = load(file)
        ROM_results[(global_refines, nt, tol)] = res
        color = next(colors)
        if color == best_color:
            color = next(colors)
        ROM_results[(global_refines, nt, tol)][0]['color'] = color
        res = res[0]
        if 'mu_ref' in res and mu_ref is None:
            mu_ref = res['mu_ref']
        elif 'mu_ref' in res:
            assert float_cmp_all(mu_ref.to_numpy(), res['mu_ref'].to_numpy())
ROM_results[(global_refines, nt, tol)][0]['color'] = best_color


mu_ref = mu_ref.to_numpy()
logger.info(f'collecten results for mu_ref={mu_ref}:')
logger.info(f' - ROM: {ROM_results.keys()}')
```

## data regarding convergence

```python
successfull_ROMS = {}

print('number of model evaluations (FOM):')
print(f'        \t{FOM_results["num_evals"]}')
print('number of model evaluations (HaPOD-VKOGA-ROM):')
for (global_refines, nt, tol), (results, statistics) in ROM_results.items():
    print(f'  eps={tol}:\t{results["num_evals"]}', end='')
    if results['results'].status == 0:
        successfull_ROMS[(global_refines, nt, tol)] = (results, statistics)
        print(' (success)')
    else:
        print()
```

## data regarding found optimum

```python
print('relative minimizer error (FOM):')
print(f'        \t{np.linalg.norm(mu_ref - FOM_results["points"][-1].to_numpy())/np.linalg.norm(mu_ref):.2e}')
print('relative minimizer error (HaPOD-VKOGA-ROM):')
for (global_refines, nt, tol), (results, statistics) in ROM_results.items():
    rel_err = np.linalg.norm(mu_ref - results["points"][-1].to_numpy())/np.linalg.norm(mu_ref)
    print(f'  eps={tol}:\t{rel_err:.2e}', end='')
    if results['results'].status == 0:
        print(' (success)')
    else:
        print()
```

```python
# since we know that J(mu_ref) = 0, absolute objective error = absolute objective
print('absolute objective error (FOM):')
print(f'        \t{FOM_results["values"][-1]:.2e}')
print('absolute objective error (HaPOD-VKOGA-ROM):')
for (global_refines_, nt_, tol), (results, statistics) in ROM_results.items():
    if global_refines_ == global_refines and nt_ == nt:
        print(f'  eps={tol}:\t{np.abs(results["values"][-1]):.2e}', end='')
        if results['results'].status == 0:
            print(' (success)')
        else:
            print()
```

## objective during minimization

```python
def add_objective_during_minimization_plot(ax):
    global global_refines, nt
    ax.set_title(f'FOM ({num_dofs[global_refines]} DoFs, nt={nt}) vs. HaPOD-VKOGA-ROM minimization for selected fixed tolerances')
    ax.set(xlabel='#model evaluations')
    ax.set(ylabel='normalized objective')

    results = FOM_results
    label = f'FOM($\\varepsilon=0$)'
    ax.semilogy(
        np.arange(len(results['values'])), np.array(results['values'])/results['values'][0],
        '-o',
        color='black',
        label=label)


    for (global_refines, nt, tol), (results, statistics) in ROM_results.items():
        label = f'HaPOD-VKOGA-ROM ($\\varepsilon=${tol})'
        ax.semilogy(
            np.arange(len(results['values'])), np.array(results['values'])/results['values'][0],
            '-' if (global_refines, nt, tol) in successfull_ROMS else '--',
            color=results['color'],
            label=label)

    ax.legend()

    _ = ax.set_xlim([0, 126])

fig, axs = plt.subplots(1, 1, figsize=(16, 8))
add_objective_during_minimization_plot(axs)
```

## data regarding internal model use

```python
print(f'FOM: {FOM_results["num_evals"]} evals')
print('HaPOD-VKOGA-ROM:')
for (global_refines_, nt_, tol), (results, statistics) in successfull_ROMS.items():
    print(f'    eps={tol}:\t{np.sum(np.array([statistics["models"]]) == "FOM")} FOM, ', end='')
    print(f'{np.sum(np.array([statistics["models"]]) == "ROM")} ROM, ', end='')
    print(f'{np.sum(np.array([statistics["models"]]) == "MLM")} MLM evals\t', end='')
    print(f'({results["num_evals"]} total)')
```

```python
labels = []
models = {
    'FOM': [],
    'ROM': [],
    'MLM': []
}
width = []
for (global_refines_, nt_, tol), (results, statistics) in successfull_ROMS.items():
    labels.append(f'{tol}')
    total_num_evals = results["num_evals"]
    width.append(0.5*total_num_evals/FOM_results["num_evals"])
    for model in 'FOM', 'ROM', 'MLM':
        num_model_evals = np.sum(np.array([statistics["models"]]) == model)
        models[model].append(num_model_evals/total_num_evals)
for kk, vv in models.items():
    models[kk] = np.array(models[kk])

def add_internal_model_use_plot(ax):
    ax.set_title('fraction of submodels used in HaPOD-VKOGA-ROM')
    ax.set_xlabel('tolerance $\\varepsilon$')
    ax.bar(labels, models['FOM'], width, label='FOM')
    ax.bar(labels, models['ROM'], width, bottom=models['FOM'], label='ROM')
    ax.bar(labels, models['MLM'], width, bottom=models['FOM']+models['ROM'], label='MLM')
    _ = ax.legend()

plt.plot()
add_internal_model_use_plot(plt.gca())
```

## Figure 1

```python
fig, axs = plt.subplots(
    nrows=1, ncols=2,
    figsize=(16, 4),
    gridspec_kw={'width_ratios':[1.75, 1], 'height_ratios':[1,]})

add_objective_during_minimization_plot(axs[0])
axs[0].set_title('convergence behaviour during minimization for various models')
add_internal_model_use_plot(axs[1])
```
