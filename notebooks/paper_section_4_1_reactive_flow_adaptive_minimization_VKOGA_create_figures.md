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

# Results for Section 4.1.2: Minimization with an adaptive tolerance $\varepsilon$

```python
from itertools import cycle

import numpy as np
np.warnings.filterwarnings('ignore') # silence numpys warnings

from matplotlib import pyplot as plt
%matplotlib inline
from matplotlib.colors import to_rgba, TABLEAU_COLORS as COLORS

from pymor.core.logger import set_log_levels, getLogger

set_log_levels({
    'main': 'DEBUG',
    'pymor': 'WARN',
})
logger = getLogger('main.main')
```

## available finished runs

```python
!ls -lh KLAIM21__RBMLROM_adaptive_tol_minimization/*.pickle
```

```python
from pymor.core.pickle import load
from pymor.tools.floatcmp import float_cmp_all

num_refines = 5
num_timesteps = 10000
MLM_opts = {
    'type': 'VKOGA',
    'training_batch_size': 1,
}
window_size = 6
max_num_violations = 10
bad_slope_threshold = -1e-15
bad_ratio_threshold = 5e-5
prefix = 'KLAIM21__RBMLROM_adaptive_tol_minimization'
filename = f"{prefix}/{prefix}_{num_refines}_{num_timesteps}_{window_size}_{max_num_violations}_{bad_slope_threshold:.2e}_{bad_ratio_threshold:.2e}_{MLM_opts['type']}_{MLM_opts['training_batch_size']}"

with open(f'{filename}.pickle', 'rb') as file:
    results, statistics = load(file)
```

```python
print('availabel results:')
for kk in results.keys():
    print(f'- {kk}')
```

```python
print('availabel statistics:')
for kk in statistics.keys():
    print(f'- {kk}')
```

```python
with open(f'{filename}_objective_bg.pickle', 'rb') as file:
    values, extent = load(file)

def plot_objective_surface(fig, ax):
    ax.set(xlabel='Da')
    ax.set(ylabel='Pe')
    im = ax.imshow(values, extent=extent, origin="lower")
    fig.colorbar(im, location='bottom', ax=ax)
```

## Figure 2

```python
from pymor.tools.floatcmp import float_cmp

method = 'Nelder-Mead'
initial_guess = statistics['mus'][0].to_numpy()

bad_slope = lambda s: s > bad_slope_threshold
bad_ratio = lambda r: r < bad_ratio_threshold

fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw={'width_ratios':[1, 1], 'height_ratios':[1, 1]})
timings_FOM_color = to_rgba('#74a9cf', 1.)
timings_ROM_color = (to_rgba('#fd8d3c', 1.), to_rgba('#fecc5c', 1.))
timings_MLM_color = (to_rgba('#006837', 1.), to_rgba('#78c679', 1.))
colors = cycle(COLORS)
color = next(colors)
plot_objective_surface(fig, axs[0][0])
axs[0][0].set_title('trajectory in parameter space')
axs[0][1].set_title(f'raw objective and running mean (width={window_size})')
objective_marker = '.'
running_mean_marker = '-'
indicator_marker = 's'
model_indicator_color = 'lightgray'
axs[0][1].plot([], [], objective_marker, color='black', label='objective')
axs[0][1].plot([], [], running_mean_marker, color='black', label='running mean')
axs[0][1].legend()
axs[1][1].set_title(f'complexity of submodels ({model_indicator_color} square: highlight of FOM)')
axs[1][0].set_title('time spent (s) in model evaluation (light) and model building (dark)')
axs[1][0].plot([], [], color=timings_FOM_color, label='FOM')
axs[1][0].plot([], [], color=timings_ROM_color[1], label='RB-ROM')
axs[1][0].plot([], [], color=timings_MLM_color[1], label='ML-ROM')
axs[1][0].legend()
axs[1][0].set_yscale('log')
fom_marker = '+'
rom_size_marker = '.'
mlm_size_marker = 'd'
axs[1][1].plot([], [], fom_marker, color='black', label='FOM: eval #')
axs[1][1].plot([], [], rom_size_marker, color='black', label='RB-ROM: |RB|')
axs[1][1].plot([], [], mlm_size_marker, color='black', label='ML-ROM: |used data|')
axs[1][1].legend()
ratio_marker = '.'
axs[0][0].plot(initial_guess[0], initial_guess[1], 'o', color='white')
for ax in axs[0][1], axs[1][1], axs[1][0]:
    ax.grid()

_global_index = [-1]

for adaptation_hierarchy, (tol, data) in enumerate(results['collected']):
    offset = np.sum([len(data['values']) for _, data in results['collected'][:adaptation_hierarchy]])
    # TOP LEFT
    for mu in data['points']:
        mu = mu.to_numpy()
        axs[0][0].plot(mu[0], mu[1], '.', color=color)
    # TOP RIGHT
    axs[0][1].semilogy(
        np.arange(len(data['values'])) + offset, data['values'], objective_marker, color=color)
    if len(data['values']) >= window_size:
        axs[0][1].plot(
            np.arange(len(data['running_mean'])) + len(data['values']) - len(data['running_mean']) + offset,
            data['running_mean'],
            running_mean_marker, color=color)

    # BOTTOM LEFT
    for idx in range(len(data['values'])):
        _global_index[0] += 1

        used_model = statistics['models'][_global_index[0]]
        assert used_model in ('FOM', 'ROM', 'MLM')

        eval_time = statistics['timings']['eval_model'][_global_index[0]]
        build_time = statistics['timings']['build_model'][_global_index[0]]
        if used_model == 'FOM':
            axs[1][0].bar(
                _global_index, eval_time, width=1., bottom=0.,
                align='edge', color=timings_FOM_color)
            axs[1][0].bar(
                _global_index, build_time[0], width=1., bottom=eval_time,
                align='edge', color=timings_ROM_color[0])
            if build_time[2]:
                axs[1][0].bar(
                    _global_index, build_time[1] + build_time[2], width=1., bottom=eval_time +build_time[0],
                    align='edge', color=timings_MLM_color[0])
        elif used_model == 'ROM':
            axs[1][0].bar(
                _global_index, eval_time, width=1., bottom=0.,
                align='edge', color=timings_ROM_color[1])
            axs[1][0].bar(
                _global_index, build_time, width=1., bottom=eval_time,
                align='edge', color=timings_MLM_color[0])
            if build_time > 100:
                list_idx_NN_training.append((idx, build_time))
                print(idx)
                print(build_time)
        else:  # MLM
            axs[1][0].bar(
                _global_index, eval_time, width=1., bottom=0.,
                align='edge', color=timings_MLM_color[1])

        # BOTTOM RIGHT
        num_fom_solves = np.sum(np.array([statistics['models'][:_global_index[0]]]) == 'FOM')
        rom_size = statistics['ROM']['dim'][_global_index[0]] if len(statistics['ROM']['dim']) > 0 else 0
        mlm_size = statistics['MLM']['size'][_global_index[0]] if len(statistics['MLM']['size']) > 0 else 0
        # plot the size of each model
        if used_model == 'FOM':
            axs[1][1].plot(
                _global_index, num_fom_solves, 's', color=model_indicator_color)
            axs[1][1].plot(
                _global_index, num_fom_solves, fom_marker, color=color)
        axs[1][1].plot(
            _global_index, rom_size, rom_size_marker, color=color)
        axs[1][1].plot(
            _global_index, mlm_size, mlm_size_marker, color=color)

    for ax in axs[0][1], axs[1][0], axs[1][1]:
        ax.set_xlim([-2, len(data['values']) + offset + 2])
    color = next(colors)
    
final_mu = results['collected'][-1][1]['points'][-1].to_numpy()
_ = axs[0][0].plot(final_mu[0], final_mu[1], 'x', color='white')
```
