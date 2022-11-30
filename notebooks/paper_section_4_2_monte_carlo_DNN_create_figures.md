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

## objective

Evaluation and visualization of the results of the ianscluster04 (ic4) run:
Instead of the pickle file etc which I added to the git, I used a more recent version of the results, where up to 12389 mu parameters were already used (in comparison to around 8000 mu parameters in the git-added version). However, since the plots mostly consider the first few thousand parameters, the should not be any difference.

<!-- 
Minimize $L^\infty$-misfit

$$\begin{align}
J_h(\mu) := \|f_* - f_h(\mu)\|_{L^\infty(0, T)} &&\text{over}&& \mathcal{P},
\end{align}$$

where we choose $f_* = f_h(\mu_*)$ for a known $\mu_* \in \mathcal{P}$. -->


## available finished runs

```python
prefix = 'MC_adaptive_model'
# FOM
num_refines = 1
num_timesteps = 1000
# adaptive model
abs_output_tol = 5e-2
training_batch_size = 200
```

```python
!ls -lh $prefix/*.pickle
```

```python
filename_pickle = f"{prefix}/{prefix}_{num_refines}_{num_timesteps}_{abs_output_tol:.2e}_ANN_{training_batch_size}_CURRENT_RUN.pickle"
```

```python
from pymor.core.pickle import load

with open(filename_pickle, 'rb') as file:
    results, statistics = load(file)
```

```python
from matplotlib.colors import to_rgba, TABLEAU_COLORS as COLORS

timings_FOM_color = to_rgba('#74a9cf', 1.)
timings_ROM_color = (to_rgba('#fd8d3c', 1.), to_rgba('#fecc5c', 1.))
timings_MLM_color = (to_rgba('#006837', 1.), to_rgba('#78c679', 1.))
```

```python
list_flags_FOM = [True if str == 'FOM' else False for str in statistics['models']]
list_flags_ROM = [True if str == 'ROM' else False for str in statistics['models']]
list_flags_MLM = [True if str == 'MLM' else False for str in statistics['models']]
```

# Reproduce the most important plot + zoom-in
For this I used code from algorithms/mc.py and modified it slightly.

```python
timings_FOM_color = to_rgba('#74a9cf', 1.)                             # eval time 
timings_ROM_color = (to_rgba('#fd8d3c', 1.), to_rgba('#fecc5c', 1.))   # build, eval time
timings_MLM_color = (to_rgba('#006837', 1.), to_rgba('#78c679', 1.))   # build, eval time

big_width = 3
MLM_eval_times = []
MLM_build_times = []


for N_plot in (2000,):
    list_idx_NN_training = []
    plt.figure()
    plt.plot([], [], color=timings_FOM_color, label='FOM')
    plt.plot([], [], color=timings_ROM_color[1], label='RB-ROM')
    plt.plot([], [], color=timings_MLM_color[1], label='ML-ROM')
    plt.yscale('log')
    plt.legend()

    # MLM
    _global_index = [-1]
    for idx in range(N_plot):
        used_model = statistics['models'][idx]
        _global_index[0] += 1
        eval_time = statistics['timings']['eval_model'][idx]
        build_time = statistics['timings']['build_model'][idx]
        if used_model == 'MLM':
            MLM_eval_times.append(eval_time)
            plt.bar(
                _global_index, eval_time, width=1., bottom=1,
                align='edge', color=timings_MLM_color[1])

    # RB
    _global_index = [-1]
    for idx in range(N_plot):
        used_model = statistics['models'][idx]
        _global_index[0] += 1
        eval_time = statistics['timings']['eval_model'][idx]
        build_time = statistics['timings']['build_model'][idx]
        if used_model == 'ROM':
            if build_time > 100: # MLM training
                list_idx_NN_training.append((idx, build_time))
                width = big_width
            else:  # MLM data append
                width = big_width
            plt.bar(
                _global_index, eval_time, width=width, bottom=1,
                align='edge', color=timings_ROM_color[1])
            plt.bar(
                _global_index, build_time, width=width, bottom=eval_time+1,
                align='edge', color=timings_MLM_color[0])

    # FOM
    _global_index = [-1]
    for idx in range(N_plot):
        used_model = statistics['models'][idx]
        _global_index[0] += 1
        eval_time = statistics['timings']['eval_model'][idx]
        build_time = statistics['timings']['build_model'][idx]
        if used_model == 'FOM':
            plt.bar(
                _global_index, eval_time, width=big_width, bottom=1,
                align='edge', color=timings_FOM_color)
            plt.bar(
                _global_index, build_time[0], width=big_width, bottom=eval_time+1,
                align='edge', color=timings_ROM_color[0])
            if build_time[2]:
                plt.bar(
                    _global_index, build_time[1 ] +build_time[2], width=big_width, bottom=eval_time +build_time[0]+1,
                    align='edge', color=timings_MLM_color[0])
```

```python
big_width = 15
MLM_eval_times = []
MLM_build_times = []


for N_plot in (10000,):
    list_idx_NN_training = []
    plt.figure()
    plt.plot([], [], color=timings_FOM_color, label='FOM')
    plt.plot([], [], color=timings_ROM_color[1], label='RB-ROM')
    plt.plot([], [], color=timings_MLM_color[1], label='ML-ROM')
    plt.yscale('log')
    plt.legend()

    # MLM
    _global_index = [-1]
    for idx in range(N_plot):
        used_model = statistics['models'][idx]
        _global_index[0] += 1
        eval_time = statistics['timings']['eval_model'][idx]
        build_time = statistics['timings']['build_model'][idx]
        if used_model == 'MLM':
            MLM_eval_times.append(eval_time)
            plt.bar(
                _global_index, eval_time, width=1., bottom=1,
                align='edge', color=timings_MLM_color[1])

    # RB
    _global_index = [-1]
    for idx in range(N_plot):
        used_model = statistics['models'][idx]
        _global_index[0] += 1
        eval_time = statistics['timings']['eval_model'][idx]
        build_time = statistics['timings']['build_model'][idx]
        if used_model == 'ROM':
            if build_time > 100: # MLM training
                list_idx_NN_training.append((idx, build_time))
                width = big_width
            else:  # MLM data append
                width = big_width
            plt.bar(
                _global_index, eval_time, width=width, bottom=1,
                align='edge', color=timings_ROM_color[1])
            plt.bar(
                _global_index, build_time, width=width, bottom=eval_time+1,
                align='edge', color=timings_MLM_color[0])

    # FOM
    _global_index = [-1]
    for idx in range(N_plot):
        used_model = statistics['models'][idx]
        _global_index[0] += 1
        eval_time = statistics['timings']['eval_model'][idx]
        build_time = statistics['timings']['build_model'][idx]
        if used_model == 'FOM':
            plt.bar(
                _global_index, eval_time, width=big_width, bottom=1,
                align='edge', color=timings_FOM_color)
            plt.bar(
                _global_index, build_time[0], width=big_width, bottom=eval_time+1,
                align='edge', color=timings_ROM_color[0])
            if build_time[2]:
                plt.bar(
                    _global_index, build_time[1 ] +build_time[2], width=big_width, bottom=eval_time +build_time[0]+1,
                    align='edge', color=timings_MLM_color[0])
```

## Get information about the ratio of MLM solves

```python
# The training of the NN took place for the following indices (with corresponding times)
list_idx_NN_training
```

```python
total_NN_optim_time = sum([tupl[1] for tupl in list_idx_NN_training])
print('Total NN optim time = {}s.'.format(total_NN_optim_time))
```

```python
N_mu_total = len(statistics['models'])

for idx in range(len(list_idx_NN_training) - 1):
    idx_mu_start = list_idx_NN_training[idx][0]
    idx_mu_end = list_idx_NN_training[idx+1][0]
    
    N_mu = idx_mu_end - idx_mu_start
    
    print('Ratio ROM solves = {:.4f}'.format(sum(list_flags_ROM[idx_mu_start : idx_mu_end]) / N_mu))
    print('Ratio MLM solves = {:.4f}'.format(sum(list_flags_MLM[idx_mu_start : idx_mu_end]) / N_mu))
    print(' ')
```

```python
print('Ratio ROM solves beginning = {:.4f}'.format(sum(list_flags_ROM[: list_idx_NN_training[0][0]]) / list_idx_NN_training[0][0]))
print('Ratio MLM solves beginning = {:.4f}'.format(sum(list_flags_MLM[: list_idx_NN_training[0][0]]) / list_idx_NN_training[0][0]))
```

```python
print('Ratio ROM solves end = {:.4f}'.format(
    sum(list_flags_ROM[list_idx_NN_training[-1][0] :]) / (N_mu_total - list_idx_NN_training[-1][0])))
print('Ratio MLM solves end = {:.4f}'.format(
    sum(list_flags_MLM[list_idx_NN_training[-1][0] :]) / (N_mu_total - list_idx_NN_training[-1][0])))
```

## Some further numbers

```python
# Evaluation timings of FOM
np.array(statistics['timings']['eval_model'])[list_flags_FOM]
```

```python
print('Mean time for FOM evaluation: {:.3f}s.'.format(np.mean(np.array(statistics['timings']['eval_model'])[list_flags_FOM])))
```

```python
# Build timings of FOM
np.array(statistics['timings']['build_model'])[list_flags_FOM]
```

```python
# Evaluation timings of ROM
np.sort(np.array(statistics['timings']['eval_model'])[list_flags_ROM])
```

```python
avg_rom_eval = np.mean(np.array(statistics['timings']['eval_model'])[list_flags_ROM])
print('Mean time for ROM evaluation: {:.3f}s.'.format(avg_rom_eval))
```

```python
# Build timings of ROM: The largest numbers are the training times for the neural network
np.sort(np.array(statistics['timings']['build_model'])[list_flags_ROM])
```

```python
# Evaluation timings of MLM
(np.array(statistics['timings']['eval_model'])[list_flags_MLM])
```

```python
avg_mlm_eval = np.mean(np.array(statistics['timings']['eval_model'])[list_flags_MLM])
print('Mean time for MLM evaluation: {:.3f}s.'.format(avg_mlm_eval))
```

```python
# Build timings of MLM
np.array(statistics['timings']['build_model'])[list_flags_MLM][:10]
```

```python
# Calculate when NN optimization pays off
total_NN_optim_time / (avg_rom_eval - avg_mlm_eval)
```

# Statistics of distribution plot
grep -o 'QoI: .*' old_MC_adaptive_model_1_1000_5.00e-02_ANN_200.txt > old_MC_adaptive_model_1_1000_5.00e-02_ANN_200_QOI.txt

```python
!grep -o 'QoI: .*' $prefix/MC_adaptive_model_1_1000_5.00e-02_ANN_200.txt > $prefix/MC_adaptive_model_1_1000_5.00e-02_ANN_200_QOI.txt
```

```python
filename_QOI = f"{prefix}/{prefix}_{num_refines}_{num_timesteps}_{abs_output_tol:.2e}_ANN_{training_batch_size}_QOI.txt"
```

```python
with open(filename_QOI) as f:
    qoi_lines = f.readlines()
```

```python
array_qoi = np.array([float(line[5:20]) for line in qoi_lines])
```

```python
def cum_mean(array):
    # Calculate the cumulative mean (or running mean?)
    cum_sum = np.cumsum(array)
    
    return cum_sum / np.arange(1, array.size + 1)
```

```python
def cum_var(array):
    # Calculate the cumulative variance (or running variance?)
    
    array_result = np.zeros(array.size)
    
    for idx in range(array.size):
        array_result[idx] = np.var(array[:idx+1])
    
    return array_result
```

```python
interval = 5
plt.figure(1)
plt.plot(array_qoi[::interval], ',', color=timings_FOM_color)
plt.plot(cum_mean(array_qoi)[::interval], '.', color=timings_FOM_color)
plt.plot(cum_var(array_qoi)[::interval], '2', color=timings_FOM_color)
plt.grid()
plt.yscale('log')
```

```python
interval = 1
final = 2000
plt.figure(1)
plt.plot(cum_mean(array_qoi)[:final:interval], '.', color=timings_FOM_color)
plt.grid()
plt.yscale('log')
```

```python
plt.figure(1)
plt.plot(cum_var(array_qoi)[:final:interval], '2', color=timings_FOM_color)
plt.grid()
plt.yscale('log')
```
