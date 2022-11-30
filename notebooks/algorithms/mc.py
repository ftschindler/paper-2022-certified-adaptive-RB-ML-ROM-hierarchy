import os, sys
from timeit import default_timer as timer
from itertools import cycle
from matplotlib.colors import to_rgba, TABLEAU_COLORS as COLORS
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from pymor.tools.floatcmp import float_cmp
from pymor.core.logger import getLogger
from pymor.core.pickle import dump

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # models
from models.adaptive import AdaptiveModel

class StagnationError(Exception):
    pass


def adaptive_monte_carlo(
    m,
    parameter_space,
    QoI,  # of the form QoI(m, mu)
    stagnation_detection=(
        None,    # width of running window
        None,    # max num violations
        None),   # bad slope indicator
    max_num_refinements=np.inf,
    min_variance_variation=np.inf,
    plotter=False,
    intermediate_dump_filename_prefix=None):

    logger = getLogger('algorithms.mc.adaptive_monte_carlo')
    assert isinstance(m, AdaptiveModel)
    if isinstance(plotter, bool) and plotter:
        plotter = lambda : None
    window_size, max_violations, bad_slope_threshold = stagnation_detection
    bad_slope = lambda s: np.abs(s) < bad_slope_threshold

    if plotter:
        timings_FOM_color = to_rgba('#74a9cf', 1.)
        timings_ROM_color = (to_rgba('#fd8d3c', 1.), to_rgba('#fecc5c', 1.))
        timings_MLM_color = (to_rgba('#006837', 1.), to_rgba('#78c679', 1.))

        colors = cycle(COLORS)
        color = next(colors)
        fig, axs = plt.subplots(3, 2, figsize=(16, 14), gridspec_kw={'width_ratios':[1, 1.5], 'height_ratios':[1, 1, 1]})
        fig.suptitle(f'adaptive Monte-Carlo')
        axs[0][0].set_title('relative jump in parameter space')
        axs[0][1].set_title('statistics of the distribution')
        objective_marker = ','
        mean_marker = '.'
        variance_marker = '2'
        running_mean_marker = '-'
        axs[0][1].plot([], [], objective_marker, color='black', label='QoI')
        axs[0][1].plot([], [], mean_marker, color='black', label='mean(QoI)')
        axs[0][1].plot([], [], variance_marker, color='black', label='var(QoI)')
        axs[0][1].legend()
        axs[1][0].set_title(f'stagnation detection: slope of smoothed var(QoI) (width={window_size})')
        running_slope_neg_marker = '.'
        indicator_marker = 's'
        indicator_color = 'red'
        model_indicator_color = 'lightgray'
        axs[1][0].plot([], [], indicator_marker, color=indicator_color, label='stagnation indicator')
        axs[1][0].plot([], [], running_slope_neg_marker, color='black', label='|slope|')
        axs[1][0].legend()
        axs[2][1].set_title(f'complexity of submodels ({model_indicator_color} square: used model)')
        fom_marker = '+'
        rom_size_marker = '.'
        mlm_size_marker = '4'
        axs[2][1].plot([], [], fom_marker, color='black', label='FOM: #evals')
        axs[2][1].plot([], [], rom_size_marker, color='black', label='ROM: |RB|')
        axs[2][1].plot([], [], mlm_size_marker, color='black', label='MLM: |used data|')
        axs[2][1].legend()
        legend_marker = 's'
        axs[0][0].grid()
        axs[0][1].grid()
        axs[2][1].grid()
        axs[1][0].grid()
        axs[2][0].set_title('from #evals: accuracy of adaptive model (color legend)')
        axs[2][0].axis('off')
        axs[1][1].set_title('time spent (s) in model evaluation (light) and model building (dark)')
        axs[1][1].plot([], [], color=timings_FOM_color, label='FOM')
        axs[1][1].plot([], [], color=timings_ROM_color[1], label='RB-ROM')
        axs[1][1].plot([], [], color=timings_MLM_color[1], label='ML-ROM')
        axs[1][1].legend()
        axs[1][1].set_yscale('log')
        plotter()

    init_data = lambda: {
        'num_evals': 0,
        'points': [],
        'values': [],
        'mean': [],
        'var': [],
        'timings': [],
        'running_mean': [],
        'running_slope': [],
        'stagnation_count': -1}

    collected_data = [(m.output_abs_tol, init_data())]
    _global_index = [-1]

    def wrapper(mu):
        data = collected_data[-1][1]
        data['num_evals'] += 1
        data['points'].append(mu)
        logger.info(f'evaluating model ...')
        logger.info(f"- used FOM evals: {np.sum(np.array([m._statistics['models']]) == 'FOM')}")
        logger.info(f"- used ROM evals: {np.sum(np.array([m._statistics['models']]) == 'ROM')}")
        logger.info(f"- used MLM evals: {np.sum(np.array([m._statistics['models']]) == 'MLM')}")
        tic = timer()
        QoI_val = QoI(m, mu=mu)
        data['timings'].append(timer() - tic)
        logger.info(f'- QoI: {QoI_val}')
        data['values'].append(QoI_val)
        data['mean'].append(np.mean(data['values']))
        data['var'].append(np.var(data['values']))
        _global_index[0] += 1
        if intermediate_dump_filename_prefix:
            with open(f'{intermediate_dump_filename_prefix}.pickle', 'wb') as file:
                dump((None, m._statistics), file)
        # BOTTOM LEFT
        if data['stagnation_count'] == -1:
            if plotter:
                axs[2][0].plot([], [], legend_marker, color=color, label=f'#{_global_index[0]}: output_abs_tol={m.output_abs_tol:.2e}')
                axs[2][0].legend()
            data['stagnation_count'] = 0
        # compute some statistics to detect stagnation
        if len(data['var']) >= window_size:
            data['running_mean'].append(np.mean(data['var'][-window_size:]))
        if len(data['running_mean']) >= window_size:
            running_mean_window = data['running_mean'][-window_size:]
            data['running_slope'].append(
                scipy_stats.linregress(np.arange(len(running_mean_window)), running_mean_window).slope
            )
            slope_is_bad = bad_slope(data['running_slope'][-1])
            if slope_is_bad > 0:
                data['stagnation_count'] += int(slope_is_bad)
            else:
                data['stagnation_count'] = 0
        if plotter:
            offset = np.sum([len(data['var']) for _, data in collected_data[:-1]])
            # TOP LEFT
            if len(data['points']) > 1:
                axs[0][0].plot(
                    _global_index,
                    np.linalg.norm(data['points'][-1].to_numpy() - data['points'][-2].to_numpy())/np.linalg.norm(data['points'][-2].to_numpy()),
                    '.', color=color)
            # TOP RIGHT
            axs[0][1].semilogy(_global_index, data['values'][-1], objective_marker, color=color)
            axs[0][1].semilogy(_global_index, data['mean'][-1], mean_marker, color=color)
            axs[0][1].semilogy(_global_index, data['var'][-1], variance_marker, color=color)
            # BOTTOM RIGHT
            used_model = m._statistics['models'][-1]
            assert used_model in ('FOM', 'ROM', 'MLM')
            num_fom_solves = np.sum(np.array([m._statistics['models']]) == 'FOM')
            rom_size = m._statistics['ROM']['dim'][-1] if len(m._statistics['ROM']['dim']) > 0 else 0
            mlm_size = m._statistics['MLM']['size'][-1] if len(m._statistics['MLM']['size']) > 0 else 0
            # first plot the currently used model
            axs[2][1].plot(
                _global_index,
                num_fom_solves if used_model == 'FOM' else rom_size if used_model == 'ROM' else mlm_size,
                indicator_marker,
                color=model_indicator_color)
            # then the size of each model
            if used_model == 'FOM':
                axs[2][1].plot(
                    _global_index, num_fom_solves, fom_marker, color=color)
            axs[2][1].plot(
                _global_index, rom_size, rom_size_marker, color=color)
            axs[2][1].plot(
                _global_index, mlm_size, mlm_size_marker, color=color)
            # MIDDLE RIGHT
            eval_time = m._statistics['timings']['eval_model'][-1]
            build_time = m._statistics['timings']['build_model'][-1]
            if used_model == 'FOM':
                axs[1][1].bar(
                    _global_index, eval_time, width=1., bottom=0.,
                    align='edge', color=timings_FOM_color)
                axs[1][1].bar(
                    _global_index, build_time[0], width=1., bottom=eval_time,
                    align='edge', color=timings_ROM_color[0])
                if build_time[2]:
                    axs[1][1].bar(
                        _global_index, build_time[1]+build_time[2], width=1., bottom=eval_time+build_time[0],
                        align='edge', color=timings_MLM_color[0])
            elif used_model == 'ROM':
                axs[1][1].bar(
                    _global_index, eval_time, width=1., bottom=0.,
                    align='edge', color=timings_ROM_color[1])
                axs[1][1].bar(
                    _global_index, build_time, width=1., bottom=eval_time,
                    align='edge', color=timings_MLM_color[0])
            else:  # MLM
                axs[1][1].bar(
                    _global_index, eval_time, width=1., bottom=0.,
                    align='edge', color=timings_MLM_color[1])
            # MIDDLE LEFT
            if len(data['running_mean']) >= window_size:
                slp = np.abs(data['running_slope'][-1])
                marker = running_slope_neg_marker
                # first, draw the red background for the bad points
                if slope_is_bad:
                    axs[1][0].semilogy(
                        _global_index,
                        slp,
                        indicator_marker, color=indicator_color)
                # then, draw the actual points
                axs[1][0].semilogy(
                    _global_index,
                    slp,
                    marker, color=color)
            for ax in axs[0][0], axs[0][1], axs[1][1], axs[1][0], axs[2][1]:
                ax.set_xlim([-1, np.ceil(max(_global_index[0], 1)/100)*100 + 1])
            plotter()
        # stagnation detection
        if data['stagnation_count'] >= max_violations:
            raise StagnationError


    tic = timer()
    num_refinements = 0
    while True:
        try:
            while True:
                wrapper(parameter_space.sample_randomly(1)[0])
        except StagnationError:
            logger.info(f'detected stagnation after {len(collected_data[-1][1]["points"])} evaluations')
            # test for final abortion
            if num_refinements >= max_num_refinements:
                logger.info(f'stopping after {num_refinements} refinements!')
                for ax in axs[0][0], axs[0][1], axs[1][1], axs[1][0], axs[2][1]:
                    ax.set_xlim([-1, _global_index[0] + 1])
                break
            if len(collected_data) > 1:
                # there has been at least one adaptation
                var_old = collected_data[-2][1]["var"][-1]
                var_new = collected_data[-1][1]["var"][-1]
                rel_improvement = var_new/var_old
                if rel_improvement < min_variance_variation:
                    logger.info(f'stopping, since last adaptation changed variance only by {100*rel_improvement}%!')
                    for ax in axs[0][0], axs[0][1], axs[1][1], axs[1][0], axs[2][1]:
                        ax.set_xlim([-1, _global_index[0] + 1])
                    break

            logger.info(f'lowering tolerance to continue ...')
            num_refinements += 1
            # lower tolerance of the adaptive model
            t = timer()
            m.update_tolerances(
                solution_abs_tol=m.solution_abs_tol/10,
                output_abs_tol=m.output_abs_tol/10)
            mlm_update_time = timer() - t
            # extend data collection
            collected_data.append((m.output_abs_tol, init_data()))
            if plotter:
                axs[1][1].bar(
                    _global_index, mlm_update_time, width=1.,
                    bottom=m._statistics['timings']['eval_model'][-1] \
                            + np.sum([tt or 0. for tt in m._statistics['timings']['build_model'][-1] or []]),
                    align='edge', color=timings_MLM_color[0])
                plotter()
                color = next(colors)

    data = {
        'collected': collected_data,
        'elapsed': timer() - tic,
    }

    return data
