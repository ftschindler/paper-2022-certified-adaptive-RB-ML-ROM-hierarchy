import os, sys
from timeit import default_timer as timer
from itertools import cycle
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, TABLEAU_COLORS as COLORS
import numpy as np
from scipy import stats as scipy_stats

from pymor.core.logger import getLogger
from pymor.tools.floatcmp import float_cmp
from pymor.core.pickle import dump

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # models
from models.adaptive import AdaptiveModel

class StagnationError(Exception):
    pass


def adaptive_minimization(
        m,
        parameter_space,
        initial_guess,
        objective,
        method,
        opts,
        stagnation_detection=(
            None,   # width of running window
            None,   # max num violations
            None,   # bad slope indicator
            None),  # bad ratio indicator
        plotter=False,
        intermediate_dump_filename_prefix=None):

    from scipy.optimize import minimize as scipy_optimize
    from scipy import stats as scipy_stats

    logger = getLogger(f'algorithms.optimization.{method.lower()}')
    assert isinstance(m, AdaptiveModel)
    if not isinstance(initial_guess, np.ndarray):
        initial_guess = initial_guess.to_numpy()
    if isinstance(plotter, bool) and plotter:
        plot_objective_surface = False
        draw_current_plot = lambda : None
    if plotter:
        plot_objective_surface, draw_current_plot = plotter
    window_size, max_violations, bad_slope_threshold, bad_ratio_threshold = stagnation_detection
    bad_slope = lambda s: s > bad_slope_threshold
    bad_ratio = lambda r: r < bad_ratio_threshold

    bounds = []
    for kk in parameter_space.ranges.keys():
        for jj in range(parameter_space.parameters[kk]):
            bounds.append((parameter_space.ranges[kk][0], parameter_space.ranges[kk][1]))
    bounds = np.array(tuple(np.array(b) for b in bounds))


    if plotter:
        logger.info('preparing plot:')
        fig, axs = plt.subplots(3, 2, figsize=(16, 14), gridspec_kw={'width_ratios':[1, 1.5], 'height_ratios':[1, 1, 1]})
        timings_FOM_color = to_rgba('#74a9cf', 1.)
        timings_ROM_color = (to_rgba('#fd8d3c', 1.), to_rgba('#fecc5c', 1.))
        timings_MLM_color = (to_rgba('#006837', 1.), to_rgba('#78c679', 1.))
        colors = cycle(COLORS)
        color = next(colors)
        plot_objective_surface(fig, axs[0][0])
        fig.suptitle(f'adaptive {method} minimization')
        axs[0][0].set_title('trajectory in parameter space')
        axs[0][1].set_title(f'raw objective and running mean (width={window_size})')
        objective_marker = '.'
        running_mean_marker = '-'
        axs[0][1].plot([], [], objective_marker, color='black', label='objective')
        axs[0][1].plot([], [], running_mean_marker, color='black', label='running mean')
        axs[0][1].legend()
        axs[1][0].set_title(f'stagnation detection: slope of smoothed objective (width={window_size})')
        running_slope_marker = 'x'
        running_slope_neg_marker = '.'
        indicator_marker = 's'
        indicator_color = 'red'
        model_indicator_color = 'lightgray'
        axs[1][0].plot([], [], indicator_marker, color=indicator_color, label='stagnation indicator')
        axs[1][0].plot([], [], running_slope_marker, color='black', label='slope')
        axs[1][0].plot([], [], running_slope_neg_marker, color='black', label='$-$ slope')
        axs[1][0].legend()
        axs[2][1].set_title(f'complexity of submodels ({model_indicator_color} square: used model)')
        axs[1][1].set_title('time spent (s) in model evaluation (light) and model building (dark)')
        axs[1][1].plot([], [], color=timings_FOM_color, label='FOM')
        axs[1][1].plot([], [], color=timings_ROM_color[1], label='RB-ROM')
        axs[1][1].plot([], [], color=timings_MLM_color[1], label='ML-ROM')
        axs[1][1].legend()
        axs[1][1].set_yscale('log')
        axs[2][0].set_title('color legend and stagnation detection: normalized slope')
        axs[2][0].plot([], [], indicator_marker, color=indicator_color, label='stagnation indicator')
        axs[2][0].legend()
        fom_marker = '+'
        rom_size_marker = '.'
        mlm_size_marker = '4'
        axs[2][1].plot([], [], fom_marker, color='black', label='FOM: #evals')
        axs[2][1].plot([], [], rom_size_marker, color='black', label='ROM: |RB|')
        axs[2][1].plot([], [], mlm_size_marker, color='black', label='MLM: |used data|')
        axs[2][1].legend()
        ratio_marker = '.'
        axs[0][0].plot(initial_guess[0], initial_guess[1], 'o', color='white')
        for ax in axs[0][1], axs[1][0], axs[2][0], axs[2][1]:
            ax.grid()
        draw_current_plot()

    init_data = lambda: {
        'num_evals': 0,
        'points': [],
        'values': [],
        'timings': [],
        'running_mean': [],
        'running_slope': [],
        'stagnation_count': -1}

    collected_data = [(m.output_abs_tol, init_data())]
    _global_index = [-1]

    def wrapper(mu):
        data = collected_data[-1][1]
        data['num_evals'] += 1
        data['points'].append(parameter_space.parameters.parse(mu))  # copy of mu required, changed inplace
        logger.info(f'evaluating model ...')
        logger.info(f"- used FOM evals: {np.sum(np.array([m._statistics['models']]) == 'FOM')}")
        logger.info(f"- used ROM evals: {np.sum(np.array([m._statistics['models']]) == 'ROM')}")
        logger.info(f"- used MLM evals: {np.sum(np.array([m._statistics['models']]) == 'MLM')}")
        tic = timer()
        QoI = objective(m, mu)
        data['timings'].append(timer() - tic)
        logger.info(f'- objective: {QoI}')
        data['values'].append(QoI)
        _global_index[0] += 1
        if intermediate_dump_filename_prefix:
            with open(f'{intermediate_dump_filename_prefix}.pickle', 'wb') as file:
                dump((None, m._statistics), file)
        if data['stagnation_count'] == -1:
            if plotter:
                # plot color legend bottom left
                axs[2][0].plot([], [], ratio_marker, color=color, label=f'output_abs_tol={m.output_abs_tol:.2e}')
                axs[2][0].legend()
            data['stagnation_count'] = 0
        # compute some statistics to detect stagnation
        if len(data['values']) >= window_size:
            data['running_mean'].append(np.mean(data['values'][-window_size:]))
        if len(data['running_mean']) >= window_size:
            running_mean_window = data['running_mean'][-window_size:]
            data['running_slope'].append(
                scipy_stats.linregress(np.arange(len(running_mean_window)), running_mean_window).slope
            )
            running_ratio = np.array(np.abs(data['running_slope'])/(np.array(data['values'])/collected_data[0][1]['values'][0])[-len(data['running_slope']):])
            ratio_is_bad = bad_ratio(running_ratio[-1])
            slope_is_bad = bad_slope(data['running_slope'][-1])
            if ratio_is_bad + slope_is_bad > 0:
                data['stagnation_count'] += ratio_is_bad + slope_is_bad
            else:
                data['stagnation_count'] = 0
        if plotter:
            # TOP LEFT
            if (bounds[0][0] <= mu[0] <= bounds[0][1]) and (bounds[1][0] <= mu[1] <= bounds[1][1]):
                axs[0][0].plot(mu[0], mu[1], '.', color=color)
            # TOP RIGHT
            axs[0][1].semilogy(_global_index, data['values'][-1], objective_marker, color=color)
            if len(data['running_mean']) >= 2:
                axs[0][1].plot([_global_index[0] - 1, _global_index[0]],
                               data['running_mean'][-2:],
                               running_mean_marker, color=color)
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
                    _global_index, eval_time, bottom=0.,
                    align='edge', color=timings_FOM_color)
                axs[1][1].bar(
                    _global_index, build_time[0], bottom=eval_time,
                    align='edge', color=timings_ROM_color[0])
                if build_time[2]:
                    axs[1][1].bar(
                        _global_index, build_time[1]+build_time[2], bottom=eval_time+build_time[0],
                        align='edge', color=timings_MLM_color[0])
            elif used_model == 'ROM':
                axs[1][1].bar(
                    _global_index, eval_time, bottom=0.,
                    align='edge', color=timings_ROM_color[1])
                axs[1][1].bar(
                    _global_index, build_time, bottom=eval_time,
                    align='edge', color=timings_MLM_color[0])
            else:  # MLM
                axs[1][1].bar(
                    _global_index, eval_time, bottom=0.,
                    align='edge', color=timings_MLM_color[1])
            # BOTTOM LEFT
            if len(data['running_mean']) >= window_size:
                # first, draw the red background for the bad points
                if ratio_is_bad:
                    axs[2][0].semilogy(
                        _global_index,
                        running_ratio[-1],
                        indicator_marker, color=indicator_color)
                # then, draw the actual points
                axs[2][0].semilogy(
                    _global_index,
                    running_ratio[-1],
                    ratio_marker, color=color)
            # MIDDLE LEFT
            if len(data['running_mean']) >= window_size:
                slp = data['running_slope'][-1]
                if slp > 0:
                    slp_to_plot = [slp,]
                    marker = running_slope_marker
                elif slp < 0:
                    slp_to_plot = [np.abs(slp),]
                    marker = running_slope_neg_marker
                else:
                    slp_to_plot = [1e-15,]
                    marker = running_slope_marker
                # first, draw the red background for the bad points
                if slope_is_bad:
                    axs[1][0].semilogy(
                        _global_index,
                        slp_to_plot,
                        indicator_marker, color=indicator_color)
                # then, draw the actual points
                axs[1][0].semilogy(
                    _global_index,
                    slp_to_plot,
                    marker, color=color)
            for ax in axs[0][1], axs[1][0], axs[1][1], axs[2][0], axs[2][1]:
                ax.set_xlim([0, np.ceil(_global_index[0]/100)*100])
            draw_current_plot()
        # stagnation detection
        if data['stagnation_count'] >= max_violations:
            raise StagnationError
        return QoI

    tic = timer()
    while True:
        try:
            results = scipy_optimize(wrapper,
                x0=initial_guess,
                method=method,
                bounds=bounds,
                options=opts)
            break
        except StagnationError:
            last_mu = collected_data[-1][1]['points'][-1]
            logger.warn(f'detected stagnation at mu={last_mu}, lowering tolerance ...')
            # continue at the last found point
            initial_guess = last_mu.to_numpy()
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
                draw_current_plot()
                color = next(colors)

    data = {
        'collected': collected_data,
        'elapsed': timer() - tic,
        'results': results,
    }

    if (results.status != 0):
        if plotter:
            x = collected_data[-1][1]['points'][-1].to_numpy()
            axs[0][0].plot(x[0], x[1], 's', color='white')
            draw_current_plot()
        logger.info(' failed!')
        logger.info(' These are the results:')
        logger.info(f'\n{results}')
    else:
        if plotter:
            axs[0][0].plot(results.x[0], results.x[1], 'x', color='white')
            draw_current_plot()
        logger.info(' succeded!')
        logger.info('  minimizer:       {}'.format(parameter_space.parameters.parse(results.x)))
        logger.info('  objective value: {}'.format(results.fun))
        logger.info(' These are the results:')
        logger.info(f'\n{results}')

    return data
