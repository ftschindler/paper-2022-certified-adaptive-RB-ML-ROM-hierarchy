A new certified hierarchical and adaptive RB-ML-ROM surrogate model for parametrized PDEs
=========================================================================================

This repository contains the supplementary material for [the publication](https://doi.org/10.1137/22M1493318
)
```
A new certified hierarchical and adaptive RB-ML-ROM surrogate model for parametrized PDEs (2022)
B. Haasdonk, H. Kleikamp, M. Ohlberger, F. Schindler, T. Wenzel
SIAM Journal on Scientific Computing, Volume 45(3), A1039 - A1065, 2023
https://doi.org/10.1137/22M1493318
```

It was used to carry out the numerical experiments and generate the figures for the publication.
The experiments were performed on Linux systems in 2022 and should work with Python versions of that time (e.g., `python3.9`).
The steps below each indicate a setup to be executed in a terminal, resulting in a URL printed to be opened in a browser to view and run the indicated notebooks.


to reproduce the figures
------------------------

The data collected during the experiments is contained within this repository (in the `notebooks/KLAIM21__RBMLROM_*` and `notebooks/MC_adaptive_model` folders), and generating the figures is straightforward.
To setup an evironment for regenerating the figures, execute:
```bash
export BASEDIR=$PWD
virtualenv -p python3.9 venv-figures  # replace python3.9 with something suitable
. venv-figures/bin/activate
pip install -r requirements-figures.txt
cd $BASEDIR/pymor && pip install -e .
cd $BASEDIR && ./start_notebook_server.sh
```
After opening the printed URL (the one with `127....`) in a browser, run
- for Figure 2: [`paper_section_4_1_reactive_flow_minimization_VKOGA_create_figures.md`](notebooks/paper_section_4_1_reactive_flow_minimization_VKOGA_create_figures.md)
- for Figure 3: [`paper_section_4_1_reactive_flow_adaptive_minimization_VKOGA_create_figures.md`](notebooks/paper_section_4_1_reactive_flow_adaptive_minimization_VKOGA_create_figures.md)
- for Figure 5: [`paper_section_4_2_monte_carlo_DNN_create_figures.md`](notebooks/paper_section_4_2_monte_carlo_DNN_create_figures.md)


to reproduce the experiments
----------------------------

Reproducing the experiments is more involved and will not yield data coinciding with the original results (due to different hardware/software setup).
To setup an evironment for reproducing the experiments, execute:
```bash
export BASEDIR=$PWD
virtualenv -p python3.9 venv-experiments  # replace python3.9 with something suitable
. venv-experiments/bin/activate
pip install -r requirements-experiments.txt
cd $BASEDIR/pymor && pip install -e .
cd $BASEDIR && ./start_notebook_server.sh
```
After opening the printed URL (the one with `127....`) in a browser, run
- for Figure 2: [`paper_section_4_1_reactive_flow_minimization_VKOGA_run_experiment.md`](notebooks/paper_section_4_1_reactive_flow_minimization_VKOGA_run_experiment.md)
- for Figure 3: [`paper_section_4_1_reactive_flow_adaptive_minimization_VKOGA_run_experiment.md`](notebooks/paper_section_4_1_reactive_flow_adaptive_minimization_VKOGA_run_experiment.md)
- for Figure 5: [`paper_section_4_2_monte_carlo_DNN_run_experiment.md`](notebooks/paper_section_4_2_monte_carlo_DNN_run_experiments.md)
