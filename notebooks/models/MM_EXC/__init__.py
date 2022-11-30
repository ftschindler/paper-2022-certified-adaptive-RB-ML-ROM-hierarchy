import numpy as np
import os, sys

from pymor.parameters.functionals import MinThetaParameterFunctional, ExpressionParameterFunctional

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # tools.py
from tools import simplify


def make_problem():
    '''yields problem, parameter_space, mu_bar'''

    from pymor.analyticalproblems.functions import BitmapFunction, ConstantFunction
    from pymor.analyticalproblems.instationary import InstationaryProblem
    from .problem import EXC_problem, set_input_dict

    parametric_quantities = {'walls': [1,2,3,4,5,6,8,9], 'windows': [], 'doors': [1,2,3,4,5,6,7,10], 'heaters': [1,2,3,4,5,6,7,8,9,10,11,12]}
    inactive_quantities = {'removed_walls': [], 'open_windows': [], 'open_doors': [], 'active_heaters': []}
    summed_quantities = {'walls': [], 'windows': [], 'doors': [], 'heaters': []}

    input_dict = set_input_dict(parametric_quantities, inactive_quantities, None, summed_quantities, True,
                            ac=0.5, owc=[0.005,0.5], iwc= [0.0025,0.1], idc=[0.01,0.1], wc=[0.01,0.1], ht=[0,100],
                                    owc_c=0.027,  iwc_c= 0.0025, idc_c=0.01,  wc_c=0.01,  ht_c=80)

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
    problem, _ = EXC_problem(input_dict, summed_quantities, outside_temperature=5,
                                        data_path = data_path,parameters_in_q=True,
                                        parameter_scaling=False,
                                        coefficient_expressions=None)
    parameter_space = problem.parameter_space

    domain_of_interest = BitmapFunction('{}/Domain_of_interest.png'.format(data_path), range=[8.57,0], bounding_box=problem.domain.domain)
    problem = problem.with_(outputs=(('l2', domain_of_interest),))  # scaling in range above for average temperature

    mu_bar = []
    for key, size in sorted(problem.parameter_space.parameters.items()):
        range_ = problem.parameter_space.ranges[key]
        if range_[0] == 0:
            value = 10**(np.log10(range_[1])/2)
        else:
            value = 10**((np.log10(range_[0]) + np.log10(range_[1]))/2)
        for i in range(size):
            mu_bar.append(value)
    mu_bar = problem.parameters.parse(mu_bar)

    problem = InstationaryProblem(problem, ConstantFunction(0, dim_domain=2), T=1, name='MMexcProblem')

    return problem, parameter_space, mu_bar


def discretize(problem, mu_bar, num_global_refines=0, nt=9, scheme='pymor-CG-P1'):
    '''yields fom, fom_data, coercivity_estimate'''
    grid_width = (np.sqrt(2)/200.)*2**-num_global_refines
    if scheme == 'pymor-CG-P1':
        from pymor.discretizers.builtin.cg import discretize_instationary_cg
        fom, fom_data = discretize_instationary_cg(
            problem, diameter=grid_width, nt=nt, mu_energy_product=mu_bar)
    elif scheme.startswith('dune') and scheme[5:9] == 'CG-P':
        order = int(scheme[9:])
        from pymor.discretizers.dunegdt.cg import discretize_instationary_cg
        fom, fom_data = discretize_instationary_cg(
            problem, diameter=grid_width, nt=nt, order=order, mu_energy_product=mu_bar)
    else:
        raise RuntimeError(f'unknown discretization scheme "{scheme}" requested!')

    fom = fom.with_(rhs=ExpressionParameterFunctional('min([2*t, 1])', {'t': 1})*fom.rhs)  # should be in the problem
    fom = fom.with_(operator=simplify(fom.operator), rhs=simplify(fom.rhs))
    fom = fom.with_(name='MMexcModel')
    coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)

    return fom, coercivity_estimator
