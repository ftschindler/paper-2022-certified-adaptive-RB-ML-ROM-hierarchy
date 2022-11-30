# ~~~
# This file is part of the paper:
#
#           "An adaptive projected Newton non-conforming dual approach
#         for trust-region reduced basis approximation of PDE-constrained
#                           parameter optimization"
#
#   https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt
#
# Copyright 2019-2020 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Luca Mechelli (2019)
#   Tim Keil      (2019 - 2020)
# ~~~

import numpy as np
from pymor.basic import *
from numbers import Number
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.parameters.base import ParametricObject
from pymor.discretizers.builtin.grids.gmsh import load_gmsh


def EXC_problem(input_dict = None, summed_quantities=None, outside_temperature = 10,
                q_inverse=1., parameters_in_q=False, parameter_scaling=False,
                data_path='../../EXC_data', coefficient_expressions=None):
    '''
    initialize EXC problem
    '''
    # The pixels of the pictures are 4727 x 2509
    bounding_box = [[0,0],[2,1]]

    # converting Bitmaps
    # Windows = Fenster
    f1  = BitmapFunction('{}/f1.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    f2  = BitmapFunction('{}/f2.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    f3  = BitmapFunction('{}/f3.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    f4  = BitmapFunction('{}/f4.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    f5  = BitmapFunction('{}/f5.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    f6  = BitmapFunction('{}/f6.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    f7  = BitmapFunction('{}/f7.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    f8  = BitmapFunction('{}/f8.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    f9  = BitmapFunction('{}/f9.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    f10 = BitmapFunction('{}/f10.png'.format(data_path), range=[1.,0.], bounding_box=bounding_box)
    f11 = BitmapFunction('{}/f11.png'.format(data_path), range=[1.,0.], bounding_box=bounding_box)
    f12 = BitmapFunction('{}/f12.png'.format(data_path), range=[1.,0.], bounding_box=bounding_box)

    # Walls = Waende
    w1 = BitmapFunction('{}/w1.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    w2 = BitmapFunction('{}/w2.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    w3 = BitmapFunction('{}/w3.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    w4 = BitmapFunction('{}/w4.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    w5 = BitmapFunction('{}/w5.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    w6 = BitmapFunction('{}/w6.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    w7 = BitmapFunction('{}/w7.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    w8 = BitmapFunction('{}/w8.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    sw = BitmapFunction('{}/sw.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    aw = BitmapFunction('{}/aw.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)

    # Doors = Tueren
    t1 = BitmapFunction('{}/t1.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    t2 = BitmapFunction('{}/t2.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    t3 = BitmapFunction('{}/t3.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    t4 = BitmapFunction('{}/t4.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    t5 = BitmapFunction('{}/t5.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    t6 = BitmapFunction('{}/t6.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)
    t7 = BitmapFunction('{}/t7.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)

    # outer Doors = Aussentueren
    at1 = BitmapFunction('{}/at1.png'.format(data_path), range=[1.,0.], bounding_box=bounding_box)
    at2 = BitmapFunction('{}/at2.png'.format(data_path), range=[1.,0.], bounding_box=bounding_box)
    it = BitmapFunction('{}/it.png'.format(data_path),   range=[1.,0.], bounding_box=bounding_box)

    #heaters
    h1  = BitmapFunction('{}/h1.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    h2  = BitmapFunction('{}/h2.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    h3  = BitmapFunction('{}/h3.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    h4  = BitmapFunction('{}/h4.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    h5  = BitmapFunction('{}/h5.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    h6  = BitmapFunction('{}/h6.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    h7  = BitmapFunction('{}/h7.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    h8  = BitmapFunction('{}/h8.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    h9  = BitmapFunction('{}/h9.png'.format(data_path),  range=[1.,0.], bounding_box=bounding_box)
    h10 = BitmapFunction('{}/h10.png'.format(data_path), range=[1.,0.], bounding_box=bounding_box)
    h11 = BitmapFunction('{}/h11.png'.format(data_path), range=[1.,0.], bounding_box=bounding_box)
    h12 = BitmapFunction('{}/h12.png'.format(data_path), range=[1.,0.], bounding_box=bounding_box)
    # a = b

    if input_dict is None:
        parametric_quantities = {'walls': [4,6], 'windows': [], 'doors': [6], 'heaters': [1]}
        active_quantities = {'removed_walls': [], 'open_windows': [12], 'open_doors': [2,7], 'active_heaters': [3, 8, 12]}
        input_dict = set_input_dict(parametric_quantities, active_quantities, parameters_in_q=parameters_in_q)

    if summed_quantities is None:
        summed_quantities = {'walls': [], 'windows': [], 'doors': [], 'heaters': []}

    # background
    if parameters_in_q:
        background = BitmapFunction('{}/background.png'.format(data_path), range=[0.,1.], bounding_box=bounding_box)
    else:
        background = BitmapFunction('{}/full_diffusion.png'.format(data_path), range=[0.,1.], bounding_box=bounding_box)

    if parameters_in_q:
        diffusion_functions = [background,
                                w1, w2, w3, w4, w5, w6, w7, w8, sw,
                                t1, t2, t3, t4, t5, t6, t7, it]
        for key in summed_quantities.keys():
            if len(summed_quantities[key]) > 0:
                for li in summed_quantities[key]:
                    idx = 0
                    if isinstance(li, str):
                        if li == 'a':
                            start_idx = 0
                            idx = 1
                            if key == 'walls':
                                li_iterate = list(np.arange(2,11))
                            if key == 'doors':
                                li_iterate = list(np.arange(2,11))
                    else:
                        if key == 'walls':
                            start_idx = 0
                            idx = li[0] + start_idx
                        if key == 'doors':
                            start_idx = 9
                            idx = li[0] + start_idx
                        li_iterate = li[1:]
                    if idx != 0:
                        new_lincomb_functions = [diffusion_functions[idx]]
                        new_lincomb_coefficients = [1.]
                        for l in li_iterate:
                            new_lincomb_functions.append(diffusion_functions[l + start_idx])
                            new_lincomb_coefficients.append(1.)
                            diffusion_functions[l + start_idx] = ConstantFunction(0,2)
                        diffusion_functions[idx] = LincombFunction(new_lincomb_functions, new_lincomb_coefficients)

    else:
        diffusion_functions = [background,
                                w1, w2, w3, w4, w5, w6, w7, w8, sw, aw,
                                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
                                t1, t2, t3, t4, t5, t6, t7, at1, at2, it]
        for key in summed_quantities.keys():
            if len(summed_quantities[key]) > 0:
                for li in summed_quantities[key]:
                    idx = 0
                    if isinstance(li, str):
                        if li == 'a':
                            start_idx = 0
                            idx = 1
                            if key == 'walls':
                                li_iterate = list(np.arange(2,11))
                            if key == 'windows':
                                li_iterate = list(np.arange(2,13))
                            if key == 'doors':
                                li_iterate = list(np.arange(2,11))
                    else:
                        if key == 'walls':
                            start_idx = 0
                            idx = li[0] + start_idx
                        if key == 'windows':
                            start_idx = 10
                            idx = li[0] + start_idx
                        if key == 'doors':
                            start_idx = 22
                            idx = li[0] + start_idx
                        li_iterate = li[1:]
                    if idx != 0:
                        new_lincomb_functions = [diffusion_functions[idx]]
                        new_lincomb_coefficients = [1.]
                        for l in li_iterate:
                            new_lincomb_functions.append(diffusion_functions[l + start_idx])
                            new_lincomb_coefficients.append(1.)
                            diffusion_functions[l + start_idx] = ConstantFunction(0,2)
                        diffusion_functions[idx] = LincombFunction(new_lincomb_functions, new_lincomb_coefficients)


    heat_functions = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12]

    if len(summed_quantities['heaters']) > 0:
        for li in summed_quantities['heaters']:
            if isinstance(li, str):
                if li == 'a':
                    start_idx = -1
                    idx = 1
                    li_iterate = list(np.arange(2,13))
            else:
                start_idx = -1
                idx = li[0] + start_idx
                li_iterate = li[1:]
            new_lincomb_functions = [heat_functions[idx]]
            new_lincomb_coefficients = [1.]
            for l in li_iterate:
                new_lincomb_functions.append(heat_functions[l + start_idx])
                new_lincomb_coefficients.append(1.)
                heat_functions[l + start_idx] = ConstantFunction(0,2)
            heat_functions[idx] = LincombFunction(new_lincomb_functions, new_lincomb_coefficients)

    scale_quotient = {}
    coefficients = {}
    parameter_type = {}
    parameter_ranges = {}
    parametric_quantities = {}
    for (key, tu) in input_dict.items():
        key_ranges = []
        key_coefficients = []
        scale_quotients = []
        shifts = []
        parametric_quantities[key]=0
        for et in tu:
            if isinstance(et, list): # e.g. [0.0025, 1]
                # Then we have a parametrized value
                if parameter_scaling: # parameter_scaling = False is currently used
                    scale_quotients.append(et[1])
                    key_ranges.append([et[0]/(et[1]),1])
                else: # e.g. 0.027
                    scale_quotients.append(1.)
                    key_ranges.append([et[0],et[1]])
                key_coefficients.append(None)
                parametric_quantities[key] += 1
            elif isinstance(et, Number):
                # Then, we have a deterministic value
                key_coefficients.append(et)
            else:
                print('wrong input given for key {}... converting to 1.'.format(key))
                key_coefficients.append(1.)
        coefficients[key] = key_coefficients
        if len(scale_quotients)>0:
            scale_quotient[key] = scale_quotients[0]  #ranges are the same for the same key
        if key_ranges != []:
            parameter_type[key] = len(key_ranges)
            parameter_ranges[key] = tuple(key_ranges[0])

    parameter_type = Parameters(parameter_type)
    for (key, list_) in coefficients.items():
        it = 0
        for (i,l) in enumerate(list_):
            if l is None:
                if coefficient_expressions is not None:
                    if key is 'heaters':
                        add_exp = '{}[{}]'
                        add_exp_derivative = '1*'
                        add_exp_second_derivative = '0*'
                    else:
                        add_exp = coefficient_expressions['function']
                        add_exp_derivative = coefficient_expressions['derivative']
                        add_exp_second_derivative = coefficient_expressions['second_derivative']

                else:
                    add_exp = '{}[{}]'
                    add_exp_derivative = '1*'
                if parametric_quantities[key] == 1:
                    if coefficient_expressions is None:
                        derivative_expressions = {key: '1*{}'.format(scale_quotient[key])}
                        second_derivative_expressions = {key: {key: ['0']}}
                    else:
                        derivative_expressions = {key: add_exp_derivative.format(key,it) + '1*{}'.format(scale_quotient[key])}
                        second_derivative_expressions = {key: {key: add_exp_second_derivative.format(key,it,key,it) + '1*{}'.format(scale_quotient[key])}}
                else:
                    expressions = np.empty((parametric_quantities[key],), dtype='<U60')
                    second_expressions = np.empty((parametric_quantities[key],), dtype=dict)
                    second_expression = np.empty((parametric_quantities[key],), dtype='<U60')
                    for k in range(parametric_quantities[key]):
                        expressions[k] = '0'
                        second_expression[k] = '0'
                    for k in range(parametric_quantities[key]):
                        second_expressions[k] = {key: second_expression.copy()}
                    second_derivative_expressions = {key: second_expressions}
                    if coefficient_expressions is None:
                        expressions[it] = '1*{}'.format(scale_quotient[key])
                    else:
                        expressions[it] = add_exp_derivative.format(key,it) + '1*{}'.format(scale_quotient[key])
                        second_derivative_expressions[key][it][key][it] = add_exp_second_derivative.format(key,it,key,it) + '1*{}'.format(scale_quotient[key])
                    derivative_expressions = {key: expressions}
                # print(derivative_expressions)
                # print(second_derivative_expressions)
                coefficients[key][i] = ExpressionParameterFunctional(add_exp.format(key,it,key,it) + '*{}'.format(scale_quotient[key]),
                                                                     {key:parameter_type[key]},
                                                                     derivative_expressions=derivative_expressions,
                                                                     second_derivative_expressions=second_derivative_expressions)
                it += 1

    if parameters_in_q:
        q_inverse_coefficients = [coefficients['walls'].pop(-1), coefficients['doors'].pop(7), coefficients['doors'].pop(7)]
    diffusion_coefficients = []
    heat_coefficients = []
    for (key,val) in coefficients.items():
        if (key == 'heaters'):
            heat_coefficients.extend(val)
        elif (key == 'windows'):
            if parameters_in_q:
                q_inverse_coefficients.extend(val)
            else:
                diffusion_coefficients.extend(val)
        else:
            diffusion_coefficients.extend(val)

    lincomb_diffusion = LincombFunction(diffusion_functions,diffusion_coefficients)
    lincomb_heat = LincombFunction(heat_functions,heat_coefficients)

    #Define problem
    # domain = RectDomain(bounding_box, left='robin', right='robin', top='robin', bottom='robin')
    domain = RectDomain(bounding_box, left='dirichlet', right='dirichlet', top='dirichlet', bottom='dirichlet')

    u_out = ConstantFunction( outside_temperature, 2)

    if parameters_in_q:
        # extracting parameters for boundary
        q_inverse_functions = [ aw, at1, at2,
                                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
        if len(summed_quantities['windows']) > 0:
            for li in summed_quantities['heaters']:
                if isinstance(li, str):
                    if li == 'a':
                        print('warning: I have summed all outside quantities together')
                        idx = 0
                        li_iterate = list(np.arange(2,16))
                else:
                    start_idx = 2
                    idx = li[0] + start_idx
                    li_iterate = li[1:]
                new_lincomb_functions = [q_inverse_functions[idx]]
                new_lincomb_coefficients = [1.]
                for l in li_iterate:
                    new_lincomb_functions += [q_inverse_functions[l + start_idx]]
                    new_lincomb_coefficients += [1.]
                    q_inverse_functions[l + start_idx] = ConstantFunction(0,2)
                q_inverse_functions[idx] = LincombFunction(new_lincomb_functions, new_lincomb_coefficients)

        q_inverse_function = LincombFunction(q_inverse_functions, q_inverse_coefficients)
        robin_data=(q_inverse_function, u_out)   # ( 1/q, u_out )
    else:
        robin_data=(ConstantFunction(q_inverse,2),u_out) # ( 1/q, u_out)

    problem = StationaryProblem(domain = domain,
                                diffusion = lincomb_diffusion,
                                rhs = lincomb_heat,
                                dirichlet_data=ConstantFunction(0, dim_domain=2),
                                # robin_data=robin_data,
                                parameter_ranges=parameter_ranges)

    return problem, scale_quotient


def set_input_dict(parametric_quantities, active_quantities, coefficient_expressions=None, summed_quantities=None, parameters_in_q=False, ac=None, owc=None, iwc=None,
                   idc=None, wc=None, ht=None, owc_c=None, iwc_c=None, idc_c=None, wc_c=None, ht_c=None):
    # ac means air conductivity
    # owc means outside wall and outside door conductivity
    # iwc means inside wall conductivity
    # idc means inside door conductivity
    # wc means window conductivity
    # ht means heat temperature

    # *_c stands for the constant value for the non parameterized case !
    # they need to be evaluated by the coefficient_expressions as well
    if coefficient_expressions is not None:
        assert 0, "This functionality is not supported in this publication!!"
        assert 'function' in coefficient_expressions, 'You might have assigned summed_quantities to coefficient_expressions. Check the input arguments of this function!!!'
        add_exp = coefficient_expressions['function']
    else:
        add_exp = '{}[{}]'

    epf = ExpressionParameterFunctional(add_exp.format('anything',0), {'anything': 1})
    # Attention ! Parametric case has priority over active case !

    # please provide the owc,iwc,idc,wc and ht in a list or use default
    def_ac, def_owc, def_iwc, def_idc, def_wc, def_ht = set_default_conductivities(parameters_in_q)

    if ac is None:
        ac = def_ac
    if owc is None:
        owc = def_owc
    if iwc is None:
        iwc = def_iwc
    if idc is None:
        idc = def_idc
    if wc is None:
        wc = def_wc
    if ht is None:
        ht = def_ht

    if owc_c is None:
        owc_c = owc[0]
    if iwc_c is None:
        iwc_c = iwc[0]
    if idc_c is None:
        idc_c = idc[0]
    if wc_c is None:
        wc_c = wc[0]
    if ht_c is None:
        ht_c = ht[1]

    # apply epf to _c
    mus = epf.parameters
    owc_c = epf(mus.parse(owc_c))
    iwc_c = epf(mus.parse(iwc_c))
    idc_c = epf(mus.parse(idc_c))
    wc_c = epf(mus.parse(wc_c))

    assert(isinstance(owc, list))
    assert(isinstance(iwc, list))
    assert(isinstance(idc, list))
    assert(isinstance(wc, list))
    assert(isinstance(ht, list))

    walls_tuple = ()
    windows_tuple = ()
    doors_tuple = ()
    heaters_tuple = ()

    walls_params = parametric_quantities['walls']
    windows_params = parametric_quantities['windows']
    doors_params = parametric_quantities['doors']
    heaters_params = parametric_quantities['heaters']

    active_quantities['walls'] = active_quantities['removed_walls']
    active_quantities['windows'] = active_quantities['open_windows']
    active_quantities['doors'] = active_quantities['open_doors']
    active_quantities['heaters'] = active_quantities['active_heaters']

    removed_walls = active_quantities['walls'] # removed walls
    open_windows = active_quantities['windows'] # open windows
    open_doors = active_quantities['doors'] # open_doors
    active_heaters = active_quantities['heaters']  # active heater

    if summed_quantities is not None:
        for key in ['walls', 'windows', 'doors', 'heaters']:
            if summed_quantities[key] == 'all':
                assert len(parametric_quantities[key]) == 1, 'only parameterize one quantity'
                assert len(active_quantities[key]) == 0, 'do not sum active quantities'
                continue
            for summand in summed_quantities[key]:
                assert isinstance(summand,list)
                assert len(summand) > 1, 'you must at least sum two quantities'
                for i in range(1,len(summand)):
                    assert summand[i] not in parametric_quantities[key], 'do not sum parametric quantities'
                    assert summand[i] not in active_quantities[key], 'do not sum active quantities'

    for i in range(1,11):
        if i in walls_params:
            if i == 10: # outside wall
                assert isinstance(owc, list) and len(owc) == 2, 'WRONG INPUT: owc has to be a list of the parametric range'
                walls_tuple += (owc,)
            else:
                assert isinstance(iwc, list) and len(iwc) == 2, 'WRONG INPUT: iwc has to be a list of the parametric range'
                walls_tuple += (iwc,)
        elif i in removed_walls:
            walls_tuple += (ac,)
        else:
            if i == 10:
                walls_tuple += (owc_c,)
            else:
                walls_tuple += (iwc_c,)
        if i in doors_params:
            assert isinstance(idc, list) and len(idc) == 2, 'WRONG INPUT: idc has to be a list of the parametric range'
            doors_tuple += (idc,)
        elif i in open_doors:
            doors_tuple += (ac,)
        else:
            if i in [8,9]:
                doors_tuple += (owc_c,)
            else:
                doors_tuple += (idc_c,)

    for i in range(1,13):
        if i in windows_params:
            assert isinstance(wc, list) and len(wc) == 2, 'WRONG INPUT: wc has to be a list of the parametric range'
            windows_tuple += (wc,)
        elif i in open_windows:
            assert isinstance(wc, list) and len(wc) == 2, 'WRONG INPUT: wc has to be a list where the first entry corresponds to open window'
            windows_tuple += (ac,)
        else:
            windows_tuple += (wc_c,)
        if i in heaters_params:
            assert isinstance(ht, list) and len(ht) == 2, 'WRONG INPUT: ht has to be a list of the parametric range'
            heaters_tuple += (ht,)
        elif i in active_heaters:
            heaters_tuple += (ht_c,)
        else:
            heaters_tuple += (0,)

    assert(len(walls_tuple) == 10)
    assert(len(windows_tuple) == 12)
    assert(len(doors_tuple) == 10)
    assert(len(heaters_tuple) == 12)
    input_dict = {'air': (ac,),
                  'walls': walls_tuple,      # must be 10
                  'windows': windows_tuple,  # must be 12
                  'doors': doors_tuple,      # must be 10
                  'heaters': heaters_tuple}  # must be 12

    return input_dict


def set_default_conductivities(parameters_in_q):
    if parameters_in_q:
        ac = 0.1            # air conductivity
        owc = [0.001,0.01]  # outside wall and outside door conductivity
        iwc = [0.005,0.05]  # inside wall conductivity
        idc = iwc           # inside door conductivity
        wc = [0.01,10]      # windows conductivity
        ht = [0,30]         # heat temperature
    else:
        ac = 100                # air conductivity
        owc = [1,100]           # outside wall and outside door conductivity
        iwc = [10,100]          # inside wall conductivity
        idc = iwc               # inside door conductivity
        wc = [1,100]            # windows conductivity
        ht = [10000,100000]     # heat temperature
    return ac, owc, iwc, idc, wc, ht
