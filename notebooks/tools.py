from IPython import display
from matplotlib import pyplot as plt
import numpy as np

from pymor.algorithms.to_matrix import to_matrix
from pymor.bindings.dunegdt import DuneXTMatrixOperator
from pymor.core.base import ImmutableObject
from pymor.models.basic import InstationaryModel
from pymor.operators.constructions import LincombOperator, VectorArrayOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ParameterFunctional


def draw_current_plot():

    display.clear_output(wait=True)
    display.display(plt.gcf())


def set_solver_options(fom, solver_options):
    assert solver_options is None or isinstance(solver_options, (str, dict))
    if isinstance(solver_options, str):
        solver_options = {'type': solver_options}
    if isinstance(solver_options, dict) and not 'inverse' in solver_options:
        solver_options = {'inverse': solver_options}

    def with_opts(op):
        if isinstance(op, LincombOperator):
            return LincombOperator(
                operators=[with_opts(oo) for oo in op.operators],
                coefficients=op.coefficients,
                solver_options=solver_options,
                name=op.name)  # the name here is extremely important, see https://github.com/pymor/pymor/discussions/1575
        elif isinstance(op, VectorArrayOperator):
            return op
        else:
            return op.with_(solver_options=solver_options)

    fom = fom.with_(operator=with_opts(fom.operator))
    if fom.products:
        products = {kk: with_opts(pp) for kk, pp in fom.products.items()}
        fom = fom.with_(products=products)
    if hasattr(fom, 'mass') and fom.mass is not None:
        fom = fom.with_(mass=with_opts(fom.mass))
    return fom


def simplify(op):
    assert isinstance(op, LincombOperator)
    parametric_ops = []
    parametric_coeffs = []
    nonparametric_ops = []
    nonparametric_coeffs = []
    for oo, cc in zip(op.operators, op.coefficients):
        if isinstance(cc, ParameterFunctional):
            parametric_ops.append(oo)
            parametric_coeffs.append(cc)
        else:
            nonparametric_ops.append(oo)
            nonparametric_coeffs.append(cc)
    if len(nonparametric_ops) == 0:
        return op
    else:
        return op.with_(
            operators=parametric_ops + [LincombOperator(nonparametric_ops, nonparametric_coeffs).assemble()],
            coefficients=parametric_coeffs + [1,])


class ConvertedVisualizer(ImmutableObject):

    def __init__(self, visualizer, vector_space):
        self.__auto_init(locals())

    def visualize(self, U, *args, **kwargs):
        V = self.vector_space.zeros(len(U))
        for ii in range(len(U)):
            v = np.array(V._list[ii].real_part.impl, copy=False)
            v[:] = U[ii].to_numpy()[:]
        self.visualizer.visualize(V, *args, **kwargs)


def to_numpy(obj):

    if isinstance(obj, (DuneXTMatrixOperator, VectorArrayOperator)):
        return NumpyMatrixOperator(to_matrix(obj))
    elif isinstance(obj, LincombOperator):
        return obj.with_(
            operators=[to_numpy(op) for op in obj.operators],
            solver_options=None,
        )
    elif isinstance(obj, InstationaryModel):
        return obj.with_(
            operator = to_numpy(obj.operator),
            rhs = to_numpy(obj.rhs),
            mass = to_numpy(obj.mass),
            products = {kk: to_numpy(vv) for kk, vv in obj.products.items()},
            initial_data = to_numpy(obj.initial_data),
            output_functional = to_numpy(obj.output_functional),
            visualizer=ConvertedVisualizer(obj.visualizer, obj.solution_space)
        )

    assert False, "We should not get here!"
