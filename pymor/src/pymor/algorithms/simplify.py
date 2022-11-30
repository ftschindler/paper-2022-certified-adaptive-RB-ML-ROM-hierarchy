# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.rules import RuleTable, match_class
from pymor.models.interface import Model
from pymor.operators.constructions import LincombOperator, ConcatenationOperator
from pymor.operators.interface import Operator


def expand(obj):
    """Expand concatenations of LincombOperators.

    To any given |Operator| or |Model|, the following
    transformations are applied recursively:

    - :class:`Concatenations <pymor.operators.constructions.ConcatenationOperator>`
      of |LincombOperators| are expanded. E.g. ::

          (O1 + O2) @ (O3 + O4)

      becomes::

          O1 @ O3 + O1 @ O4 + O2 @ O3 + O2 @ O4

    - |LincombOperators| inside |LincombOperators| are merged into a single
      |LincombOperator|

    - |ConcatenationOperators| inside |ConcatenationOperators| are merged into a
      single |ConcatenationOperator|.

    Parameters
    ----------
    obj
        Either a |Model| or an |Operator| to which the expansion rules are
        applied recursively for all :meth:`children <pymor.algorithms.rules.RuleTable.get_children>`.

    Returns
    -------
    The transformed object.
    """
    return ExpandRules().apply(obj)


class ExpandRules(RuleTable):

    def __init__(self):
        super().__init__(use_caching=True)

    @match_class(LincombOperator)
    def action_LincombOperator(self, op):
        # recursively expand all children
        op = self.replace_children(op)

        # merge child LincombOperators
        if any(isinstance(o, LincombOperator) for o in op.operators):
            ops, coeffs = [], []
            for c, o in zip(op.coefficients, op.operators):
                if isinstance(o, LincombOperator):
                    coeffs.extend(c * cc for cc in o.coefficients)
                    ops.extend(o.operators)
                else:
                    coeffs.append(c)
                    ops.append(o)
            op = op.with_(operators=ops, coefficients=coeffs)
        return op

    @match_class(ConcatenationOperator)
    def action_ConcatenationOperator(self, op):
        op = self.replace_children(op)

        # merge child ConcatenationOperators
        if any(isinstance(o, ConcatenationOperator) for o in op.operators):
            ops = []
            for o in ops:
                if isinstance(o, ConcatenationOperator):
                    ops.extend(o.operators)
                else:
                    ops.append(o)
            op = op.with_operators(ops)

        # expand concatenations with LincombOperators
        if any(isinstance(o, LincombOperator) for o in op.operators):
            i = next(iter(i for i, o in enumerate(op.operators) if isinstance(o, LincombOperator)))
            left, right = op.operators[:i], op.operators[i+1:]
            ops = [ConcatenationOperator(left + (o,) + right) for o in op.operators[i].operators]
            op = op.operators[i].with_(operators=ops)

            # there can still be LincombOperators within the summands so we recurse ..
            op = self.apply(op)

        return op

    @match_class(Model, Operator)
    def action_recurse(self, op):
        return self.replace_children(op)
