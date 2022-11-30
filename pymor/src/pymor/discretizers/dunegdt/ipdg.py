from pymor.core.config import config, is_jupyter


if config.HAVE_DUNEGDT:
    import numpy as np
    from functools import partial
    from numbers import Number

    from dune.xt.grid import (
            ApplyOnCustomBoundaryIntersections,
            ApplyOnInnerIntersections,
            ApplyOnInnerIntersectionsOnce,
            Dim,
            DirichletBoundary,
            NeumannBoundary,
            RobinBoundary,
            Walker,
            )
    from dune.xt.functions import GridFunction as GF
    from dune.xt.la import Istl
    from dune.gdt import (
            DiscontinuousLagrangeSpace,
            DiscreteFunction,
            LocalCouplingIntersectionIntegralBilinearForm,
            LocalCouplingIntersectionRestrictedIntegralBilinearForm,
            LocalElementIntegralBilinearForm,
            LocalElementIntegralFunctional,
            LocalElementProductIntegrand,
            LocalIPDGBoundaryPenaltyIntegrand,
            LocalIPDGInnerPenaltyIntegrand,
            LocalIntersectionIntegralBilinearForm,
            LocalIntersectionIntegralFunctional,
            LocalIntersectionRestrictedIntegralBilinearForm,
            LocalIntersectionProductIntegrand,
            LocalLaplaceIPDGDirichletCouplingIntegrand,
            LocalLaplaceIPDGInnerCouplingIntegrand,
            LocalLaplaceIntegrand,
            LocalLinearAdvectionUpwindDirichletCouplingIntegrand,
            LocalLinearAdvectionUpwindInnerCouplingIntegrand,
            LocalLinearAdvectionIntegrand,
            LocalIntersectionRestrictedIntegralFunctional,
            MatrixOperator,
            VectorFunctional,
            estimate_combined_inverse_trace_inequality_constant,
            estimate_element_to_intersection_equivalence_constant,
            make_element_and_intersection_sparsity_pattern,
            )

    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.instationary import InstationaryProblem
    from pymor.bindings.dunegdt import DuneXTMatrixOperator
    from pymor.core.logger import getLogger
    from pymor.discretizers.dunegdt.functions import LincombDuneGridFunction
    from pymor.discretizers.dunegdt.gui import (
            DuneGDT1dAsNumpyVisualizer, DuneGDTK3dVisualizer, DuneGDTParaviewVisualizer)
    from pymor.core.base import ImmutableObject
    from pymor.discretizers.dunegdt.gui import (
            DuneGDT1dAsNumpyVisualizer, DuneGDTK3dVisualizer, DuneGDTParaviewVisualizer)
    from pymor.discretizers.dunegdt.functions import DuneGridFunction
    from pymor.discretizers.dunegdt.problems import InstationaryDuneProblem, StationaryDuneProblem
    from pymor.models.basic import InstationaryModel, StationaryModel
    from pymor.operators.constructions import ConstantOperator, LincombOperator, VectorArrayOperator
    from pymor.tools.floatcmp import float_cmp


    def discretize_stationary_ipdg(analytical_problem, diameter=None, domain_discretizer=None,
                                   grid_type=None, grid=None, boundary_info=None,
                                   order=1, data_approximation_order=2, la_backend=Istl(), symmetry_factor=1,
                                   weight=None, penalty_parameter=None, mu_energy_product=None,
                                   solver_options=None):
        """Discretizes a |StationaryProblem| with dune-gdt using an interior penalty (IP) discontinuous Galerkin (DG)
           method based on Lagrange finite elements.

        The type of IPDG scheme is determined by `symmetry_factor` and `weight`:

        * with `weight==None` we obtain

          - `symmetry_factor==-1`: non-symmetric interior penalty scheme (NIPDG)
          - `symmetry_factor==0`: incomplete interior penalty scheme (IIPDG)
          - `symmetry_factor==1`: symmetric interior penalty scheme (SIPDG)

        * with `weight!=None`, we expect `weight` to be a |Parameter| compatible to the diffusion of
          the analytical problem, to create a nonparametric weight function (see below), and obtain

          - `symmetry_factor==1`: symmetric weighted interior penalty scheme (SWIPDG)

        Note: Data functions might be replaced by their respective interpolations.

        Note: We currently only support linear advection, which is discretized with an upwind numerical flux.

        Parameters
        ----------
        analytical_problem
            The |StationaryProblem| to discretize.
        diameter
            If not `None`, `diameter` is passed as an argument to the `domain_discretizer`.
        domain_discretizer
            Discretizer to be used for discretizing the analytical domain. This has
            to be a function `domain_discretizer(domain_description, diameter, ...)`.
            If `None`, |discretize_domain_default| is used.
        grid_type
            If not `None`, this parameter is forwarded to `domain_discretizer` to specify
            the type of the generated |Grid|.
        grid
            Instead of using a domain discretizer, the |Grid| can also be passed directly
            using this parameter.
        boundary_info
            A |BoundaryInfo| specifying the boundary types of the grid boundary entities.
            Must be provided if `grid` is specified.
        order
            Order of the Finite Element space.
        data_approximation_order
            Polynomial order (on each grid element) for the interpolation of the data functions.
        la_backend
            Tag to determine which linear algebra backend from dune-xt is used.
        symmetry_factor
            Usually one of -1, 0, 1, determines the IPDG scheme (see above).
        weight
            Determines the IPDG scheme (see above), either None or compatible with the diffusion in the sense that:
            ```
            p = analytical_problem
            mu_weight = p.diffusion.parameters.parse(weight)
            weight = LincombFunction(p.diffusion.functions, p.diffusion.evaluate_coefficients(mu_weight))
        penalty_parameter
            Positive number to ensure coercivity of the resulting diffusion bilinear form. Is determined automatically
            if `None`.
            ```

        Returns
        -------
        m
            The |Model| that has been generated.
        data
            Dictionary with the following entries:

                :grid:           The generated |Grid|.
                :boundary_info:  The generated |BoundaryInfo|.
        """

        # currently limited to non-parametric Dirichlet data
        assert analytical_problem.dirichlet_data is None or not analytical_problem.dirichlet_data.parametric

        # convert problem, creates grid, boundary info and checks and converts all data functions
        assert isinstance(analytical_problem, StationaryProblem)
        p = StationaryDuneProblem.from_pymor(
                analytical_problem,
                data_approximation_order=data_approximation_order,
                diameter=diameter, domain_discretizer=domain_discretizer,
                grid_type=grid_type, grid=grid, boundary_info=boundary_info)

        return _discretize_stationary_ipdg_dune(
                p, order=order, la_backend=la_backend, symmetry_factor=symmetry_factor,
                weight=weight, penalty_parameter=penalty_parameter,
                mu_energy_product=mu_energy_product, solver_options=solver_options)


    def _discretize_stationary_ipdg_dune(
            dune_problem, order=1, la_backend=Istl(), symmetry_factor=1, weight=None, penalty_parameter=None,
            mu_energy_product=None, solver_options=None):
        """Discretizes a |StationaryProblem| with dune-gdt using an interior penalty (IP) discontinuous Galerkin (DG)
           method based on Lagrange finite elements.

           Note: usually not to be used directly, see :meth:`discretize_stationary_ipdg` instead.
        """
        logger = getLogger('pymor.discretizers.dunegdt.ipdg.discretize_stationary_ipdg')

        assert isinstance(dune_problem, StationaryDuneProblem)
        assert symmetry_factor in (-1, 0, 1)

        assert solver_options is None or isinstance(solver_options, (str, dict))
        if isinstance(solver_options, str):
            solver_options = {'type': solver_options}
        if isinstance(solver_options, dict) and not 'inverse' in solver_options:
            solver_options = {'inverse': solver_options}
        def ensure_opts(op):
            if solver_options:
                op = op.with_(solver_options=solver_options)
            return op

        p = dune_problem
        grid, boundary_info = p.grid, p.boundary_info
        d = grid.dimension

        # some preparations
        space = DiscontinuousLagrangeSpace(grid, order=order, dim_range=Dim(1))
        sparsity_pattern = make_element_and_intersection_sparsity_pattern(space)
        lhs_ops = []
        lhs_coeffs = []
        rhs_ops = []
        rhs_coeffs = []
        name = 'IIPDG'

        # diffusion part
        if p.diffusion:
            # penalty parameter for the diffusion part of the IPDG scheme
            if penalty_parameter is None:
                if symmetry_factor == -1:
                    name = 'NIPDG'
                    penalty_parameter = 1 # any positive number will do (the smaller the better)
                else:
                    name = 'SIPDG'
                    # TODO: check if we need to include diffusion for the coercivity here!
                    # TODO: each is a grid walk, compute this in one grid walk with the sparsity pattern
                    logger.debug('computing grid quality estimate ...')
                    C_G = estimate_element_to_intersection_equivalence_constant(grid)
                    logger.debug('computing inverse trace inequality estimate ...')
                    C_M_times_1_plus_C_T = estimate_combined_inverse_trace_inequality_constant(space)
                    penalty_parameter = C_G*C_M_times_1_plus_C_T
                    if symmetry_factor == 1:
                        penalty_parameter *= 4
            assert isinstance(penalty_parameter, Number)
            assert penalty_parameter > 0

            # weight for the diffusion part of the IPDG scheme (see above)
            if weight is None:
                weight = GF(grid, 1., (Dim(d), Dim(d)))
            else:
                assert symmetry_factor == 1
                name = 'SWIPDG'
                # try:
                if DuneGridFunction.is_base_of(weight):
                    weight = GF(grid, weight, (Dim(d), Dim(d)))
                else:
                    mu_weight = p.diffusion.parameters.parse(weight)
                    weight = p.diffusion.with_(mu=mu_weight)
                    weight = GF(grid, weight, (Dim(d), Dim(d)))

            # contributions to the left hand side
            def make_diffusion_operator_parametric_part(func):
                op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
                op += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(GF(grid, func, (Dim(d), Dim(d)))))
                op += (LocalCouplingIntersectionIntegralBilinearForm(LocalLaplaceIPDGInnerCouplingIntegrand(
                            symmetry_factor, GF(grid, func, (Dim(d), Dim(d))), weight)),
                       {}, ApplyOnInnerIntersectionsOnce(grid))
                op += (LocalIntersectionIntegralBilinearForm(LocalLaplaceIPDGDirichletCouplingIntegrand(
                            symmetry_factor, GF(grid, func, (Dim(d), Dim(d))))),
                       {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                return op

            # diffusion_funcs, diffusion_coeffs = interpolate(p.diffusion)
            lhs_ops += [make_diffusion_operator_parametric_part(func) for func in p.diffusion.functions]
            lhs_coeffs += list(p.diffusion.coefficients)

            def make_diffusion_operator_nonparametric_part():
                op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
                op += (LocalCouplingIntersectionIntegralBilinearForm(LocalIPDGInnerPenaltyIntegrand(
                            penalty_parameter, weight)),
                       {}, ApplyOnInnerIntersectionsOnce(grid))
                op += (LocalIntersectionIntegralBilinearForm(LocalIPDGBoundaryPenaltyIntegrand(
                            symmetry_factor, weight)),
                       {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                return op

            lhs_ops += [make_diffusion_operator_nonparametric_part()]
            lhs_coeffs += [1.]

            # contributions to the right hand side
            if p.dirichlet_data:
                def make_ipdg_dirichlet_penalty_functional(func):
                    op = VectorFunctional(grid, space, la_backend)
                    op += (LocalIntersectionIntegralFunctional(
                                LocalIntersectionProductIntegrand(GF(grid, penalty_parameter)).with_ansatz(GF(grid,
                                    func))), {},
                           ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                    return op

                # dirichlet_funcs, dirichlet_coeffs = interpolate(p.dirichlet_data)
                rhs_ops += [make_ipdg_dirichlet_penalty_functional(func) for func in p.dirichlet_data.functions]
                rhs_coeffs += list(p.dirichlet_data.coefficients)

                def make_laplace_ipdg_dirichlet_coupling_functional(dirichlet_func, diffusion_func):
                    op = VectorFunctional(grid, space, la_backend)
                    op += (LocalIntersectionIntegralFunctional(LocalLaplaceIPDGDirichletCouplingIntegrand(
                                symmetry_factor, GF(grid, diffusion_func, (Dim(d), Dim(d))), GF(grid, dirichlet_func))),
                           {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
                    return op

                rhs_ops += [make_laplace_ipdg_dirichlet_coupling_functional(dirichlet_func, diffusion_func)
                            for diffusion_func in p.diffusion.functions
                            for dirichlet_func in p.dirichlet_data.functions]
                rhs_coeffs += [dirichlet_coeff*diffusion_coeff
                               for diffusion_coeff in p.diffusion.coefficients
                               for dirichlet_coeff in p.dirichlet_data.coefficients]

        # advection part
        if p.advection:
            assert False
            # # TODO: the filter is probably not runtime efficient, due to the temporary FieldVector/list conversion
            # # x_local is in reference intersection coordinates
            # def restrict_to_inflow(func):
            #     return lambda intersection, x_local: \
            #             func.evaluate(intersection.to_global(x_local)).dot(intersection.unit_outer_normal(x_local)) < 0

            # # we do not simply want to use interpolate() since we require the pyMOR function for the filter above
            # # alongside the dune function

            # # contributions to the left hand side
            # def make_advection_operator(pymor_func, dune_func):
            #     op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
            #     op += LocalElementIntegralBilinearForm(LocalLinearAdvectionIntegrand(GF(grid, dune_func)))
            #         # logging_prefix='volume'))
            #     op += (LocalCouplingIntersectionRestrictedIntegralBilinearForm(restrict_to_inflow(pymor_func),
            #         LocalLinearAdvectionUpwindInnerCouplingIntegrand(GF(grid, dune_func))), #logging_prefix='inner')),
            #            {}, ApplyOnInnerIntersections(grid))
            #     op += (LocalIntersectionRestrictedIntegralBilinearForm(restrict_to_inflow(pymor_func),
            #         LocalLinearAdvectionUpwindDirichletCouplingIntegrand(GF(grid, dune_func))),
            #             # logging_prefix='dirichlet_lhs')),
            #            {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
            #     return op

            # advection_funcs_P, advection_coeffs = to_lincomb(p.advection)
            # advection_funcs_D = [interpolate_single(ff) for ff in advection_funcs_P]
            # lhs_ops += [make_advection_operator(pymor_func, dune_func)
            #             for pymor_func, dune_func in zip(advection_funcs_P, advection_funcs_D)]
            # lhs_coeffs += list(advection_coeffs)

            # # contributions to the right hand side
            # if p.dirichlet_data:
            #     def make_advection_dirichlet_boundary_functional(
            #             pymor_direction_func, dune_direction_func, dirichlet_func):
            #         op = VectorFunctional(grid, space, la_backend)
            #         op += (LocalIntersectionRestrictedIntegralFunctional(restrict_to_inflow(pymor_direction_func),
            #                 LocalLinearAdvectionUpwindDirichletCouplingIntegrand(
            #                     GF(grid, dune_direction_func), GF(grid, dirichlet_func))), #logging_prefix='dirichlet_rhs')),
            #                {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))
            #         return op

            #     if not p.diffusion:
            #         dirichlet_funcs, dirichlet_coeffs = interpolate(p.dirichlet_data)
            #     rhs_ops += [make_advection_dirichlet_boundary_functional(
            #         pymor_advection_func, dune_advection_func, dirichlet_func)
            #                 for dirichlet_func in dirichlet_funcs
            #                 for pymor_advection_func, dune_advection_func in zip(advection_funcs_P, advection_funcs_D)]
            #     rhs_coeffs += [advection_coeff*dirichlet_coeff
            #                    for dirichlet_coeff in dirichlet_coeffs
            #                    for advection_coeff in advection_coeffs]

        # reaction part
        def make_weighted_l2_operator(func):
             op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
             op += LocalElementIntegralBilinearForm(LocalElementProductIntegrand(GF(grid, func)))
             return op

        if p.reaction:
            lhs_ops += [make_weighted_l2_operator(func) for func in p.reaction.function]
            lhs_coeffs += list(p.reaction.coefficients)

        # robin boundaries
        if p.robin_data:
            assert isinstance(p.robin_data, tuple) and len(p.robin_data) == 2
            robin_parameter, robin_boundary_values = p.robin_data
            # robin_parameter_funcs, robin_parameter_coeffs = interpolate(p.robin_data[0])
            # robin_boundary_values_funcs, robin_boundary_values_coeffs = interpolate(p.robin_data[1])

            # contributions to the left hand side
            def make_weighted_l2_robin_boundary_operator(func):
                op = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
                op += (LocalIntersectionIntegralBilinearForm(LocalIntersectionProductIntegrand(GF(grid, func))), {},
                       ApplyOnCustomBoundaryIntersections(grid, boundary_info, RobinBoundary()))
                return op

            lhs_ops += [make_weighted_l2_robin_boundary_operator(func) for func in robin_parameter.functions]
            lhs_coeffs += list(robin_parameter.coefficients)

            # contributions to the right hand side
            def make_weighted_l2_robin_boundary_functional(r_param_func, r_bv_func):
                op = VectorFunctional(grid, space, la_backend)
                op += (LocalIntersectionIntegralFunctional(
                            LocalIntersectionProductIntegrand(GF(grid, r_param_func)).with_ansatz(r_bv_func)), {},
                       ApplyOnCustomBoundaryIntersections(grid, boundary_info, RobinBoundary()))
                return op

            for r_param_func, r_param_coeff in zip(robin_parameter.function, robin_parametercoefficients):
                for r_bv_func, r_bv_coeff in zip(robin_boundary_values.functions, robin_boundary_values.coefficients):
                    rhs_ops += [make_weighted_l2_robin_boundary_functional(r_param_func, r_bv_func)]
                    rhs_coeffs += [r_param_coeff*r_bv_coeff]

        # source contribution
        if p.rhs:
            def make_l2_functional(func):
                op = VectorFunctional(grid, space, la_backend)
                op += LocalElementIntegralFunctional(
                        LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(GF(grid, func)))
                return op

            # source_funcs, source_coeffs = interpolate(p.rhs)
            rhs_ops += [make_l2_functional(func) for func in p.rhs.functions]
            rhs_coeffs += list(p.rhs.coefficients)

        # neumann boundaries
        if p.neumann_data:
            def make_l2_neumann_boundary_functional(func):
                op = VectorFunctional(grid, space, la_backend)
                op += (LocalIntersectionIntegralFunctional(
                            LocalIntersectionProductIntegrand(GF(grid, -1)).with_ansatz(GF(grid, func))), {},
                       ApplyOnCustomBoundaryIntersections(grid, boundary_info, NeumannBoundary()))
                return op

            # neumann_data_funcs, neumann_data_coeffs = interpolate(p.neumann_data)
            rhs_ops += [make_l2_neumann_boundary_functional(func) for func in p.neumann_data.functions]
            rhs_coeffs += list(p.neumann_data.coefficients)

        # products
        l2_product = make_weighted_l2_operator(1)
        h1_product = make_weighted_l2_operator(1)
        h1_product += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(GF(grid, 1, (Dim(d), Dim(d)))))
        if p.diffusion:
            weighted_h1_semi_penalty_product = MatrixOperator(grid, space, space, la_backend, sparsity_pattern)
            weighted_h1_semi_penalty_product += LocalElementIntegralBilinearForm(LocalLaplaceIntegrand(weight))
            weighted_h1_semi_penalty_product += (
                    LocalCouplingIntersectionIntegralBilinearForm(LocalIPDGInnerPenaltyIntegrand(
                        penalty_parameter, weight)),
                    {}, ApplyOnInnerIntersectionsOnce(grid))
            weighted_h1_semi_penalty_product += (
                    LocalIntersectionIntegralBilinearForm(LocalIPDGBoundaryPenaltyIntegrand(
                        symmetry_factor, weight)),
                    {}, ApplyOnCustomBoundaryIntersections(grid, boundary_info, DirichletBoundary()))

        # output functionals
        outputs = []
        if p.outputs:
            if any(v[0] not in ('l2', 'l2_boundary') for v in p.outputs):
                raise NotImplementedError(f'I do not know how to discretize a {v[0]} output!')
            for output_type, output_data in p.outputs:
                assert isinstance(output_data, DuneGridFunction)  # as in: not LincombDuneGridFunction
                output_data = output_data.impl
                if output_type == 'l2':
                    op = VectorFunctional(grid, space, la_backend)
                    op += LocalElementIntegralFunctional(LocalElementProductIntegrand(GF(grid, 1)).with_ansatz(
                            GF(grid, output_data)))
                    outputs.append(op)
                elif output_type == 'l2_boundary':
                    op = VectorFunctional(grid, space, la_backend)
                    op += (LocalIntersectionIntegralFunctional(
                            LocalIntersectionProductIntegrand(GF(grid, 1)).with_ansatz(GF(grid, output_data))), {},
                            ApplyOnBoundaryIntersections(grid))
                    outputs.append(op)
                else:
                    raise NotImplementedError(f'I do not know how to discretize a {v[0]} output!')

        logger.debug('walking the grid ...')
        # assemble all of the above in one grid walk
        walker = Walker(grid)
        for op in lhs_ops:
            walker.append(op)
        for op in rhs_ops:
            walker.append(op)
        walker.append(l2_product)
        walker.append(h1_product)
        if p.diffusion:
            walker.append(weighted_h1_semi_penalty_product)
        for op in outputs:
            walker.append(op)
        walker.walk(thread_parallel=False) # support not stable/enabled yet

        # wrap everything as pyMOR operators
        lhs_ops = [ensure_opts(DuneXTMatrixOperator(op.matrix)) for op in lhs_ops]
        L = ensure_opts(LincombOperator(operators=lhs_ops, coefficients=lhs_coeffs, name='ellipticOperator'))

        rhs_ops = [VectorArrayOperator(lhs_ops[0].range.make_array([op.vector])) for op in rhs_ops]
        F = LincombOperator(operators=rhs_ops, coefficients=rhs_coeffs, name='rhsOperator')

        products = {'l2': ensure_opts(DuneXTMatrixOperator(l2_product.matrix, name='l2')),
                    'h1': ensure_opts(DuneXTMatrixOperator(h1_product.matrix, name='h1'))}
        if p.diffusion:
            products['weighted_h1_semi_penalty'] = ensure_opts(DuneXTMatrixOperator(
                    weighted_h1_semi_penalty_product.matrix, name='weighted_h1_semi_penalty'))

        outputs = [DuneXTVector(op) for op in outputs]
        if len(outputs) == 0:
            output_functional = None
        elif len(outputs) == 1:
            output_functional = outputs[0]
        else:
            from pymor.operators.block import BlockColumnOperator
            output_functional = BlockColumnOperator(outputs)

        # visualizer
        if d == 1:
            visualizer = DuneGDT1dMatplotlibVisualizer(space)
        else:
            visualizer = DuneGDTK3dVisualizer(grid, space) if is_jupyter() else DuneGDTParaviewVisualizer(space)

        m  = StationaryModel(L, F, output_functional=output_functional, products=products, visualizer=visualizer,
                             name=f'{p.name}_{name}')

        # for convenience: an interpolation of data functions into the solution space
        space_interpolation_points = space.interpolation_points() # cache
        def interpolate(func):
            df = DiscreteFunction(space, la_backend)
            np_view = np.array(df.dofs.vector, copy=False)
            np_view[:] = func.evaluate(space_interpolation_points)[:].ravel()
            return m.solution_space.make_array([df.dofs.vector,])

        data = {'grid': grid,
                'boundary_info': boundary_info,
                'space': space,
                'interpolate': interpolate}

        return m, data


