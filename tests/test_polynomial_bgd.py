from fractions import Fraction
import unittest

import numpy as np

from distributions import (
    BGD,
    MUD,
    PolynomialMUD,
    leq_sum,
    symbolic_polynomial_bgd_template,
)
from semantics import (
    ConstraintContext,
    DomainPolynomialConstraint,
    ParameterConstraint,
    ParameterPolynomial,
    Relation,
    StatePolynomial,
)


def constant(ndim: int, value) -> StatePolynomial:
    return StatePolynomial.constant(ndim, value)


def make_polynomial_bgd() -> BGD:
    u = StatePolynomial.variable(1, 0)
    E = np.empty((3,), dtype=object)
    E[0] = PolynomialMUD([[0, 2]], [1 + u])
    E[1] = PolynomialMUD([[10, 12]], [2 - u])
    E[2] = PolynomialMUD([[0, 3]], [constant(1, 1)])
    return BGD(E, [Fraction(1, 2)], [Fraction(1, 4)])


def make_center_only_polynomial_bgd(
    center: StatePolynomial,
    *,
    center_breakpoints=(0, 1),
) -> BGD:
    zero = StatePolynomial.zero(1)
    E = np.empty((3,), dtype=object)
    E[0] = PolynomialMUD([[0, 1]], [zero])
    E[1] = PolynomialMUD([center_breakpoints], [center])
    E[2] = PolynomialMUD([[0, 1]], [zero])
    return BGD(E, [Fraction(1, 2)], [Fraction(1, 3)])


class TestPolynomialBGDConstruction(unittest.TestCase):
    def test_constructor_accepts_one_polynomial_cell_family(self):
        bgd = make_polynomial_bgd()

        self.assertIs(bgd.cell_family, PolynomialMUD)
        self.assertTrue(
            all(isinstance(block, PolynomialMUD) for block in bgd.E.flat)
        )
        self.assertEqual(bgd.mass(), 13)

    def test_constructor_rejects_mixed_cell_families(self):
        bgd = make_polynomial_bgd()
        E = bgd.E.copy()
        E[1] = MUD([[10, 12]], [1])

        with self.assertRaisesRegex(TypeError, "same GridMUD cell family"):
            BGD(E, bgd.alpha, bgd.beta)

    def test_exact_convolution_rejects_upper_bound_options(self):
        bgd = make_polynomial_bgd()

        with self.assertRaisesRegex(ValueError, "upper-bound options"):
            bgd.convolve_uniform(
                0,
                0,
                1,
                max_interval=Fraction(1, 2),
            )


class TestPolynomialBGDCore(unittest.TestCase):
    def assert_polynomial_family(self, bgd: BGD) -> None:
        self.assertIs(bgd.cell_family, PolynomialMUD)
        self.assertTrue(
            all(isinstance(block, PolynomialMUD) for block in bgd.E.flat)
        )

    def test_scale_and_add_preserve_exact_polynomials(self):
        bgd = make_polynomial_bgd()
        theta = ParameterPolynomial.variable("theta")

        scaled = bgd.scale(theta)
        added = bgd + bgd

        self.assert_polynomial_family(scaled)
        self.assert_polynomial_family(added)
        self.assertEqual(
            scaled.C.P[0].polynomial,
            bgd.C.P[0].polynomial * theta,
        )
        self.assertEqual(added.mass(), 26)

    def test_restrict_preserves_family_and_reparameterizes_center(self):
        bgd = make_polynomial_bgd()
        u = StatePolynomial.variable(1, 0)

        restricted = bgd.restrict(0, ">=", 11)

        self.assert_polynomial_family(restricted)
        self.assertTrue(restricted.E[0].is_empty)
        self.assertEqual(restricted.center_lefts, (Fraction(11),))
        self.assertEqual(
            restricted.C.P[0].polynomial,
            Fraction(3, 2) - u / 2,
        )
        self.assertEqual(restricted.mass(), Fraction(21, 4))

    def test_frame_alignment_preserves_family_and_mass(self):
        bgd = make_polynomial_bgd()

        center_aligned = bgd.align_center_domain([8], [14])
        period_aligned = bgd.align_edge_periods([4], [6])

        self.assert_polynomial_family(center_aligned)
        self.assert_polynomial_family(period_aligned)
        self.assertEqual(center_aligned.mass(), bgd.mass())
        self.assertEqual(period_aligned.mass(), bgd.mass())
        self.assertEqual(center_aligned.center_lefts, (Fraction(8),))
        self.assertEqual(center_aligned.center_rights, (Fraction(14),))
        self.assertEqual(period_aligned.left_lengths, (Fraction(4),))
        self.assertEqual(period_aligned.right_lengths, (Fraction(6),))

    def test_standardize_moves_dirac_boundary_without_losing_family(self):
        E = np.empty((3,), dtype=object)
        E[0] = PolynomialMUD(
            [[0, 2, 2]],
            [constant(1, 0), constant(1, 5)],
        )
        E[1] = PolynomialMUD([[10, 12]], [constant(1, 0)])
        E[2] = PolynomialMUD([[0, 3]], [constant(1, 0)])
        bgd = BGD(E, [Fraction(1, 2)], [Fraction(1, 4)])

        standardized = bgd.standardize()

        self.assert_polynomial_family(standardized)
        self.assertEqual(standardized.mass(), bgd.mass())
        self.assertEqual(
            standardized.E[0].S[0],
            (Fraction(0), Fraction(0), Fraction(2)),
        )
        self.assertEqual(
            standardized.E[0].P[0].polynomial,
            constant(1, Fraction(5, 2)),
        )
        self.assertEqual(
            standardized.C.S[0],
            (Fraction(10), Fraction(10), Fraction(12)),
        )
        self.assertEqual(
            standardized.C.P[0].polynomial,
            constant(1, 5),
        )

    def test_standardize_keeps_symbolic_decay_as_polynomial_coefficient(self):
        alpha = ParameterPolynomial.variable("alpha")
        E = np.empty((3,), dtype=object)
        E[0] = PolynomialMUD(
            [[0, 2, 2]],
            [constant(1, 0), constant(1, 5)],
        )
        E[1] = PolynomialMUD([[10, 12]], [constant(1, 0)])
        E[2] = PolynomialMUD([[0, 3]], [constant(1, 0)])

        standardized = BGD(
            E,
            [alpha],
            [ParameterPolynomial.constant(Fraction(1, 4))],
        ).standardize()

        self.assertEqual(
            standardized.E[0].P[0].polynomial,
            constant(1, 5 * alpha),
        )

    def test_dimension_operations_preserve_polynomial_family(self):
        bgd = make_polynomial_bgd()

        joint = bgd.independent_product(bgd)
        permuted = joint.permute_dims([1, 0])
        marginalized = joint.marginalize(1)

        self.assert_polynomial_family(joint)
        self.assert_polynomial_family(permuted)
        self.assert_polynomial_family(marginalized)
        self.assertEqual(joint.mass(), 169)
        self.assertEqual(permuted.mass(), 169)
        self.assertEqual(marginalized.mass(), 169)

    def test_cross_family_add_and_product_are_rejected_early(self):
        polynomial = make_polynomial_bgd()
        E = np.empty((3,), dtype=object)
        E[0] = MUD([[0, 2]], [1])
        E[1] = MUD([[10, 12]], [1])
        E[2] = MUD([[0, 3]], [1])
        mass = BGD(E, polynomial.alpha, polynomial.beta)

        with self.assertRaisesRegex(TypeError, "same GridMUD cell family"):
            polynomial.add(mass)
        with self.assertRaisesRegex(TypeError, "same GridMUD cell family"):
            polynomial.independent_product(mass)


class TestPolynomialBGDConstraints(unittest.TestCase):
    def test_nonnegative_constraints_cover_decays_and_cell_domains(self):
        bgd = make_polynomial_bgd()

        constraints = bgd.nonnegative_constraints()

        self.assertEqual(len(constraints), 7)
        self.assertTrue(
            all(
                isinstance(constraint, ParameterConstraint)
                for constraint in constraints[:4]
            )
        )
        self.assertEqual(
            [constraint.relation for constraint in constraints[:4]],
            [Relation.GE, Relation.GT, Relation.GE, Relation.GT],
        )
        self.assertTrue(
            all(
                isinstance(constraint, DomainPolynomialConstraint)
                for constraint in constraints[4:]
            )
        )
        self.assertTrue(
            all(constraint.evaluate({}) for constraint in constraints[:4])
        )
        self.assertTrue(
            all(
                constraint.evaluate_at([Fraction(1, 2)])
                for constraint in constraints[4:]
            )
        )

    def test_le_constraints_return_region_polynomial_differences(self):
        lower = make_polynomial_bgd()
        upper = lower.scale(2)

        constraints = lower.le_constraints(upper)

        self.assertTrue(
            all(
                isinstance(constraint, ParameterConstraint)
                for constraint in constraints[:2]
            )
        )
        cell_constraints = constraints[2:]
        self.assertEqual(len(cell_constraints), 3)
        self.assertTrue(
            all(
                isinstance(constraint, DomainPolynomialConstraint)
                for constraint in cell_constraints
            )
        )
        self.assertEqual(
            cell_constraints[1].polynomial,
            lower.C.P[0].polynomial,
        )

    def test_leq_sum_adds_cell_polynomials_without_symbolic_max(self):
        lower = make_polynomial_bgd()
        upper = lower.scale(2).align_center_domain([8], [14])

        constraints = leq_sum([lower, lower], upper)

        self.assertEqual(len(constraints), 11)
        self.assertTrue(
            all(
                isinstance(constraint, ParameterConstraint)
                for constraint in constraints[:4]
            )
        )
        for constraint in constraints[4:]:
            self.assertIsInstance(
                constraint,
                DomainPolynomialConstraint,
            )
            self.assertTrue(constraint.polynomial.is_zero)
            self.assertTrue(
                constraint.evaluate_at([Fraction(1, 2)])
            )

    def test_leq_sum_supports_empty_lower_sum(self):
        upper = make_polynomial_bgd()

        constraints = leq_sum([], upper)

        self.assertEqual(len(constraints), 3)
        self.assertTrue(
            all(
                isinstance(constraint, DomainPolynomialConstraint)
                for constraint in constraints
            )
        )

    def test_leq_sum_compares_decay_powers_after_period_alignment(self):
        alpha_lower = ParameterPolynomial.variable("alpha_lower")
        alpha_upper = ParameterPolynomial.variable("alpha_upper")
        lower_shape = make_polynomial_bgd()
        lower = BGD(
            lower_shape.E,
            [alpha_lower],
            lower_shape.beta,
        )
        upper_shape = lower_shape.align_edge_periods([4], [6])
        upper = BGD(
            upper_shape.E,
            [alpha_upper],
            upper_shape.beta,
        )

        constraints = leq_sum([lower], upper)

        self.assertEqual(
            constraints[0].polynomial,
            alpha_upper - alpha_lower**2,
        )
        self.assertEqual(constraints[0].relation, Relation.GE)

    def test_polynomial_constraints_reject_solver_factory(self):
        bgd = make_polynomial_bgd()

        with self.assertRaisesRegex(TypeError, "constraint_factory"):
            bgd.le_constraints(
                bgd,
                constraint_factory=lambda left, right, name: (
                    name,
                    left,
                    right,
                ),
            )


class TestPolynomialBGDUniformConvolution(unittest.TestCase):
    def test_center_density_convolves_exactly_without_mass_upper_bound(self):
        u = StatePolynomial.variable(1, 0)
        bgd = make_center_only_polynomial_bgd(
            StatePolynomial.constant(1, 1)
        )

        result = bgd.convolve_uniform(0, 0, 1)

        self.assertIs(result.cell_family, PolynomialMUD)
        self.assertEqual(result.center_lefts, (Fraction(0),))
        self.assertEqual(result.center_rights, (Fraction(2),))
        self.assertEqual(
            result.C.S,
            ((Fraction(0), Fraction(1), Fraction(2)),),
        )
        self.assertEqual(result.C.P[0].polynomial, u)
        self.assertEqual(result.C.P[1].polynomial, 1 - u)
        self.assertEqual(result.mass(), bgd.mass())

    def test_linear_center_density_grows_degree_exactly(self):
        u = StatePolynomial.variable(1, 0)
        bgd = make_center_only_polynomial_bgd(2 * u)

        result = bgd.convolve_uniform(0, 0, 1)

        self.assertEqual(result.C.P[0].polynomial, u**2)
        self.assertEqual(result.C.P[1].polynomial, 1 - u**2)
        self.assertEqual(result.C.P[0].polynomial.degree(0), 2)
        self.assertEqual(result.mass(), bgd.mass())

    def test_dirac_center_becomes_continuous_uniform_density(self):
        bgd = make_center_only_polynomial_bgd(
            StatePolynomial.constant(1, 6),
            center_breakpoints=(2, 2),
        )

        result = bgd.convolve_uniform(0, 0, 2)

        self.assertEqual(result.center_lefts, (Fraction(2),))
        self.assertEqual(result.center_rights, (Fraction(4),))
        self.assertEqual(
            result.C.P[0].polynomial,
            StatePolynomial.constant(1, 3),
        )
        self.assertEqual(result.mass(), bgd.mass())

    def test_tail_convolution_sums_finitely_many_symbolic_decay_blocks(self):
        u = StatePolynomial.variable(1, 0)
        alpha = ParameterPolynomial.variable("alpha")
        zero = StatePolynomial.zero(1)
        E = np.empty((3,), dtype=object)
        E[0] = PolynomialMUD(
            [[0, 1]],
            [StatePolynomial.constant(1, 1)],
        )
        E[1] = PolynomialMUD([[0, 1]], [zero])
        E[2] = PolynomialMUD([[0, 1]], [zero])
        bgd = BGD(E, [alpha], [Fraction(1, 3)])

        result = bgd.convolve_uniform(0, 0, 1)

        self.assertEqual(
            result.E[0].P[0].polynomial,
            alpha + (1 - alpha) * u,
        )
        self.assertEqual(
            result.C.P[0].polynomial,
            1 - u,
        )

    def test_wide_noise_crosses_multiple_tail_blocks_and_preserves_mass(self):
        bgd = make_polynomial_bgd()

        result = bgd.convolve_uniform(0, -3, 4)

        self.assertEqual(result.center_lefts, (Fraction(7),))
        self.assertEqual(result.center_rights, (Fraction(16),))
        self.assertEqual(result.left_lengths, bgd.left_lengths)
        self.assertEqual(result.right_lengths, bgd.right_lengths)
        self.assertEqual(result.mass(), bgd.mass())
        self.assertGreater(result.C.shape[0], bgd.C.shape[0])

    def test_multidimensional_convolution_changes_only_selected_axis(self):
        one_dimensional = make_center_only_polynomial_bgd(
            StatePolynomial.constant(1, 1)
        )
        joint = one_dimensional.independent_product(one_dimensional)
        u = StatePolynomial.variable(2, 0)

        result = joint.convolve_uniform(0, 0, 1)

        self.assertEqual(result.center_lefts, (Fraction(0), Fraction(0)))
        self.assertEqual(result.center_rights, (Fraction(2), Fraction(1)))
        self.assertEqual(
            result.C.S[0],
            (Fraction(0), Fraction(1), Fraction(2)),
        )
        self.assertEqual(result.C.S[1], joint.C.S[1])
        self.assertEqual(result.C.P[0, 0].polynomial, u)
        self.assertEqual(result.C.P[1, 0].polynomial, 1 - u)
        self.assertEqual(result.mass(), joint.mass())


class TestSymbolicPolynomialBGDTemplate(unittest.TestCase):
    def test_template_declares_coefficients_decays_and_validity_constraints(self):
        context = ConstraintContext()

        template = symbolic_polynomial_bgd_template(
            make_polynomial_bgd(),
            2,
            context,
            name_prefix="inv",
        )
        problem = context.build()

        self.assertIs(template.cell_family, PolynomialMUD)
        self.assertEqual(len(problem.variables), 11)
        self.assertEqual(len(problem.constraints), 7)
        self.assertEqual(
            set(template.C.P[0].polynomial.terms),
            {(0,), (1,), (2,)},
        )
        self.assertIn("inv_alpha_0", {var.name for var in problem.variables})
        self.assertIn(
            "inv_E_1_cell_0_coef_2",
            {var.name for var in problem.variables},
        )

    def test_template_uses_total_degree_multivariate_basis(self):
        shape = make_polynomial_bgd().independent_product(
            make_polynomial_bgd()
        )
        context = ConstraintContext()

        template = symbolic_polynomial_bgd_template(
            shape,
            2,
            context,
            name_prefix="joint",
        )

        self.assertEqual(
            set(template.C.P[0, 0].polynomial.terms),
            {
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (2, 0),
            },
        )

    def test_template_accepts_per_dimension_degree_limits(self):
        shape = make_polynomial_bgd().independent_product(
            make_polynomial_bgd()
        )
        context = ConstraintContext()

        template = symbolic_polynomial_bgd_template(
            shape,
            (1, 2),
            context,
            name_prefix="axis",
        )

        self.assertEqual(
            set(template.C.P[0, 0].polynomial.terms),
            {
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
            },
        )

    def test_template_excludes_state_powers_on_dirac_dimensions(self):
        E = np.empty((3,), dtype=object)
        E[0] = MUD([[0, 1]], [0])
        E[1] = MUD([[10, 10]], [1])
        E[2] = MUD([[0, 1]], [0])
        shape = BGD(E, [Fraction(1, 2)], [Fraction(1, 3)])
        context = ConstraintContext()

        template = symbolic_polynomial_bgd_template(
            shape,
            3,
            context,
            name_prefix="dirac",
        )

        self.assertEqual(
            set(template.C.P[0].polynomial.terms),
            {(0,)},
        )
        center_constraint = template.nonnegative_constraints()[5]
        self.assertEqual(center_constraint.domain.active_dims, ())

    def test_template_rejects_invalid_degree_and_duplicate_names(self):
        shape = make_polynomial_bgd()
        context = ConstraintContext()

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            symbolic_polynomial_bgd_template(shape, -1, context)
        with self.assertRaisesRegex(ValueError, "length"):
            symbolic_polynomial_bgd_template(shape, (0, 1), context)
        with self.assertRaisesRegex(TypeError, "integers"):
            symbolic_polynomial_bgd_template(shape, (False,), context)
        symbolic_polynomial_bgd_template(
            shape,
            0,
            context,
            name_prefix="same",
        )
        with self.assertRaisesRegex(KeyError, "already exists"):
            symbolic_polynomial_bgd_template(
                shape,
                0,
                context,
                name_prefix="same",
            )


if __name__ == "__main__":
    unittest.main()
