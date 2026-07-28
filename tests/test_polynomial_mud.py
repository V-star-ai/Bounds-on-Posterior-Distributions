from fractions import Fraction
import unittest

import numpy as np

from distributions import (
    MUD,
    PolynomialCell,
    PolynomialMUD,
)
from semantics import (
    DomainPolynomialConstraint,
    ParameterPolynomial,
    ParameterVariable,
    Relation,
    StatePolynomial,
)


class TestPolynomialMUDConstruction(unittest.TestCase):
    def test_constructor_accepts_state_polynomial_payloads(self):
        u = StatePolynomial.variable(1, 0)

        mud = PolynomialMUD([[0, 1]], [1 + u])

        self.assertIsInstance(mud.P[0], PolynomialCell)
        self.assertEqual(mud.P[0].polynomial, 1 + u)

    def test_constructor_rejects_wrong_polynomial_dimension(self):
        with self.assertRaisesRegex(ValueError, "2 state dimensions"):
            PolynomialMUD(
                [[0, 1], [0, 1]],
                [[StatePolynomial.constant(1, 1)]],
            )

    def test_constructor_rejects_dependence_on_dirac_dimension(self):
        u = StatePolynomial.variable(1, 0)

        with self.assertRaisesRegex(ValueError, "Dirac dimension 0"):
            PolynomialMUD([[2, 2]], [1 + u])


class TestPolynomialMUDCore(unittest.TestCase):
    def test_mass_integrates_density_and_physical_volume(self):
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)
        mud = PolynomialMUD(
            [[0, 2], [10, 13]],
            [[1 + u + 2 * v]],
        )

        self.assertEqual(mud.mass(), 15)

    def test_mass_uses_dirac_measure_factor_one(self):
        v = StatePolynomial.variable(2, 1)
        mud = PolynomialMUD(
            [[4, 4], [0, 2]],
            [[3 + 2 * v]],
        )

        self.assertEqual(mud.mass(), 8)

    def test_align_reparameterizes_each_subcell_exactly(self):
        u = StatePolynomial.variable(1, 0)
        mud = PolynomialMUD([[0, 2]], [1 + 2 * u])

        aligned = mud.align([[0, 1, 2]])

        self.assertEqual(aligned.P[0].polynomial, 1 + u)
        self.assertEqual(aligned.P[1].polynomial, 2 + u)
        self.assertEqual(aligned.mass(), mud.mass())

    def test_restrict_reparameterizes_density_without_renormalizing(self):
        u = StatePolynomial.variable(1, 0)
        mud = PolynomialMUD([[0, 2]], [1 + 2 * u])

        restricted = mud.restrict(0, ">=", 1)

        self.assertEqual(restricted.S, ((Fraction(1), Fraction(2)),))
        self.assertEqual(restricted.P[0].polynomial, 2 + u)
        self.assertEqual(restricted.mass(), Fraction(5, 2))

    def test_restrict_dirac_respects_strictness(self):
        mud = PolynomialMUD(
            [[2, 2]],
            [StatePolynomial.constant(1, 7)],
        )

        self.assertEqual(mud.restrict(0, ">=", 2).mass(), 7)
        self.assertTrue(mud.restrict(0, ">", 2).is_empty)

    def test_empty_restrict_keeps_polynomial_cell_family(self):
        mud = PolynomialMUD(
            [[0, 1], [2, 4]],
            [[StatePolynomial.constant(2, 3)]],
        )

        result = mud.restrict(0, ">", 2)

        self.assertTrue(result.is_empty)
        self.assertIsInstance(result, PolynomialMUD)
        self.assertEqual(result.S[0], (Fraction(2),))
        self.assertEqual(result.S[1], (Fraction(2), Fraction(4)))

    def test_align_preserves_mixed_dirac_cell(self):
        v = StatePolynomial.variable(2, 1)
        mud = PolynomialMUD(
            [[1, 1], [0, 2]],
            [[2 + v]],
        )

        result = mud.align(
            [[0, 1, 1, 2], [0, 1, 2]],
        )
        refined_v = StatePolynomial.variable(2, 1)

        self.assertEqual(result.P[1, 0].polynomial, 2 + refined_v / 2)
        self.assertEqual(result.P[1, 1].polynomial, Fraction(5, 2) + refined_v / 2)
        self.assertEqual(result.mass(), mud.mass())

    def test_add_aligns_grids_and_adds_polynomials(self):
        u = StatePolynomial.variable(1, 0)
        left = PolynomialMUD([[0, 2]], [1 + u])
        right = PolynomialMUD([[0, 1, 2]], [u, 2 * u])

        result = left + right

        self.assertEqual(result.S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(result.P[0].polynomial, 1 + Fraction(1, 2) * u + u)
        self.assertEqual(
            result.P[1].polynomial,
            Fraction(3, 2) + Fraction(1, 2) * u + 2 * u,
        )
        self.assertEqual(result.mass(), left.mass() + right.mass())

    def test_scale_accepts_parameter_polynomial(self):
        theta_variable = ParameterVariable("theta")
        theta = ParameterPolynomial.variable(theta_variable)
        u = StatePolynomial.variable(1, 0)
        mud = PolynomialMUD([[0, 1]], [1 + u])

        scaled = mud.scale(theta)

        self.assertEqual(scaled.P[0].polynomial, theta + theta * u)
        self.assertEqual(
            scaled.mass().evaluate({theta_variable: 4}),
            6,
        )

    def test_evaluate_uses_physical_to_local_coordinate_map(self):
        theta_variable = ParameterVariable("theta")
        theta = ParameterPolynomial.variable(theta_variable)
        u = StatePolynomial.variable(1, 0)
        mud = PolynomialMUD([[2, 6]], [theta + 2 * u])

        value = mud.ops.evaluate(
            mud.P[0],
            mud._intervals_for_index((0,)),
            [3],
        )

        self.assertEqual(value, theta + Fraction(1, 2))


class TestPolynomialMUDDimensions(unittest.TestCase):
    def test_independent_product_concatenates_polynomial_variables(self):
        left_u = StatePolynomial.variable(1, 0)
        right_v = StatePolynomial.variable(1, 0)
        left = PolynomialMUD([[0, 2]], [1 + left_u])
        right = PolynomialMUD([[10, 13]], [2 + right_v])

        result = left.independent_product(right)
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)

        self.assertEqual(result.P[0, 0].polynomial, (1 + u) * (2 + v))
        self.assertEqual(result.mass(), left.mass() * right.mass())

    def test_marginalize_integrates_removed_continuous_dimension(self):
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)
        mud = PolynomialMUD(
            [[0, 2], [10, 13]],
            [[1 + u + 2 * v]],
        )

        result = mud.marginalize(0)
        remaining = StatePolynomial.variable(1, 0)

        self.assertEqual(result.P[0].polynomial, 3 + 4 * remaining)
        self.assertEqual(result.mass(), mud.mass())

    def test_marginalize_removes_dirac_dimension_without_zero_factor(self):
        v = StatePolynomial.variable(2, 1)
        mud = PolynomialMUD(
            [[4, 4], [0, 2]],
            [[3 + 2 * v]],
        )

        result = mud.marginalize(0)
        remaining = StatePolynomial.variable(1, 0)

        self.assertEqual(result.P[0].polynomial, 3 + 2 * remaining)
        self.assertEqual(result.mass(), mud.mass())

    def test_marginalize_sums_multiple_removed_cells(self):
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)
        mud = PolynomialMUD(
            [[0, 1, 3], [0, 1]],
            [[1 + u], [2 + v]],
        )

        result = mud.marginalize(0)
        remaining = StatePolynomial.variable(1, 0)

        self.assertEqual(
            result.P[0].polynomial,
            Fraction(11, 2) + 2 * remaining,
        )
        self.assertEqual(result.mass(), mud.mass())

    def test_permute_dims_reorders_grid_and_polynomial_variables(self):
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)
        mud = PolynomialMUD(
            [[0, 2], [10, 13]],
            [[1 + 2 * u + 3 * v]],
        )

        result = mud.permute_dims([1, 0])
        new_u = StatePolynomial.variable(2, 0)
        new_v = StatePolynomial.variable(2, 1)

        self.assertEqual(
            result.S,
            (
                (Fraction(10), Fraction(13)),
                (Fraction(0), Fraction(2)),
            ),
        )
        self.assertEqual(
            result.P[0, 0].polynomial,
            1 + 3 * new_u + 2 * new_v,
        )
        self.assertEqual(result.mass(), mud.mass())


class TestPolynomialMUDEmbeddingAndConstraints(unittest.TestCase):
    def test_mass_mud_embedding_is_exact_for_continuous_and_dirac_cells(self):
        mass_mud = MUD(
            [[0, 2, 2], [10, 13]],
            np.array([[12], [5]], dtype=object),
        )

        result = PolynomialMUD.from_mass_mud(mass_mud)

        self.assertEqual(
            result.P[0, 0].polynomial,
            StatePolynomial.constant(2, 2),
        )
        self.assertEqual(
            result.P[1, 0].polynomial,
            StatePolynomial.constant(2, Fraction(5, 3)),
        )
        self.assertEqual(result.mass(), mass_mud.mass())

    def test_nonnegative_constraint_uses_only_continuous_dimensions(self):
        v = StatePolynomial.variable(2, 1)
        mud = PolynomialMUD(
            [[4, 4], [0, 2]],
            [[1 + v]],
        )

        constraint = mud.ops.nonnegative_constraint(
            mud.P[0, 0],
            mud._intervals_for_index((0, 0)),
        )

        self.assertIsInstance(constraint, DomainPolynomialConstraint)
        self.assertEqual(constraint.relation, Relation.GE)
        self.assertEqual(constraint.domain.active_dims, (1,))

    def test_le_constraint_keeps_polynomial_difference(self):
        u = StatePolynomial.variable(1, 0)
        left = PolynomialMUD([[0, 1]], [1 + u])
        right = PolynomialMUD([[0, 1]], [2 + 3 * u])

        constraint = left.ops.le_constraint(
            left.P[0],
            right.P[0],
            left._intervals_for_index((0,)),
        )

        self.assertEqual(constraint.polynomial, 1 + 2 * u)
        self.assertTrue(constraint.evaluate_at([Fraction(1, 3)]))


class TestPolynomialMUDUniformConvolution(unittest.TestCase):
    def test_constant_density_convolves_to_exact_triangle(self):
        u = StatePolynomial.variable(1, 0)
        mud = PolynomialMUD(
            [[0, 1]],
            [StatePolynomial.constant(1, 1)],
        )

        result = mud.convolve_uniform(0, 0, 1)

        self.assertEqual(
            result.S,
            ((Fraction(0), Fraction(1), Fraction(2)),),
        )
        self.assertEqual(result.P[0].polynomial, u)
        self.assertEqual(result.P[1].polynomial, 1 - u)
        self.assertEqual(result.mass(), mud.mass())

    def test_linear_density_increases_convolved_degree_by_one(self):
        u = StatePolynomial.variable(1, 0)
        theta = ParameterPolynomial.variable("theta")
        mud = PolynomialMUD([[0, 1]], [theta * u])

        result = mud.convolve_uniform(0, 0, 1)

        self.assertEqual(
            result.P[0].polynomial,
            theta * u**2 / 2,
        )
        self.assertEqual(
            result.P[1].polynomial,
            theta * (1 - u**2) / 2,
        )
        self.assertEqual(result.P[0].polynomial.degree(0), 2)
        self.assertEqual(result.mass(), mud.mass())

    def test_nonunit_source_and_noise_lengths_use_physical_jacobians(self):
        u = StatePolynomial.variable(1, 0)
        mud = PolynomialMUD(
            [[0, 2]],
            [StatePolynomial.constant(1, 1)],
        )

        result = mud.convolve_uniform(0, -1, 1)

        self.assertEqual(
            result.S,
            (
                (
                    Fraction(-1),
                    Fraction(1),
                    Fraction(3),
                ),
            ),
        )
        self.assertEqual(result.P[0].polynomial, u)
        self.assertEqual(result.P[1].polynomial, 1 - u)
        self.assertEqual(result.mass(), 2)

    def test_multiple_source_cells_sum_on_shared_target_piece(self):
        u = StatePolynomial.variable(1, 0)
        mud = PolynomialMUD(
            [[0, 1, 2]],
            [
                StatePolynomial.constant(1, 1),
                StatePolynomial.constant(1, 2),
            ],
        )

        result = mud.convolve_uniform(0, 0, 1)

        self.assertEqual(
            result.S,
            ((Fraction(0), Fraction(1), Fraction(2), Fraction(3)),),
        )
        self.assertEqual(result.P[0].polynomial, u)
        self.assertEqual(result.P[1].polynomial, 1 + u)
        self.assertEqual(result.P[2].polynomial, 2 - 2 * u)
        self.assertEqual(result.mass(), 3)

    def test_dirac_source_becomes_uniform_density(self):
        mud = PolynomialMUD(
            [[2, 2]],
            [StatePolynomial.constant(1, 6)],
        )

        result = mud.convolve_uniform(0, 0, 2)

        self.assertEqual(
            result.S,
            ((Fraction(2), Fraction(4)),),
        )
        self.assertEqual(
            result.P[0].polynomial,
            StatePolynomial.constant(1, 3),
        )
        self.assertEqual(result.mass(), mud.mass())

    def test_multidimensional_convolution_preserves_other_state_variables(self):
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)
        mud = PolynomialMUD(
            [[0, 1], [0, 2]],
            [[1 + u + 2 * v]],
        )

        result = mud.convolve_uniform(0, 0, 1)

        self.assertEqual(
            result.P[0, 0].polynomial,
            u + u**2 / 2 + 2 * u * v,
        )
        self.assertEqual(
            result.P[1, 0].polynomial,
            Fraction(3, 2) + 2 * v - u - u**2 / 2 - 2 * u * v,
        )
        self.assertEqual(result.S[1], mud.S[1])
        self.assertEqual(result.mass(), mud.mass())

    def test_empty_and_invalid_convolution_inputs(self):
        empty = PolynomialMUD(
            [[3]],
            np.empty((0,), dtype=object),
        )

        result = empty.convolve_uniform(0, -1, 2)

        self.assertTrue(result.is_empty)
        self.assertIsInstance(result, PolynomialMUD)
        self.assertEqual(result.S, ((Fraction(2),),))
        with self.assertRaisesRegex(ValueError, "dim out of range"):
            result.convolve_uniform(1, 0, 1)
        with self.assertRaisesRegex(ValueError, "low < high"):
            PolynomialMUD(
                [[0, 1]],
                [StatePolynomial.constant(1, 1)],
            ).convolve_uniform(0, 1, 1)
        nonempty = PolynomialMUD(
            [[0, 1]],
            [StatePolynomial.constant(1, 1)],
        )
        with self.assertRaisesRegex(ValueError, "convolution breakpoint"):
            nonempty.ops.convolve_uniform_dim(
                nonempty.P[0],
                nonempty._intervals_for_index((0,)),
                0,
                Fraction(0),
                Fraction(1),
                (Fraction(0), Fraction(2)),
            )


if __name__ == "__main__":
    unittest.main()
