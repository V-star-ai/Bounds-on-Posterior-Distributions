from fractions import Fraction
import unittest

from semantics import (
    ConstraintContext,
    DomainPolynomialConstraint,
    ParameterPolynomial,
    ParameterVariable,
    PolynomialIdentity,
    Relation,
    StatePolynomial,
    UnitBoxDomain,
)


class TestParameterPolynomial(unittest.TestCase):
    def test_exact_arithmetic_and_canonicalization(self):
        theta = ParameterVariable("theta")
        value = ParameterPolynomial.variable(theta)

        result = (value + 1) * (value - 1)

        self.assertEqual(result, value**2 - 1)
        self.assertEqual(result.degree(), 2)
        self.assertEqual(result.degree(theta), 2)
        self.assertEqual(result.evaluate({theta: 3}), 8)

    def test_float_coefficients_use_decimal_fraction(self):
        self.assertEqual(
            ParameterPolynomial.constant(0.1).constant_value,
            Fraction(1, 10),
        )

    def test_zero_terms_are_removed(self):
        theta = ParameterPolynomial.variable("theta")

        self.assertTrue((theta - theta).is_zero)
        self.assertEqual((theta - theta).terms, {})

    def test_division_only_accepts_constant(self):
        theta = ParameterPolynomial.variable("theta")

        self.assertEqual((2 * theta) / 4, theta / 2)
        with self.assertRaisesRegex(TypeError, "nonconstant"):
            _ = theta / (theta + 1)

    def test_differentiation_is_exact(self):
        theta = ParameterPolynomial.variable("theta")

        derivative = (3 * theta**3 + 2 * theta).differentiate("theta")

        self.assertEqual(derivative, 9 * theta**2 + 2)


class TestStatePolynomial(unittest.TestCase):
    def test_multivariate_arithmetic_and_evaluation(self):
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)
        polynomial = 1 + 2 * u + 3 * v + 4 * u * v

        self.assertEqual(
            polynomial.evaluate([Fraction(1, 2), Fraction(1, 3)]),
            Fraction(11, 3),
        )
        self.assertEqual(polynomial.degree(0), 1)
        self.assertEqual(polynomial.degree(1), 1)
        self.assertEqual(polynomial.degree(), 2)

    def test_parameter_polynomial_coefficients_are_preserved(self):
        theta_variable = ParameterVariable("theta")
        theta = ParameterPolynomial.variable(theta_variable)
        u = StatePolynomial.variable(1, 0)
        polynomial = theta * u + theta**2

        self.assertEqual(
            polynomial.evaluate([Fraction(1, 2)], {theta_variable: 2}),
            5,
        )
        self.assertEqual(polynomial.parameter_variables, (theta_variable,))

    def test_affine_substitution_is_exact(self):
        u = StatePolynomial.variable(1, 0)
        polynomial = 1 + 2 * u + u**2

        result = polynomial.affine_substitute(0, Fraction(1), Fraction(2))

        self.assertEqual(result, 4 + 8 * u + 4 * u**2)
        for point in (Fraction(0), Fraction(1, 3), Fraction(1)):
            self.assertEqual(
                result.evaluate([point]),
                polynomial.evaluate([1 + 2 * point]),
            )

    def test_integrate_unit_can_keep_or_remove_dimension(self):
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)
        polynomial = 1 + 2 * u + 3 * v + 4 * u * v

        kept = polynomial.integrate_unit(0)
        removed = polynomial.integrate_unit(0, remove=True)

        self.assertEqual(kept, 2 + 5 * v)
        expected_removed = 2 + 5 * StatePolynomial.variable(1, 0)
        self.assertEqual(removed, expected_removed)
        self.assertEqual(removed.ndim, 1)

    def test_antiderivative(self):
        u = StatePolynomial.variable(1, 0)
        antiderivative = (3 + 2 * u + 6 * u**2).antiderivative(0)

        self.assertEqual(
            antiderivative,
            3 * u + u**2 + 2 * u**3,
        )

    def test_independent_product_concatenates_state_variables(self):
        left_u = StatePolynomial.variable(1, 0)
        right_v = StatePolynomial.variable(1, 0)

        result = (1 + left_u).independent_product(2 + 3 * right_v)
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)

        self.assertEqual(result, 2 + 2 * u + 3 * v + 3 * u * v)

    def test_permute_dims_reorders_exponents(self):
        u = StatePolynomial.variable(2, 0)
        v = StatePolynomial.variable(2, 1)

        result = (2 * u + 3 * v**2).permute_dims([1, 0])
        new_u = StatePolynomial.variable(2, 0)
        new_v = StatePolynomial.variable(2, 1)

        self.assertEqual(result, 3 * new_u**2 + 2 * new_v)

    def test_state_dimension_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            _ = StatePolynomial.variable(1, 0) + StatePolynomial.variable(2, 0)


class TestPolynomialConstraints(unittest.TestCase):
    def test_domain_constraint_keeps_full_polynomial(self):
        theta_variable = ParameterVariable("theta")
        theta = ParameterPolynomial.variable(theta_variable)
        u = StatePolynomial.variable(1, 0)
        polynomial = theta + u - u**2

        constraint = DomainPolynomialConstraint(
            polynomial,
            Relation.GE,
            UnitBoxDomain(1),
        )

        self.assertIs(constraint.polynomial, polynomial)
        self.assertTrue(
            constraint.evaluate_at(
                [Fraction(1, 2)],
                {theta_variable: Fraction(1, 4)},
            )
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            constraint.evaluate_at([Fraction(3, 2)], {theta_variable: 0})

    def test_inactive_dirac_dimension_cannot_appear_in_polynomial(self):
        u = StatePolynomial.variable(2, 0)
        dirac_axis = StatePolynomial.variable(2, 1)
        domain = UnitBoxDomain(2, active_dims=[0])

        DomainPolynomialConstraint(u + 1, Relation.GE, domain)
        with self.assertRaisesRegex(ValueError, "inactive domain dimension 1"):
            DomainPolynomialConstraint(u + dirac_axis, Relation.GE, domain)

    def test_polynomial_identity_expands_to_coefficient_equalities(self):
        theta = ParameterPolynomial.variable("theta")
        u = StatePolynomial.variable(1, 0)
        identity = PolynomialIdentity(theta * u + 1, 2 * u + 1)

        coefficient_constraints = identity.coefficient_constraints()

        self.assertEqual(len(coefficient_constraints), 1)
        self.assertTrue(
            coefficient_constraints[0].evaluate({"theta": 2})
        )
        self.assertFalse(
            coefficient_constraints[0].evaluate({"theta": 3})
        )

    def test_context_builds_exact_positive_quotient_constraints(self):
        context = ConstraintContext()
        theta_variable = context.declare("theta")
        theta = ParameterPolynomial.variable(theta_variable)

        quotient = context.exact_positive_quotient(
            1 + theta,
            1 - theta,
            prefix="tail",
        )
        problem = context.build()

        self.assertEqual(
            tuple(variable.name for variable in problem.variables),
            ("theta", "tail_0"),
        )
        values = {"theta": Fraction(1, 2), "tail_0": 3}
        self.assertTrue(all(constraint.evaluate(values) for constraint in problem.constraints))
        self.assertEqual(quotient.evaluate(values), 3)

    def test_context_rejects_undeclared_parameters(self):
        context = ConstraintContext()
        context.constrain_parameter(
            ParameterPolynomial.variable("missing"),
            Relation.GE,
        )

        with self.assertRaisesRegex(ValueError, "undeclared parameters: missing"):
            context.build()


if __name__ == "__main__":
    unittest.main()
