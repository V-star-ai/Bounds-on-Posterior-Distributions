import unittest
from fractions import Fraction

from analyzer import ProgramStructure
from distributions import PolynomialMUD
from semantics import ConstraintProblem, StatePolynomial


def build(source: str):
    return ProgramStructure(source).build_polynomial_semantics()


def point_mass(distribution, dim: int, point):
    return (
        distribution.restrict(dim, ">=", point)
        .restrict(dim, "<=", point)
        .mass()
    )


class TestPolynomialProgramPriors(unittest.TestCase):
    def test_uniform_and_mapping_priors_form_exact_joint_distribution(self):
        result = build(
            """
            prior:
                x ~ Uniform(0, 2);
                y ~ {0: 1/4, 2: 3/4};
            program:
                skip
            """
        )

        self.assertEqual(result.variable_order, ("x", "y"))
        self.assertIs(result.distribution.cell_family, PolynomialMUD)
        self.assertEqual(result.distribution.mass(), 1)
        self.assertEqual(
            result.distribution.C.P[0, 0].polynomial,
            StatePolynomial.constant(2, Fraction(1, 8)),
        )
        self.assertEqual(
            result.distribution.C.P[0, 2].polynomial,
            StatePolynomial.constant(2, Fraction(3, 8)),
        )
        self.assertIsInstance(result.constraints, ConstraintProblem)
        self.assertEqual(result.constraints.variables, ())
        self.assertEqual(result.constraints.constraints, ())

    def test_non_polynomial_prior_is_rejected_in_exact_mode(self):
        with self.assertRaisesRegex(
            ValueError,
            "no exact finite piecewise-polynomial representation",
        ):
            build(
                """
                prior:
                    x ~ Normal(0, 1);
                program:
                    skip
                """
            )


class TestLoopFreePolynomialProgramSemantics(unittest.TestCase):
    def test_uniform_addition_uses_exact_polynomial_convolution(self):
        result = build(
            """
            prior:
                x ~ Uniform(0, 1);
            program:
                x := x + Uniform(0, 1)
            """
        )
        u = StatePolynomial.variable(1, 0)

        self.assertEqual(
            result.distribution.C.S,
            ((Fraction(0), Fraction(1), Fraction(2)),),
        )
        self.assertEqual(result.distribution.C.P[0].polynomial, u)
        self.assertEqual(result.distribution.C.P[1].polynomial, 1 - u)
        self.assertEqual(result.distribution.mass(), 1)

    def test_uniform_subtraction_reflects_noise_interval(self):
        result = build(
            """
            prior:
                x := 0;
            program:
                x := x - Uniform(1, 3)
            """
        )

        self.assertEqual(result.distribution.center_lefts, (Fraction(-3),))
        self.assertEqual(result.distribution.center_rights, (Fraction(-1),))
        self.assertEqual(
            result.distribution.C.P[0].polynomial,
            StatePolynomial.constant(1, Fraction(1, 2)),
        )
        self.assertEqual(result.distribution.mass(), 1)

    def test_shift_and_distribution_replacement_preserve_joint_mass(self):
        result = build(
            """
            prior:
                x ~ Uniform(0, 1);
                y := 0;
            program:
                x := x + 2;
                y := Uniform(3, 5)
            """
        )

        self.assertEqual(result.distribution.center_lefts, (Fraction(2), Fraction(3)))
        self.assertEqual(result.distribution.center_rights, (Fraction(3), Fraction(5)))
        self.assertEqual(result.distribution.mass(), 1)

    def test_state_condition_preserves_branch_correlation(self):
        result = build(
            """
            prior:
                x ~ Uniform(0, 1);
                y := 0;
            program:
                if (x <= 1/2) {
                    y := y + 1
                } else {
                    y := y + 2
                }
            """
        )

        self.assertEqual(result.distribution.mass(), 1)
        self.assertEqual(point_mass(result.distribution, 1, 1), Fraction(1, 2))
        self.assertEqual(point_mass(result.distribution, 1, 2), Fraction(1, 2))

    def test_probabilistic_choice_and_numeric_if_mix_exactly(self):
        result = build(
            """
            prior:
                x := 0;
            program:
                { x := x + 1 } [1/4] { x := x + 2 };
                if (1/2) {
                    x := x + 10
                } else {
                    x := x + 20
                }
            """
        )

        expected = {
            11: Fraction(1, 8),
            12: Fraction(3, 8),
            21: Fraction(1, 8),
            22: Fraction(3, 8),
        }
        self.assertEqual(result.distribution.mass(), 1)
        for point, mass in expected.items():
            self.assertEqual(
                point_mass(result.distribution, 0, point),
                mass,
            )

    def test_observe_returns_unnormalized_subdistribution(self):
        result = build(
            """
            prior:
                x ~ Uniform(0, 2);
            program:
                observe((x >= 1/2) & (x < 3/2))
            """
        )

        self.assertEqual(result.distribution.center_lefts, (Fraction(1, 2),))
        self.assertEqual(result.distribution.center_rights, (Fraction(3, 2),))
        self.assertEqual(result.distribution.mass(), Fraction(1, 2))

    def test_while_builds_park_constraints_and_mass_objective(self):
        result = build(
            """
            prior:
                x := 0;
            program:
                while (x < 1) {
                    x := x + 1
                }
            """
        )

        self.assertGreater(len(result.constraints.variables), 0)
        self.assertGreater(len(result.constraints.constraints), 0)
        self.assertGreaterEqual(result.objective.degree(), 1)
        self.assertEqual(result.distribution.center_lefts, (Fraction(1),))
        self.assertEqual(result.distribution.center_rights, (Fraction(1),))
        self.assertEqual(result.loop_template_degrees, ((0,),))

    def test_while_infers_per_variable_degrees_and_applies_increment(self):
        source = """
            prior:
                y := 0;
                x := 0;
            program:
                while ((y < 2) || (y > 2)) {
                    x := x + Uniform(0, 1);
                    y := y + 1
                }
        """

        inferred = ProgramStructure(
            source,
            loop_unroll_iterations=2,
        ).build_polynomial_semantics()
        raised = ProgramStructure(
            source,
            loop_unroll_iterations=2,
            polynomial_loop_degree_increment=1,
        ).build_polynomial_semantics()

        self.assertEqual(inferred.variable_order, ("y", "x"))
        self.assertEqual(inferred.loop_template_degrees, ((0, 1),))
        self.assertEqual(raised.loop_template_degrees, ((1, 2),))

    def test_while_shift_preserves_periodic_dirac_tail_shape(self):
        cases = (
            ("x + 1", 2, (Fraction(0), Fraction(1), Fraction(1))),
            ("x - 1", 0, (Fraction(0), Fraction(0), Fraction(1))),
        )
        for assignment, edge, expected in cases:
            with self.subTest(assignment=assignment):
                result = ProgramStructure(
                    f"""
                    prior:
                        x := 0;
                    program:
                        while (1/2) {{
                            x := {assignment}
                        }}
                    """,
                    loop_unroll_iterations=2,
                ).build_polynomial_semantics()

                self.assertEqual(
                    result.distribution.E[edge].S[0],
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
