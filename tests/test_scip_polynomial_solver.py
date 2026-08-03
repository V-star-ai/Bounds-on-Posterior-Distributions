import importlib.util
import unittest
from fractions import Fraction

import numpy as np

from distributions import BGD, PolynomialMUD
from semantics import (
    ConstraintContext,
    ConstraintProblem,
    ParameterConstraint,
    ParameterPolynomial,
    ParameterVariable,
    Relation,
    StatePolynomial,
)
from solvers import SCIPPolynomialSolver


HAS_PYSCIPOPT = importlib.util.find_spec("pyscipopt") is not None


@unittest.skipUnless(HAS_PYSCIPOPT, "PySCIPOpt is not installed")
class TestSCIPPolynomialSolver(unittest.TestCase):
    def solver(self, bounds, **options):
        return SCIPPolynomialSolver(
            bounds,
            factor_bound=10,
            strict_epsilon=1e-7,
            feasibility_tolerance=1e-8,
            relative_gap=1e-6,
            time_limit=10,
            **options,
        )

    def test_nonlinear_parameter_objective(self):
        context = ConstraintContext()
        x_variable = context.declare("x")
        x = ParameterPolynomial.variable(x_variable)
        context.constrain_parameter(x, Relation.GE)

        result = self.solver({x_variable: (-2, 2)}).solve(
            context.build(),
            objective=(x - 1) ** 2,
        )

        self.assertTrue(result.is_optimal)
        self.assertAlmostEqual(result.values["x"], 1, places=5)
        self.assertAlmostEqual(result.objective_value, 0, places=6)
        self.assertEqual(result.factor_variable_count, 0)

    def test_univariate_domain_nonnegativity(self):
        context = ConstraintContext()
        t_variable = context.declare("t")
        t = ParameterPolynomial.variable(t_variable)
        u = StatePolynomial.variable(1, 0)
        context.constrain_domain(
            StatePolynomial.constant(1, t) - u * (1 - u),
            Relation.GE,
        )

        result = self.solver({t_variable: (0, 1)}).solve(
            context.build(),
            objective=t,
        )

        self.assertTrue(result.is_optimal)
        self.assertAlmostEqual(result.values["t"], 0.25, places=5)
        self.assertGreater(result.factor_variable_count, 0)
        self.assertLess(result.max_certificate_residual, 1e-5)

    def test_complete_factor_solution_can_be_used_as_warm_start(self):
        context = ConstraintContext()
        t_variable = context.declare("t")
        t = ParameterPolynomial.variable(t_variable)
        u = StatePolynomial.variable(1, 0)
        context.constrain_domain(
            StatePolynomial.constant(1, t) - u * (1 - u),
            Relation.GE,
        )
        solver = self.solver({t_variable: (0, 1)})

        first = solver.solve(context.build(), objective=t)
        warm_start = {**first.values, **first.factor_values}
        second = solver.solve(
            context.build(),
            objective=t,
            initial_values=warm_start,
        )

        self.assertTrue(second.has_solution)
        self.assertEqual(
            set(second.factor_values),
            set(first.factor_values),
        )
        self.assertAlmostEqual(second.objective_value, 0.25, places=5)
        self.assertLess(second.max_certificate_residual, 1e-5)

    def test_joint_decay_and_polynomial_coefficient_optimization(self):
        context = ConstraintContext()
        alpha_variable = context.declare("alpha")
        coefficient_variable = context.declare("coefficient")
        t_variable = context.declare("t")
        alpha = ParameterPolynomial.variable(alpha_variable)
        coefficient = ParameterPolynomial.variable(coefficient_variable)
        t = ParameterPolynomial.variable(t_variable)
        u = StatePolynomial.variable(1, 0)

        context.constrain_parameter(alpha + coefficient, Relation.EQ, 1)
        context.constrain_parameter(
            alpha * coefficient / 4 - t,
            Relation.GE,
        )
        context.constrain_domain(
            StatePolynomial.constant(1, t)
            - alpha * coefficient * u * (1 - u),
            Relation.GE,
        )

        result = self.solver(
            {
                alpha_variable: (0.1, 0.9),
                coefficient_variable: (0.1, 0.9),
                t_variable: (0, 0.25),
            }
        ).solve(
            context.build(),
            objective=t,
            sense="maximize",
        )

        self.assertTrue(result.is_optimal)
        self.assertAlmostEqual(result.values["alpha"], 0.5, places=4)
        self.assertAlmostEqual(
            result.values["coefficient"],
            0.5,
            places=4,
        )
        self.assertAlmostEqual(result.values["t"], 0.0625, places=5)

    def test_solves_constraints_emitted_by_polynomial_bgd(self):
        context = ConstraintContext()
        t_variable = context.declare("t")
        t = ParameterPolynomial.variable(t_variable)
        u = StatePolynomial.variable(1, 0)
        zero = StatePolynomial.zero(1)

        lower_blocks = np.empty((3,), dtype=object)
        upper_blocks = np.empty((3,), dtype=object)
        for blocks in (lower_blocks, upper_blocks):
            blocks[0] = PolynomialMUD([[0, 1]], [zero])
            blocks[2] = PolynomialMUD([[0, 1]], [zero])
        lower_blocks[1] = PolynomialMUD([[0, 1]], [u * (1 - u)])
        upper_blocks[1] = PolynomialMUD(
            [[0, 1]],
            [StatePolynomial.constant(1, t)],
        )
        lower = BGD(
            lower_blocks,
            [Fraction(1, 2)],
            [Fraction(1, 2)],
        )
        upper = BGD(
            upper_blocks,
            [Fraction(1, 2)],
            [Fraction(1, 2)],
        )
        for constraint in lower.le_constraints(upper):
            context.add(constraint)

        result = self.solver({t_variable: (0, 1)}).solve(
            context.build(),
            objective=t,
        )

        self.assertTrue(result.is_optimal)
        self.assertAlmostEqual(result.values["t"], 0.25, places=5)

    def test_false_constant_constraint_is_infeasible(self):
        context = ConstraintContext()
        context.constrain_parameter(-1, Relation.GE)

        result = self.solver({}).solve(context.build())

        self.assertEqual(result.status, "infeasible")
        self.assertFalse(result.has_solution)

    def test_true_constant_problem_has_a_solution(self):
        context = ConstraintContext()
        context.constrain_parameter(1, Relation.GE)

        result = self.solver({}).solve(context.build())

        self.assertTrue(result.is_optimal)
        self.assertTrue(result.has_solution)
        self.assertEqual(result.values, {})

    def test_rejects_undeclared_constraint_parameters(self):
        x_variable = ParameterVariable("x")
        x = ParameterPolynomial.variable(x_variable)
        problem = ConstraintProblem(
            (),
            (ParameterConstraint(x, Relation.GE),),
        )

        with self.assertRaisesRegex(ValueError, "undeclared"):
            self.solver({}).solve(problem)


if __name__ == "__main__":
    unittest.main()
