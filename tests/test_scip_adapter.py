import importlib.util
import unittest
from fractions import Fraction

import numpy as np

from Adapter import SCIPAdapter
from Adapter.expr import Expr, Var
from analyzer import ProgramStructure
from distributions import BGD, MUD
from main import build_solver, get_solver_config
from semantics import ConstraintContext, ParameterPolynomial, Relation
from semantics.program import default_polynomial_variable_bounds


HAS_PYSCIPOPT = importlib.util.find_spec("pyscipopt") is not None


@unittest.skipUnless(HAS_PYSCIPOPT, "PySCIPOpt is not installed")
class TestSCIPAdapter(unittest.TestCase):
    def adapter(self, **options):
        config = {
            "variable_upper_bound": 10,
            "strict_epsilon": 1e-7,
            "time_limit": 10,
            "relative_gap": 1e-6,
            "feasibility_tolerance": 1e-8,
            "require_optimal": True,
        }
        config.update(options)
        return SCIPAdapter(**config)

    def test_legacy_expr_nonlinear_objective(self):
        adapter = self.adapter()
        x = Var("x")
        y = Var("y")

        result = adapter.solve(
            {"x": "x", "y": "y"},
            [x + y <= 1, x + y >= 1],
            objective=(x - y) ** 2,
        )

        self.assertAlmostEqual(result["x"], 0.5, places=5)
        self.assertAlmostEqual(result["y"], 0.5, places=5)
        self.assertEqual(adapter.last_stats["status"], "optimal")

    def test_max_uses_exact_binary_formulation(self):
        adapter = self.adapter()
        x = Var("x")
        y = Var("y")

        result = adapter.solve(
            {"x": "x", "y": "y"},
            [x <= 0.2, x >= 0.2, y <= 0.8, y >= 0.8],
            objective=Expr.max(x, y),
        )

        self.assertAlmostEqual(result["x"], 0.2, places=6)
        self.assertAlmostEqual(result["y"], 0.8, places=6)
        self.assertAlmostEqual(
            adapter.last_stats["objective_value"],
            0.8,
            places=6,
        )
        self.assertEqual(adapter.last_stats["binary_variables"], 1)

    def test_bgd_adapter_lifecycle(self):
        blocks = np.empty((3,), dtype=object)
        blocks[0] = MUD([[0, 1]], [0])
        blocks[1] = MUD([[0, 1]], [Fraction(1, 2)])
        blocks[2] = MUD([[0, 1]], [0])
        constant = BGD(
            blocks,
            [Fraction(1, 2)],
            [Fraction(1, 2)],
        )
        adapter = self.adapter(
            variable_upper_bound=2,
            require_optimal=False,
        )

        symbolic, environment = adapter.build_bgd_leq(
            constant,
            template=constant,
            name_prefix="probe",
        )
        solved = adapter.solve_bgd_expr(symbolic, environment)

        self.assertAlmostEqual(solved.C.P[0], 0.5, places=6)
        self.assertAlmostEqual(solved.mass(), 0.5, places=5)

    def test_exact_polynomial_ir_delegates_to_scip_solver(self):
        context = ConstraintContext()
        x_variable = context.declare("x")
        x = ParameterPolynomial.variable(x_variable)
        context.constrain_parameter(x, Relation.GE)
        adapter = self.adapter()

        result = adapter.solve_polynomial(
            context.build(),
            variable_bounds={x_variable: (-2, 2)},
            objective=(x - 1) ** 2,
        )

        self.assertTrue(result.is_optimal)
        self.assertAlmostEqual(result.values["x"], 1, places=5)
        self.assertIs(adapter.last_polynomial_result, result)

    def test_polynomial_loop_preserves_shifted_dirac_tail(self):
        semantics = ProgramStructure(
            """
            prior:
                x := 0;
            program:
                while (1/2) {
                    x := x + 1
                }
            """,
            loop_unroll_iterations=2,
        ).build_polynomial_semantics()
        adapter = self.adapter(
            variable_upper_bound=20,
            require_optimal=False,
        )
        bounds = default_polynomial_variable_bounds(
            semantics.constraints,
            coefficient_bound=20,
            mass_bound=20,
            strict_epsilon=1e-7,
        )

        result = adapter.solve_polynomial(
            semantics.constraints,
            variable_bounds=bounds,
            objective=semantics.objective,
        )

        self.assertTrue(result.has_solution)
        self.assertAlmostEqual(result.objective_value, 1, places=5)

    def test_program_structure_park_loop_uses_scip_adapter(self):
        program = ProgramStructure(
            """
            prior:
                x ~ {0: 1};
            program:
                while(1/2) {
                    x := x - 1
                }
            """,
            loop_unroll_iterations=1,
            template_dirac_iterations=1,
        )
        adapter = self.adapter(
            variable_upper_bound=10,
            relative_gap=1e-4,
            require_optimal=False,
        )

        result = program.solve_bgd(adapter, method="Park")

        self.assertAlmostEqual(result.mass(), 1, places=5)
        self.assertAlmostEqual(result.alpha[0], 0.5, places=5)
        self.assertIn(adapter.last_stats["status"], ("optimal", "gaplimit"))

    def test_main_solver_configuration_registers_scip(self):
        config = get_solver_config(
            {
                "solver": {
                    "name": "scip",
                    "scip": {
                        "variable_upper_bound": 12,
                        "time_limit": 3,
                        "use_symmetry": False,
                        "require_optimal": True,
                    },
                }
            }
        )

        solver = build_solver(config)

        self.assertIsInstance(solver, SCIPAdapter)
        self.assertEqual(solver.variable_upper_bound, 12)
        self.assertEqual(solver.time_limit, 3)
        self.assertFalse(solver.use_symmetry)
        self.assertTrue(solver.require_optimal)


if __name__ == "__main__":
    unittest.main()
