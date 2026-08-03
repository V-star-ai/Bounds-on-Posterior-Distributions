import importlib.util
import unittest
from pathlib import Path

from Adapter import SCIPAdapter
from analyzer import ProgramStructure
from benchmarks.validate_add_uniform_polynomial import exact_exit_density
from semantics.program import (
    default_polynomial_variable_bounds,
    evaluate_polynomial_bgd,
)
from visualize_bgd import _eval_bgd_at


HAS_PYSCIPOPT = importlib.util.find_spec("pyscipopt") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_PYSCIPOPT, "PySCIPOpt is not installed")
class TestAddUniformPolynomialBenchmark(unittest.TestCase):
    def test_zero_degree_park_upper_bound(self):
        source = (
            ROOT / "benchmarks" / "PLDI22" / "add_uniform.txt"
        ).read_text(encoding="utf-8")
        semantics = ProgramStructure(
            source,
            polynomial_loop_degree=0,
            loop_unroll_iterations=2,
        ).build_polynomial_semantics()
        adapter = SCIPAdapter(
            variable_upper_bound=20,
            strict_epsilon=1e-7,
            factor_bound=20,
            time_limit=10,
            relative_gap=1e-4,
            feasibility_tolerance=1e-7,
        )
        bounds = default_polynomial_variable_bounds(
            semantics.constraints,
            coefficient_bound=20,
            mass_bound=20,
            strict_epsilon=1e-7,
        )

        solved = adapter.solve_polynomial(
            semantics.constraints,
            variable_bounds=bounds,
            objective=semantics.objective,
        )

        self.assertTrue(solved.has_solution)
        self.assertEqual(len(semantics.constraints.variables), 29)
        self.assertEqual(len(semantics.constraints.constraints), 90)
        self.assertGreater(solved.objective_value, 2.6)
        self.assertLess(solved.objective_value, 2.7)

        upper = evaluate_polynomial_bgd(
            semantics.distribution,
            solved.values,
        )
        for x in (0.25, 0.75, 1.25, 2.25, 4.25, 7.75):
            with self.subTest(x=x):
                self.assertGreaterEqual(
                    _eval_bgd_at(upper, [2, x], value="density") + 1e-6,
                    exact_exit_density(x),
                )


if __name__ == "__main__":
    unittest.main()
