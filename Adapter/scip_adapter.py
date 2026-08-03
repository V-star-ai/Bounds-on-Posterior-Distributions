from __future__ import annotations

from fractions import Fraction
from math import isfinite
from numbers import Real
from typing import Mapping

import numpy as np

from Adapter.adapter import Adapter
from Adapter.expr import (
    Add,
    CompareOp,
    Const,
    Constraint,
    Div,
    Expr,
    FractionConst,
    Max,
    Mul,
    Pow,
    Sub,
    Var,
    ensure_expr,
)
from semantics import ConstraintProblem
from solvers import SCIPPolynomialResult, SCIPPolynomialSolver


class SCIPAdapter(Adapter):
    """SCIP integration for both legacy Expr and exact polynomial IR."""

    def __init__(
        self,
        *,
        variable_lower_bound=0,
        variable_upper_bound=1000,
        variable_bounds: Mapping[str, tuple[Real, Real]] | None = None,
        strict_epsilon=1e-7,
        factor_bound=10,
        certificate_degree: int | None = None,
        time_limit: Real | None = None,
        relative_gap: Real | None = None,
        feasibility_tolerance: Real | None = None,
        feasibility_emphasis=False,
        use_symmetry=True,
        display=False,
        require_optimal=False,
    ):
        self.variable_lower_bound = _finite_float(
            variable_lower_bound,
            "variable_lower_bound",
        )
        self.variable_upper_bound = _finite_float(
            variable_upper_bound,
            "variable_upper_bound",
        )
        if self.variable_lower_bound >= self.variable_upper_bound:
            raise ValueError(
                "variable_lower_bound must be smaller than "
                "variable_upper_bound"
            )
        self.variable_bounds = _normalize_bound_overrides(variable_bounds)
        self.strict_epsilon = _positive_float(
            strict_epsilon,
            "strict_epsilon",
        )
        self.factor_bound = _positive_float(factor_bound, "factor_bound")
        if certificate_degree is not None:
            if (
                not isinstance(certificate_degree, int)
                or isinstance(certificate_degree, bool)
                or certificate_degree < 0
                or certificate_degree % 2
            ):
                raise ValueError(
                    "certificate_degree must be a nonnegative even integer"
                )
        self.certificate_degree = certificate_degree
        self.time_limit = _optional_positive_float(time_limit, "time_limit")
        self.relative_gap = _optional_nonnegative_float(
            relative_gap,
            "relative_gap",
        )
        self.feasibility_tolerance = _optional_positive_float(
            feasibility_tolerance,
            "feasibility_tolerance",
        )
        self.feasibility_emphasis = bool(feasibility_emphasis)
        self.use_symmetry = bool(use_symmetry)
        self.display = bool(display)
        self.require_optimal = bool(require_optimal)
        self.last_stats = {}
        self.last_polynomial_result: SCIPPolynomialResult | None = None

    def build_var(self, name):
        return name

    def var_max(self, a, b):
        return max(a, b)

    def solve(self, vars, constraints, objective=None):
        try:
            from pyscipopt import Model
        except ImportError as exc:
            raise ImportError(
                "PySCIPOpt is required; install it with "
                "`python -m pip install -r requirements-scip.txt`"
            ) from exc

        model = Model("legacy_bgd")
        if not self.display:
            model.hideOutput()
        if self.time_limit is not None:
            model.setRealParam("limits/time", self.time_limit)
        if self.relative_gap is not None:
            model.setRealParam("limits/gap", self.relative_gap)
        if self.feasibility_tolerance is not None:
            model.setRealParam(
                "numerics/feastol",
                self.feasibility_tolerance,
            )

        scip_variables = {}
        bounds = {}
        for name in vars:
            lower, upper = self._bounds_for_name(name)
            bounds[name] = (lower, upper)
            scip_variables[name] = model.addVar(
                name=name,
                lb=lower,
                ub=upper,
                vtype="C",
            )

        compiler = _LegacySCIPCompiler(
            model,
            scip_variables,
            bounds,
            strict_epsilon=self.strict_epsilon,
        )
        raw_constraints = []
        for index, constraint in enumerate(constraints):
            if isinstance(constraint, (bool, np.bool_)):
                if not bool(constraint):
                    raise RuntimeError(
                        "Constraints are infeasible (constant False)"
                    )
                continue
            if not isinstance(constraint, Constraint):
                raise TypeError(constraint)
            if constraint.op is CompareOp.NE:
                raise ValueError("SCIPAdapter does not support '!=' constraints")
            compiler.add_constraint(constraint, f"constraint_{index}")
            raw_constraints.append(constraint)

        objective_expr = ensure_expr(0 if objective is None else objective)
        objective_compiled = compiler.expression(objective_expr)
        objective_lower, objective_upper = compiler.expression_bounds(
            objective_expr
        )
        objective_variable = model.addVar(
            name="__scip_objective",
            lb=objective_lower,
            ub=objective_upper,
            vtype="C",
        )
        model.addCons(
            objective_variable == objective_compiled,
            name="__objective_definition",
        )
        model.setObjective(objective_variable, "minimize")
        model.optimize()

        status = str(model.getStatus())
        solution_count = int(model.getNSols())
        if solution_count == 0:
            raise RuntimeError(
                f"SCIP failed to find a feasible solution (status={status})"
            )
        if self.require_optimal and status != "optimal":
            raise RuntimeError(
                f"SCIP did not prove global optimality (status={status})"
            )

        solution = model.getBestSol()
        result = {
            name: float(model.getSolVal(solution, variable))
            for name, variable in scip_variables.items()
        }
        max_violation = max(
            (
                self._constraint_violation(constraint, result)
                for constraint in raw_constraints
            ),
            default=0.0,
        )
        validation_tolerance = max(
            1e-6,
            10
            * (
                self.feasibility_tolerance
                if self.feasibility_tolerance is not None
                else 1e-6
            ),
        )
        if max_violation > validation_tolerance:
            raise RuntimeError(
                "SCIP solution failed independent constraint validation: "
                f"max_violation={max_violation}"
            )

        self.last_stats = {
            "status": status,
            "solution_count": solution_count,
            "objective_value": float(
                model.getSolVal(solution, objective_variable)
            ),
            "dual_bound": _finite_or_none(model.getDualbound()),
            "relative_gap": _finite_or_none(model.getGap()),
            "solve_time": float(model.getSolvingTime()),
            "nodes": int(model.getNNodes()),
            "variables": len(scip_variables),
            "auxiliary_variables": compiler.auxiliary_variable_count,
            "binary_variables": compiler.binary_variable_count,
            "constraints": compiler.constraint_count + 1,
            "max_violation": max_violation,
        }
        print(
            "[SCIPAdapter] "
            f"status={status}, variables={len(scip_variables)}, "
            f"auxiliary={compiler.auxiliary_variable_count}, "
            f"binary={compiler.binary_variable_count}, "
            f"constraints={compiler.constraint_count + 1}, "
            f"objective={self.last_stats['objective_value']}, "
            f"gap={self.last_stats['relative_gap']}, "
            f"seconds={self.last_stats['solve_time']:.4f}",
            flush=True,
        )
        return result

    def solve_bgd_expr(self, bgd_expr, envs):
        solved_vars = self.solve(
            envs.vars,
            envs.constraints_list,
            objective=bgd_expr.mass(),
        )
        return self._eval_bgd_expr_with_vars(bgd_expr, solved_vars)

    def solve_polynomial(
        self,
        problem: ConstraintProblem,
        *,
        variable_bounds: Mapping | None = None,
        objective=0,
        sense="minimize",
        initial_values: Mapping[str, Real] | None = None,
    ) -> SCIPPolynomialResult:
        if variable_bounds is None:
            declared_names = {variable.name for variable in problem.variables}
            variable_bounds = {
                name: bounds
                for name, bounds in self.variable_bounds.items()
                if name in declared_names
            }
        solver = SCIPPolynomialSolver(
            variable_bounds,
            factor_bound=self.factor_bound,
            strict_epsilon=self.strict_epsilon,
            certificate_degree=self.certificate_degree,
            time_limit=self.time_limit,
            relative_gap=self.relative_gap,
            feasibility_tolerance=self.feasibility_tolerance,
            feasibility_emphasis=self.feasibility_emphasis,
            use_symmetry=self.use_symmetry,
            display=self.display,
        )
        result = solver.solve(
            problem,
            objective=objective,
            sense=sense,
            initial_values=initial_values,
        )
        if self.require_optimal and not result.is_optimal:
            raise RuntimeError(
                "SCIP did not prove global optimality "
                f"(status={result.status})"
            )
        self.last_polynomial_result = result
        return result

    def _bounds_for_name(self, name: str) -> tuple[float, float]:
        if name in self.variable_bounds:
            return self.variable_bounds[name]
        if self._is_unit_interval_variable(name):
            return 0.0, 1.0 - self.strict_epsilon
        return self.variable_lower_bound, self.variable_upper_bound

    @staticmethod
    def _is_unit_interval_variable(name: str) -> bool:
        return (
            "_alpha_" in name
            or "_beta_" in name
            or name.startswith("c_w")
        )

    def _constraint_violation(self, constraint, values) -> float:
        left = float(self.eval_expr(constraint.left, values))
        right = float(self.eval_expr(constraint.right, values))
        if constraint.op is CompareOp.LE:
            return max(left - right, 0.0)
        if constraint.op is CompareOp.LT:
            return max(left - right + self.strict_epsilon, 0.0)
        if constraint.op is CompareOp.EQ:
            return abs(left - right)
        if constraint.op is CompareOp.GE:
            return max(right - left, 0.0)
        if constraint.op is CompareOp.GT:
            return max(right - left + self.strict_epsilon, 0.0)
        raise TypeError(constraint.op)


ScipAdapter = SCIPAdapter


class _LegacySCIPCompiler:
    def __init__(
        self,
        model,
        variables,
        bounds,
        *,
        strict_epsilon: float,
    ):
        self.model = model
        self.variables = variables
        self.bounds = bounds
        self.strict_epsilon = strict_epsilon
        self.auxiliary_variable_count = 0
        self.binary_variable_count = 0
        self.constraint_count = 0
        self._expression_cache = {}
        self._bounds_cache = {}
        self._max_counter = 0

    def expression(self, expr):
        expr = ensure_expr(expr)
        cached = self._expression_cache.get(id(expr))
        if cached is not None and cached[0] is expr:
            return cached[1]

        if isinstance(expr, Var):
            result = self.variables[expr.name]
        elif isinstance(expr, (Const, FractionConst)):
            result = float(expr.value)
        elif isinstance(expr, Add):
            result = self.expression(expr.left) + self.expression(expr.right)
        elif isinstance(expr, Sub):
            result = self.expression(expr.left) - self.expression(expr.right)
        elif isinstance(expr, Mul):
            result = self.expression(expr.left) * self.expression(expr.right)
        elif isinstance(expr, Div):
            denominator_bounds = self.expression_bounds(expr.right)
            if denominator_bounds[0] <= 0 <= denominator_bounds[1]:
                raise ValueError(
                    "SCIPAdapter division denominator may contain zero; "
                    "provide tighter variable_bounds"
                )
            result = self.expression(expr.left) / self.expression(expr.right)
        elif isinstance(expr, Pow):
            exponent = _integer_exponent(expr.right)
            if exponent < 0:
                base_bounds = self.expression_bounds(expr.left)
                if base_bounds[0] <= 0 <= base_bounds[1]:
                    raise ValueError(
                        "SCIPAdapter negative power base may contain zero"
                    )
            result = self.expression(expr.left) ** exponent
        elif isinstance(expr, Max):
            result = self._max_expression(expr)
        else:
            raise TypeError(expr)

        self._expression_cache[id(expr)] = (expr, result)
        return result

    def expression_bounds(self, expr) -> tuple[float, float]:
        expr = ensure_expr(expr)
        cached = self._bounds_cache.get(id(expr))
        if cached is not None and cached[0] is expr:
            return cached[1]

        if isinstance(expr, Var):
            result = self.bounds[expr.name]
        elif isinstance(expr, (Const, FractionConst)):
            value = float(expr.value)
            result = (value, value)
        elif isinstance(expr, Add):
            left = self.expression_bounds(expr.left)
            right = self.expression_bounds(expr.right)
            result = (left[0] + right[0], left[1] + right[1])
        elif isinstance(expr, Sub):
            left = self.expression_bounds(expr.left)
            right = self.expression_bounds(expr.right)
            result = (left[0] - right[1], left[1] - right[0])
        elif isinstance(expr, Mul):
            result = _multiply_intervals(
                self.expression_bounds(expr.left),
                self.expression_bounds(expr.right),
            )
        elif isinstance(expr, Div):
            denominator = self.expression_bounds(expr.right)
            if denominator[0] <= 0 <= denominator[1]:
                raise ValueError(
                    "SCIPAdapter division denominator may contain zero; "
                    "provide tighter variable_bounds"
                )
            reciprocal = (1 / denominator[1], 1 / denominator[0])
            result = _multiply_intervals(
                self.expression_bounds(expr.left),
                reciprocal,
            )
        elif isinstance(expr, Pow):
            result = _power_interval(
                self.expression_bounds(expr.left),
                _integer_exponent(expr.right),
            )
        elif isinstance(expr, Max):
            left = self.expression_bounds(expr.left)
            right = self.expression_bounds(expr.right)
            result = (max(left[0], right[0]), max(left[1], right[1]))
        else:
            raise TypeError(expr)

        if not all(isfinite(value) for value in result):
            raise ValueError("SCIPAdapter expression bounds are not finite")
        self._bounds_cache[id(expr)] = (expr, result)
        return result

    def add_constraint(self, constraint: Constraint, name: str) -> None:
        left = self.expression(constraint.left)
        right = self.expression(constraint.right)
        if constraint.op is CompareOp.LE:
            compiled = left <= right
        elif constraint.op is CompareOp.LT:
            compiled = left <= right - self.strict_epsilon
        elif constraint.op is CompareOp.EQ:
            compiled = left == right
        elif constraint.op is CompareOp.GE:
            compiled = left >= right
        elif constraint.op is CompareOp.GT:
            compiled = left >= right + self.strict_epsilon
        else:
            raise TypeError(constraint.op)
        if isinstance(compiled, (bool, np.bool_)):
            if not bool(compiled):
                raise RuntimeError(
                    "Constraints are infeasible (constant False)"
                )
            return
        self.model.addCons(compiled, name=name)
        self.constraint_count += 1

    def _max_expression(self, expr: Max):
        left_bounds = self.expression_bounds(expr.left)
        right_bounds = self.expression_bounds(expr.right)
        if left_bounds[0] >= right_bounds[1]:
            return self.expression(expr.left)
        if right_bounds[0] >= left_bounds[1]:
            return self.expression(expr.right)

        left = self.expression(expr.left)
        right = self.expression(expr.right)
        lower, upper = self.expression_bounds(expr)
        index = self._max_counter
        self._max_counter += 1
        result = self.model.addVar(
            name=f"__scip_max_{index}",
            lb=lower,
            ub=upper,
            vtype="C",
        )
        branch = self.model.addVar(
            name=f"__scip_max_branch_{index}",
            vtype="B",
        )
        relax_left = max(0.0, right_bounds[1] - left_bounds[0])
        relax_right = max(0.0, left_bounds[1] - right_bounds[0])
        self.model.addCons(result >= left)
        self.model.addCons(result >= right)
        self.model.addCons(result <= left + relax_left * branch)
        self.model.addCons(
            result <= right + relax_right * (1 - branch)
        )
        self.auxiliary_variable_count += 1
        self.binary_variable_count += 1
        self.constraint_count += 4
        return result


def _integer_exponent(expr) -> int:
    if isinstance(expr, FractionConst):
        exponent = expr.value
    elif isinstance(expr, Const):
        exponent = Fraction(str(expr.value))
    else:
        raise ValueError("SCIPAdapter only supports constant integer powers")
    if exponent.denominator != 1:
        raise ValueError("SCIPAdapter only supports constant integer powers")
    return int(exponent)


def _multiply_intervals(left, right) -> tuple[float, float]:
    candidates = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(candidates), max(candidates)


def _power_interval(bounds, exponent: int) -> tuple[float, float]:
    if exponent == 0:
        return 1.0, 1.0
    if exponent < 0:
        if bounds[0] <= 0 <= bounds[1]:
            raise ValueError("SCIPAdapter negative power base may contain zero")
        positive = _power_interval(bounds, -exponent)
        return 1 / positive[1], 1 / positive[0]
    if exponent % 2:
        return bounds[0] ** exponent, bounds[1] ** exponent
    if bounds[0] <= 0 <= bounds[1]:
        return 0.0, max(abs(bounds[0]), abs(bounds[1])) ** exponent
    values = bounds[0] ** exponent, bounds[1] ** exponent
    return min(values), max(values)


def _normalize_bound_overrides(variable_bounds):
    if variable_bounds is None:
        return {}
    if not isinstance(variable_bounds, Mapping):
        raise TypeError("variable_bounds must be a mapping")
    result = {}
    for name, bounds in variable_bounds.items():
        if not isinstance(name, str) or not name:
            raise TypeError("variable bound keys must be non-empty strings")
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise TypeError(f"bounds for {name!r} must be a pair")
        lower = _finite_float(bounds[0], f"{name} lower bound")
        upper = _finite_float(bounds[1], f"{name} upper bound")
        if lower >= upper:
            raise ValueError(f"bounds for {name!r} are not ordered")
        result[name] = (lower, upper)
    return result


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _optional_positive_float(value, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name)


def _optional_nonnegative_float(value, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_or_none(value) -> float | None:
    value = float(value)
    return value if isfinite(value) else None
