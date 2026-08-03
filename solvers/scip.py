"""Experimental SCIP compiler for the exact polynomial semantic IR.

Parameter constraints are sent to SCIP as bounded polynomial constraints.
Universal unit-box constraints use a finite-degree Putinar certificate whose
SOS matrices are represented by bounded lower-triangular factors. The semantic
IR remains exact, but SCIP solves the resulting nonconvex model in floating
point; a returned solution is therefore a numerical certificate candidate, not
an independently verified exact proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isfinite
from numbers import Real
from typing import Mapping, Sequence

from semantics.constraints import (
    ConstraintProblem,
    DomainPolynomialConstraint,
    ParameterConstraint,
    PolynomialIdentity,
    Relation,
)
from semantics.polynomial import (
    ParameterPolynomial,
    ParameterVariable,
    StatePolynomial,
)


@dataclass(frozen=True)
class SCIPPolynomialResult:
    status: str
    values: dict[str, float]
    solution_count: int
    objective_value: float | None
    primal_bound: float | None
    dual_bound: float | None
    relative_gap: float | None
    solve_time: float
    nodes: int
    max_certificate_residual: float | None
    parameter_variable_count: int
    factor_variable_count: int
    constraint_count: int
    factor_values: dict[str, float]

    @property
    def has_solution(self) -> bool:
        return self.solution_count > 0

    @property
    def is_optimal(self) -> bool:
        return self.status == "optimal"


class SCIPPolynomialSolver:
    """Compile exact polynomial IR to a bounded factorized-SOS SCIP model."""

    def __init__(
        self,
        variable_bounds: Mapping[
            ParameterVariable | str,
            tuple[Real, Real],
        ],
        *,
        factor_bound: Real,
        strict_epsilon: Real,
        certificate_degree: int | None = None,
        time_limit: Real | None = None,
        relative_gap: Real | None = None,
        feasibility_tolerance: Real | None = None,
        feasibility_emphasis: bool = False,
        use_symmetry: bool = True,
        display: bool = False,
    ):
        self.variable_bounds = _normalize_variable_bounds(variable_bounds)
        self.factor_bound = _positive_float(factor_bound, "factor_bound")
        self.strict_epsilon = _positive_float(
            strict_epsilon,
            "strict_epsilon",
        )
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
        self.time_limit = _optional_positive_float(
            time_limit,
            "time_limit",
        )
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

    def solve(
        self,
        problem: ConstraintProblem,
        *,
        objective=0,
        sense: str = "minimize",
        initial_values: Mapping[str, Real] | None = None,
    ) -> SCIPPolynomialResult:
        if not isinstance(problem, ConstraintProblem):
            raise TypeError("problem must be a ConstraintProblem")
        if sense not in ("minimize", "maximize"):
            raise ValueError("sense must be 'minimize' or 'maximize'")
        objective = ParameterPolynomial.coerce(objective)
        self._validate_problem_bounds(problem, objective)

        try:
            from pyscipopt import (
                Model,
                SCIP_PARAMEMPHASIS,
                SCIP_PARAMSETTING,
            )
        except ImportError as exc:
            raise ImportError(
                "PySCIPOpt is required; install it with "
                "`python -m pip install pyscipopt`"
            ) from exc

        model = Model("polynomial_constraints")
        if self.feasibility_emphasis:
            model.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)
            model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
        if not self.use_symmetry:
            # Factorized SOS matrices can introduce large artificial
            # column/sign symmetry groups. On large models SCIP's generic
            # detector may add tens of thousands of constraints before root.
            model.setIntParam("misc/usesymmetry", 0)
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

        variables = {}
        for variable in problem.variables:
            lower, upper = self.variable_bounds[variable.name]
            variables[variable.name] = model.addVar(
                name=variable.name,
                lb=float(lower),
                ub=float(upper),
                vtype="C",
            )

        compiler = _SCIPCompiler(
            model,
            variables,
            factor_bound=self.factor_bound,
            strict_epsilon=self.strict_epsilon,
            certificate_degree=self.certificate_degree,
        )
        for index, constraint in enumerate(problem.constraints):
            compiler.add_semantic_constraint(constraint, index)

        objective_expression = compiler.parameter_expression(objective)
        objective_lower, objective_upper = _polynomial_interval(
            objective,
            self.variable_bounds,
        )
        objective_variable = model.addVar(
            name="__objective",
            lb=float(objective_lower),
            ub=float(objective_upper),
            vtype="C",
        )
        compiler.add_equality(
            objective_variable,
            objective_expression,
            "__objective_definition",
        )
        model.setObjective(objective_variable, sense)
        if initial_values is not None:
            initial_solution = model.createSol()
            initial_model_variables = {
                **variables,
                **compiler.factor_variables,
            }
            unknown = sorted(
                set(initial_values) - set(initial_model_variables)
            )
            if unknown:
                raise ValueError(
                    "initial_values contains unknown variables: "
                    f"{', '.join(unknown)}"
                )
            for name, value in initial_values.items():
                model.setSolVal(
                    initial_solution,
                    initial_model_variables[name],
                    float(value),
                )
            if all(
                variable.name in initial_values
                for variable in objective.variables
            ):
                model.setSolVal(
                    initial_solution,
                    objective_variable,
                    float(objective.evaluate(initial_values)),
                )
            model.addSol(initial_solution, free=True)
        model.optimize()

        status = str(model.getStatus())
        solution = model.getBestSol() if model.getNSols() else None
        values = {}
        objective_value = None
        primal_bound = None
        max_residual = None
        factor_values = {}
        if solution is not None:
            values = {
                name: float(model.getSolVal(solution, variable))
                for name, variable in variables.items()
            }
            objective_value = float(
                model.getSolVal(solution, objective_variable)
            )
            primal_bound = float(model.getObjVal())
            max_residual = compiler.max_equality_residual(solution)
            factor_values = {
                name: float(model.getSolVal(solution, variable))
                for name, variable in compiler.factor_variables.items()
            }

        dual_bound = _finite_or_none(model.getDualbound())
        gap = _finite_or_none(model.getGap())
        return SCIPPolynomialResult(
            status=status,
            values=values,
            solution_count=int(model.getNSols()),
            objective_value=objective_value,
            primal_bound=primal_bound,
            dual_bound=dual_bound,
            relative_gap=gap,
            solve_time=float(model.getSolvingTime()),
            nodes=int(model.getNNodes()),
            max_certificate_residual=max_residual,
            parameter_variable_count=len(variables),
            factor_variable_count=compiler.factor_variable_count,
            constraint_count=compiler.constraint_count,
            factor_values=factor_values,
        )

    def _validate_problem_bounds(
        self,
        problem: ConstraintProblem,
        objective: ParameterPolynomial,
    ) -> None:
        declared_names = {variable.name for variable in problem.variables}
        referenced_names = {
            variable.name
            for constraint in problem.constraints
            for variable in _constraint_parameter_variables(constraint)
        }
        undeclared_constraints = sorted(referenced_names - declared_names)
        if undeclared_constraints:
            raise ValueError(
                "constraints reference undeclared parameters: "
                f"{', '.join(undeclared_constraints)}"
            )
        missing = sorted(declared_names - self.variable_bounds.keys())
        extra = sorted(self.variable_bounds.keys() - declared_names)
        if missing:
            raise ValueError(
                "finite bounds are required for every parameter variable; "
                f"missing: {', '.join(missing)}"
            )
        if extra:
            raise ValueError(
                "variable_bounds contains undeclared parameters: "
                f"{', '.join(extra)}"
            )
        objective_names = {
            variable.name for variable in objective.variables
        }
        undeclared = sorted(objective_names - declared_names)
        if undeclared:
            raise ValueError(
                "objective references undeclared parameters: "
                f"{', '.join(undeclared)}"
            )


class _SCIPCompiler:
    def __init__(
        self,
        model,
        variables,
        *,
        factor_bound: float,
        strict_epsilon: float,
        certificate_degree: int | None,
    ):
        self.model = model
        self.variables = variables
        self.factor_bound = factor_bound
        self.strict_epsilon = strict_epsilon
        self.certificate_degree = certificate_degree
        self.factor_variable_count = 0
        self.factor_variables = {}
        self.constraint_count = 0
        self._equalities = []
        self._infeasibility_variable = None

    def add_semantic_constraint(self, constraint, index: int) -> None:
        name = f"constraint_{index}"
        if isinstance(constraint, ParameterConstraint):
            self._add_parameter_constraint(constraint, name)
            return
        if isinstance(constraint, PolynomialIdentity):
            for coefficient_index, coefficient_constraint in enumerate(
                constraint.coefficient_constraints()
            ):
                self._add_parameter_constraint(
                    coefficient_constraint,
                    f"{name}_coefficient_{coefficient_index}",
                )
            return
        if isinstance(constraint, DomainPolynomialConstraint):
            self._add_domain_constraint(constraint, name)
            return
        raise TypeError(
            f"unsupported semantic constraint: {type(constraint).__name__}"
        )

    def parameter_expression(self, polynomial: ParameterPolynomial):
        expression = 0.0
        for monomial, coefficient in polynomial.terms.items():
            term = float(coefficient)
            for variable, exponent in monomial:
                term *= self.variables[variable.name] ** exponent
            expression += term
        return expression

    def add_equality(self, left, right, name: str) -> None:
        difference = left - right
        if isinstance(difference, Real):
            if float(difference) != 0.0:
                self._add_infeasibility_constraint(name)
            return
        self.model.addCons(difference == 0, name=name)
        self._equalities.append(difference)
        self.constraint_count += 1

    def _add_infeasibility_constraint(self, name: str) -> None:
        if self._infeasibility_variable is None:
            self._infeasibility_variable = self.model.addVar(
                name="__infeasible",
                lb=0.0,
                ub=0.0,
                vtype="C",
            )
        self.model.addCons(
            self._infeasibility_variable >= 1.0,
            name=name,
        )
        self.constraint_count += 1

    def _add_inequality(self, expression, relation: Relation, name: str) -> None:
        if relation is Relation.GE:
            right = 0.0
            satisfied = expression >= right
        elif relation is Relation.GT:
            right = self.strict_epsilon
            satisfied = expression >= right
        elif relation is Relation.LE:
            right = 0.0
            satisfied = expression <= right
        elif relation is Relation.LT:
            right = -self.strict_epsilon
            satisfied = expression <= right
        else:
            raise TypeError(f"unsupported relation: {relation}")

        if isinstance(expression, Real):
            if not satisfied:
                self._add_infeasibility_constraint(name)
            return
        self.model.addCons(satisfied, name=name)
        self.constraint_count += 1

    def max_equality_residual(self, solution) -> float:
        if not self._equalities:
            return 0.0
        return max(
            abs(float(self.model.getSolVal(solution, expression)))
            for expression in self._equalities
        )

    def _add_parameter_constraint(
        self,
        constraint: ParameterConstraint,
        name: str,
    ) -> None:
        expression = self.parameter_expression(constraint.polynomial)
        relation = constraint.relation
        if relation is Relation.EQ:
            self.add_equality(expression, 0, name)
            return
        self._add_inequality(expression, relation, name)

    def _add_domain_constraint(
        self,
        constraint: DomainPolynomialConstraint,
        name: str,
    ) -> None:
        polynomial = constraint.polynomial
        relation = constraint.relation
        if relation is Relation.EQ:
            for coefficient_index, coefficient in enumerate(
                polynomial.terms.values()
            ):
                self.add_equality(
                    self.parameter_expression(coefficient),
                    0,
                    f"{name}_coefficient_{coefficient_index}",
                )
            return

        if relation in (Relation.LE, Relation.LT):
            polynomial = -polynomial
        if relation in (Relation.GT, Relation.LT):
            polynomial = polynomial - self.strict_epsilon
        if relation not in (
            Relation.GE,
            Relation.GT,
            Relation.LE,
            Relation.LT,
        ):
            raise TypeError(f"unsupported relation: {relation}")

        if not constraint.domain.active_dims:
            coefficient = polynomial.coefficient((0,) * polynomial.ndim)
            self._add_inequality(
                self.parameter_expression(coefficient),
                Relation.GE,
                name,
            )
            return

        target_degree = max(polynomial.degree(), 0)
        degree = self.certificate_degree
        if degree is None:
            degree = target_degree if target_degree % 2 == 0 else target_degree + 1
        if degree < target_degree:
            raise ValueError(
                f"{name} degree {target_degree} exceeds "
                f"certificate_degree {degree}"
            )

        right_coefficients = self._factorized_sos_coefficients(
            polynomial.ndim,
            constraint.domain.active_dims,
            degree,
            f"{name}_s0",
        )
        if degree >= 2:
            for dim in constraint.domain.active_dims:
                multiplier = self._factorized_sos_coefficients(
                    polynomial.ndim,
                    constraint.domain.active_dims,
                    degree - 2,
                    f"{name}_s{dim + 1}",
                )
                for exponents, expression in multiplier.items():
                    linear_exponents = list(exponents)
                    linear_exponents[dim] += 1
                    _add_expression(
                        right_coefficients,
                        tuple(linear_exponents),
                        expression,
                    )
                    quadratic_exponents = list(exponents)
                    quadratic_exponents[dim] += 2
                    _add_expression(
                        right_coefficients,
                        tuple(quadratic_exponents),
                        -expression,
                    )

        all_exponents = sorted(
            set(polynomial.terms) | set(right_coefficients)
        )
        for coefficient_index, exponents in enumerate(all_exponents):
            left = self.parameter_expression(
                polynomial.coefficient(exponents)
            )
            right = right_coefficients.get(exponents, 0.0)
            self.add_equality(
                left,
                right,
                f"{name}_coefficient_{coefficient_index}",
            )

    def _factorized_sos_coefficients(
        self,
        ndim: int,
        active_dims: Sequence[int],
        sos_degree: int,
        name: str,
    ):
        basis = _total_degree_exponents(
            ndim,
            active_dims,
            sos_degree // 2,
        )
        factors = []
        for column in range(len(basis)):
            factor = []
            for row in range(column, len(basis)):
                lower = 0.0 if row == column else -self.factor_bound
                variable_name = f"{name}_L_{row}_{column}"
                variable = self.model.addVar(
                    name=variable_name,
                    lb=lower,
                    ub=self.factor_bound,
                    vtype="C",
                )
                self.factor_variables[variable_name] = variable
                factor.append((basis[row], variable))
                self.factor_variable_count += 1
            factors.append(factor)

        coefficients = {}
        for factor in factors:
            for left_index, (left_exponents, left_variable) in enumerate(
                factor
            ):
                for right_index in range(left_index, len(factor)):
                    right_exponents, right_variable = factor[right_index]
                    exponents = tuple(
                        left + right
                        for left, right in zip(
                            left_exponents,
                            right_exponents,
                        )
                    )
                    scale = 1 if left_index == right_index else 2
                    _add_expression(
                        coefficients,
                        exponents,
                        scale * left_variable * right_variable,
                    )
        return coefficients


def _total_degree_exponents(
    ndim: int,
    active_dims: Sequence[int],
    total_degree: int,
) -> list[tuple[int, ...]]:
    result = []
    active_dims = tuple(active_dims)
    for active_exponents in product(
        range(total_degree + 1),
        repeat=len(active_dims),
    ):
        if sum(active_exponents) > total_degree:
            continue
        exponents = [0] * ndim
        for dim, exponent in zip(active_dims, active_exponents):
            exponents[dim] = exponent
        result.append(tuple(exponents))
    return sorted(result, key=lambda exponents: (sum(exponents), exponents))


def _add_expression(coefficients, exponents, expression) -> None:
    coefficients[exponents] = coefficients.get(exponents, 0.0) + expression


def _normalize_variable_bounds(variable_bounds):
    if not isinstance(variable_bounds, Mapping):
        raise TypeError("variable_bounds must be a mapping")
    result = {}
    for variable, bounds in variable_bounds.items():
        name = variable.name if isinstance(variable, ParameterVariable) else variable
        if not isinstance(name, str) or not name:
            raise TypeError("variable bound keys must be parameter names")
        if name in result:
            raise ValueError(f"duplicate bounds for parameter {name!r}")
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise TypeError(
                f"bounds for {name!r} must be a (lower, upper) pair"
            )
        lower = _finite_fraction(bounds[0], f"{name} lower bound")
        upper = _finite_fraction(bounds[1], f"{name} upper bound")
        if lower > upper:
            raise ValueError(f"bounds for {name!r} are not ordered")
        result[name] = (lower, upper)
    return result


def _polynomial_interval(
    polynomial: ParameterPolynomial,
    bounds: Mapping[str, tuple[Fraction, Fraction]],
) -> tuple[Fraction, Fraction]:
    lower = Fraction(0)
    upper = Fraction(0)
    for monomial, coefficient in polynomial.terms.items():
        term_lower = Fraction(1)
        term_upper = Fraction(1)
        for variable, exponent in monomial:
            power_lower, power_upper = _power_interval(
                bounds[variable.name],
                exponent,
            )
            candidates = (
                term_lower * power_lower,
                term_lower * power_upper,
                term_upper * power_lower,
                term_upper * power_upper,
            )
            term_lower = min(candidates)
            term_upper = max(candidates)
        if coefficient >= 0:
            lower += coefficient * term_lower
            upper += coefficient * term_upper
        else:
            lower += coefficient * term_upper
            upper += coefficient * term_lower
    return lower, upper


def _power_interval(
    bounds: tuple[Fraction, Fraction],
    exponent: int,
) -> tuple[Fraction, Fraction]:
    lower, upper = bounds
    if exponent == 0:
        return Fraction(1), Fraction(1)
    if exponent % 2:
        return lower**exponent, upper**exponent
    if lower <= 0 <= upper:
        return Fraction(0), max(abs(lower), abs(upper)) ** exponent
    values = (lower**exponent, upper**exponent)
    return min(values), max(values)


def _finite_fraction(value, name: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
        return Fraction(str(value))
    try:
        result = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    return result


def _positive_float(value, name: str) -> float:
    result = float(_finite_fraction(value, name))
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
    result = float(_finite_fraction(value, name))
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_or_none(value) -> float | None:
    value = float(value)
    return value if isfinite(value) else None


def _constraint_parameter_variables(constraint):
    if isinstance(constraint, ParameterConstraint):
        return constraint.polynomial.variables
    if isinstance(constraint, DomainPolynomialConstraint):
        return constraint.polynomial.parameter_variables
    if isinstance(constraint, PolynomialIdentity):
        return tuple(
            set(constraint.left.parameter_variables)
            | set(constraint.right.parameter_variables)
        )
    raise TypeError(
        f"unsupported semantic constraint: {type(constraint).__name__}"
    )
