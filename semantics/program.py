from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from distributions import (
    BGD,
    PolynomialMUD,
    fraction_lcm,
    leq_sum,
    symbolic_polynomial_bgd_template,
)
from distributions.mud import merge_breakpoints
from intervals import (
    const_int_value,
    interval_complement,
    interval_intersect,
    interval_is_empty,
    interval_union,
)
from preprocessing.polynomial_prior_prep import (
    distribution_to_polynomial_bgd,
    prior_to_polynomial_bgd,
)
from probably.pgcl.ast import Program
from probably.pgcl.ast.expressions import (
    Binop,
    BinopExpr,
    NatLitExpr,
    RealLitExpr,
    Unop,
    UnopExpr,
    VarExpr,
)
from probably.pgcl.ast.instructions import (
    AsgnInstr,
    ChoiceInstr,
    IfInstr,
    LoopInstr,
    ObserveInstr,
    SkipInstr,
    TickInstr,
    WhileInstr,
)
from semantics.constraints import ConstraintContext, ConstraintProblem
from semantics.polynomial import ParameterPolynomial, StatePolynomial


@dataclass(frozen=True)
class ShiftAssignment:
    variable: str
    offset: Fraction


@dataclass(frozen=True)
class UniformConvolutionAssignment:
    variable: str
    low: Fraction
    high: Fraction


@dataclass(frozen=True)
class ReplaceDistributionAssignment:
    variable: str
    distribution: tuple


Assignment = (
    ShiftAssignment
    | UniformConvolutionAssignment
    | ReplaceDistributionAssignment
)


@dataclass(frozen=True)
class PolynomialProgramResult:
    distribution: BGD
    constraints: ConstraintProblem
    variable_order: tuple[str, ...]
    objective: ParameterPolynomial
    loop_template_degrees: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, BGD):
            raise TypeError("distribution must be a BGD")
        if self.distribution.cell_family is not PolynomialMUD:
            raise TypeError("distribution must use PolynomialMUD cells")
        if not isinstance(self.constraints, ConstraintProblem):
            raise TypeError("constraints must be a ConstraintProblem")
        if len(self.variable_order) != self.distribution.ndim:
            raise ValueError(
                "variable_order length must match distribution dimensions"
            )
        if not isinstance(self.objective, ParameterPolynomial):
            raise TypeError("objective must be a ParameterPolynomial")
        for degrees in self.loop_template_degrees:
            if len(degrees) != self.distribution.ndim:
                raise ValueError(
                    "loop template degree length must match distribution dimensions"
                )
            if any(degree < 0 for degree in degrees):
                raise ValueError("loop template degrees must be nonnegative")


def numeric_literal_value(expr, *, role: str = "numeric value") -> Fraction:
    if isinstance(expr, NatLitExpr):
        return Fraction(int(expr.value))
    if isinstance(expr, RealLitExpr):
        if expr.is_infinite():
            raise ValueError(f"{role} must be finite")
        return expr.to_fraction()
    raise ValueError(f"{role} must be a numeric literal")


def choice_probability(expr) -> Fraction:
    value = numeric_literal_value(expr, role="choice probability")
    if value < 0 or value > 1:
        raise ValueError("choice probability must satisfy 0 <= c <= 1")
    return value


def condition_intervals(expr):
    if isinstance(expr, BinopExpr):
        operator = expr.operator
        if operator in (Binop.AND, Binop.OR):
            left_variable, left_intervals = condition_intervals(expr.lhs)
            right_variable, right_intervals = condition_intervals(expr.rhs)
            if left_variable != right_variable:
                raise ValueError("condition must use a single variable")
            if operator == Binop.AND:
                intervals = interval_intersect(
                    left_intervals,
                    right_intervals,
                )
            else:
                intervals = interval_union(
                    left_intervals,
                    right_intervals,
                )
            return left_variable, intervals

        if operator in (
            Binop.LT,
            Binop.LEQ,
            Binop.GT,
            Binop.GEQ,
            Binop.EQ,
        ):
            if isinstance(expr.lhs, VarExpr) and isinstance(
                expr.rhs,
                (NatLitExpr, RealLitExpr),
            ):
                return _atomic_condition(
                    expr.lhs.var,
                    operator,
                    const_int_value(expr.rhs),
                )
            if isinstance(expr.lhs, (NatLitExpr, RealLitExpr)) and isinstance(
                expr.rhs,
                VarExpr,
            ):
                reverse = {
                    Binop.LT: Binop.GT,
                    Binop.LEQ: Binop.GEQ,
                    Binop.GT: Binop.LT,
                    Binop.GEQ: Binop.LEQ,
                    Binop.EQ: Binop.EQ,
                }[operator]
                return _atomic_condition(
                    expr.rhs.var,
                    reverse,
                    const_int_value(expr.lhs),
                )
            raise ValueError(
                "condition must compare one variable with one numeric literal"
            )
        raise ValueError(
            "condition must use <, <=, >, >=, or = with logical combination"
        )

    if isinstance(expr, UnopExpr) and expr.operator == Unop.NEG:
        variable, intervals = condition_intervals(expr.expr)
        return variable, interval_complement(intervals)
    raise ValueError("condition must compare a variable with a numeric literal")


def classify_assignment(
    instr: AsgnInstr,
    distribution_map: Mapping[str, tuple],
) -> Assignment:
    if not isinstance(instr, AsgnInstr):
        raise TypeError("instr must be an AsgnInstr")

    variable = instr.lhs
    rhs = instr.rhs
    direct_distribution = _placeholder_distribution(rhs, distribution_map)
    if direct_distribution is not None:
        return ReplaceDistributionAssignment(variable, direct_distribution)

    if not isinstance(rhs, BinopExpr) or rhs.operator not in (
        Binop.PLUS,
        Binop.MINUS,
    ):
        raise ValueError(
            f"assignment must be {variable} := {variable} + c, "
            f"{variable} := {variable} +/- Uniform(a,b), or "
            f"{variable} := Distribution(...)"
        )

    distribution_term = _distribution_addition(
        variable,
        rhs,
        distribution_map,
    )
    if distribution_term is not None:
        dist_name, params, sign = distribution_term
        if dist_name != "Uniform":
            raise ValueError(
                "only Uniform(a,b) is supported in distribution addition"
            )
        low, high = (_as_fraction(value) for value in params)
        if sign < 0:
            low, high = -high, -low
        if low >= high:
            raise ValueError("Uniform addition requires low < high")
        return UniformConvolutionAssignment(variable, low, high)

    if isinstance(rhs.lhs, VarExpr) and rhs.lhs.var == variable:
        offset = numeric_literal_value(
            rhs.rhs,
            role="assignment constant",
        )
        if rhs.operator == Binop.MINUS:
            offset = -offset
        return ShiftAssignment(variable, offset)

    if (
        rhs.operator == Binop.PLUS
        and isinstance(rhs.rhs, VarExpr)
        and rhs.rhs.var == variable
    ):
        return ShiftAssignment(
            variable,
            numeric_literal_value(
                rhs.lhs,
                role="assignment constant",
            ),
        )
    raise ValueError(
        f"assignment must be {variable} := {variable} + c, "
        f"{variable} := {variable} +/- Uniform(a,b), or "
        f"{variable} := Distribution(...)"
    )


def restrict_intervals(
    distribution: BGD,
    dim: int,
    intervals,
    *,
    max_fn=None,
) -> BGD:
    if max_fn is None and distribution.cell_family is PolynomialMUD:
        max_fn = _equal_polynomial_decay
    pieces = []
    for low, low_closed, high, high_closed in interval_union(intervals, []):
        interval = (low, low_closed, high, high_closed)
        if interval_is_empty(interval):
            continue

        piece = distribution
        if low is not None:
            piece = piece.restrict(
                dim,
                ">=" if low_closed else ">",
                low,
            )
        if high is not None:
            piece = piece.restrict(
                dim,
                "<=" if high_closed else "<",
                high,
            )
        pieces.append(piece)

    if not pieces:
        point = distribution.center_lefts[dim]
        return distribution.restrict(dim, ">=", point).restrict(
            dim,
            "<",
            point,
        )

    result = pieces[0]
    for piece in pieces[1:]:
        result = result.add(piece, max_fn=max_fn)
    return result


def _equal_polynomial_decay(left, right, name: str):
    if ParameterPolynomial.coerce(left) != ParameterPolynomial.coerce(right):
        raise ValueError(
            f"{name} differs across disjoint polynomial restrictions"
        )
    return left


def build_polynomial_program_semantics(
    prior: dict,
    program: Program,
    distribution_map: Mapping[str, tuple],
    *,
    loop_template_degree: int | Sequence[int] | str | None = "infer",
    loop_template_degree_increment: int = 0,
    loop_unroll_iterations: int = 2,
) -> PolynomialProgramResult:
    if not isinstance(program, Program):
        raise TypeError("program must be a pGCL Program")

    distribution, variable_order = prior_to_polynomial_bgd(prior)
    variable_map = {
        variable: dim for dim, variable in enumerate(variable_order)
    }
    context = ConstraintContext()
    configured_degrees = _normalize_loop_template_degrees(
        loop_template_degree,
        distribution.ndim,
    )
    if (
        not isinstance(loop_template_degree_increment, Integral)
        or isinstance(loop_template_degree_increment, bool)
        or loop_template_degree_increment < 0
    ):
        raise ValueError(
            "loop_template_degree_increment must be a nonnegative integer"
        )
    loop_template_degree_increment = int(loop_template_degree_increment)
    if (
        not isinstance(loop_unroll_iterations, int)
        or isinstance(loop_unroll_iterations, bool)
        or loop_unroll_iterations < 1
    ):
        raise ValueError("loop_unroll_iterations must be a positive integer")
    loop_counter = 0
    selected_loop_degrees = []

    def execute_block(instructions: Sequence, state: BGD) -> BGD:
        for instruction in instructions:
            state = execute(instruction, state)
        return state

    def execute(instruction, state: BGD) -> BGD:
        nonlocal loop_counter
        if isinstance(instruction, SkipInstr):
            return state
        if isinstance(instruction, AsgnInstr):
            action = classify_assignment(instruction, distribution_map)
            dim = _variable_dimension(action.variable, variable_map)
            if isinstance(action, ShiftAssignment):
                return state.add_constant(dim, action.offset)
            if isinstance(action, UniformConvolutionAssignment):
                return state.convolve_uniform(dim, action.low, action.high)
            replacement = distribution_to_polynomial_bgd(
                action.distribution
            )
            return state.replace_dim(dim, replacement)
        if isinstance(instruction, IfInstr):
            if isinstance(instruction.cond, (NatLitExpr, RealLitExpr)):
                probability = choice_probability(instruction.cond)
                true_state = execute_block(instruction.true, state)
                false_state = execute_block(instruction.false, state)
                return true_state.scale(probability).add(
                    false_state.scale(1 - probability),
                    max_fn=_equal_polynomial_decay,
                )

            variable, intervals = condition_intervals(instruction.cond)
            dim = _variable_dimension(variable, variable_map)
            true_state = restrict_intervals(state, dim, intervals)
            false_state = restrict_intervals(
                state,
                dim,
                interval_complement(intervals),
            )
            return execute_block(instruction.true, true_state).add(
                execute_block(instruction.false, false_state),
                max_fn=_equal_polynomial_decay,
            )
        if isinstance(instruction, ChoiceInstr):
            probability = choice_probability(instruction.prob)
            left_state = execute_block(instruction.lhs, state)
            right_state = execute_block(instruction.rhs, state)
            return left_state.scale(probability).add(
                right_state.scale(1 - probability),
                max_fn=_equal_polynomial_decay,
            )
        if isinstance(instruction, ObserveInstr):
            variable, intervals = condition_intervals(instruction.cond)
            dim = _variable_dimension(variable, variable_map)
            return restrict_intervals(state, dim, intervals)
        if isinstance(instruction, WhileInstr):
            current_loop = loop_counter
            loop_counter += 1
            if isinstance(instruction.cond, (NatLitExpr, RealLitExpr)):
                probability = choice_probability(instruction.cond)
                guard = lambda value: value.scale(probability)
                exit_guard = lambda value: value.scale(1 - probability)
                guard_dim = None
                guard_intervals = ()
            else:
                variable, guard_intervals = condition_intervals(
                    instruction.cond
                )
                guard_dim = _variable_dimension(variable, variable_map)
                complement = interval_complement(guard_intervals)
                guard = lambda value: restrict_intervals(
                    value,
                    guard_dim,
                    guard_intervals,
                )
                exit_guard = lambda value: restrict_intervals(
                    value,
                    guard_dim,
                    complement,
                )

            seed, center_lefts, center_rights = _loop_geometry_seed(
                state,
                instruction.body,
                distribution_map,
                variable_map,
                guard_dim,
                guard_intervals,
            )
            probes = []
            probe = seed
            for _ in range(loop_unroll_iterations):
                probe = execute_block(instruction.body, guard(probe))
                probes.append(probe)

            base_degrees = (
                _infer_polynomial_degrees((seed, *probes))
                if configured_degrees is None
                else configured_degrees
            )
            template_degrees = tuple(
                degree + loop_template_degree_increment
                for degree in base_degrees
            )
            selected_loop_degrees.append(template_degrees)
            shape = _loop_template_shape(
                seed,
                probes,
                center_lefts,
                center_rights,
                guard_dim,
                guard_intervals,
            )
            invariant = symbolic_polynomial_bgd_template(
                shape,
                template_degrees,
                context,
                name_prefix=f"w{current_loop}",
            )
            body_result = execute_block(
                instruction.body,
                guard(invariant),
            )
            for constraint in leq_sum(
                [state, body_result],
                invariant,
            ):
                context.add(constraint)
            return exit_guard(invariant)
        if isinstance(instruction, LoopInstr):
            raise ValueError("unbounded loop is not supported by polynomial semantics")
        if isinstance(instruction, TickInstr):
            raise ValueError("tick is not supported by polynomial semantics")
        raise ValueError(
            f"unsupported statement in polynomial semantics: "
            f"{type(instruction).__name__}"
        )

    result = execute_block(program.instructions, distribution)
    objective = polynomial_bgd_mass(result, context)
    return PolynomialProgramResult(
        distribution=result,
        constraints=context.build(),
        variable_order=variable_order,
        objective=objective,
        loop_template_degrees=tuple(selected_loop_degrees),
    )


def _normalize_loop_template_degrees(
    value: int | Sequence[int] | str | None,
    ndim: int,
) -> tuple[int, ...] | None:
    if value is None or (
        isinstance(value, str) and value.strip().lower() == "infer"
    ):
        return None
    if isinstance(value, Integral) and not isinstance(value, bool):
        value = int(value)
        if value < 0:
            raise ValueError("loop_template_degree must be nonnegative")
        return (value,) * ndim
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "loop_template_degree must be 'infer', an integer, or a sequence"
        )
    if len(value) != ndim:
        raise ValueError(
            "loop_template_degree sequence length must match program variables"
        )
    degrees = []
    for degree in value:
        if not isinstance(degree, Integral) or isinstance(degree, bool):
            raise TypeError("loop template degrees must be integers")
        degree = int(degree)
        if degree < 0:
            raise ValueError("loop template degrees must be nonnegative")
        degrees.append(degree)
    return tuple(degrees)


def _infer_polynomial_degrees(
    distributions: Sequence[BGD],
) -> tuple[int, ...]:
    if not distributions:
        raise ValueError("at least one distribution is required")
    ndim = distributions[0].ndim
    degrees = [0] * ndim
    for distribution in distributions:
        if distribution.ndim != ndim:
            raise ValueError("probe distribution dimensions do not match")
        if distribution.cell_family is not PolynomialMUD:
            raise TypeError("degree inference requires PolynomialMUD cells")
        for mud in distribution.E.flat:
            for cell in mud.P.flat:
                for dim in range(ndim):
                    degrees[dim] = max(
                        degrees[dim],
                        cell.polynomial.degree(dim),
                    )
    return tuple(degrees)


def _atomic_condition(variable, operator, constant):
    if operator == Binop.LT:
        intervals = [(None, False, constant, False)]
    elif operator == Binop.LEQ:
        intervals = [(None, False, constant, True)]
    elif operator == Binop.GT:
        intervals = [(constant, False, None, False)]
    elif operator == Binop.GEQ:
        intervals = [(constant, True, None, False)]
    elif operator == Binop.EQ:
        intervals = [(constant, True, constant, True)]
    else:
        raise ValueError("unsupported condition comparison")
    return variable, intervals


def _placeholder_distribution(expr, distribution_map):
    if not isinstance(expr, VarExpr):
        return None
    return distribution_map.get(expr.var)


def _distribution_addition(variable, rhs, distribution_map):
    if isinstance(rhs.lhs, VarExpr) and rhs.lhs.var == variable:
        distribution = _placeholder_distribution(
            rhs.rhs,
            distribution_map,
        )
        if distribution is not None:
            sign = 1 if rhs.operator == Binop.PLUS else -1
            return distribution[0], distribution[1], sign
    if (
        rhs.operator == Binop.PLUS
        and isinstance(rhs.rhs, VarExpr)
        and rhs.rhs.var == variable
    ):
        distribution = _placeholder_distribution(
            rhs.lhs,
            distribution_map,
        )
        if distribution is not None:
            return distribution[0], distribution[1], 1
    return None


def _variable_dimension(variable: str, variable_map) -> int:
    try:
        return variable_map[variable]
    except KeyError as exc:
        raise ValueError(f"unknown program variable: {variable}") from exc


def _as_fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def polynomial_bgd_mass(
    distribution: BGD,
    context: ConstraintContext,
) -> ParameterPolynomial:
    """Represent a symbolic BGD's geometric tail mass polynomially."""

    total = ParameterPolynomial.zero()
    for edge_index in np.ndindex(distribution.E.shape):
        term = ParameterPolynomial.coerce(
            distribution.E[edge_index].mass()
        )
        if term.is_zero:
            continue
        direction = distribution.index_to_direction(edge_index)
        for dim, side in enumerate(direction):
            if side < 0:
                decay = ParameterPolynomial.coerce(distribution.alpha[dim])
            elif side > 0:
                decay = ParameterPolynomial.coerce(distribution.beta[dim])
            else:
                continue
            denominator = 1 - decay
            if denominator.is_constant:
                term = term / denominator.constant_value
            else:
                term = context.exact_positive_quotient(
                    term,
                    denominator,
                    prefix=f"objective_tail_E{'_'.join(map(str, edge_index))}",
                )
        total += term
    return total


def default_polynomial_variable_bounds(
    problem: ConstraintProblem,
    *,
    coefficient_bound=100,
    mass_bound=100,
    strict_epsilon=Fraction(1, 10**7),
):
    coefficient_bound = _positive_fraction(
        coefficient_bound,
        "coefficient_bound",
    )
    mass_bound = _positive_fraction(mass_bound, "mass_bound")
    strict_epsilon = _positive_fraction(
        strict_epsilon,
        "strict_epsilon",
    )
    if strict_epsilon >= 1:
        raise ValueError("strict_epsilon must be smaller than 1")

    result = {}
    for variable in problem.variables:
        name = variable.name
        if "_alpha_" in name or "_beta_" in name:
            result[variable] = (Fraction(0), 1 - strict_epsilon)
        elif name.startswith("objective_tail_"):
            result[variable] = (Fraction(0), mass_bound)
        else:
            result[variable] = (-coefficient_bound, coefficient_bound)
    return result


def evaluate_polynomial_bgd(
    distribution: BGD,
    parameter_values,
) -> BGD:
    """Substitute a SCIP parameter solution into a PolynomialBGD."""

    result_E = np.empty(distribution.E.shape, dtype=object)
    for edge_index in np.ndindex(distribution.E.shape):
        mud = distribution.E[edge_index]
        payloads = np.empty(mud.shape, dtype=object)
        for cell_index in np.ndindex(mud.shape):
            polynomial = mud.P[cell_index].polynomial
            payloads[cell_index] = StatePolynomial(
                polynomial.ndim,
                {
                    exponents: coefficient.evaluate(parameter_values)
                    for exponents, coefficient in polynomial.terms.items()
                },
            )
        result_E[edge_index] = PolynomialMUD(mud.S, payloads)

    def evaluate_decay(value):
        value = ParameterPolynomial.coerce(value)
        return value.evaluate(parameter_values)

    return BGD(
        result_E,
        [evaluate_decay(value) for value in distribution.alpha],
        [evaluate_decay(value) for value in distribution.beta],
    )


def _loop_geometry_seed(
    state: BGD,
    instructions: Sequence,
    distribution_map,
    variable_map,
    guard_dim,
    guard_intervals,
):
    left_lengths = list(state.left_lengths)
    right_lengths = list(state.right_lengths)
    center_lefts = list(state.center_lefts)
    center_rights = list(state.center_rights)

    if guard_dim is not None:
        for low, _low_closed, high, _high_closed in guard_intervals:
            for point in (low, high):
                if point is not None:
                    center_lefts[guard_dim] = min(
                        center_lefts[guard_dim],
                        point,
                    )
                    center_rights[guard_dim] = max(
                        center_rights[guard_dim],
                        point,
                    )

    def combine_period(current, required):
        if required <= 0:
            return current
        if current <= 0:
            return required
        return fraction_lcm(current, required)

    def scan(block):
        for instruction in block:
            if isinstance(instruction, AsgnInstr):
                action = classify_assignment(instruction, distribution_map)
                dim = _variable_dimension(action.variable, variable_map)
                if isinstance(action, ShiftAssignment):
                    if action.offset < 0:
                        left_lengths[dim] = combine_period(
                            left_lengths[dim],
                            -action.offset,
                        )
                        center_lefts[dim] = min(
                            center_lefts[dim],
                            state.center_lefts[dim] + action.offset,
                        )
                    elif action.offset > 0:
                        right_lengths[dim] = combine_period(
                            right_lengths[dim],
                            action.offset,
                        )
                        center_rights[dim] = max(
                            center_rights[dim],
                            state.center_rights[dim] + action.offset,
                        )
                elif isinstance(action, UniformConvolutionAssignment):
                    width = action.high - action.low
                    if action.low < 0:
                        left_lengths[dim] = combine_period(
                            left_lengths[dim],
                            width,
                        )
                        center_lefts[dim] = min(
                            center_lefts[dim],
                            state.center_lefts[dim] + action.low,
                        )
                    if action.high > 0:
                        right_lengths[dim] = combine_period(
                            right_lengths[dim],
                            width,
                        )
                        center_rights[dim] = max(
                            center_rights[dim],
                            state.center_rights[dim] + action.high,
                        )
                else:
                    raise ValueError(
                        "distribution replacement inside a polynomial loop "
                        "is not yet supported"
                    )
            elif isinstance(instruction, IfInstr):
                scan(instruction.true)
                scan(instruction.false)
            elif isinstance(instruction, ChoiceInstr):
                scan(instruction.lhs)
                scan(instruction.rhs)
            elif isinstance(instruction, (WhileInstr, LoopInstr)):
                raise ValueError(
                    "nested loops are not yet supported by polynomial semantics"
                )

    scan(instructions)
    seed = state.align_frame(
        state.center_lefts,
        state.center_rights,
        left_lengths,
        right_lengths,
    )
    return seed, tuple(center_lefts), tuple(center_rights)


def _loop_template_shape(
    seed: BGD,
    probes: Sequence[BGD],
    center_lefts,
    center_rights,
    guard_dim,
    guard_intervals,
) -> BGD:
    shape = seed.align_center_domain(center_lefts, center_rights)
    result_E = shape._copy_E()
    center_index = (1,) * shape.ndim
    center = result_E[center_index]
    target_S = list(center.S)

    for dim in range(shape.ndim):
        ordinary_points = set()
        dirac_points = set()
        for probe in probes:
            ordinary_points.update(
                _global_breakpoints_in_range(
                    probe,
                    dim,
                    center_lefts[dim],
                    center_rights[dim],
                )
            )
            dirac_points.update(
                point
                for point in _global_dirac_points(probe, dim)
                if center_lefts[dim] <= point <= center_rights[dim]
            )
        target_S[dim] = merge_breakpoints(
            target_S[dim],
            tuple(sorted(ordinary_points)),
            *((point, point) for point in sorted(dirac_points)),
            preserve_dirac=True,
        )

    if guard_dim is not None:
        boundary_points = sorted(
            {
                point
                for low, _low_closed, high, _high_closed in guard_intervals
                for point in (low, high)
                if point is not None
            }
        )
        target_S[guard_dim] = merge_breakpoints(
            target_S[guard_dim],
            *((point, point) for point in boundary_points),
            preserve_dirac=True,
        )

    result_E[center_index] = center.align(tuple(target_S))
    result = BGD(result_E, shape.alpha, shape.beta).standardize(
        skip_static_zero=False
    ).align_center_subdivisions()
    return _add_probe_diracs_to_loop_shape(
        result,
        (seed, *probes),
    )


def _add_probe_diracs_to_loop_shape(
    shape: BGD,
    probes: Sequence[BGD],
) -> BGD:
    """Add probe atoms to center cells and their periodic tail phases."""

    global_points = [set() for _ in range(shape.ndim)]
    for probe in probes:
        for dim in range(shape.ndim):
            global_points[dim].update(
                _global_dirac_points(probe, dim)
            )

    result_E = shape._copy_E()
    for edge_index in np.ndindex(result_E.shape):
        mud = result_E[edge_index]
        target_S = list(mud.S)
        changed = False
        for dim, points in enumerate(global_points):
            additions = []
            for point in points:
                local_point = _loop_shape_local_dirac(
                    shape,
                    edge_index,
                    dim,
                    point,
                )
                if local_point is not None:
                    additions.append((local_point, local_point))
            if additions:
                merged = merge_breakpoints(
                    target_S[dim],
                    *additions,
                    preserve_dirac=True,
                )
                if merged != target_S[dim]:
                    target_S[dim] = merged
                    changed = True
        if changed:
            result_E[edge_index] = mud.align(tuple(target_S))

    return BGD(result_E, shape.alpha, shape.beta).standardize(
        skip_static_zero=False
    ).align_center_subdivisions()


def _loop_shape_local_dirac(
    shape: BGD,
    edge_index: tuple[int, ...],
    dim: int,
    point: Fraction,
) -> Fraction | None:
    """Map one global atom to the local coordinate of a BGD direction block."""

    center_index = (1,) * shape.ndim
    if edge_index == center_index:
        if shape.center_lefts[dim] <= point <= shape.center_rights[dim]:
            return point
        return None

    direction = shape.index_to_direction(edge_index)[dim]
    if direction < 0:
        period = shape.left_lengths[dim]
        if period <= 0 or point >= shape.center_lefts[dim]:
            return None
        distance = shape.center_lefts[dim] - point
        phase = distance % period
        return Fraction(0) if phase == 0 else period - phase
    if direction > 0:
        period = shape.right_lengths[dim]
        if period <= 0 or point <= shape.center_rights[dim]:
            return None
        return (point - shape.center_rights[dim]) % period

    if shape.center_lefts[dim] <= point <= shape.center_rights[dim]:
        return point - shape.center_lefts[dim]
    return None


def _global_breakpoints_in_range(
    distribution: BGD,
    dim: int,
    low: Fraction,
    high: Fraction,
) -> set[Fraction]:
    points = set()
    for edge_index in np.ndindex(distribution.E.shape):
        mud = distribution.E[edge_index]
        direction = distribution.index_to_direction(edge_index)
        side = direction[dim]
        if edge_index == (1,) * distribution.ndim:
            offsets = (Fraction(0),)
        elif side < 0:
            length = distribution.left_lengths[dim]
            if length <= 0:
                continue
            first = (distribution.center_lefts[dim] - high) // length
            last = (distribution.center_lefts[dim] - low) // length + 1
            offsets = tuple(
                distribution.center_lefts[dim] - block * length
                for block in range(max(1, first), max(1, last) + 1)
            )
        elif side > 0:
            length = distribution.right_lengths[dim]
            if length <= 0:
                continue
            first = (low - distribution.center_rights[dim]) // length
            last = (high - distribution.center_rights[dim]) // length + 1
            offsets = tuple(
                distribution.center_rights[dim] + block * length
                for block in range(max(0, first), max(0, last) + 1)
            )
        else:
            offsets = (distribution.center_lefts[dim],)
        for offset in offsets:
            for point in mud.S[dim]:
                global_point = point + offset
                if low <= global_point <= high:
                    points.add(global_point)
    return points


def _global_dirac_points(distribution: BGD, dim: int) -> set[Fraction]:
    points = set()
    center_index = (1,) * distribution.ndim
    for edge_index in np.ndindex(distribution.E.shape):
        mud = distribution.E[edge_index]
        direction = distribution.index_to_direction(edge_index)
        if edge_index == center_index:
            offset = Fraction(0)
        elif direction[dim] < 0:
            offset = (
                distribution.center_lefts[dim]
                - distribution.left_lengths[dim]
            )
        elif direction[dim] > 0:
            offset = distribution.center_rights[dim]
        else:
            offset = distribution.center_lefts[dim]
        for cell_index, (left, right) in enumerate(
            zip(mud.S[dim], mud.S[dim][1:])
        ):
            if left != right:
                continue
            slicer = [slice(None)] * mud.ndim
            slicer[dim] = cell_index
            if any(
                not mud.ops.is_static_zero(value)
                for value in np.asarray(
                    mud.P[tuple(slicer)],
                    dtype=object,
                ).flat
            ):
                points.add(offset + left)
    return points


def _positive_fraction(value, name: str) -> Fraction:
    value = _as_fraction(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = (
    "Assignment",
    "PolynomialProgramResult",
    "ReplaceDistributionAssignment",
    "ShiftAssignment",
    "UniformConvolutionAssignment",
    "build_polynomial_program_semantics",
    "choice_probability",
    "classify_assignment",
    "condition_intervals",
    "default_polynomial_variable_bounds",
    "evaluate_polynomial_bgd",
    "numeric_literal_value",
    "polynomial_bgd_mass",
    "restrict_intervals",
)
