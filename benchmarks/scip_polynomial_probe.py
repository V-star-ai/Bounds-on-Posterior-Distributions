from __future__ import annotations

import argparse
import json
import sys
from functools import reduce
from operator import mul
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantics import ConstraintContext, ParameterPolynomial, Relation, StatePolynomial
from solvers import SCIPPolynomialSolver


def box_peak_problem(ndim: int):
    context = ConstraintContext()
    t_variable = context.declare("t")
    t = ParameterPolynomial.variable(t_variable)
    coordinates = [
        StatePolynomial.variable(ndim, dim)
        for dim in range(ndim)
    ]
    peak = reduce(
        mul,
        (coordinate * (1 - coordinate) for coordinate in coordinates),
    )
    context.constrain_domain(
        StatePolynomial.constant(ndim, t) - peak,
        Relation.GE,
    )
    return context.build(), t_variable, t


def run_probe(ndim: int, time_limit: float, display: bool) -> dict:
    problem, t_variable, objective = box_peak_problem(ndim)
    result = SCIPPolynomialSolver(
        {t_variable: (0, 1)},
        factor_bound=10,
        strict_epsilon=1e-7,
        certificate_degree=2 * ndim,
        time_limit=time_limit,
        relative_gap=1e-5,
        feasibility_tolerance=1e-8,
        display=display,
    ).solve(problem, objective=objective)
    return {
        "dimension": ndim,
        "degree": 2 * ndim,
        "expected_optimum": 4 ** (-ndim),
        "status": result.status,
        "candidate": result.objective_value,
        "dual_bound": result.dual_bound,
        "relative_gap": result.relative_gap,
        "seconds": result.solve_time,
        "nodes": result.nodes,
        "parameter_variables": result.parameter_variable_count,
        "factor_variables": result.factor_variable_count,
        "constraints": result.constraint_count,
        "max_coefficient_residual": result.max_certificate_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe SCIP on multivariate polynomial box envelopes."
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[1, 2],
    )
    parser.add_argument("--time-limit", type=float, default=10)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    if any(dimension <= 0 for dimension in args.dimensions):
        parser.error("dimensions must be positive")
    if args.time_limit <= 0:
        parser.error("time-limit must be positive")

    results = [
        run_probe(dimension, args.time_limit, args.display)
        for dimension in args.dimensions
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
