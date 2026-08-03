from __future__ import annotations

import argparse
import json
import sys
from math import comb, factorial
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyzer import ProgramStructure
from main import (
    get_solver_config,
    get_template_config,
    load_config,
    solve_with_config,
)
from visualize_bgd import _eval_bgd_at


def irwin_hall_density(count: int, value: float) -> float:
    if value <= 0 or value >= count:
        return 0.0
    return sum(
        (-1) ** index
        * comb(count, index)
        * (value - index) ** (count - 1)
        for index in range(int(value) + 1)
    ) / factorial(count - 1)


def exact_exit_density(x: float) -> float:
    """Density at y=2 for the PLDI22 add_uniform benchmark."""

    result = 0.0
    for failures in range(int(2 * x) + 1):
        uniform_count = failures + 2
        irwin_hall_value = 2 * x - failures
        failure_probability = (failures + 1) / 2 ** (failures + 2)
        result += (
            failure_probability
            * 2
            * irwin_hall_density(uniform_count, irwin_hall_value)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bgd_config.json")
    parser.add_argument("--xmax", type=float, default=8)
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()
    if args.xmax <= 0 or args.samples <= 0:
        parser.error("xmax and samples must be positive")

    config = load_config(args.config)
    template = get_template_config(config)
    source = (
        ROOT / "benchmarks" / "PLDI22" / "add_uniform.txt"
    ).read_text(encoding="utf-8")
    program = ProgramStructure(
        source,
        loop_unroll_iterations=template["loop_unroll_iterations"],
        polynomial_loop_degree=template["polynomial_loop_degree"],
        polynomial_loop_degree_increment=template[
            "polynomial_loop_degree_increment"
        ],
    )
    upper = solve_with_config(program, get_solver_config(config))

    worst = {
        "margin": float("inf"),
        "x": None,
        "exact_density": None,
        "upper_density": None,
    }
    for index in range(args.samples):
        x = (index + 0.5) * args.xmax / args.samples
        exact = exact_exit_density(x)
        bound = _eval_bgd_at(upper, [2, x], value="density")
        margin = bound - exact
        if margin < worst["margin"]:
            worst = {
                "margin": margin,
                "x": x,
                "exact_density": exact,
                "upper_density": bound,
            }

    mass = upper.mass()
    if not mass.is_constant:
        raise RuntimeError("solved upper bound still has symbolic mass")
    print(
        json.dumps(
            {
                "upper_mass": float(mass.constant_value),
                "xmax": args.xmax,
                "samples": args.samples,
                "worst_sample": worst,
                "sampled_upper_bound_valid": worst["margin"] >= -1e-6,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
