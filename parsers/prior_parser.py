import re
import numpy as np

from fractions import Fraction
from typing import Tuple, Union

from parsers.parser_utils import parse_number, parse_object_sequence_string, split_top_level
from distributions import Normal, Uniform, Exponential, EED


def parse_mapping_string(s: str):
    """
    Parse a mapping string into a Python dict.
    Keys may be numbers or tuples of numbers. Values must be numbers.

    Example: {1:2,0:1,(1,2):3/4}' -> {1: 2, 0: 1, (1, 2): Fraction(3, 4)}
    """

    s = "".join(s.split())
    if s == "":
        return {}

    if s[0] != "{" or s[-1] != "}":
        raise ValueError("Mapping string must be enclosed in {...}")

    inner = s[1:-1]
    if inner == "":
        return {}

    def parse_key(x: str) -> tuple[int, ...]:
        if not x:
            raise ValueError("Empty key encountered")

        if x[0] == "(" and x[-1] == ")":
            body = x[1:-1]
            raw = tuple(parse_object_sequence_string(body))
            if not raw:
                raise ValueError("Empty key encountered")
        else:
            raw = (parse_number(x),)

        if not all(
            isinstance(v, int)
            or (isinstance(v, float) and v.is_integer())
            or (isinstance(v, Fraction) and v.denominator == 1)
            for v in raw
        ):
            raise ValueError("The support of a discrete distribution must consist of integers.")

        return tuple(int(v) for v in raw)

    result = {}
    for item in split_top_level(inner):
        parts = split_top_level(item, sep=":")
        if len(parts) != 2:
            raise ValueError(f"Each mapping item must contain exactly one top-level ':', got {item}")

        k_str, v_str = parts
        k = parse_key(k_str)
        v = parse_number(v_str)

        if v < 0:
            raise ValueError("Probability values must be nonnegative.")
        elif v > 0:
            # Drop entries with zero probability.
            result[k] = v

    if not result:
        raise ValueError("An empty mapping cannot represent a distribution.")

    return result


def parse_prior_line(line: str) -> Tuple[Tuple[str, ...], Union[Normal, Uniform, Exponential, EED]]:
    """
    Parse one prior assignment line, e.g.
      "x=Normal(0,1)"
      "x=Uniform(0,1)"
      "x=Exponential(1)"
      "x={0:0.2,1:0.5,3:0.3}"
      "x,y=EED([[0,1],[0,1/2]],[[0.2,0.1,0.3],[0.1,0.4,0.1]],[0.1,0.2],[0.3,0.4])"

    Returns: (vars_tuple, dist_instance)
    """

    line = "".join(line.split())
    if not line:
        return tuple(), None

    # local registry (edit here when adding new distributions)
    DIST_NAMES = ("Normal", "Uniform", "Exponential", "EED")

    # split into LHS and RHS around '='
    lhs, rhs = line.split("=", 1)
    if not lhs:
        raise ValueError("Missing variable(s) on the left-hand side.")

    # ensure no more than one distribution name occurs
    hits = [name for name in DIST_NAMES if name in rhs]
    if len(hits) > 1:
        raise ValueError("A line must not contain more than one distribution name.")
    dist_name = hits[0] if hits else None

    vars_tuple = tuple(v for v in lhs.split(",") if v)
    if not vars_tuple:
        raise ValueError("No variables found on the left-hand side.")

    if dist_name is not None:
        rhs = rhs.replace(dist_name, "", 1)

        if not (rhs.startswith("(") and rhs.endswith(")")):
            raise ValueError(f"Expected '{dist_name}(...)'.")
        args_str = rhs[1:-1]

        if dist_name == "EED":
            args = parse_object_sequence_string(args_str, {0: "fraction"})
            n = len(args[0])

            if len(args) == 4:
                dist_obj = EED(*args)
            elif len(args) == 5:
                discrete_dims = set(args[4])
                args[4] = [any(x == i for x in discrete_dims) for i in range(n)]
                dist_obj = EED(*args)
            else:
                raise ValueError("EED expects 4 or 5 arguments in the input string.")

        elif dist_name == "Normal":
            args = parse_object_sequence_string(args_str)
            dist_obj = Normal(*args)

        elif dist_name == "Uniform":
            args = parse_object_sequence_string(args_str, {0: "fraction", 1: "fraction"})
            dist_obj = Uniform(*args)

        elif dist_name == "Exponential":
            args = parse_object_sequence_string(args_str)
            dist_obj = Exponential(*args)

    else:
        mapping = parse_mapping_string(rhs)

        if not mapping:
            raise ValueError("Discrete distribution mapping must not be empty.")

        # Infer dimension
        n = len(next(iter(mapping)))

        # Build S from the exact per-dimension support values.
        # Geometric tails outside the listed support are represented by alpha/beta.
        S = [sorted({pt[i] for pt in mapping}) for i in range(n)]
        P = np.zeros(tuple(len(si) for si in S), dtype=float)

        # Fill probability table
        for pt, prob in mapping.items():
            idx = tuple(S[i].index(pt[i]) for i in range(n))
            P[idx] = prob

        dist_obj = EED(S, P, [0] * n, [0] * n, [True] * n)

    return vars_tuple, dist_obj


def parse_prior(prior):
    """Parse the prior section into a dict mapping vars_tuple to a distribution instance."""

    prior_items = re.split(r"[\n;]+", prior)
    prior_dict = {}

    for item in prior_items:
        vars_tuple, dist_obj = parse_prior_line(item)
        if vars_tuple:
            prior_dict[vars_tuple] = dist_obj

    return prior_dict
