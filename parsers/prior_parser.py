import re
from fractions import Fraction
from typing import Union
from parsers.parser_utils import parse_number


def parse_mapping_string(s: str):
    """
    Parse a mapping string into a Python dict.
    Keys may be numbers or tuples of numbers. Values must be numbers.
    """

    s = "".join(s.split())
    if s == "":
        return {}

    if s[0] != "{" or s[-1] != "}":
        raise ValueError("Mapping string must be enclosed in {...}")

    inner = s[1:-1]
    if inner == "":
        return {}

    result = {}
    for item in inner.split(","):
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid mapping format: {s}")

        k = parse_number(parts[0])
        v = parse_number(parts[1])
        if v < 0:
            raise ValueError("Probability values must be nonnegative.")
        elif v > 0:
            # Drop entries with zero probability.
            result[k] = v

    if not result:
        result = {0: 0}

    return result


def parse_prior_line(line: str) -> tuple[tuple[str, ...], tuple]:
    """
    Parse one prior assignment line, e.g.
      "x=0"
      "x~Normal(0,1)"
      "x~Uniform(0,1)"
      "x~Exponential(1)"
      "x~{0:0.2,1:0.5,3:0.3}"

    Returns: (vars_tuple, dist_obj)
    """

    line = "".join(line.split())
    if not line:
        return tuple(), None

    # split into LHS and RHS around '=' or '~'
    lhs, rhs = re.split(r"[=~]", line)
    if not lhs:
        raise ValueError("Missing variable(s) on the left-hand side.")

    # local registry (edit here when adding new distributions)
    DIST_NAMES = ("Normal", "Uniform", "Exponential")

    # ensure no more than one distribution name occurs
    hits = [name for name in DIST_NAMES if name in rhs]
    if len(hits) > 1:
        raise ValueError("A line must not contain more than one distribution name.")
    dist_name = hits[0] if hits else None

    vars_tuple = tuple(v for v in lhs.split(",") if v)
    if not vars_tuple:
        raise ValueError("No variables found on the left-hand side.")

    if dist_name:
        rhs = rhs.replace(dist_name, "", 1)
        if not (rhs.startswith("(") and rhs.endswith(")")):
            raise ValueError(f"Expected '{dist_name}(...)'.")
        args_str = rhs[1:-1]
        args = parse_object_sequence_string(args_str)
        dist_obj = (dist_name, args)

    else:
        if '{' in rhs:
            mapping = parse_mapping_string(rhs)
            if not mapping:
                raise ValueError("Discrete distribution mapping must not be empty.")
            dist_obj = ('Dict', mapping)
        
        else:
            dist_obj = ('Num', parse_number(rhs))

    return vars_tuple, dist_obj


def parse_prior(prior: str):
    """Parse the prior section into a dict mapping vars_tuple to a distribution instance."""

    prior_items = [x for x in re.split(r"[\n;]+", prior) if x.strip()]
    prior_dict = {}

    for item in prior_items:
        vars_tuple, dist_obj = parse_prior_line(item)
        if vars_tuple:
            prior_dict[vars_tuple] = dist_obj

    return prior_dict
