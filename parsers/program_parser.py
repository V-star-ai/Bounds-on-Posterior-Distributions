import re
from typing import Any
from parsers.parser_utils import parse_object_sequence_string
from distributions import EED, Normal, Uniform, Exponential
from probably.pgcl import parse_pgcl


def replace_distributions(code: str) -> tuple[str, dict[str, Any]]:
    """
    Replace distribution constructor calls such as `Normal(...)`, `Uniform(...)`,
    and `Exponential(...)` with fresh placeholder variables `distribution_i`.

    The returned mapping stores the corresponding EED objects for each placeholder.
    """

    # Local registry (edit here when adding new distributions).
    DIST_NAMES = ("Normal", "Uniform", "Exponential")

    name_pattern = "|".join(re.escape(name) for name in DIST_NAMES)
    pattern = re.compile(rf"\b({name_pattern})\s*\(\s*([^()]*)\s*\)")

    distribution_map: dict[str, EED] = {}
    counter = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal counter

        dist_name = match.group(1)
        args_str = match.group(2)
        args_str = "".join(args_str.split())

        if dist_name == "Normal":
            args = parse_object_sequence_string(args_str)
            dist_obj = Normal(*args)

        elif dist_name == "Uniform":
            args = parse_object_sequence_string(args_str, {0: "fraction", 1: "fraction"})
            dist_obj = Uniform(*args)

        elif dist_name == "Exponential":
            args = parse_object_sequence_string(args_str)
            dist_obj = Exponential(*args)

        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")

        placeholder = f"distribution_{counter}"
        distribution_map[placeholder] = dist_obj
        counter += 1
        return placeholder

    new_code = pattern.sub(repl, code)
    return new_code, distribution_map


def parse_program(program_str: str) -> tuple[dict, Program]:
    """
    Parse a program block of the DSL into a pGCL program.

    Distribution constructor calls are first replaced by placeholder variables so
    that `parse_pgcl` can parse the program. 
    """

    program_str, distribution_map = replace_distributions(program_str)
    prog = parse_pgcl(program_str)
    
    return prog, distribution_map
