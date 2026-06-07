import re
from parsers.parser_utils import parse_number
from probably.pgcl import parse_pgcl
from probably.pgcl.ast import Program


def replace_distributions(code: str) -> tuple[str, dict[str, tuple]]:
    """
    Replace distribution constructor calls such as `Normal(...)`, `Uniform(...)`,
    and `Exponential(...)` with placeholders `distribution_i`.
    """

    # Local registry (edit here when adding new distributions).
    DIST_NAMES = ("Normal", "Uniform", "Exponential")

    name_pattern = "|".join(re.escape(name) for name in DIST_NAMES)
    pattern = re.compile(rf"\b({name_pattern})\s*\(\s*([^()]*)\s*\)")

    distribution_map: dict[str, tuple] = {}
    counter = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal counter

        dist_name = match.group(1)
        args_str = match.group(2)
        
        args_str = "".join(args_str.split())
        args = tuple(parse_number(x) for x in args_str.split(","))
        dist_obj = (dist_name, args)

        placeholder = f"distribution_{counter}"
        distribution_map[placeholder] = dist_obj
        counter += 1
        return placeholder

    new_code = pattern.sub(repl, code)
    return new_code, distribution_map


def parse_program(program_str: str) -> tuple[Program, dict[str, tuple]]:
    """
    Parse a program str into a pGCL program.
    """

    program_str, distribution_map = replace_distributions(program_str)
    prog = parse_pgcl(program_str)
    return prog, distribution_map
