import re
from typing import Any

from parsers.parser_utils import parse_object_sequence_string
from distributions import EED, Normal, Uniform, Exponential

from probably.pgcl import parse_pgcl
from probably.pgcl.ast import (
    Program,
    AsgnInstr,
    ChoiceInstr,
    IfInstr,
    VarExpr,
    WhileInstr,
)


def replace_distributions(code: str) -> tuple[str, dict[str, EED]]:
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
            dist_obj = Normal(*args).to_eed()

        elif dist_name == "Uniform":
            args = parse_object_sequence_string(args_str, {0: "fraction", 1: "fraction"})
            dist_obj = Uniform(*args).to_eed()

        elif dist_name == "Exponential":
            args = parse_object_sequence_string(args_str)
            dist_obj = Exponential(*args).to_eed()

        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")

        placeholder = f"distribution_{counter}"
        distribution_map[placeholder] = dist_obj
        counter += 1
        return placeholder

    new_code = pattern.sub(repl, code)
    return new_code, distribution_map


def reverse_replace_distributions(syntax_tree: Any, distribution_map: dict[str, EED]) -> None:
    """
    Reverse the placeholder substitution performed by `replace_distributions`.

    This traverses the parsed pGCL syntax tree and replaces RHS placeholder
    variables such as `distribution_0` with the corresponding EED objects.
    """
    
    if isinstance(syntax_tree, list):
        for item in syntax_tree:
            reverse_replace_distributions(item, distribution_map)

    elif isinstance(syntax_tree, WhileInstr):
        reverse_replace_distributions(syntax_tree.body, distribution_map)

    elif isinstance(syntax_tree, IfInstr):
        reverse_replace_distributions(syntax_tree.true, distribution_map)
        reverse_replace_distributions(syntax_tree.false, distribution_map)

    elif isinstance(syntax_tree, ChoiceInstr):
        reverse_replace_distributions(syntax_tree.lhs, distribution_map)
        reverse_replace_distributions(syntax_tree.rhs, distribution_map)

    elif isinstance(syntax_tree, AsgnInstr):
        # Only replace if RHS is a placeholder variable: VarExpr("distribution_i").
        if isinstance(syntax_tree.rhs, VarExpr) and re.match(r"^distribution_\d+$", syntax_tree.rhs.var):
            syntax_tree.rhs = distribution_map[syntax_tree.rhs.var]


def parse_program(program_str: str) -> Program:
    """
    Parse a program block of the DSL into a pGCL program.

    Distribution constructor calls are first replaced by placeholder variables so
    that `parse_pgcl` can parse the program. After parsing, the placeholders are
    substituted back with their corresponding EED objects.
    """

    program_str, distribution_map = replace_distributions(program_str)
    prog = parse_pgcl(program_str)
    reverse_replace_distributions(prog.instructions, distribution_map)

    return prog
