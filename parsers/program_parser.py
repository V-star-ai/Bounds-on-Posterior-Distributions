import re
from typing import Any
from parsers.parser_utils import parse_object_sequence_string
from distributions import Normal, Uniform, Exponential
from probably.pgcl import parse_pgcl
from probably.pgcl.ast import Program


def replace_distributions(code: str) -> tuple[str, dict[str, Any]]:
    """
    Replace distribution constructor calls such as `Normal(...)`, `Uniform(...)`,
    and `Exponential(...)` with fresh placeholder variables `distribution_i`.

    The returned mapping stores the corresponding dist objects for each placeholder.
    """

    # Local registry (edit here when adding new distributions).
    DIST_NAMES = ("Normal", "Uniform", "Exponential")

    name_pattern = "|".join(re.escape(name) for name in DIST_NAMES)
    pattern = re.compile(rf"\b({name_pattern})\s*\(\s*([^()]*)\s*\)")

    distribution_map: dict[str, Any] = {}
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


def collect_continualized_vars(syntax_tree: Any, conted_vars: set[str]):
    """
    Traverse pGCL syntax tree, collect variables that are continualized.
    """

    if isinstance(syntax_tree, list):
        for item in syntax_tree:
            collect_continualized_vars(item, conted_vars)

    elif isinstance(syntax_tree, WhileInstr):
        collect_continualized_vars(syntax_tree.body, conted_vars)

    elif isinstance(syntax_tree, IfInstr):
        collect_continualized_vars(syntax_tree.true, conted_vars)
        collect_continualized_vars(syntax_tree.false, conted_vars)

    elif isinstance(syntax_tree, ChoiceInstr):
        collect_continualized_vars(syntax_tree.lhs, conted_vars)
        collect_continualized_vars(syntax_tree.rhs, conted_vars)

    elif isinstance(syntax_tree, AsgnInstr):
        rhs_expr = syntax_tree.rhs
        target_var = syntax_tree.lhs.var
        has_dist_placeholder = False

        # Check direct distribution placeholder VarExpr
        if isinstance(rhs_expr, VarExpr):
            if re.match(r"^distribution_\d+$", rhs_expr.var):
                has_dist_placeholder = True

        # Check distribution placeholder in BinopExpr lhs / rhs
        elif isinstance(rhs_expr, BinopExpr):
            if isinstance(rhs_expr.lhs, VarExpr) and re.match(r"^distribution_\d+$", rhs_expr.lhs.var):
                has_dist_placeholder = True
            if isinstance(rhs_expr.rhs, VarExpr) and re.match(r"^distribution_\d+$", rhs_expr.rhs.var):
                has_dist_placeholder = True

        # Add lhs variable to continualized set if placeholder exists
        if has_dist_placeholder:
            conted_vars.add(target_var)


def parse_program(program_str: str) -> tuple[dict, Program]:
    """
    Parse a program block of the DSL into a pGCL program.

    Distribution constructor calls are first replaced by placeholder variables so
    that `parse_pgcl` can parse the program. 
    """

    program_str, distribution_map = replace_distributions(program_str)
    prog = parse_pgcl(program_str)

    conted_vars = set()
    collect_continualized_vars(prog, conted_vars)
    
    return prog, distribution_map, conted_vars
