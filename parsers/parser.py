import re
from typing import Any, Tuple
from probably.pgcl.ast import Program

from parsers.prior_parser import parse_prior
from parsers.program_parser import parse_program


def split_program(src_str: str) -> Tuple[str, str]:
    """
    Split a DSL source into (prior, program).

    The keywords 'prior:' and 'program:' must each appear exactly once.
    """

    if len(re.findall(r"\bprior\s*:", src_str)) != 1 or len(re.findall(r"\bprogram\s*:", src_str)) != 1:
        raise ValueError("Keywords 'prior:' and 'program:' must each appear exactly once.")

    m = re.match(r"^\s*prior\s*:\s*(.*?)\s*\bprogram\s*:\s*(.*)$", src_str, flags=re.DOTALL)
    if not m:
        raise ValueError("Invalid format: expected 'prior: ... program: ...'")

    prior_str = m.group(1)
    program_str = m.group(2)
    return prior_str, program_str


def parse_src(src_str: str) -> Tuple[dict, dict, Program]:
    """
    Parse a full DSL source into (prior_dict, distribution_map, pgcl_program), where prior_dict maps
    tuples of variable names to their initial priors, distribution_map maps placeholders in the program 
    to corresponding distribution instances, and pgcl_program is the parsed pGCL AST.
    """
    
    prior_str, program_str = split_program(src_str)
    prior_dict = parse_prior(prior_str)
    pgcl_program, distribution_map = parse_program(program_str)
    return prior_dict, pgcl_program, distribution_map
