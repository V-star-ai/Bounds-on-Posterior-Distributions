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


def parse_src(src_str: str) -> Tuple[dict, Program]:
    """
    Parse a full DSL source into (prior_dict, pgcl_program), where prior_dict maps
    tuples of variable names to their initial priors, and pgcl_program is the parsed pGCL AST.

    Example input:

    prior:
    x1 = Normal(0,1)
    x2 = Uniform(0,1)
    x3 = Exponential(1)
    x4 = {0: 0.2, 1: 0.5, 2: 0.3}
    x5,x6 = EED([[0,1/2],[0,1]],[[0.2,0.3,0.3],[0.1,0.5,0.2],[0.2,0.1,0.1]],[0.3,0.2],[0.4,0.6])
    program:
    while(1/3){
        if(x3>0.5){
            x1:=Exponential(2)
            x2:=Normal(0,1)
        }else{
            x4:=x4+1
            x5:=Uniform(0,1)
        }
    }
    """

    prior_str, program_str = split_program(src_str)
    return parse_prior(prior_str), parse_program(program_str)
