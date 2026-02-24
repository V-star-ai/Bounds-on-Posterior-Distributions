from typing import List, Union, Tuple, Dict, Any
from fractions import Fraction
import re
from probably.pgcl import parse_pgcl


def _split_top_level_commas(s: str) -> List[str]:
    """Split by commas at top level. Assumes no whitespace."""
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch == "[":
            depth += 1
            buf.append(ch)
        elif ch == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced ']'")
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if depth != 0:
        raise ValueError("Unbalanced '['")
    tail = "".join(buf)
    if tail:
        parts.append(tail)
    return parts


def _parse_item(tok: str) -> Union[str, List]:
    # tok has no whitespace
    if tok.startswith("["):
        if not tok.endswith("]"):
            raise ValueError(f"List token missing closing bracket: {tok}")
        inner = tok[1:-1]
        if inner == "":
            return []
        sub = _split_top_level_commas(inner)
        return [_parse_item(t) for t in sub]

    if tok == "":
        raise ValueError("Empty token")
        
    # if it looks like a fraction, just try Fraction
    if "/" in tok:
        return Fraction(tok)

    return tok


def parse_number_list_string(s: str) -> List[Union[str, List[str], List[List[str]]]]:
    """
    Parse a comma-separated string into a list of items.
    Elements are kept as strings, except tokens containing '/' which are parsed as Fraction.

    Example: "[[2.1]],[[1,2/3],[3,4.1]]" -> [[["2.1"]], [["1", Fraction(2, 3)], ["3", "4.1"]]]
    """
    parts = _split_top_level_commas(s)
    return [_parse_item(p) for p in parts]


def parse_prior_line(line_no_ws: str) -> Tuple[Tuple[str, ...], Any]:
    """
    Parse one prior assignment line，Assumes no whitespace, e.g.
      "x,y=Normal(0,1)"
      "z=UniformBox([[0,1]],[[1]])"
      "x,y,z=EventualExp([[0,1],[0,1/2]],[[0.2,0.1,0.3],[0.1,0.4,0.1]],[0.1,0.2],[0.3,0.4])"

    Returns: (vars_tuple, dist_instance)
    """
    # local registry (edit here when adding new distributions)
    DIST_NAMES = ("Normal", "UniformBox", "EventualExp")
    dist_constructors: Dict[str, Any] = {
        "Normal": Normal,
        "UniformBox": UniformBox,
        "EventualExp": EventualExp,
    }

    # 1) ensure exactly one distribution name occurs
    hits = [name for name in DIST_NAMES if name in line_no_ws]
    if len(hits) != 1:
        raise ValueError("Exactly one of Normal/UniformBox/EventualExp must appear in the line.")
    dist_name = hits[0]

    if line_no_ws.count(dist_name) != 1:
        raise ValueError("Distribution name must appear exactly once.")

    # 2) split into LHS and RHS around the distribution name
    lhs, rhs = line_no_ws.split(dist_name, 1)

    # lhs: "x,y=" -> "x,y"
    if "=" in lhs:
        lhs = lhs.split("=", 1)[0]
    if not lhs:
        raise ValueError("Missing variable(s) on the left-hand side.")

    vars_tuple = tuple(v for v in lhs.split(",") if v)
    if not vars_tuple:
        raise ValueError("No variables found on the left-hand side.")

    # 3) rhs: "(...)" required
    if not (rhs.startswith("(") and rhs.endswith(")")):
        raise ValueError(f"Expected '{dist_name}(...)'.")

    args_str = rhs[1:-1]

    # 4) parse argument list
    args = parse_number_list_string(args_str)

    # 5) instantiate
    dist_obj = dist_constructors[dist_name](*args)
    return vars_tuple, dist_obj

def parse_prior(prior):
    """
    Parse the raw 'prior' section body into a dict mapping vars_tuple to a distribution instance.
    """
    prior_items = re.split(r"[\n;]+", prior)
    prior_dict = {}
    for item in prior_items:
        item = re.sub(r"\s+", "", item)
        if not item:
            continue
        vars_tuple, dist_obj = parse_prior_line(item)
        prior_dict[vars_tuple] = dist_obj
    return prior_dict


def split_program(src: str) -> Tuple[str, str]:
    """
    Split a DSL source string into (prior, program);
    'prior:' and 'program:' are keywords and must each appear exactly once.
    """
    if src.count("prior:") != 1 or src.count("program:") != 1:
        raise ValueError("Keywords 'prior:' and 'program:' must each appear exactly once.")

    m = re.match(r"^\s*prior\s*:\s*(.*?)\s*program\s*:\s*(.*)", src, flags=re.DOTALL)
    if not m:
        raise ValueError("Invalid format: expected 'prior: ... program: ...'")

    prior = m.group(1)
    program = m.group(2)
    return prior, program

            
def parse_program(program_str):
    """
    Parse a full DSL source into (prior_dict, pgcl_program):
      - prior section is parsed by parse_prior()
      - program section is parsed by parse_pgcl()
      
    Example input:

    prior:
    x1 = Normal(0,1)
    x2 = UniformBox([[0,1]],[1])
    x3,x4 = EventualExp([[0,1/2],[0,1]],[[0.2,0.3,0.3],[0.1,0.5,0.2],[0.2,0.1,0.1]],[0.3,0.2],[0.4,0.6])
    program:
    while(x1<0){
      if(x2>0.5){
        x3:=x3+0.2
      }else{
        x4:=x4+0.3
      }
    }
    """
    prior, program = split_program(program_str)
    return parse_prior(prior), parse_pgcl(program)
